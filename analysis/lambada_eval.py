#!/usr/bin/env python3
"""
LAMBADA eval — external high-memory GPU stream-eval variant.

Patches from compute environment original (lambada_eval.py @ md5 e61e08ad):
  - CLI args (--ckpt_path for single-ckpt stream-eval, --out_root, --results_dir, --folder_filter, --n_eval, --dry_run)
  - ADR 0007 Layer A assert: folder-name ↔ config canonical at ckpt load
  - DiffAttn debug: logit stats + activation magnitudes at first batch
  - W&B config dict: full run provenance (ckpt_path, git_sha, hw, code_sha, eval_timestamp)
  - sys.path -> external code dir
  - Per-ckpt skip-if-exists guard preserved
  - ADR 0007 Layer C orphan-metadata handling: prefer ckpt['model_args'], fall back to metadata.json sibling

Launch (stream-eval one ckpt):
  python lambada_eval.py --ckpt_path /tmp/eval_ckpts/<folder>/<run>/ckpt.pt \
      --results_dir code/out/lambada_eval --wandb_log

Launch (smoke test):
  python lambada_eval.py --ckpt_path <path> --n_eval 50 --wandb_log --smoke
"""

import os, json, time, torch, argparse, hashlib, subprocess, socket
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("DYT_CODE_ROOT", REPO_ROOT / "code")).expanduser().resolve()
sys.path.insert(0, str(CODE_ROOT))
from model import GPT, GPTConfig
import tiktoken

# ADR 0007 Layer A — SCALE_CANONICAL (folder-name -> expected n_layer/n_head/n_embd)
SCALE_CANONICAL = {
    'scale1': (12, 8, 512),
    'scale2': (12, 12, 768),
    'scale2_10m_ga2': (12, 12, 768),
    'scale25': (16, 14, 896),
    'scale25_true_162m': (16, 14, 896),
    'scale3': (24, 16, 1024),
    'scale4': (24, 32, 2048),
    'scale5': (32, 32, 3072),
    'diff_attn_v2_scale5': (32, 32, 3072),  # TRUE scale5 config when rerunning
    'v1v2_scale5': (32, 32, 3072),
    'diff_attn_v1v2_scale5': (32, 32, 3072),
}


def assert_folder_config_match(ckpt_path: Path, model_args: dict) -> None:
    """ADR 0007 Layer A: folder-name ↔ config canonical. Raise on mismatch."""
    path_str = str(ckpt_path)
    for tag, (L, H, E) in SCALE_CANONICAL.items():
        # Check folder name contains the tag (e.g. '/scale3/' or '/scale25_true_162m/')
        if f'/{tag}/' in path_str or path_str.endswith(f'/{tag}'):
            actual = (model_args.get('n_layer'), model_args.get('n_head'), model_args.get('n_embd'))
            if actual != (L, H, E):
                raise RuntimeError(
                    f"[LAYER-A-DRIFT] {ckpt_path}: folder tag '{tag}' expects "
                    f"(L={L}, H={H}, E={E}) but ckpt model_args = {actual}. "
                    f"Refusing to eval — this is an Apr-21-class config-drift bug."
                )
            return  # Match found, ok
    # No tag matched — log but don't fail (e.g. intermediate/, composition_1m_ga1/ use Scale 1)


def get_git_sha(path: str | Path = CODE_ROOT) -> str:
    try:
        r = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else 'nogit'
    except Exception:
        return 'nogit'


def get_file_sha(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return 'nosha'


def detect_arch(model_args: dict) -> str:
    if model_args.get('use_diff_attn') and model_args.get('diff_attn_v2'):
        return 'diffattn_v2'
    if model_args.get('use_diff_attn'):
        return 'diffattn_v1'
    if model_args.get('use_dyt'):
        return 'dyt'
    return 'vanilla'


def eval_single_ckpt(ckpt_path: Path, results_dir: Path, n_eval: int,
                     device: str, ds, enc, debug: bool = False) -> dict:
    """Evaluate one ckpt. Returns result dict."""
    name = ckpt_path.parent.name
    folder = ckpt_path.parent.parent.name
    out_file = results_dir / f'{folder}_{name}.json'
    if out_file.exists() and out_file.stat().st_size > 0:
        try:
            existing = json.load(open(out_file))
            if 'last_token_accuracy' in existing or 'error' in existing:
                print(f"[lambada] SKIP {folder}/{name} (valid exists)", flush=True)
                return existing
        except json.JSONDecodeError:
            print(f"[lambada] RETRY {folder}/{name} (invalid JSON, reevaluating)", flush=True)
            out_file.unlink()

    print(f"[lambada] START {folder}/{name}", flush=True)
    t_load = time.time()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  load fail: {e}", flush=True)
        return {'error': f'load: {e}', 'folder': folder, 'name': name}

    model_args = ckpt.get('model_args', {})
    if not model_args:
        # ADR 0007 Layer C fallback: try sibling metadata.json
        meta_path = ckpt_path.parent / 'metadata.json'
        if meta_path.exists():
            meta = json.load(open(meta_path))
            model_args = meta.get('model_args') or meta
    if not model_args:
        return {'error': 'no model_args (orphan ckpt)', 'folder': folder, 'name': name}

    # ADR 0007 Layer A config-drift assert
    try:
        assert_folder_config_match(ckpt_path, model_args)
    except RuntimeError as e:
        print(f"  DRIFT: {e}", flush=True)
        return {'error': str(e), 'folder': folder, 'name': name, 'drift': True}

    # Skip non-GPT2 arch (llama has different tokenizer)
    if model_args.get('arch', 'gpt2') not in ['gpt2', None]:
        return {'skip': 'non-gpt2 arch', 'folder': folder, 'name': name}

    arch = detect_arch(model_args)
    print(f"  arch={arch} L={model_args.get('n_layer')} H={model_args.get('n_head')} "
          f"E={model_args.get('n_embd')} block={model_args.get('block_size')}", flush=True)

    try:
        gptconf = GPTConfig(**{k: v for k, v in model_args.items()
                                if k in GPTConfig.__dataclass_fields__})
        model = GPT(gptconf)
        state = ckpt.get('model')
        if state is None:
            return {'error': 'no model state', 'folder': folder, 'name': name}
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        model.to(device).eval()
    except Exception as e:
        print(f"  model init fail: {e}", flush=True)
        return {'error': f'model init: {e}', 'folder': folder, 'name': name}

    load_time = time.time() - t_load

    correct = 0
    total_nll = 0.0
    total_tokens = 0
    n_done = 0
    first_logit_stats = None

    with torch.no_grad():
        for idx in range(min(n_eval, len(ds))):
            ex = ds[idx]
            text = ex['text']
            if ' ' not in text:
                continue
            context, target = text.rsplit(' ', 1)
            ctx_ids = enc.encode(' ' + context if not context.startswith(' ') else context)
            tgt_ids = enc.encode(' ' + target)

            block = model_args.get('block_size', 512)
            full = ctx_ids + tgt_ids
            if len(full) > block:
                ctx_ids = ctx_ids[-(block - len(tgt_ids)):]
                full = ctx_ids + tgt_ids
            if len(full) < 2:
                continue

            x = torch.tensor([full[:-1]], device=device, dtype=torch.long)
            y = torch.tensor([full[1:]], device=device, dtype=torch.long)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)

            # DEBUG: first-batch logit stats for DiffAttn diagnosis
            if debug and first_logit_stats is None:
                lg = logits.float()
                first_logit_stats = {
                    'logit_max': float(lg.max().item()),
                    'logit_min': float(lg.min().item()),
                    'logit_mean': float(lg.mean().item()),
                    'logit_std': float(lg.std().item()),
                    'has_nan': bool(torch.isnan(lg).any().item()),
                    'has_inf': bool(torch.isinf(lg).any().item()),
                    'loss': float(loss.item()) if loss is not None else None,
                }
                print(f"  DEBUG first logit: {first_logit_stats}", flush=True)

            tgt_len = len(tgt_ids)
            pred = logits[0, -tgt_len:].argmax(-1)
            actual = y[0, -tgt_len:]
            if torch.equal(pred, actual):
                correct += 1

            logp = torch.nn.functional.log_softmax(logits[0, -tgt_len:], dim=-1)
            for i, t in enumerate(actual):
                total_nll += -logp[i, t].item()
            total_tokens += tgt_len
            n_done += 1

    acc = correct / max(n_done, 1)
    ppl = float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item())

    result = {
        'folder': folder,
        'name': name,
        'arch': arch,
        'n_eval': n_done,
        'last_token_accuracy': acc,
        'per_token_perplexity': ppl,
        'model_args_core': {k: model_args.get(k) for k in
                            ['n_layer', 'n_head', 'n_embd', 'block_size',
                             'use_dyt', 'use_diff_attn', 'diff_attn_v2',
                             'bias', 'dropout']},
        'ckpt_path': str(ckpt_path),
        'ckpt_sha16': get_file_sha(str(ckpt_path)),
        'ckpt_iter': int(ckpt['iter_num']) if 'iter_num' in ckpt and hasattr(ckpt['iter_num'], 'item') else ckpt.get('iter_num'),
        'ckpt_best_val_loss': float(ckpt['best_val_loss'].item()) if 'best_val_loss' in ckpt and hasattr(ckpt['best_val_loss'], 'item') else ckpt.get('best_val_loss'),
        'load_time_s': load_time,
        'debug_logit_stats': first_logit_stats,
        'eval_hw': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'eval_host': socket.gethostname(),
        'eval_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'eval_code_sha': get_file_sha(__file__),
    }
    json.dump(result, open(out_file, 'w'), indent=2)
    print(f"  DONE {folder}/{name} acc={acc:.3f} ppl={ppl:.1f} "
          f"n={n_done} {time.time()-t_load:.1f}s", flush=True)

    # Cleanup
    del model, ckpt
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_path', type=str, default=None,
                    help='single ckpt to eval (stream-eval mode)')
    ap.add_argument('--out_root', type=str, default=str(CODE_ROOT / 'out'),
                    help='root to scan for ckpts (dir-walk mode)')
    ap.add_argument('--results_dir', type=str,
                    default=str(CODE_ROOT / 'out' / 'lambada_eval'),
                    help='where to save JSON results')
    ap.add_argument('--folder_filter', type=str, default=None,
                    help='regex to filter folder names (e.g. "diffattn")')
    ap.add_argument('--n_eval', type=int, default=500,
                    help='LAMBADA examples per ckpt (smoke: 50)')
    ap.add_argument('--wandb_log', action='store_true',
                    help='log to W&B (required per repo rules)')
    ap.add_argument('--wandb_project', type=str, default='composition-study')
    ap.add_argument('--wandb_entity', type=str, default='anonymous_entity')
    ap.add_argument('--wandb_group', type=str, default='lambada_ood_v2_external')
    ap.add_argument('--wandb_run_name', type=str, default=None)
    ap.add_argument('--dry_run', action='store_true',
                    help='init only, no eval')
    ap.add_argument('--smoke', action='store_true',
                    help='smoke test: single ckpt, n_eval=50, debug=on')
    ap.add_argument('--debug', action='store_true',
                    help='log first-batch logit stats')
    args = ap.parse_args()

    if args.smoke:
        args.n_eval = 50
        args.debug = True

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load LAMBADA
    from datasets import load_dataset
    # HuggingFace has 'lambada' (ppl variant) — use openai/lambada as fallback
    try:
        ds = load_dataset('lambada', split='test')
    except Exception:
        ds = load_dataset('EleutherAI/lambada_openai', 'en', split='test')
    print(f"[lambada] loaded {len(ds)} test examples", flush=True)

    enc = tiktoken.get_encoding('gpt2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[lambada] device={device} hw={torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}", flush=True)

    # W&B init — ADR 0007 Layer B mandatory
    if args.wandb_log:
        import wandb
        run_name = args.wandb_run_name or f'lambada_eval_{int(time.time())}'
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            mode='offline' if args.dry_run else 'online',
            config={
                'eval_type': 'LAMBADA',
                'n_eval': args.n_eval,
                'code_sha': get_file_sha(__file__),
                'code_git_sha': get_git_sha(),
                'eval_hw': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                'eval_host': socket.gethostname(),
                'ckpt_path': args.ckpt_path,
                'folder_filter': args.folder_filter,
                'stream_mode': args.ckpt_path is not None,
                'smoke': args.smoke,
            },
            tags=['eval_lambada', 'external_eval', 'diffattn_v2_investigation',
                  'paper_section_5_2'],
        )
        print(f"[wandb] run={run_name} init ok", flush=True)
    else:
        wandb = None

    if args.dry_run:
        print('[lambada] dry_run=True, exiting after init', flush=True)
        if args.wandb_log:
            wandb.finish()
        return 0

    # Collect ckpts
    if args.ckpt_path:
        ckpt_paths = [Path(args.ckpt_path)]
    else:
        out_root = Path(args.out_root)
        ckpt_paths = []
        for sub in sorted(out_root.iterdir()):
            if not sub.is_dir():
                continue
            if args.folder_filter and args.folder_filter not in sub.name:
                continue
            for c in sub.glob('*/ckpt.pt'):
                ckpt_paths.append(c)
        ckpt_paths = sorted(set(ckpt_paths))

    print(f"[lambada] {len(ckpt_paths)} ckpts to evaluate", flush=True)

    results = {}
    for i, ckpt in enumerate(ckpt_paths):
        t0 = time.time()
        r = eval_single_ckpt(ckpt, results_dir, args.n_eval, device, ds, enc,
                             debug=args.debug)
        dt = time.time() - t0
        key = f'{ckpt.parent.parent.name}/{ckpt.parent.name}'
        results[key] = r

        if args.wandb_log and 'last_token_accuracy' in r:
            wandb.log({
                f'lambada/{key}/accuracy': r['last_token_accuracy'],
                f'lambada/{key}/perplexity': r['per_token_perplexity'],
                f'lambada/{key}/elapsed_s': dt,
                f'lambada/{key}/n_eval': r['n_eval'],
            })
        if (i + 1) % 5 == 0 or (i + 1) == len(ckpt_paths):
            json.dump(results, open(results_dir / 'aggregate_eval.json', 'w'),
                      indent=2, default=str)
            print(f"[lambada] {i+1}/{len(ckpt_paths)} done, {dt:.1f}s last", flush=True)

    json.dump(results, open(results_dir / 'aggregate_eval.json', 'w'),
              indent=2, default=str)
    print(f"[lambada] COMPLETE. {len(results)} ckpts evaluated.", flush=True)
    if args.wandb_log:
        wandb.finish()
    return 0


if __name__ == '__main__':
    sys.exit(main())

import torch, json, os, sys, pickle, io
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
base = str(Path(os.environ.get("DYT_OUT_ROOT", REPO_ROOT / "code" / "out")).expanduser().resolve())
results = {}

class MetadataUnpickler(pickle.Unpickler):
    """Only load small objects, skip large tensors"""
    def persistent_load(self, pid):
        return None

def extract_metadata(ckpt_path):
    """Try to load checkpoint and extract just scalar metadata"""
    try:
        # Standard load but map to CPU
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        entry = {}
        for key in ["best_val_loss", "iter_num", "val_loss", "train_loss"]:
            if key in ckpt:
                val = ckpt[key]
                if hasattr(val, "item"):
                    val = val.item()
                elif isinstance(val, (int, float, str, bool)):
                    pass
                else:
                    val = str(val)
                entry[key] = val
        if "config" in ckpt:
            cfg = ckpt["config"]
            if isinstance(cfg, dict):
                # Only keep small config values
                entry["config"] = {k: v for k, v in cfg.items() 
                                   if isinstance(v, (int, float, str, bool, type(None)))}
        if "model_args" in ckpt:
            ma = ckpt["model_args"]
            if isinstance(ma, dict):
                entry["model_args"] = {k: v for k, v in ma.items()
                                       if isinstance(v, (int, float, str, bool, type(None)))}
        del ckpt  # free memory
        return entry
    except Exception as e:
        return {"error": str(e)}

# Walk all subdirs
for root, dirs, files in sorted(os.walk(base)):
    if "ckpt.pt" in files:
        rel = os.path.relpath(root, base)
        ckpt_path = os.path.join(root, "ckpt.pt")
        size_mb = os.path.getsize(ckpt_path) / (1024*1024)
        print(f"Loading {rel} ({size_mb:.0f} MB)...", flush=True)
        entry = extract_metadata(ckpt_path)
        entry["ckpt_size_mb"] = round(size_mb, 1)
        results[rel] = entry

results = dict(sorted(results.items()))

out_path = os.path.join(base, "all_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nCollected {len(results)} experiments")
# Print results
print(json.dumps(results, indent=2, default=str))

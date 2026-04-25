"""Prepare wikitext-103 data + subsets for local experiments."""
import os
import numpy as np

def prepare():
    # Check if already prepared
    if os.path.exists('data/wikitext/train.bin'):
        train = np.fromfile('data/wikitext/train.bin', dtype=np.uint16)
        print(f'Wikitext already prepared: {len(train)/1e6:.1f}M tokens')
    else:
        import tiktoken
        from datasets import load_dataset
        enc = tiktoken.get_encoding('gpt2')
        print('Loading wikitext-103...')
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', trust_remote_code=True)
        os.makedirs('data/wikitext', exist_ok=True)
        for split_name, out_name in [('train', 'train'), ('validation', 'val')]:
            tokens = []
            for doc in ds[split_name]:
                if doc['text'].strip():
                    tokens.extend(enc.encode_ordinary(doc['text']))
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(f'data/wikitext/{out_name}.bin')
            print(f'  {out_name}: {len(arr)/1e6:.2f}M tokens')

    # Create subsets
    train = np.fromfile('data/wikitext/train.bin', dtype=np.uint16)
    val = np.fromfile('data/wikitext/val.bin', dtype=np.uint16)

    for size_m in [1, 10, 50]:
        dirname = f'data/wikitext_{size_m}m'
        os.makedirs(dirname, exist_ok=True)
        size = size_m * 1_000_000
        train_sub = train[:size]
        train_sub.tofile(f'{dirname}/train.bin')
        val_size = max(10000, size // 100)
        val_sub = val[:val_size]
        val_sub.tofile(f'{dirname}/val.bin')
        print(f'  {dirname}: train={len(train_sub)/1e6:.1f}M, val={len(val_sub)/1e6:.1f}M')

    print('Done!')

if __name__ == '__main__':
    prepare()

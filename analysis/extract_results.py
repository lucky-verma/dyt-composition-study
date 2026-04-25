import torch, json, os, glob, sys

base = "<CODE_ROOT>/out"
results = {}

# Walk all subdirs looking for ckpt.pt
for root, dirs, files in os.walk(base):
    if "ckpt.pt" in files:
        rel = os.path.relpath(root, base)
        ckpt_path = os.path.join(root, "ckpt.pt")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            entry = {}
            for key in ["best_val_loss", "iter_num", "val_loss", "train_loss", "config"]:
                if key in ckpt:
                    val = ckpt[key]
                    if hasattr(val, "item"):
                        val = val.item()
                    entry[key] = val
            # Also check for model_args
            if "model_args" in ckpt:
                entry["model_args"] = ckpt["model_args"]
            results[rel] = entry
        except Exception as e:
            results[rel] = {"error": str(e)}

# Sort by key
results = dict(sorted(results.items()))

with open(os.path.join(base, "all_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Collected {len(results)} experiments")
print(json.dumps(results, indent=2, default=str))

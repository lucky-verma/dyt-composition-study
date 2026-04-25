"""
ViT experiment: Does DyT's regime-dependent regularization hold for vision?
- CIFAR-10: 50K images, ViT-Small -> overfitting regime
- CIFAR-100: 50K images, 100 classes -> less overfitting (more classes)
- Compare: ViT with LayerNorm vs ViT with DyT (replace all LN with DyT)

OOM safe: ViT-Small ~22M params, ~1GB on H100.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import json, os, sys, gc, time, math

DEVICE = 'cuda'
DTYPE = torch.bfloat16

# --- DyT replacement for LayerNorm ---
class DynamicTanh(nn.Module):
    def __init__(self, ndim, alpha_init=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight + self.bias

# --- Minimal ViT ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_dyt=False, alpha_init=2.0):
        super().__init__()
        self.norm1 = DynamicTanh(dim, alpha_init) if use_dyt else nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = DynamicTanh(dim, alpha_init) if use_dyt else nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, dim=384, depth=12, heads=6, use_dyt=False, alpha_init=2.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, use_dyt=use_dyt, alpha_init=alpha_init) for _ in range(depth)])
        self.norm = DynamicTanh(dim, alpha_init) if use_dyt else nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.use_dyt = use_dyt

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x)

def train_vit(dataset_name, use_dyt, seed, epochs=100, batch_size=128, lr=1e-3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Data
    transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform_test = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    num_classes = 10 if dataset_name == 'cifar10' else 100
    DS = torchvision.datasets.CIFAR10 if dataset_name == 'cifar10' else torchvision.datasets.CIFAR100
    train_ds = DS(root='./data/vision', train=True, download=True, transform=transform_train)
    test_ds = DS(root='./data/vision', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Model: ViT-Small (depth=6, dim=256, heads=4) -- fits on any GPU
    model = SimpleViT(num_classes=num_classes, dim=256, depth=6, heads=4, use_dyt=use_dyt).to(DEVICE, dtype=DTYPE)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    config_name = "DyT" if use_dyt else "LayerNorm"
    print(f"[{dataset_name}/{config_name}/s{seed}] {params:.1f}M params, GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0; train_correct = 0; train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE, dtype=DTYPE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=DTYPE):
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        scheduler.step()

        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_loss = 0; val_correct = 0; val_total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(DEVICE, dtype=DTYPE), labels.to(DEVICE)
                    with torch.cuda.amp.autocast(dtype=DTYPE):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                    val_loss += loss.item() * imgs.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += imgs.size(0)

            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            entry = {'epoch': epoch+1, 'train_acc': round(train_acc,2), 'val_acc': round(val_acc,2),
                     'train_loss': round(train_loss/train_total,4), 'val_loss': round(val_loss/val_total,4)}
            history.append(entry)
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.1f}% val={val_acc:.1f}% train_loss={train_loss/train_total:.4f} val_loss={val_loss/val_total:.4f}", flush=True)

    return history, params

# --- Run all conditions ---
results = {}
os.makedirs('out/vit_experiments', exist_ok=True)

conditions = [
    ('cifar10', False, 1337),
    ('cifar10', True, 1337),
    ('cifar10', False, 42),
    ('cifar10', True, 42),
    ('cifar100', False, 1337),
    ('cifar100', True, 1337),
    ('cifar100', False, 42),
    ('cifar100', True, 42),
]

for dataset_name, use_dyt, seed in conditions:
    config = "dyt" if use_dyt else "ln"
    key = f"{dataset_name}_{config}_s{seed}"
    print(f"\n{'='*50}\n{key}\n{'='*50}", flush=True)

    try:
        history, params = train_vit(dataset_name, use_dyt, seed, epochs=100)
        results[key] = {'history': history, 'params': params, 'final_train_acc': history[-1]['train_acc'], 'final_val_acc': history[-1]['val_acc']}
        print(f"  FINAL: train={history[-1]['train_acc']:.1f}% val={history[-1]['val_acc']:.1f}%", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        results[key] = {'error': str(e)}
    finally:
        gc.collect(); torch.cuda.empty_cache()

with open('out/vit_experiments/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== VIT EXPERIMENTS COMPLETE ===")
# Summary
for key, data in results.items():
    if 'error' not in data:
        print(f"  {key}: train={data['final_train_acc']:.1f}% val={data['final_val_acc']:.1f}%")

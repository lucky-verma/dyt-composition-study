"""
ViT alpha sweep -- test if DyT works on ViT with different alpha values.
The default alpha=2.0 killed ViT. Try lower values.
"""
import torch, torch.nn as nn, torch.optim as optim, json, os, sys, gc
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T

DEVICE = 'cuda'
DTYPE = torch.bfloat16

class DynamicTanh(nn.Module):
    def __init__(self, ndim, alpha_init=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight + self.bias

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
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
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(self, dim, num_heads, use_dyt=False, alpha_init=2.0):
        super().__init__()
        self.norm1 = DynamicTanh(dim, alpha_init) if use_dyt else nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = DynamicTanh(dim, alpha_init) if use_dyt else nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SimpleViT(nn.Module):
    def __init__(self, num_classes=10, dim=256, depth=6, heads=4, use_dyt=False, alpha_init=2.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        num_patches = (32 // 4) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, use_dyt, alpha_init) for _ in range(depth)])
        self.norm = DynamicTanh(dim, alpha_init) if use_dyt else nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))

def train_vit(use_dyt, alpha_init, seed, epochs=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform_test = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_ds = torchvision.datasets.CIFAR10(root='./data/vision', train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root='./data/vision', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = SimpleViT(num_classes=10, use_dyt=use_dyt, alpha_init=alpha_init).to(DEVICE, dtype=DTYPE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    for epoch in range(epochs):
        model.train()
        train_correct = train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE, dtype=DTYPE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_correct += (model(imgs).argmax(1) == labels).sum().item()
            train_total += len(labels)
        scheduler.step()

        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(DEVICE, dtype=DTYPE), labels.to(DEVICE)
                    val_correct += (model(imgs).argmax(1) == labels).sum().item()
                    val_total += len(labels)
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            best_val = max(best_val, val_acc)
            config = f"DyT(a={alpha_init})" if use_dyt else "LN"
            print(f"  [{config}/s{seed}] Epoch {epoch+1}: train={train_acc:.1f}% val={val_acc:.1f}%", flush=True)
    return best_val

# Run alpha sweep for ViT
results = {}
os.makedirs('out/vit_alpha', exist_ok=True)

conditions = [
    (False, 0, "LN"),           # LayerNorm baseline
    (True, 0.1, "DyT_a0.1"),   # Very weak DyT
    (True, 0.3, "DyT_a0.3"),   # Weak DyT
    (True, 0.5, "DyT_a0.5"),   # DyT paper default
    (True, 1.0, "DyT_a1.0"),   # Our GPT sweet spot
    (True, 2.0, "DyT_a2.0"),   # What killed ViT before
]

for use_dyt, alpha, name in conditions:
    print(f"\n{'='*50}\n{name}\n{'='*50}", flush=True)
    try:
        best = train_vit(use_dyt, alpha, seed=1337, epochs=100)
        results[name] = {'best_val_acc': best, 'alpha': alpha, 'use_dyt': use_dyt}
        print(f"  BEST VAL: {best:.1f}%", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results[name] = {'error': str(e)}
    gc.collect(); torch.cuda.empty_cache()

with open('out/vit_alpha/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== VIT ALPHA SWEEP COMPLETE ===")
for name, data in results.items():
    if 'best_val_acc' in data:
        print(f"  {name}: {data['best_val_acc']:.1f}%")

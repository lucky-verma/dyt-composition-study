# Scale 4: 1.3B parameters (GPT-2 XL-like)
n_layer = 24
n_head = 32
n_embd = 2048
block_size = 512
batch_size = 4
gradient_accumulation_steps = 16
max_iters = 5000
eval_interval = 500
eval_iters = 50
learning_rate = 1e-4
compile = True
log_interval = 500
wandb_log = False
bias = False
dropout = 0.0
dtype = 'bfloat16'

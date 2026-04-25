# Scale 2: 124M parameters (GPT-2 architecture)
n_layer = 12
n_head = 12
n_embd = 768
block_size = 512
batch_size = 32
gradient_accumulation_steps = 2
max_iters = 5000
eval_interval = 500
eval_iters = 50
learning_rate = 3e-4
compile = True
log_interval = 500
wandb_log = False
bias = False
dropout = 0.0
dtype = 'bfloat16'

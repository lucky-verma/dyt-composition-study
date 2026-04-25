# Scale 3: 354M parameters (GPT-2 Medium)
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 512
batch_size = 16
gradient_accumulation_steps = 4
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

# Scale 5: 3.78B parameters, GPT-2-family stress-test configuration.
n_layer = 32
n_head = 32
n_embd = 3072
block_size = 512
batch_size = 1
gradient_accumulation_steps = 64
max_iters = 5000
eval_interval = 500
eval_iters = 50
learning_rate = 1e-4
compile = True
log_interval = 100
wandb_log = False
bias = False
dropout = 0.0
dtype = 'bfloat16'

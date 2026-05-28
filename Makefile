PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help install validate hf-validate smoke-model smoke-train-cpu data

help:
	@echo "Targets:"
	@echo "  make install          Install Python dependencies"
	@echo "  make validate         Run dependency-light artifact checks"
	@echo "  make hf-validate      Check live Hugging Face dataset metadata"
	@echo "  make smoke-model      Instantiate all model variants on CPU"
	@echo "  make smoke-train-cpu  Run a tiny CPU training smoke"
	@echo "  make data             Prepare WikiText data"

install:
	$(PIP) install -r requirements.txt

validate:
	$(PYTHON) scripts/validate_repo.py

hf-validate:
	$(PYTHON) scripts/validate_hf_dataset.py

smoke-model:
	$(PYTHON) scripts/smoke_model.py

smoke-train-cpu:
	@test -f code/data/wikitext_1m/train.bin || \
		(echo "Missing code/data/wikitext_1m/train.bin. Run 'make data' first."; exit 2)
	cd code && $(PYTHON) train.py --dataset=wikitext_1m \
		--n_layer=2 --n_head=2 --n_embd=128 --block_size=128 \
		--batch_size=4 --gradient_accumulation_steps=16 \
		--max_iters=20 --eval_interval=10 --eval_iters=2 \
		--learning_rate=3e-4 --compile=False --device=cpu \
		--wandb_log=False --out_dir=out/smoke

data:
	cd code && $(PYTHON) prepare_wikitext.py

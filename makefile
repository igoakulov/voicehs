.PHONY: infer clean

# ====== CONFIG ======
t ?= Hello, world!
CKPT_PATH := ./models/base/F5TTS_v1_Base_v2/model_last_inference.safetensors
OUT_DIR := ./output

# ====== COMMANDS ======

infer:
	@echo "Running F5-TTS inference..."
	@PYTHONWARNINGS=ignore f5-tts_infer-cli \
		--ckpt_file $(CKPT_PATH) \
		--gen_text "$(t)" \
		--output_dir $(OUT_DIR)
	@echo "✅ Output saved in $(OUT_DIR)"

# Remove junk before commit
clean:
	@echo "🧹 Cleaning caches, build artifacts, and temp files..."
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
	rm -rf $(OUT_DIR)
	rm -rf .pytest_cache .mypy_cache .ipynb_checkpoints .cache
	@echo "✅ Clean complete."

# GenFAR Inference Quickstart

1. **Create a clean environment**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
      uv pip sync --index-strategy unsafe-best-match \
     --extra-index-url https://download.pytorch.org/whl/cu121 \
     requirements.txt
   ```

3. **Run a test inference on bundled data**
   ```bash
   uv run python run_inference.py \
     --data-dir data \
     --models-dir models \
     --config-path configs/model_features.yaml \
     --output-dir outputs \
     --device auto
   ```

Outputs will appear under `outputs/`.

# Local models

This directory is kept in Git with `.gitkeep`, but model files are ignored. Keep large
downloaded or exported weights out of the repository so clones stay small.

TinySearch currently uses an ONNX embedding bundle compatible with
`sentence-transformers/all-MiniLM-L6-v2`. By default it lives at:

```text
models/all-minilm-l6-v2-onnx/
```

That bundle contains `model.onnx` plus tokenizer files and gives a much faster cold
start than loading PyTorch through `SentenceTransformer`.

## Automatic download

When `embedding_backend` in `configs/tinysearch_config.json` is `onnx` and
`embedding_model` is `fast`, starting **`servers/mcp_server.py`** or
**`servers/fastapi_server.py`** downloads the bundle once from Hugging Face
(`onnx-models/all-MiniLM-L6-v2-onnx`) into the default bundle directory above.

Override the directory with `TINYSEARCH_ONNX_MODEL_DIR` if needed.

## Manual export (optional)

From the repo root, with export-only dependencies installed:

```bash
pip install torch transformers onnx
```

```bash
python scripts/export_embedding_onnx.py
```

This pulls weights from `sentence-transformers/all-MiniLM-L6-v2`, then writes
`model.onnx` plus tokenizer artifacts to the default bundle directory using PyTorch.
Inference still uses `onnxruntime` only.

## Override path

Set `TINYSEARCH_ONNX_MODEL_DIR` to an absolute directory that contains the same layout:
`model.onnx`, `tokenizer.json`, plus companion tokenizer files as shipped by the repos above.

## Licensing

The MiniLM model is Apache-2.0. Keep attribution when redistributing model files, even
though this repository does not commit them.

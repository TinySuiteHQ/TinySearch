"""Download and persist local ONNX embedding bundles under ``models/``."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import snapshot_download

from services.embedding_service import (
    _onnx_bundle_ready,
    clear_onnx_runtime_cache,
    resolve_local_embedding_model_spec,
)

_LOCK_NAME = ".download.lock"
_STALE_LOCK_SEC = 1800.0
_LOCK_WAIT_SEC = 3600.0
_POLL_SEC = 0.25


@contextmanager
def _exclusive_bundle_lock(bundle_dir: Path):
    """Serialize concurrent downloads into ``bundle_dir`` (cross-process)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    lock_path = bundle_dir / _LOCK_NAME
    deadline = time.monotonic() + _LOCK_WAIT_SEC
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for ONNX bundle lock {lock_path}"
                ) from None
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > _STALE_LOCK_SEC:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            time.sleep(_POLL_SEC)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _copy_tree(src: Path, dest: Path) -> None:
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def ensure_onnx_bundle_sync(embedding_model: str | None = None) -> None:
    """If the ONNX bundle is missing, download it once into the bundle directory."""
    if _onnx_bundle_ready(embedding_model):
        return

    spec = resolve_local_embedding_model_spec(embedding_model)
    dest = spec.local_dir
    with _exclusive_bundle_lock(dest):
        if _onnx_bundle_ready(embedding_model):
            return

        print(
            "[tinysearch] downloading ONNX embedding bundle "
            f"model={spec.requested_model!r} repo={spec.repo_id!r}; "
            "see Hugging Face model card for license. One-time fetch.",
            file=sys.stderr,
            flush=True,
        )

        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tinysearch-onnx-dl-") as td_raw:
            td = Path(td_raw)
            snapshot_download(
                repo_id=spec.repo_id,
                local_dir=str(td),
                local_dir_use_symlinks=False,
                allow_patterns=list(spec.allow_patterns),
            )
            _copy_tree(td, dest)

        if not _onnx_bundle_ready(embedding_model):
            raise RuntimeError(
                f"ONNX bundle for {spec.requested_model!r} is incomplete after "
                f"download from {spec.repo_id!r} under {dest}. Expected an ONNX "
                "model plus tokenizer files."
            )

    clear_onnx_runtime_cache()

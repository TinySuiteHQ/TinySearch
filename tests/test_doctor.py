from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from tinysearch import doctor


class DoctorOutputTests(unittest.TestCase):
    def test_run_never_writes_to_stdout(self) -> None:
        with patch.object(
            doctor, "_check_chromium", return_value=(True, "ok")
        ), patch.object(doctor, "_check_model", return_value=(True, "ok")), patch.object(
            doctor, "_check_writable", return_value=(True, "ok")
        ):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = doctor.run()

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(exit_code, 0)

    def test_run_returns_nonzero_when_any_check_fails(self) -> None:
        with patch.object(
            doctor, "_check_chromium", return_value=(False, "missing")
        ), patch.object(doctor, "_check_model", return_value=(True, "ok")), patch.object(
            doctor, "_check_writable", return_value=(True, "ok")
        ):
            exit_code = doctor.run()

        self.assertEqual(exit_code, 1)

    def test_run_never_downloads_the_model(self) -> None:
        with patch(
            "tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync"
        ) as ensure, patch.object(
            doctor, "_check_chromium", return_value=(True, "ok")
        ), patch.object(doctor, "_check_writable", return_value=(True, "ok")):
            doctor.run()

        ensure.assert_not_called()


class DoctorCheckModelTests(unittest.TestCase):
    def test_reports_missing_bundle(self) -> None:
        config = {"embedding_backend": "onnx", "embedding_model": "fast"}
        with patch("tinysearch.doctor.resolve_local_embedding_model_spec") as resolve:
            resolve.return_value.local_dir = Path("/nonexistent/bundle/dir")
            resolve.return_value.onnx_paths = ("model.onnx",)
            ok, message = doctor._check_model(config)

        self.assertFalse(ok)
        self.assertIn("missing", message)

    def test_skips_non_onnx_backend(self) -> None:
        ok, _message = doctor._check_model(
            {"embedding_backend": "openai_compatible", "embedding_model": "x"}
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()

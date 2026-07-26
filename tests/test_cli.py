from __future__ import annotations

import unittest
from unittest.mock import patch

from tinysearch.cli import build_parser, main


class CliParserTests(unittest.TestCase):
    def test_no_subcommand_has_none_command(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.command)

    def test_setup_with_system_deps_flag(self) -> None:
        args = build_parser().parse_args(["setup", "--with-system-deps"])
        self.assertEqual(args.command, "setup")
        self.assertTrue(args.with_system_deps)

    def test_setup_without_flag_defaults_false(self) -> None:
        args = build_parser().parse_args(["setup"])
        self.assertFalse(args.with_system_deps)

    def test_doctor_and_serve_subcommands_parse(self) -> None:
        self.assertEqual(build_parser().parse_args(["doctor"]).command, "doctor")
        self.assertEqual(build_parser().parse_args(["serve"]).command, "serve")


class CliDispatchTests(unittest.TestCase):
    def test_no_args_runs_mcp_stdio(self) -> None:
        with patch("tinysearch.cli._run_mcp_stdio", return_value=0) as run_mcp:
            with self.assertRaises(SystemExit) as cm:
                main([])
        run_mcp.assert_called_once()
        self.assertEqual(cm.exception.code, 0)

    def test_setup_dispatches_with_flag(self) -> None:
        with patch("tinysearch.cli._run_setup", return_value=0) as run_setup:
            with self.assertRaises(SystemExit):
                main(["setup", "--with-system-deps"])
        run_setup.assert_called_once_with(True)

    def test_doctor_dispatches(self) -> None:
        with patch("tinysearch.cli._run_doctor", return_value=1) as run_doctor:
            with self.assertRaises(SystemExit) as cm:
                main(["doctor"])
        run_doctor.assert_called_once()
        self.assertEqual(cm.exception.code, 1)

    def test_serve_dispatches(self) -> None:
        with patch("tinysearch.cli._run_serve", return_value=0) as run_serve:
            with self.assertRaises(SystemExit):
                main(["serve"])
        run_serve.assert_called_once()


if __name__ == "__main__":
    unittest.main()

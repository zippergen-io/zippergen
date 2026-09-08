"""Offline acceptance checks using the project's ZipperGen CLI."""

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent


def zg(*args):
    completed = subprocess.run(
        ["zippergen", *args], cwd=ROOT, capture_output=True, text=True,
        check=True,
    )
    return completed.stdout


class CoinChainTests(unittest.TestCase):
    def test_protocol(self):
        report = json.loads(zg("validate", "--json"))
        self.assertTrue(report["valid"])
        self.assertEqual(report["lifelines"], ["L0", "L1", "L2", "L3"])
        self.assertEqual(report["inputs"], [
            {"name": "outcome", "type": "str", "lifeline": "L0"},
            {"name": "token", "type": "str", "lifeline": "L0"},
        ])
        self.assertEqual(report["outputs"], [
            {"name": "result", "type": "str", "lifeline": "L3"},
        ])
        communications = zg("show", "--communications")
        self.assertEqual(
            [line.strip() for line in communications.splitlines() if ">>" in line],
            ["L0(token) >> L1(token)", "L1(token) >> L2(token)",
             "L2(token) >> L3(token)"],
        )
        for owner in ("L0", "L1", "L2"):
            self.assertNotIn("if ", report["projections"][owner])
        self.assertIn("if At[L0].outcome == 'heads':", report["projections"]["L3"])

    def test_both_outcomes_with_identical_token(self):
        for outcome in ("heads", "tails"):
            with self.subTest(outcome=outcome):
                result = json.loads(zg(
                    "run", "--input", f"outcome={outcome}",
                    "--input", "token=opaque",
                ))
                self.assertEqual(result, {"result": outcome})


if __name__ == "__main__":
    unittest.main()

"""Exhaustive black-box checks of all three-update histories."""

import itertools
from pathlib import Path
import subprocess
import unittest


class StatusChainTests(unittest.TestCase):
    def test_all_status_histories(self):
        for statuses in itertools.product(("passed", "checking", "failed"), repeat=3):
            with self.subTest(statuses=statuses):
                decisive = [status for status in statuses if status != "checking"]
                expected = "accept" if decisive and decisive[-1] == "passed" else "reject"
                command = ["zippergen", "run"]
                for name, value in zip(
                    ("first_status", "second_status", "third_status"), statuses
                ):
                    command.extend(("--input", f"{name}={value}"))
                command.extend(("--input", "token=opaque"))
                result = subprocess.run(
                    command, cwd=Path(__file__).parent,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
                self.assertNotIn("reject" if expected == "accept" else "accept", result.stdout)


if __name__ == "__main__":
    unittest.main()

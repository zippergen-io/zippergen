"""Exhaustive deterministic checks of the three-update protocol."""

from itertools import product
import subprocess
import unittest

from zippergen import run
from workflow import L0, L1, L2, L3, status_chain


class StatusChainTests(unittest.TestCase):
    def test_every_status_sequence(self):
        for statuses in product(("passed", "checking", "failed"), repeat=3):
            # Independent oracle: the last non-checking status determines validity.
            decisive = [value for value in statuses if value != "checking"]
            expected = "accept" if decisive and decisive[-1] == "passed" else "reject"
            for token in ("opaque", "passed failed checking", ""):
                with self.subTest(statuses=statuses, token=token):
                    actual = run(
                        status_chain,
                        [L0, L1, L2, L3],
                        {"L0": dict(zip(
                            ("first_status", "second_status", "third_status", "token"),
                            (*statuses, token),
                        ))},
                    )
                    self.assertEqual(actual, expected)

    def test_application_message_architecture(self):
        view = subprocess.check_output(
            ["zippergen", "show", "--communications"], text=True
        )
        messages = [line.strip() for line in view.splitlines() if ">>" in line]
        self.assertEqual(messages, [
            "L0(token) >> L1(token)",
            "L1(token) >> L2(token)",
            "L2(token) >> L3(token)",
        ])


if __name__ == "__main__":
    unittest.main()

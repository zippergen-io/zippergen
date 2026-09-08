"""Offline behavior and protocol checks; no models or services required."""

import itertools
import subprocess
import unittest

from zippergen import run
from workflow import L0, L1, L2, L3, status_chain


class StatusChainTests(unittest.TestCase):
    def test_every_status_history(self):
        for statuses in itertools.product(("passed", "checking", "failed"), repeat=3):
            with self.subTest(statuses=statuses):
                decisive = [value for value in statuses if value != "checking"]
                expected = "accept" if decisive and decisive[-1] == "passed" else "reject"
                inputs = dict(zip(("first_status", "second_status", "third_status"), statuses))
                inputs["token"] = "opaque: passed failed checking"
                self.assertEqual(
                    run(status_chain, [L0, L1, L2, L3], {"L0": inputs}),
                    expected,
                )

    def test_application_message_architecture(self):
        rendered = subprocess.check_output(
            ["zippergen", "show", "--communications"], text=True
        )
        messages = [line.strip() for line in rendered.splitlines() if ">>" in line]
        self.assertEqual(messages, [
            "L0(token) >> L1(token)",
            "L1(token) >> L2(token)",
            "L2(token) >> L3(token)",
        ])


if __name__ == "__main__":
    unittest.main()

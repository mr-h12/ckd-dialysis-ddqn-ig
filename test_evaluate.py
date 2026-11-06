"""Module 3 checks: rule-based cap and tabular bins."""
import unittest
from types import SimpleNamespace

import numpy as np

from env import ACTIONS, N_STEPS
from evaluate import rule_based, to_bins


class EvaluateTests(unittest.TestCase):
    def test_rule_capped_at_13(self):
        heavy = SimpleNamespace(p=dict(overload0=4.0, weight=45.0))   # needs ~22 mL/kg/h
        self.assertEqual(ACTIONS[rule_based(None, heavy)], 13)

    def test_rule_lower_rate_for_light_overload(self):
        light = SimpleNamespace(p=dict(overload0=1.5, weight=110.0))  # needs ~3.4 mL/kg/h
        self.assertEqual(ACTIONS[rule_based(None, light)], 5)

    def test_bins_in_range(self):
        for state in np.random.default_rng(0).uniform(0, 1.5, size=(1000, 8)):
            o, s, t = to_bins(state)
            self.assertTrue(0 <= o <= 4 and 0 <= s <= 4 and 0 <= t < N_STEPS)


if __name__ == '__main__':
    unittest.main()

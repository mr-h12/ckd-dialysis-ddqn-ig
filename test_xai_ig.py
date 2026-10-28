"""Numerical tests with analytically known attributions and a saturating ReLU."""
import unittest
import numpy as np
import torch
from xai_ig import explain, baseline_for


class AttributionTests(unittest.TestCase):
    def test_linear_exactness_and_target(self):
        model = torch.nn.Linear(8, 5)
        with torch.no_grad():
            model.weight.copy_(torch.arange(40).reshape(5, 8) / 40)
            model.bias.copy_(torch.arange(5).float())
        x = np.linspace(.1, .8, 8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        model.train()
        result = explain(model, x, b)
        expected = x * model.weight[result['action']].detach().numpy()
        np.testing.assert_allclose(result['attributions'], expected, atol=1e-6)
        self.assertEqual(result['action'], 4)
        self.assertLess(abs(result['residual']), 1e-5)
        self.assertTrue(model.training)

    def test_identical_reference_is_zero(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 10), torch.nn.ReLU(), torch.nn.Linear(10, 5))
        x = np.ones(8, dtype=np.float32)
        result = explain(model, x, x)
        np.testing.assert_array_equal(result['attributions'], np.zeros(8))

    def test_reference_preserves_demographics(self):
        x = np.linspace(.1, .8, 8, dtype=np.float32)
        np.testing.assert_array_equal(baseline_for(x)[5:], x[5:])

    def test_integrates_through_saturation(self):
        class Saturated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, x):
                score = (1 - torch.relu(1 - x[:, :1])) * self.scale
                return torch.cat([score + 2, score * 0, score * 0, score * 0, score * 0], 1)
        x = np.zeros(8, dtype=np.float32)
        x[0] = 2
        result = explain(Saturated(), x, np.zeros(8, dtype=np.float32))
        self.assertAlmostEqual(float(result['attributions'][0]), 1, places=5)

    def test_rejects_invalid_state(self):
        with self.assertRaises(ValueError):
            explain(torch.nn.Linear(8, 5), [float('nan')] * 8)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from opm_adjoint_chainrule import MILLIDARCY_TO_M2, PermeabilityMap


class PullbackTests(unittest.TestCase):
    def test_directional_derivative_including_tied_components_and_si_units(self):
        mapping = PermeabilityMap(("PERMX",), ((1,), (1,), (0.001,)))
        m = np.log([[200.], [800.]])
        direction = np.array([[0.3], [-0.7]])
        weights = np.array([[0., 2., 300.], [1., -0.4, 500.]])

        def objective(parameters):
            k_si = mapping.physical_permeability(parameters) * MILLIDARCY_TO_M2
            return np.sum(weights * np.sin(k_si / 1e-12))

        k_si = mapping.physical_permeability(m) * MILLIDARCY_TO_M2
        raw_gradient = weights * np.cos(k_si / 1e-12) / 1e-12
        original = raw_gradient.copy()
        analytic = np.sum(mapping.pullback(m, raw_gradient) * direction)
        step = 1e-5
        numeric = (objective(m + step * direction) - objective(m - step * direction)) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-8)
        np.testing.assert_array_equal(raw_gradient, original)

    def test_independent_mapping_does_not_mix_components(self):
        mapping = PermeabilityMap(("PERMX", "PERMY", "PERMZ"), tuple(map(tuple, np.eye(3))))
        result = mapping.pullback(np.log([[2, 3, 4]]), np.array([[1., 0., 0.]]))
        np.testing.assert_allclose(result, [[2 * MILLIDARCY_TO_M2, 0, 0]], atol=0)

    def test_invalid_gradient_fails(self):
        mapping = PermeabilityMap(("PERMX",), ((1,), (1,), (0.001,)))
        for gradient in (np.ones((2, 1)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                mapping.pullback(np.zeros((2, 1)), gradient)


if __name__ == "__main__":
    unittest.main()

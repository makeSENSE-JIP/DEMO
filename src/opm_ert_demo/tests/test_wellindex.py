import math
import unittest

from opm_ert_demo.wellindex import peaceman_well_index, frozen_well_indexes


class PeacemanTests(unittest.TestCase):
    def test_isotropic_uniform_cell_matches_hand_computation(self):
        r0 = 0.28 * math.sqrt(100.0**2 + 100.0**2) / 2.0
        denominator = math.log(r0 / 0.25)
        expected = 2.0 * math.pi * 500.0 * 1.0 / denominator
        self.assertAlmostEqual(
            peaceman_well_index(500.0, 500.0, 100.0, 100.0, 1.0), expected, places=12
        )

    def test_isotropic_radius_independent_of_permeability(self):
        self.assertAlmostEqual(
            peaceman_well_index(250.0, 250.0, 100.0, 100.0, 1.0) / 250.0,
            peaceman_well_index(800.0, 800.0, 100.0, 100.0, 1.0) / 800.0,
            places=12,
        )

    def test_anisotropy_changes_the_factor(self):
        iso = peaceman_well_index(500.0, 500.0, 100.0, 100.0, 1.0)
        aniso = peaceman_well_index(500.0, 125.0, 100.0, 100.0, 1.0)
        self.assertGreater(iso, aniso)

    def test_frozen_indexes_exist_for_all_demo_wells(self):
        indexes = frozen_well_indexes(500.0, 100.0, 100.0, 1.0)
        self.assertEqual(
            set(indexes), {"INJ1", "P1", "P2", "P3", "P4"}
        )
        self.assertTrue(all(value > 0 for value in indexes.values()))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            peaceman_well_index(0.0, 500.0, 100.0, 100.0, 1.0)


if __name__ == "__main__":
    unittest.main()

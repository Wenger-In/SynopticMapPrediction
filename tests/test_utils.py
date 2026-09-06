import tempfile
import unittest
from pathlib import Path

import numpy as np

from smp.config import load_config
from smp.utils import min_max_normalize, moving_average, require_file


class UtilityTests(unittest.TestCase):
    def test_default_paths_are_absolute(self) -> None:
        config = load_config()
        self.assertTrue(config.path("sunspot").is_absolute())
        self.assertTrue(config.path("model_output").is_absolute())

    def test_min_max_normalize(self) -> None:
        result = min_max_normalize(np.array([-2.0, 0.0, 2.0]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_min_max_normalize_rejects_constant_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "constant"):
            min_max_normalize(np.ones(3))

    def test_moving_average_preserves_length(self) -> None:
        self.assertEqual(moving_average(np.arange(5.0), 3).shape, (5,))

    def test_require_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.dat"
            with self.assertRaisesRegex(FileNotFoundError, "config/local.json"):
                require_file(missing)


if __name__ == "__main__":
    unittest.main()

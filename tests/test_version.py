import re
import unittest
from pathlib import Path

import spatialmath


class TestVersion(unittest.TestCase):
    def test_version_is_a_non_empty_string(self):
        self.assertIsInstance(spatialmath.__version__, str)
        self.assertTrue(spatialmath.__version__)

    def test_version_matches_pyproject(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
        self.assertIsNotNone(match, f"couldn't find a version in {pyproject}")
        self.assertEqual(spatialmath.__version__, match.group(1))


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()

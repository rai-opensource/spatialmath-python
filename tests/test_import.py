import subprocess
import sys
import unittest


class TestImport(unittest.TestCase):
    def test_import_does_not_load_scipy(self):
        # scipy is slow to import and only a few functions need it, so it is
        # imported by those functions and not when spatialmath is imported
        code = "import sys, spatialmath; sys.exit(any(m.split('.')[0] == 'scipy' for m in sys.modules))"
        result = subprocess.run([sys.executable, "-c", code])
        self.assertEqual(result.returncode, 0, "import spatialmath loaded scipy")


if __name__ == "__main__":  # pragma nocover
    unittest.main()

"""Compiling the package must not raise SyntaxWarning or DeprecationWarning.

A non-raw docstring containing LaTeX such as ``\\mat`` or ``\\vec`` gives an
"invalid escape sequence" warning, and ``\\t`` or ``\\v`` (as in ``\\times`` and
``\\vec``) are valid escapes that silently put a tab or vertical tab into the
documentation.  Use ``r\"\"\"`` for docstrings that contain backslashes.
"""

import pathlib
import unittest
import warnings

import spatialmath


class TestSourceWarnings(unittest.TestCase):
    def test_package_compiles_without_warnings(self):
        root = pathlib.Path(spatialmath.__file__).parent
        problems = []
        for path in sorted(root.rglob("*.py")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            problems += [
                f"{path.relative_to(root)}:{w.lineno}: {w.message}"
                for w in caught
                if issubclass(w.category, (SyntaxWarning, DeprecationWarning))
            ]
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()

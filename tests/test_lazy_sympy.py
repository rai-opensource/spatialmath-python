"""Regression tests for optional SymPy loading in fresh interpreters."""

import subprocess
import sys
import textwrap

import pytest


def run_python(source):
    """Run an isolated check with the same interpreter as pytest."""
    subprocess.run([sys.executable, "-c", textwrap.dedent(source)], check=True)


def test_numeric_import_does_not_load_sympy():
    run_python(
        """
        import sys
        import numpy as np
        import spatialmath
        from spatialmath import base as smb
        assert not any(n == 'sympy' or n.startswith('sympy.') for n in sys.modules)
        assert smb.isscalar(1.0)
        assert smb.isvector([1, 2])
        assert smb.isnumberlist([1, 2])
        assert smb.norm([3, 4]) == 5
        assert np.allclose(smb.rot2(0), np.eye(2))
        assert not any(n == 'sympy' or n.startswith('sympy.') for n in sys.modules)
    """
    )


@pytest.mark.parametrize("sympy_first", [False, True])
def test_symbolic_operations_in_both_import_orders(sympy_first):
    prefix = "import sympy as sp\n" if sympy_first else ""
    run_python(
        prefix
        + """
import numpy as np
from spatialmath import base as smb
import sympy as sp
x = sp.Symbol('x', real=True)
assert smb.issymbol(x)
assert smb.issymbol([1, x])
assert smb.issymbol((x, 1))
assert not smb.issymbol([1, 2])
assert smb.isscalar(x)
assert not smb.isscalar([x])
assert not smb.isscalar((x,))
assert smb.isvector([x, 1])
assert smb.isnumberlist([x, 1])
assert smb.getvector(x)[0] == x
assert smb.getvector([x, 1])[0] == x
assert smb.sin(x) == sp.sin(x)
assert smb.cos(x) == sp.cos(x)
assert smb.tan(x) == sp.tan(x)
assert smb.sqrt(x) == sp.sqrt(x)
assert smb.norm([x, 1]) == sp.sqrt(x*x + 1)
assert smb.det(np.array([[x, 0], [0, 1]], dtype=object)) == x
T = smb.trot2(x)
assert sp.simplify(smb.tr2xyt(T)[2] - sp.atan2(sp.sin(x), sp.cos(x))) == 0
assert smb.simplify(sp.sin(x)**2 + sp.cos(x)**2) == 1
"""
    )


@pytest.mark.parametrize("helper", ["symbol", "zero", "one", "negative_one", "pi"])
def test_explicit_symbolic_helpers_load_sympy(helper):
    run_python(
        textwrap.dedent(f"helper = {helper!r}\n")
        + textwrap.dedent(
            """
        import sys
        from spatialmath.base import symbolic as sym
        assert 'sympy' not in sys.modules
        getattr(sym, helper)(*(['x'] if helper == 'symbol' else []))
        assert 'sympy' in sys.modules
        x = sym.symbol('x')
        import sympy as sp
        assert isinstance(x, sp.Symbol)
        from spatialmath.base.symbolic import Symbol, symtype
        assert Symbol is sp.Symbol
        assert symtype == (sp.Expr,)
        assert sym.zero() == sp.S.Zero
        assert sym.one() == sp.S.One
        assert sym.negative_one() == sp.S.NegativeOne
        assert sym.pi() == sp.pi
    """
        )
    )


def test_numeric_operations_without_sympy():
    run_python(
        """
        import importlib.abc
        import sys
        class BlockSympy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'sympy' or fullname.startswith('sympy.'):
                    raise ModuleNotFoundError('SymPy blocked for this test')
        sys.meta_path.insert(0, BlockSympy())
        from spatialmath import base as smb
        assert not smb.issymbol(1)
        assert smb.norm([3, 4]) == 5
        assert smb.sin(0) == 0
        assert smb.simplify(2) == 2
        assert smb.isscalar(1)
        assert smb.isvector([1, 2])
    """
    )

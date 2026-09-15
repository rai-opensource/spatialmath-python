# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
This package provides a light-weight wrapper to support use of SymPy.  It
generalizes some common functions so that they can accept numerical or
Symbolic arguments.

If SymPy is not installed then only the standard numeric operations are
supported.
"""

import math
from spatialmath.base.types import *

try:  # pragma: no cover
    # print('Using SymPy')
    import sympy

    _symbolics = True
    symtype = (sympy.Expr,)
    from sympy import Symbol

except ImportError:  # pragma: no cover
    # SymPy is not installed
    _symbolics = False
    symtype = ()
    Symbol = Any


# ---------------------------------------------------------------------------------------#

if _symbolics:

    def symbol(
        name: str, real: Optional[bool] = True
    ) -> Union[Symbol, Tuple[Symbol, ...]]:
        """
        Create symbolic variables

        :param name: symbol names
        :type name: str
        :param real: assume variable is real, defaults to True
        :type real: bool, optional
        :return: SymPy symbols
        :rtype: sympy

        .. runblock:: pycon

            >>> from spatialmath.base.symbolic import *
            >>> theta = symbol('theta')
            >>> theta
            >>> theta, psi = symbol('theta psi')
            >>> theta
            >>> psi
            >>> q = symbol('q_:6')
            >>> q

        .. note:: In Jupyter symbols are pretty printed.

            - symbols named after greek letters will appear as greek letters
            - underscore means subscript as it does in LaTex, so the symbols ``q``
              above will be subscripted.

        :seealso: :func:`sympy.symbols`
        """
        return sympy.symbols(name, real=real)


def issymbol(var: Any) -> bool:
    """
    Test if variable is symbolic

    :param var: variable to test
    :return: whether variable is symbolic
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> theta = symbol('theta')
        >>> issymbol(theta)
        >>> issymbol(3.4)

    """
    if _symbolics:
        if isinstance(var, (list, tuple)):
            return any([isinstance(x, symtype) for x in var])
        else:
            return isinstance(var, symtype)
    else:
        return False


@overload
def sin(theta: float) -> float:
    ...


@overload
def sin(theta: Symbol) -> Symbol:
    ...


def sin(theta):
    """
    Generalized sine function

    :param θ: argument
    :type θ: float or symbolic
    :return: sin(θ)
    :rtype: float or symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> theta = symbol('theta')
        >>> sin(theta)
        >>> sin(0.5)

    :seealso: :func:`sympy.sin`
    """
    if issymbol(theta):
        return sympy.sin(theta)
    else:
        return math.sin(theta)


@overload
def cos(theta: float) -> float:
    ...


@overload
def cos(theta: Symbol) -> Symbol:
    ...


def cos(theta):
    """
    Generalized cosine function

    :param θ: argument
    :type θ: float or symbolic
    :return: cos(θ)
    :rtype: float or symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> theta = symbol('theta')
        >>> cos(theta)
        >>> cos(0.5)

    :seealso: :func:`sympy.cos`
    """
    if issymbol(theta):
        return sympy.cos(theta)
    else:
        return math.cos(theta)


@overload
def tan(theta: float) -> float:
    ...


@overload
def tan(theta: Symbol) -> Symbol:
    ...


def tan(theta):
    """
    Generalized tangent function

    :param θ: argument
    :type θ: float or symbolic
    :return: tan(θ)
    :rtype: float or symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> theta = symbol('theta')
        >>> tan(theta)
        >>> tan(0.5)

    :seealso: :func:`sympy.cos`
    """
    if issymbol(theta):
        return sympy.tan(theta)
    else:
        return math.tan(theta)


@overload
def sqrt(theta: float) -> float:
    ...


@overload
def sqrt(theta: Symbol) -> Symbol:
    ...


def sqrt(v):
    """
    Generalized sqrt function

    :param v: argument
    :type v: float or symbolic
    :return: √ v
    :rtype: float or symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> x = symbol('x')
        >>> sqrt(x ** 2)
        >>> sqrt(4)

    :seealso: :func:`sympy.sqrt`
    """
    if issymbol(v):
        return sympy.sqrt(v)
    else:
        return math.sqrt(v)


def zero() -> Symbol:
    """
    Symbolic constant: zero

    :return: 0
    :rtype: symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> x = symbol('x')
        >>> zero()
        >>> x + zero()

    :seealso: :func:`sympy.S.Zero`
    """
    return sympy.S.Zero


def one() -> Symbol:
    """
    Symbolic constant: one

    :return: 1
    :rtype: symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> x = symbol('x')
        >>> one()
        >>> one() * x

    :seealso: :func:`sympy.S.One`
    """
    return sympy.S.One


def negative_one() -> Symbol:
    """
    Symbolic constant: negative one

    :return: -1
    :rtype: symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> x = symbol('x')
        >>> negative_one()
        >>> negative_one() * x

    :seealso: :func:`sympy.S.NegativeOne`
    """
    return sympy.S.NegativeOne


def pi() -> Symbol:
    """
    Symbolic constant: pi

    :return: π
    :rtype: symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> import math
        >>> sin(pi())
        >>> sin(math.pi)

    :seealso: :func:`sympy.S.Pi`
    """
    return sympy.S.Pi


def simplify(x: Symbol) -> Symbol:
    """
    Symbolic simplification

    :param x: expression to simplify
    :type x: symbolic
    :return: -1
    :rtype: symbolic

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> x = symbol('x')
        >>> y = (x - 1) * (x + 1) - x ** 2
        >>> y
        >>> simplify(y)

    :seealso: :func:`sympy.simplify`
    """
    if _symbolics:
        return sympy.simplify(x)
    else:
        return x


def det(x):
    """
    Symbolic determinant

    :param m: matrix
    :type x: ndarray with symbolic elements
    :return: determinant
    :rtype: ndarray with symbolic elements

    .. runblock:: pycon

        >>> from spatialmath.base.symbolic import *
        >>> from spatialmath.base import rot2
        >>> theta = symbol('theta')
        >>> R = rot2(theta)
        >>> print(R)
        >>> print(det(R))
        >>> print(simplify(det(R)))

    .. note:: Converts to a SymPy ``Matrix`` and then back again.
    """

    return sympy.Matrix(x).det()

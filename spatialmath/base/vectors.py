# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
Functions to manipulate vectors

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.
"""

# pylint: disable=invalid-name

import math
import numpy as np
from spatialmath.base.argcheck import getvector
from spatialmath.base.types import *

try:  # pragma: no cover
    # print('Using SymPy')
    import sympy

    _symbolics = True

except ImportError:  # pragma: no cover
    _symbolics = False

_eps = np.finfo(np.float64).eps


def norm(v: ArrayLikePure) -> float:
    """
    Norm of vector

    :param v: any vector
    :type v: array_like(n)
    :return: norm of vector
    :rtype: float

    ``norm(v)`` is the 2-norm (length or magnitude) of the vector ``v``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> norm([3, 4])

    .. note:: This function does not use NumPy, it is ~2x faster than
        `numpy.linalg.norm()` for a 3-vector

    :seealso: :func:`~spatialmath.base.unit`

    :SymPy: supported
    """
    sum = 0
    for x in v:
        sum += x * x

    if _symbolics and isinstance(sum, sympy.Expr):
        return sympy.sqrt(sum)
    else:
        return math.sqrt(sum)


def normsq(v: ArrayLikePure) -> float:
    """
    Squared norm of vector

    :param v: any vector
    :type v: array_like(n)
    :return: norm of vector
    :rtype: float

    ``norm(sq)`` is the sum of squared elements of the vector ``v``
    or :math:`|v|^2`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> normsq([2, 3])

    .. note:: This function does not use NumPy, it is ~2x faster than
        `numpy.linalg.norm() ** 2` for a 3-vector

    :seealso: :func:`~spatialmath.base.unit`

    :SymPy: supported
    """
    sum = 0
    for x in v:
        sum += x * x

    return sum


def cross(u: ArrayLike3, v: ArrayLike3) -> R3:
    """
    Cross product of vectors

    :param u: any vector
    :type u: array_like(3)
    :param v: any vector
    :type v: array_like(3)
    :return: cross product
    :rtype: nd.array(3)

    ``cross(u, v)`` is the cross product of the vectors ``u`` and ``v``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> cross([1, 0, 0], [0, 1, 0])

    .. note:: This function does not use NumPy, it is ~1.5x faster than
        `numpy.cross()`

    :seealso: :func:`~spatialmath.base.unit`

    :SymPy: supported
    """
    return np.r_[
        u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]
    ]


def colvec(v: ArrayLike) -> NDArray:
    """
    Create a column vector

    :param v: any vector
    :type v: array_like(n)
    :return: a column vector
    :rtype: ndarray(n,1)

    Convert input to a column vector.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> colvec([1, 2, 3])
    """
    v = getvector(v)
    return np.array(v).reshape((len(v), 1))


def unitvec(v: ArrayLike, tol: float = 20) -> NDArray:
    """
    Create a unit vector

    :param v: any vector
    :type v: array_like(n)
    :param tol: Tolerance in units of eps for zero-norm case, defaults to 20
    :type: float
    :return: a unit-vector parallel to ``v``.
    :rtype: ndarray(n)
    :raises ValueError: for zero length vector

    ``unitvec(v)`` is a vector parallel to `v` of unit length.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unitvec([3, 4])

    :seealso: :func:`~numpy.linalg.norm`

    """

    v = getvector(v)
    n = norm(v)

    if n > tol * _eps:  # if greater than eps
        return v / n
    else:
        raise ValueError("zero norm vector")


def unitvec_norm(v: ArrayLike, tol: float = 20) -> Tuple[NDArray, float]:
    """
    Create a unit vector

    :param v: any vector
    :type v: array_like(n)
    :param tol: Tolerance in units of eps for zero-norm case, defaults to 20
    :type: float
    :return: a unit-vector parallel to ``v`` and the norm
    :rtype: (ndarray(n), float)
    :raises ValueError: for zero length vector

    ``unitvec(v)`` is a vector parallel to `v` of unit length.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unitvec([3, 4])

    :seealso: :func:`~numpy.linalg.norm`

    """

    v = getvector(v)
    nm = norm(v)

    if nm > tol * _eps:  # if greater than eps
        return (v / nm, nm)
    else:
        raise ValueError("zero norm vector")


def isunitvec(v: ArrayLike, tol: float = 20) -> bool:
    """
    Test if vector has unit length

    :param v: vector to test
    :type v: ndarray(n)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isunitvec([1, 0])
        >>> isunitvec([1, 2])

    :seealso: unit, iszerovec, isunittwist
    """
    return bool(abs(np.linalg.norm(v) - 1) < tol * _eps)


def iszerovec(v: ArrayLike, tol: float = 20) -> bool:
    """
    Test if vector has zero length

    :param v: vector to test
    :type v: ndarray(n)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether vector has zero length
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> iszerovec([0, 0])
        >>> iszerovec([1, 2])

    :seealso: unit, isunitvec, isunittwist
    """
    return bool(np.linalg.norm(v) < tol * _eps)


def iszero(v: float, tol: float = 20) -> bool:
    """
    Test if scalar is zero

    :param v: value to test
    :type v: float
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether value is zero
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> iszero(0)
        >>> iszero(1)

    :seealso: unit, iszerovec, isunittwist
    """
    return bool(abs(v) < tol * _eps)


def isunittwist(v: ArrayLike6, tol: float = 20) -> bool:
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)

    :param v: twist vector to test
    :type v: array_like(6)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether twist has unit length
    :rtype: bool
    :raises ValueError: for incorrect vector length


    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).

    A unit twist can be a:

    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isunittwist([1, 2, 3, 1, 0, 0])
        >>> isunittwist([0, 0, 0, 2, 0, 0])

    :seealso: unit, isunitvec
    """
    v = getvector(v)

    if len(v) == 6:
        # test for SE(3) twist
        return isunitvec(v[3:6], tol=tol) or (
            iszerovec(v[3:6], tol=tol) and isunitvec(v[0:3], tol=tol)
        )
    else:
        raise ValueError


def isunittwist2(v: ArrayLike3, tol: float = 20) -> bool:
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)

    :param v: twist vector to test
    :type v: array_like(3)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool
    :raises ValueError: for incorrect vector length

    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).

    A unit twist can be a:

    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isunittwist2([1, 2, 1])
        >>> isunittwist2([0, 0, 2])

    :seealso: unit, isunitvec
    """
    v = getvector(v)

    if len(v) == 3:
        # test for SE(2) twist
        return isunitvec(v[2], tol=tol) or (
            iszero(v[2], tol=tol) and isunitvec(v[0:2], tol=tol)
        )
    else:
        raise ValueError


def unittwist(S: ArrayLike6, tol: float = 20) -> Union[R6, None]:
    """
    Convert twist to unit twist

    :param S: twist vector
    :type S: array_like(6)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: unit twist
    :rtype: ndarray(6)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unittwist([2, 4, 6, 2, 0, 0])
        >>> unittwist([2, 0, 0, 0, 0, 0])

    Returns None if the twist has zero magnitude
    """

    S = getvector(S, 6)

    if iszerovec(S, tol=tol):
        return None

    v = S[0:3]
    w = S[3:6]

    if iszerovec(w, tol=tol):
        th = norm(v)
    else:
        th = norm(w)

    return S / th


def unittwist_norm(
    S: Union[R6, ArrayLike6], tol: float = 20
) -> Tuple[Union[R6, None], Union[float, None]]:
    """
    Convert twist to unit twist and norm

    :param S: twist vector
    :type S: array_like(6)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: unit twist and scalar motion
    :rtype: tuple (ndarray(6), float)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> S, n = unittwist_norm([1, 2, 3, 1, 0, 0])
        >>> print(S, n)
        >>> S, n = unittwist_norm([0, 0, 0, 2, 0, 0])
        >>> print(S, n)
        >>> S, n = unittwist_norm([0, 0, 0, 0, 0, 0])
        >>> print(S, n)

    .. note:: Returns (None,None) if the twist has zero magnitude
    """

    S = getvector(S, 6)

    if iszerovec(S, tol=tol):
        return (None, None)  # according to "note" in docstring.

    v = S[0:3]
    w = S[3:6]

    if iszerovec(w, tol=tol):
        th = norm(v)
    else:
        th = norm(w)

    return (S / th, th)


def unittwist2(S: ArrayLike3, tol: float = 20) -> Union[R3, None]:
    """
    Convert twist to unit twist

    :param S: twist vector
    :type S: array_like(3)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: unit twist
    :rtype: ndarray(3)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unittwist2([2, 4, 2])
        >>> unittwist2([2, 0, 0])

    .. note:: Returns None if the twist has zero magnitude
    """

    S = getvector(S, 3)

    if iszerovec(S, tol=tol):
        return None

    v = S[0:2]
    w = S[2]

    if iszero(w, tol=tol):
        th = norm(v)
    else:
        th = abs(w)

    return S / th


def unittwist2_norm(
    S: ArrayLike3, tol: float = 20
) -> Tuple[Union[R3, None], Union[float, None]]:
    """
    Convert twist to unit twist

    :param S: twist vector
    :type S: array_like(3)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: unit twist and scalar motion
    :rtype: tuple (ndarray(3), float)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unittwist2([2, 4, 2])
        >>> unittwist2([2, 0, 0])

    .. note:: Returns (None, None) if the twist has zero magnitude
    """

    S = getvector(S, 3)

    if iszerovec(S, tol=tol):
        return (None, None)

    v = S[0:2]
    w = S[2]

    if iszero(w, tol=tol):
        th = norm(v)
    else:
        th = abs(w)

    return (S / th, th)


def wrap_0_pi(theta: ArrayLike) -> Union[float, NDArray]:
    r"""
    Wrap angle to range :math:`[0, \pi]`

    :param theta: input angle
    :type theta: scalar or ndarray
    :return: angle wrapped into range :math:`[0, \pi)`
    :rtype: scalar or ndarray

    This is used to fold angles of colatitude.  If zero is the angle of the
    north pole, colatitude increases to :math:`\pi` at the south pole then
    decreases to :math:`0` as we head back to the north pole.

    :seealso: :func:`wrap_mpi2_pi2` :func:`wrap_0_2pi` :func:`wrap_mpi_pi` :func:`angle_wrap`
    """
    theta = np.abs(getvector(theta))
    n = theta / np.pi
    if isinstance(n, np.ndarray):
        n = n.astype(int)
    else:
        n = np.fix(n).astype(int)

    y = np.where(np.bitwise_and(n, 1) == 0, theta - n * np.pi, (n + 1) * np.pi - theta)
    if isinstance(y, np.ndarray) and y.size == 1:
        return float(y[0])
    else:
        return y


def wrap_mpi2_pi2(theta: ArrayLike) -> Union[float, NDArray]:
    r"""
    Wrap angle to range :math:`[-\pi/2, \pi/2]`

    :param theta: input angle
    :type theta: scalar or ndarray
    :return: angle wrapped into range :math:`[-\pi/2, \pi/2]`
    :rtype: scalar or ndarray

    This is used to fold angles of latitude.

    :seealso: :func:`wrap_0_pi` :func:`wrap_0_2pi` :func:`wrap_mpi_pi` :func:`angle_wrap`

    """
    theta = getvector(theta)
    n = theta / np.pi * 2
    if isinstance(n, np.ndarray):
        n = n.astype(int)
    else:
        n = np.fix(n).astype(int)

    y = np.where(np.bitwise_and(n, 1) == 0, theta - n * np.pi, n * np.pi - theta)
    if isinstance(y, np.ndarray) and len(y) == 1:
        return float(y[0])
    else:
        return y


def wrap_0_2pi(theta: ArrayLike) -> Union[float, NDArray]:
    r"""
    Wrap angle to range :math:`[0, 2\pi)`

    :param theta: input angle
    :type theta: scalar or ndarray
    :return: angle wrapped into range :math:`[0, 2\pi)`
    :rtype: scalar or ndarray

    :seealso: :func:`wrap_mpi_pi` :func:`wrap_0_pi` :func:`wrap_mpi2_pi2` :func:`angle_wrap`
    """
    theta = getvector(theta)
    y = theta - 2.0 * math.pi * np.floor(theta / 2.0 / np.pi)
    if isinstance(y, np.ndarray) and len(y) == 1:
        return float(y[0])
    else:
        return y


def wrap_mpi_pi(theta: ArrayLike) -> Union[float, NDArray]:
    r"""
    Wrap angle to range :math:`[-\pi, \pi)`

    :param theta: input angle
    :type theta: scalar or ndarray
    :return: angle wrapped into range :math:`[-\pi, \pi)`
    :rtype: scalar or ndarray

    :seealso: :func:`wrap_0_2pi` :func:`wrap_0_pi` :func:`wrap_mpi2_pi2` :func:`angle_wrap`
    """
    theta = getvector(theta)
    y = np.mod(theta + math.pi, 2 * math.pi) - np.pi
    if isinstance(y, np.ndarray) and len(y) == 1:
        return float(y[0])
    else:
        return y


# @overload
# def angdiff(a:ArrayLike):
#     ...


@overload
def angdiff(a: ArrayLike, b: ArrayLike) -> NDArray:
    ...


@overload
def angdiff(a: ArrayLike) -> NDArray:
    ...


def angdiff(a, b=None):
    r"""
    Angular difference

    :param a: angle in radians
    :type a: scalar or array_like
    :param b: angle in radians
    :type b: scalar or array_like
    :return: angular difference a-b
    :rtype: scalar or array_like

    - ``angdiff(a, b)`` is the difference ``a - b`` wrapped to the range
      :math:`[-\pi, \pi)`.  This is the operator :math:`a \circleddash b` used
      in the RVC book

        - If ``a`` and ``b`` are both scalars, the result is scalar
        - If ``a`` is array_like, the result is a NumPy array ``a[i]-b``
        - If ``a`` is array_like, the result is a NumPy array ``a-b[i]``
        - If ``a`` and ``b`` are both vectors of the same length, the result is
          a NumPy array ``a[i]-b[i]``

    - ``angdiff(a)`` is the angle or vector of angles ``a`` wrapped to the range
      :math:`[-\pi, \pi)`.

        - If ``a`` is a scalar, the result is scalar
        - If ``a`` is array_like, the result is a NumPy array

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> from math import pi
        >>> angdiff(0, 2 * pi)
        >>> angdiff(0.9 * pi, -0.9 * pi) / pi
        >>> angdiff(3 * pi)

    :seealso: :func:`vector_diff` :func:`wrap_mpi_pi`
    """
    a = getvector(a)
    if b is not None:
        b = getvector(b)
        a = a - b  # cannot use -= here, numpy wont broadcast

    y = np.mod(a + math.pi, 2 * math.pi) - math.pi
    if isinstance(y, np.ndarray) and len(y) == 1:
        return float(y[0])
    else:
        return y


def angle_std(theta: ArrayLike) -> float:
    r"""
    Standard deviation of angular values

    :param theta: angular values
    :type theta: array_like
    :return: circular standard deviation
    :rtype: float

    .. math::

        \sigma_{\theta} = \sqrt{-2 \log \| \left[ \frac{\sum \sin \theta_i}{N}, \frac{\sum \sin \theta_i}{N} \right] \|} \in [0, \infty)

    :seealso: :func:`angle_mean`
    """
    X = np.cos(theta).mean()
    Y = np.sin(theta).mean()
    R = np.sqrt(X**2 + Y**2)

    return np.sqrt(-2 * np.log(R))


def angle_mean(theta: ArrayLike) -> float:
    r"""
    Mean of angular values

    :param theta: angular values
    :type v: array_like
    :return: circular mean
    :rtype: float

    The circular mean is given by

    .. math::

        \bar{\theta} = \tan^{-1} \frac{\sum \sin \theta_i}{\sum \cos \theta_i} \in [-\pi, \pi)]

    :seealso: :func:`angle_std`
    """
    X = np.cos(theta).sum()
    Y = np.sin(theta).sum()
    return np.arctan2(Y, X)


def angle_wrap(theta: ArrayLike, mode: str = "-pi:pi") -> Union[float, NDArray]:
    """
    Generalized angle-wrapping

    :param v: angles to wrap
    :type v: array_like
    :param mode: wrapping mode, one of: ``"0:2pi"``, ``"0:pi"``, ``"-pi/2:pi/2"`` or ``"-pi:pi"`` [default]
    :type mode: str, optional
    :return: wrapped angles
    :rtype: ndarray

    .. note:: The modes ``"0:pi"`` and ``"-pi/2:pi/2"`` are used to wrap angles of
        colatitude and latitude respectively.

    :seealso: :func:`wrap_0_2pi` :func:`wrap_mpi_pi` :func:`wrap_0_pi` :func:`wrap_mpi2_pi2`
    """
    if mode == "0:2pi":
        return wrap_0_2pi(theta)
    elif mode == "-pi:pi":
        return wrap_mpi_pi(theta)
    elif mode == "0:pi":
        return wrap_0_pi(theta)
    elif mode == "-pi/2:pi/2":
        return wrap_mpi2_pi2(theta)
    else:
        raise ValueError("bad method specified")


def vector_diff(v1: ArrayLike, v2: ArrayLike, mode: str) -> NDArray:
    """
    Generalized vector differnce

    :param v1: first vector
    :type v1: array_like(n)
    :param v2: second vector
    :type v2: array_like(n)
    :param mode: subtraction mode
    :type mode: str of length n

    ==============  ====================================
    mode character  purpose
    ==============  ====================================
    r               real number, don't wrap
    c               angle on circle, wrap to [-π, π)
    C               angle on circle, wrap to [0, 2π)
    l               latitude angle, wrap to [-π/2, π/2]
    L               colatitude angle, wrap to [0, π]
    ==============  ====================================

    :seealso: :func:`angdiff` :func:`wrap_0_2pi` :func:`wrap_mpi_pi` :func:`wrap_0_pi` :func:`wrap_mpi2_pi2`
    """
    v = getvector(v1) - getvector(v2)
    for i, m in enumerate(mode):
        if m == "r":
            pass
        elif m == "c":
            v[i] = wrap_mpi_pi(v[i])
        elif m == "C":
            v[i] = wrap_0_2pi(v[i])
        elif m == "l":
            v[i] = wrap_mpi2_pi2(v[i])
        elif m == "L":
            v[i] = wrap_0_pi(v[i])
        else:
            raise ValueError("bad mode character")

    return v


def removesmall(v: ArrayLike, tol: float = 20) -> NDArray:
    """
    Set small values to zero

    :param v: any vector
    :type v: array_like(n) or ndarray(n,m)
    :param tol: Tolerance in units of eps, defaults to 20
    :type tol: int, optional
    :return: vector with small values set to zero
    :rtype: ndarray(n) or ndarray(n,m)

    Values with absolute value less than ``tol`` will be set to zero.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> a = np.r_[1, 2, 3, 1e-16]
        >>> print(a)
        >>> a = removesmall(a)
        >>> print(a)
        >>> print(a[3])

    """
    return np.where(np.abs(v) < tol * _eps, 0, v)


def project(v1: ArrayLike3, v2: ArrayLike3) -> ArrayLike3:
    """
    Projects vector v1 onto v2. Returns a vector parallel to v2.

    :param v1: vector to be projected
    :type v1: array_like(n)
    :param v2: vector to be projected onto
    :type v2: array_like(n)
    :return: vector projection of v1 onto v2 (parrallel to v2)
    :rtype: ndarray(n)
    """
    return np.dot(v1, v2) * v2


def orthogonalize(v1: ArrayLike3, v2: ArrayLike3, normalize: bool = True) -> ArrayLike3:
    """
    Orthoginalizes vector v1 with respect to v2 with minimum rotation.
    Returns a the nearest vector to v1 that is orthoginal to v2.

    :param v1: vector to be orthoginalized
    :type v1: array_like(n)
    :param v2: vector that returned vector will be orthoginal to
    :type v2: array_like(n)
    :param normalize: whether to normalize the output vector
    :type normalize: bool
    :return: nearest vector to v1 that is orthoginal to v2
    :rtype: ndarray(n)
    """
    v_orth = v1 - project(v1, v2)
    if normalize:
        v_orth = v_orth / np.linalg.norm(v_orth)
    return v_orth


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_vectors.py"
        ).read()
    )  # pylint: disable=exec-used

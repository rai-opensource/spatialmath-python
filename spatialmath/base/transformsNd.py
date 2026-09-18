# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
This modules contains functions to operate on special matrices in 2D or 3D, for
example SE(n), SO(n), se(n) and so(n) where n is 2 or 3.

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.
"""
# pylint: disable=invalid-name

import math
import numpy as np
from spatialmath.base.types import *
from spatialmath.base.argcheck import getvector, isvector
from spatialmath.base.vectors import iszerovec, unitvec_norm

# from spatialmath.base.symbolic import issymbol
# from spatialmath.base.transforms3d import transl
# from spatialmath.base.transforms2d import transl2


try:  # pragma: no cover
    # print('Using SymPy')
    from sympy import Matrix

    _symbolics = True

except ImportError:  # pragma: no cover
    _symbolics = False

_eps = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------------------#
@overload
def r2t(R: SO2Array, check: bool = False) -> SE2Array:
    ...


@overload
def r2t(R: SO3Array, check: bool = False) -> SE3Array:
    ...


def r2t(R, check=False):
    """
    Convert SO(n) to SE(n)

    :param R: rotation matrix
    :type R: ndarray(2,2) or ndarray(3,3)
    :param check: check if rotation matrix is valid (default False, no check)
    :type check: bool
    :return: homogeneous transformation matrix
    :rtype: ndarray(3,3) or ndarray(4,4)
    :raises ValueError: bad argument

    ``T = r2t(R)`` is an SE(2) or SE(3) homogeneous transform equivalent to an
    SO(2) or SO(3) orthonormal rotation matrix ``R`` with a zero translational
    component

    - if ``R`` is 2x2 then ``T`` is 3x3: SO(2) → SE(2)
    - if ``R`` is 3x3 then ``T`` is 4x4: SO(3) → SE(3)

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> R = rot2(0.3)
        >>> R
        >>> r2t(R)

    :seealso: t2r, rt2tr
    """
    if not isinstance(R, np.ndarray):
        raise ValueError("argument must be NumPy array")
    dim = R.shape
    if dim[0] != dim[1]:
        raise ValueError("Matrix must be square")
    n = dim[0] + 1
    m = dim[0]

    if R.dtype == "O":
        # symbolic matrix
        T = np.zeros((n, n), dtype="O")
    else:
        # numeric matrix
        if not isinstance(R, np.ndarray):
            raise ValueError("Argument must be a NumPy array")
        if check and not isR(R):
            raise ValueError("Invalid SO(3) matrix ")

        # T = np.pad(R, (0, 1), mode='constant')
        # T[-1, -1] = 1.0
        T = np.zeros((n, n))
    T[:m, :m] = R
    T[-1, -1] = 1

    return T


# ---------------------------------------------------------------------------------------#
@overload
def t2r(T: SE2Array, check: bool = False) -> SO2Array:
    ...


@overload
def t2r(T: SE3Array, check: bool = False) -> SO3Array:
    ...


def t2r(T: SEnArray, check: bool = False) -> SOnArray:
    """
    Convert SE(n) to SO(n)

    :param T: homogeneous transformation matrix
    :type T: ndarray(3,3) or ndarray(4,4)
    :param check: check if rotation matrix is valid (default False, no check)
    :type check: bool
    :return: rotation matrix
    :rtype: ndarray(2,2) or ndarray(3,3)
    :raises ValueError: bad argument

    ``R = T2R(T)`` is the orthonormal rotation matrix component of homogeneous
    transformation matrix ``T``

    - if ``T`` is 3x3 then ``R`` is 2x2: SE(2) → SO(2)
    - if ``T`` is 4x4 then ``R`` is 3x3: SE(3) → SO(3)

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = trot2(0.3, t=[1,2])
        >>> T
        >>> t2r(T)

    .. note:: Any translational component of T is lost.

    :seealso: r2t, tr2rt
    """
    if not isinstance(T, np.ndarray):
        raise ValueError("argument must be NumPy array")
    dim = T.shape
    if dim[0] != dim[1]:
        raise ValueError("Matrix must be square")

    if dim[0] == 3:
        R = T[:2, :2]
    elif dim[0] == 4:
        R = T[:3, :3]
    else:
        raise ValueError("Value must be an SE(3) matrix")

    if check and not isR(R):
        raise ValueError("Invalid rotation submatrix")

    return R


a = t2r(np.eye(4, dtype="float"))

b = t2r(np.eye(3))

# ---------------------------------------------------------------------------------------#


@overload
def tr2rt(T: SE2Array, check=False) -> Tuple[SO2Array, R2]:
    ...


@overload
def tr2rt(T: SE3Array, check=False) -> Tuple[SO3Array, R3]:
    ...


def tr2rt(T: SEnArray, check=False) -> Tuple[SOnArray, Rn]:
    """
    Convert SE(n) to SO(n) and translation

    :param T: SE(n) matrix
    :type T: ndarray(3,3) or ndarray(4,4)
    :param check: check if SO(3) submatrix is valid (default False, no check)
    :type check: bool
    :return: SO(n) matrix and translation vector
    :rtype: tuple: (ndarray(2,2), ndarray(2)) or (ndarray(3,3), ndarray(3))
    :raises ValueError: bad argument

    (R,t) = tr2rt(T) splits a homogeneous transformation matrix (NxN) into an orthonormal
    rotation matrix R (MxM) and a translation vector T (Mx1), where N=M+1.

    - if ``T`` is 3x3 - in SE(2) - then ``R`` is 2x2 and ``t`` is 2x1.
    - if ``T`` is 4x4 - in SE(3) - then ``R`` is 3x3 and ``t`` is 3x1.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = trot2(0.3, t=[1,2])
        >>> T
        >>> R, t = tr2rt(T)
        >>> R
        >>> t

    :seealso: rt2tr, tr2r
    """
    if not isinstance(T, np.ndarray):
        raise ValueError("argument must be NumPy array")
    dim = T.shape
    if dim[0] != dim[1]:
        raise ValueError("Matrix must be square")

    if dim[0] == 3:
        R = t2r(T, check)
        t = T[:2, 2]
    elif dim[0] == 4:
        R = t2r(T, check)
        t = T[:3, 3]
    else:
        raise ValueError("T must be an SE2 or SE3 homogeneous transformation matrix")

    return (R, t)


# ---------------------------------------------------------------------------------------#


@overload
def rt2tr(R: SO2Array, t: ArrayLike2, check=False) -> SE2Array:
    ...


@overload
def rt2tr(R: SO3Array, t: ArrayLike3, check=False) -> SE3Array:
    ...


def rt2tr(R, t, check=False):
    """
    Convert SO(n) and translation to SE(n)

    :param R: SO(n) matrix
    :type R: ndarray(2,2) or ndarray(3,3)
    :param t: translation vector
    :type R: ndarray(2) or ndarray(3)
    :param check: check if SO(3) matrix is valid (default False, no check)
    :type check: bool
    :return: SE(3) matrix
    :rtype: ndarray(4,4) or (3,3)
    :raises ValueError: bad argument

    ``T = rt2tr(R, t)`` is a homogeneous transformation matrix (N+1xN+1) formed from an
    orthonormal rotation matrix ``R`` (NxN) and a translation vector ``t``
    (Nx1).

    - If ``R`` is 2x2 and ``t`` is 2x1, then ``T`` is 3x3
    - If ``R`` is 3x3 and ``t`` is 3x1, then ``T`` is 4x4

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> R = rot2(0.3)
        >>> t = [1, 2]
        >>> rt2tr(R, t)

    :seealso: rt2m, tr2rt, r2t
    """
    t = getvector(t, dim=None, out="array")
    if not isinstance(R, np.ndarray):
        raise ValueError("Rotation matrix not a NumPy array")
    if R.shape[0] != t.shape[0]:
        raise ValueError("R and t must have the same number of rows")
    if check and not isR(R):
        raise ValueError("Invalid rotation matrix")

    if R.dtype == "O":
        if R.shape == (2, 2):
            T = np.pad(R, ((0, 1), (0, 1)), "constant")  # type: ignore
            T[:2, 2] = t
            T[2, 2] = 1
        elif R.shape == (3, 3):
            T = np.pad(R, ((0, 1), (0, 1)), "constant")  # type: ignore
            T[:3, 3] = t
            T[3, 3] = 1
        else:
            raise ValueError("R must be an SO2 or SO3 rotation matrix")
    else:
        if R.shape == (2, 2):
            T = np.eye(3)
            T[:2, :2] = R
            T[:2, 2] = t
        elif R.shape == (3, 3):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
        else:
            raise ValueError(
                f"R must be an SO(2) or SO(3) rotation matrix, not {R.shape}"
            )

    return T


# ---------------------------------------------------------------------------------------#


def Ab2M(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pack matrix and vector to matrix

    :param A: square matrix
    :type A: ndarray(3,3) or ndarray(2,2)
    :param b: translation vector
    :type b: ndarray(3) or ndarray(2)
    :return: matrix
    :rtype: ndarray(4,4) or ndarray(3,3)
    :raises ValueError: bad arguments

    ``M = Ab2M(A, b)`` is a matrix (N+1xN+1) formed from a matrix ``R`` (NxN) and a vector ``t``
    (Nx1).  The bottom row is all zeros.

    - If ``A`` is 2x2 and ``b`` is 2x1, then ``M`` is 3x3
    - If ``A`` is 3x3 and ``b`` is 3x1, then ``M`` is 4x4

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> A = np.c_[[1, 2], [3, 4]].T
        >>> b = [5, 6]
        >>> Ab2M(A, b)

    :seealso: rt2tr, tr2rt, r2t
    """
    b = getvector(b, dim=None, out="array")
    if not isinstance(A, np.ndarray):
        raise ValueError("Rotation matrix not a NumPy array")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have the same number of rows")

    if A.shape == (2, 2):
        T = np.zeros((3, 3))
        T[:2, :2] = A
        T[:2, 2] = b
    elif A.shape == (3, 3):
        T = np.zeros((4, 4))
        T[:3, :3] = A
        T[:3, 3] = b
    else:
        raise ValueError("A must be 2x2 or 3x3")

    return T


# ======================= predicates


def isR(R: NDArray, tol: float = 20) -> bool:  # -> TypeGuard[SOnArray]:
    r"""
    Test if matrix belongs to SO(n)

    :param R: matrix to test
    :type R: ndarray(2,2) or ndarray(3,3)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether matrix is a proper orthonormal rotation matrix
    :rtype: bool

    Checks orthogonality, ie. :math:`{\bf R} {\bf R}^T = {\bf I}` and :math:`\det({\bf R}) > 0`.
    For the first test we check that the norm of the residual is less than ``tol * eps``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isR(np.eye(3))
        >>> isR(rot2(0.5))
        >>> isR(np.zeros((3,3)))

    :seealso: isrot2, isrot
    """
    n = R.shape[0]
    if n == 3:
        # explicit cofactor expansion avoids the dispatch overhead of
        # np.linalg.det/norm, which dominates cost for such a small matrix
        det = (
            R[0, 0] * (R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1])
            - R[0, 1] * (R[1, 0] * R[2, 2] - R[1, 2] * R[2, 0])
            + R[0, 2] * (R[1, 0] * R[2, 1] - R[1, 1] * R[2, 0])
        )
        if det <= 0:
            return False
        D = R @ R.T
        D[0, 0] -= 1.0
        D[1, 1] -= 1.0
        D[2, 2] -= 1.0
        return bool(np.sum(D * D) < (tol * _eps) ** 2)
    elif n == 2:
        det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
        if det <= 0:
            return False
        D = R @ R.T
        D[0, 0] -= 1.0
        D[1, 1] -= 1.0
        return bool(np.sum(D * D) < (tol * _eps) ** 2)
    else:
        return bool(
            np.linalg.norm(R @ R.T - np.eye(n)) < tol * _eps and np.linalg.det(R) > 0
        )


def isskew(S: NDArray, tol: float = 20) -> bool:  # -> TypeGuard[sonArray]:
    r"""
    Test if matrix belongs to so(n)

    :param S: matrix to test
    :type S: ndarray(2,2) or ndarray(3,3)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool

    Checks skew-symmetry, ie. :math:`{\bf S} + {\bf S}^T = {\bf 0}`.
    We check that the norm of the residual is less than ``tol * eps``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> isskew(np.zeros((3,3)))
        >>> isskew(np.array([[0, -2], [2, 0]]))
        >>> isskew(np.eye(3))

    :seealso: isskewa
    """
    return bool(np.linalg.norm(S + S.T) < tol * _eps)


def isskewa(S: NDArray, tol: float = 20) -> bool:  # -> TypeGuard[senArray]:
    r"""
    Test if matrix belongs to se(n)

    :param S: matrix to test
    :type S: ndarray(3,3) or ndarray(4,4)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool

    Check if matrix is augmented skew-symmetric, ie. the top left (n-1xn-1) partition ``S`` is
    skew-symmetric :math:`{\bf S} + {\bf S}^T = {\bf 0}`, and the bottom row is zero
    We check that the norm of the residual is less than ``tol * eps``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> isskewa(np.zeros((3,3)))
        >>> isskewa(np.array([[0, -2], [2, 0]])) # this matrix is skew but not skewa
        >>> isskewa(np.array([[0, -2, 5], [2, 0, 6], [0, 0, 0]]))

    :seealso: isskew
    """
    return bool(np.linalg.norm(S[0:-1, 0:-1] + S[0:-1, 0:-1].T) < tol * _eps) and all(
        S[-1, :] == 0
    )


def iseye(S: NDArray, tol: float = 20) -> bool:
    """
    Test if matrix is identity

    :param S: matrix to test
    :type S: ndarray(n,n)
    :param tol: tolerance in units of eps, defaults to 20
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool

    Check if matrix is an identity matrix.
    We check that the sum of the absolute value of the residual is less than ``tol * eps``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> iseye(np.array([[1,0], [0,1]]))
        >>> iseye(np.array([[1,2], [0,1]]))

    :seealso: isskew, isskewa
    """
    s = S.shape
    if len(s) != 2 or s[0] != s[1]:
        return False  # not a square matrix
    return bool(np.abs(S - np.eye(s[0])).sum() < tol * _eps)


# ---------------------------------------------------------------------------------------#
@overload
def skew(v: float) -> se2Array:
    ...


@overload
def skew(v: ArrayLike3) -> se3Array:
    ...


def skew(v):
    r"""
    Create skew-symmetric metrix from vector

    :param v: vector
    :type v: array_like(1) or array_like(3)
    :return: skew-symmetric matrix in so(2) or so(3)
    :rtype: ndarray(2,2) or ndarray(3,3)
    :raises ValueError: bad argument

    ``skew(V)`` is a skew-symmetric matrix formed from the elements of ``V``.

    - ``len(V)``  is 1 then ``S`` = :math:`\left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]`
    - ``len(V)`` is 3 then ``S`` = :math:`\left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]`

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> skew(2)
        >>> skew([1, 2, 3])

    .. note::

        - This is the inverse of the function ``vex()``.
        - These are the generator matrices for the Lie algebras so(2) and so(3).

    :seealso: :func:`vex` :func:`skewa`
    :SymPy: supported
    """
    v = getvector(v, None, "sequence")
    if len(v) == 1:
        # fmt: off
        return np.array([
            [0.0,  -v[0]],
            [v[0],  0.0]
        ])  # type: ignore
        # fmt: on
    elif len(v) == 3:
        # fmt: off
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0]
        ])  # type: ignore
        # fmt: on
    else:
        raise ValueError("argument must be a 1- or 3-vector")


# ---------------------------------------------------------------------------------------#
@overload
def vex(s: so2Array, check: bool = False) -> R1:
    ...


@overload
def vex(s: so3Array, check: bool = False) -> R3:
    ...


def vex(s, check=False):
    r"""
    Convert skew-symmetric matrix to vector

    :param s: skew-symmetric matrix
    :type s: ndarray(2,2) or ndarray(3,3)
    :param check: check if matrix is skew symmetric (default False, no check)
    :type check: bool
    :return: vector of unique values
    :rtype: ndarray(1) or ndarray(3)
    :raises ValueError: bad argument

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.

    - ``S`` is 2x2 - so(2) case - where ``S`` :math:`= \left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]` then return :math:`[v]`
    - ``S`` is 3x3 - so(3) case -  where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]` then return :math:`[v_x, v_y, v_z]`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> S = skew(2)
        >>> print(S)
        >>> vex(S)
        >>> S = skew([1, 2, 3])
        >>> print(S)
        >>> vex(S)

    .. note::

        - This is the inverse of the function ``skew()``.
        - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
          is actually skew-symmetric.
        - The function takes the mean of the two elements that correspond to each unique
          element of the matrix.

    :seealso: :func:`skew` :func:`vexa`
    :SymPy: supported
    """
    if s.shape == (3, 3):
        if check and not isskew(s):
            raise ValueError("Argument is not skew symmetric")
        return np.array([s[2, 1] - s[1, 2], s[0, 2] - s[2, 0], s[1, 0] - s[0, 1]]) / 2
    elif s.shape == (2, 2):
        return np.array([s[1, 0] - s[0, 1]]) / 2
    else:
        raise ValueError("Argument must be 2x2 or 3x3 matrix")


# ---------------------------------------------------------------------------------------#
@overload
def skewa(v: ArrayLike3) -> se2Array:
    ...


@overload
def skewa(v: ArrayLike6) -> se3Array:
    ...


def skewa(v: Union[ArrayLike3, ArrayLike6]) -> Union[se2Array, se3Array]:
    r"""
    Create augmented skew-symmetric metrix from vector

    :param v: vector
    :type v: array_like(3), array_like(6)
    :return: augmented skew-symmetric matrix in se(2) or se(3)
    :rtype: ndarray(3,3) or ndarray(4,4)
    :raises ValueError: bad argument

    ``skewa(V)`` is an augmented skew-symmetric matrix formed from the elements of ``V``.

    - ``len(V)`` is 3 then S = :math:`\left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]`
    - ``len(V)`` is 6 then S = :math:`\left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]`

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> skewa([1, 2, 3])
        >>> skewa([1, 2, 3, 4, 5, 6])

    .. note::

        - This is the inverse of the function ``vexa()``.
        - These are the generator matrices for the Lie algebras se(2) and se(3).
        - Map twist vectors in 2D and 3D space to se(2) and se(3).

    :seealso: :func:`vexa` :func:`skew`
    :SymPy: supported
    """

    v = getvector(v, None)
    if len(v) == 3:
        omega = np.zeros((3, 3), dtype=v.dtype)
        omega[:2, :2] = skew(v[2])
        omega[:2, 2] = v[0:2]
        return omega
    elif len(v) == 6:
        omega = np.zeros((4, 4), dtype=v.dtype)
        omega[:3, :3] = skew(v[3:6])
        omega[:3, 3] = v[0:3]
        return omega
    else:
        raise ValueError("expecting a 3- or 6-vector")


@overload
def vexa(Omega: se2Array, check: bool = False) -> R3:
    ...


@overload
def vexa(Omega: se3Array, check: bool = False) -> R6:
    ...


def vexa(Omega: senArray, check: bool = False) -> Union[R3, R6]:
    r"""
    Convert skew-symmetric matrix to vector

    :param s: augmented skew-symmetric matrix
    :type s: ndarray(3,3) or ndarray(4,4)
    :param check: check if matrix is skew symmetric part is valid (default False, no check)
    :type check: bool
    :return: vector of unique values
    :rtype: ndarray(3) or ndarray(6)
    :raises ValueError: bad argument

    ``vexa(S)`` is the vector which has the corresponding augmented skew-symmetric matrix ``S``.

    - ``S`` is 3x3 - se(2) case - where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]` then return :math:`[v_1, v_2, v_3]`.
    - ``S`` is 4x4 - se(3) case -  where ``S`` :math:`= \left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]` then return :math:`[v_1, v_2, v_3, v_4, v_5, v_6]`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> S = skewa([1, 2, 3])
        >>> print(S)
        >>> vexa(S)
        >>> S = skewa([1, 2, 3, 4, 5, 6])
        >>> print(S)
        >>> vexa(S)

    .. note::

        - This is the inverse of the function ``skewa``.
        - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
          is actually skew-symmetric.
        - The function takes the mean of the two elements that correspond to each unique
          element of the matrix.

    :seealso: :func:`skewa` :func:`vex`
    :SymPy: supported
    """
    if Omega.shape == (4, 4):
        return np.hstack((Omega[:3, 3], vex(Omega[:3, :3], check=check)))
    elif Omega.shape == (3, 3):
        return np.hstack((Omega[:2, 2], vex(Omega[:2, :2], check=check)))
    else:
        raise ValueError("expecting a 3x3 or 4x4 matrix")


def h2e(v: NDArray) -> NDArray:
    """
    Convert from homogeneous to Euclidean form

    :param v: homogeneous vector or matrix
    :type v: array_like(n), ndarray(n,m)
    :return: Euclidean vector
    :rtype: ndarray(n-1), ndarray(n-1,m)

    - If ``v`` is an N-vector, return an (N-1)-column vector where the elements have
      all been scaled by the last element of ``v``.
    - If ``v`` is a matrix (NxM), return a matrix (N-1xM), where each column has
      been scaled by its last element.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> h2e([2, 4, 6, 1])
        >>> h2e([2, 4, 6, 2])
        >>> h = np.c_[[1,2,1], [3,4,2], [5,6,1]]
        >>> h
        >>> h2e(h)

    .. note:: The result is always a 2D array, a 1D input results in a column vector.

    :seealso: e2h
    """
    if isinstance(v, np.ndarray) and len(v.shape) == 2:
        # dealing with matrix
        return v[:-1, :] / v[-1, :][np.newaxis, :]

    elif isvector(v):
        # dealing with shape (N,) array
        v = getvector(v, out="col")
        return v[0:-1] / v[-1]

    else:
        raise ValueError("bad type")


def e2h(v: NDArray) -> NDArray:
    """
    Convert from Euclidean to homogeneous form

    :param v: Euclidean vector or matrix
    :type v: array_like(n), ndarray(n,m)
    :return: homogeneous vector
    :rtype: ndarray(n+1,m)

    - If ``v`` is an N-vector, return an (N+1)-column vector where a value of 1 has
      been appended as the last element.
    - If ``v`` is a matrix (NxM), return a matrix (N+1xM), where each column has
      been appended with a value of 1, ie. a row of ones has been appended to the matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> e2h([2, 4, 6])
        >>> e = np.c_[[1,2], [3,4], [5,6]]
        >>> e
        >>> e2h(e)

    .. note:: The result is always a 2D array, a 1D input results in a column vector.

    :seealso: e2h
    """
    if isinstance(v, np.ndarray) and len(v.shape) == 2:
        # dealing with matrix
        return np.vstack([v, np.ones((1, v.shape[1]))])

    elif isvector(v):
        # dealing with shape (N,) array
        v = getvector(v, out="col")
        return np.vstack((v, 1))

    else:
        raise ValueError("bad type")


def homtrans(T: SEnArray, p: np.ndarray) -> np.ndarray:
    r"""
    Apply a homogeneous transformation to a Euclidean vector

    :param T: homogeneous transformation
    :type T: Numpy array (n,n)
    :param p: Vector(s) to be transformed
    :type p: array_like(n-1), ndarray(n-1,m)
    :return: transformed Euclidean vector(s)
    :rtype: ndarray(n-1,m)
    :raises ValueError: bad argument

    - ``homtrans(T, p)`` applies the homogeneous transformation ``T`` to the Euclidean points
      stored columnwise in the array ``p``.

    - ``homtrans(T, v)`` as above but ``v`` is a 1D array considered to be a column vector, and the
      retured value will be a column vector.


    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = trotx(0.3)
        >>> v = [1, 2, 3]
        >>> h2e( T @ e2h(v))
        >>> homtrans(T, v)

    .. note::

        - If T is a homogeneous transformation defining the pose of {B} with respect to {A},
          then the points are defined with respect to frame {B} and are transformed to be
          with respect to frame {A}.

    :seealso: :func:`e2h` :func:`h2e`
    """
    p = e2h(p)
    if p.shape[0] != T.shape[0]:
        raise ValueError("matrices and point data do not conform")

    return h2e(T @ p)


def det(m: np.ndarray) -> float:
    """
    Determinant of matrix

    :param m: any square matrix
    :type v: array_like(n,n)
    :return: determinant
    :rtype: float

    ``det(v)`` is the determinant of the matrix ``m``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> norm([3, 4])

    :seealso: :func:`~numpy.linalg.det`

    :SymPy: supported
    """
    if m.dtype.kind == "O" and _symbolics:
        return Matrix(m).det()
    else:
        return np.linalg.det(m)


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_transformsNd.py"
        ).read()
    )  # pylint: disable=exec-used

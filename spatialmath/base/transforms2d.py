# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
These functions create and manipulate 2D rotation matrices and rigid-body
transformations as 2x2 SO(2) matrices and 3x3 SE(2) matrices respectively.
These matrices are represented as 2D NumPy arrays.

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.

"""

# pylint: disable=invalid-name

import sys
import math
import warnings
import numpy as np

try:
    import matplotlib.pyplot as plt

    _matplotlib_exists = True
except ImportError:
    _matplotlib_exists = False

import spatialmath.base as smb
from spatialmath.base.types import *
from spatialmath.base.transformsNd import rt2tr
from spatialmath.base.vectors import unitvec

_eps = np.finfo(np.float64).eps

try:  # pragma: no cover
    # print('Using SymPy')
    import sympy

    _symbolics = True

except ImportError:  # pragma: no cover
    _symbolics = False


# ---------------------------------------------------------------------------------------#
def rot2(theta: float, unit: str = "rad") -> SO2Array:
    """
    Create SO(2) rotation

    :param theta: rotation angle
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: SO(2) rotation matrix
    :rtype: ndarray(2,2)

    - ``rot2(θ)`` is an SO(2) rotation matrix (2x2) representing a rotation of θ radians.
    - ``rot2(θ, 'deg')`` as above but θ is in degrees.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> rot2(0.3)
        >>> rot2(45, 'deg')
    """
    theta = smb.getunit(theta, unit, vector=False)
    ct = smb.sym.cos(theta)
    st = smb.sym.sin(theta)
    # fmt: off
    R = np.array([
        [ct, -st],
        [st,  ct]])
    # fmt: on
    return R


# ---------------------------------------------------------------------------------------#
def trot2(theta: float, unit: str = "rad", t: Optional[ArrayLike2] = None) -> SE2Array:
    """
    Create SE(2) pure rotation

    :param theta: rotation angle about X-axis
    :type θ: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: 2D translation vector, defaults to [0,0]
    :type t: array_like(2)
    :return: 3x3 homogeneous transformation matrix
    :rtype: ndarray(3,3)

    - ``trot2(θ)`` is a homogeneous transformation (3x3) representing a rotation of
      θ radians.
    - ``trot2(θ, 'deg')`` as above but θ is in degrees.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> trot2(0.3)
        >>> trot2(45, 'deg', t=[1,2])

    .. note:: By default, the translational component is zero but it can be
        set to a non-zero value.

    :seealso: xyt2tr
    """
    T = np.pad(rot2(theta, unit), (0, 1), mode="constant")
    if t is not None:
        T[:2, 2] = smb.getvector(t, 2, "array")
    T[2, 2] = 1  # integer to be symbolic friendly
    return T


def xyt2tr(xyt: ArrayLike3, unit: str = "rad") -> SE2Array:
    """
    Create SE(2) pure rotation

    :param xyt: 2d translation and rotation
    :type xyt: array_like(3)
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: SE(2) matrix
    :rtype: ndarray(3,3)

    - ``xyt2tr([x,y,θ])`` is a homogeneous transformation (3x3) representing a rotation of
      θ radians and a translation of (x,y).

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> xyt2tr([1,2,0.3])
        >>> xyt2tr([1,2,45], 'deg')

    :seealso: tr2xyt
    """
    xyt = smb.getvector(xyt, 3)
    T = np.pad(rot2(xyt[2], unit), (0, 1), mode="constant")
    T[:2, 2] = xyt[0:2]
    T[2, 2] = 1.0
    return T


def tr2xyt(T: SE2Array, unit: str = "rad") -> R3:
    """
    Convert SE(2) to x, y, theta

    :param T: SE(2) matrix
    :type T: ndarray(3,3)
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: [x, y, θ]
    :rtype: ndarray(3)

    - ``tr2xyt(T)`` is a vector giving the equivalent 2D translation and
      rotation for this SO(2) matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = xyt2tr([1, 2, 0.3])
        >>> T
        >>> tr2xyt(T)

    :seealso: trot2
    """

    if T.dtype == "O" and _symbolics:
        angle = sympy.atan2(T[1, 0], T[0, 0])
    else:
        angle = math.atan2(T[1, 0], T[0, 0])
    return np.r_[T[0, 2], T[1, 2], angle]


# ---------------------------------------------------------------------------------------#
@overload  # pragma: no cover
def transl2(x: float, y: float) -> SE2Array:
    ...


@overload  # pragma: no cover
def transl2(x: ArrayLike2) -> SE2Array:
    ...


@overload  # pragma: no cover
def transl2(x: SE2Array) -> R2:
    ...


def transl2(x, y=None):
    """
    Create SE(2) pure translation, or extract translation from SE(2) matrix

    **Create a translational SE(2) matrix**

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :return: SE(2) matrix
    :rtype: ndarray(3,3)

    - ``T = transl2([X, Y])`` is an SE(2) homogeneous transform (3x3)
      representing a pure translation.
    - ``T = transl2( V )`` as above but the translation is given by a 2-element
      list, dict, or a numpy array, row or column vector.


    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> transl2(3, 4)
        >>> transl2([3, 4])
        >>> transl2(np.array([3, 4]))

    **Extract the translational part of an SE(2) matrix**

    :param x: SE(2) transform matrix
    :type x: ndarray(3,3)
    :return: translation elements of SE(2) matrix
    :rtype: ndarray(2)

    - ``t = transl2(T)`` is the translational part of the SE(3) matrix ``T`` as a
      2-element NumPy array.


    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        >>> transl2(T)

    .. note:: This function is compatible with the MATLAB version of the Toolbox.  It
        is unusual/weird in doing two completely different things inside the one
        function.

    :seealso: :func:`tr2pos2` :func:`pos2tr2`
    """

    if smb.isscalar(x) and smb.isscalar(y):
        # (x, y) -> SE(2)
        t = np.array([x, y])
    elif smb.isvector(x, 2):
        # R2 -> SE(2)
        t = cast(NDArray, smb.getvector(x, 2))
    elif smb.ismatrix(x, (3, 3)):
        # SE(2) -> R2
        return x[:2, 2]
    else:
        raise ValueError("bad argument")

    if t.dtype != "O":
        t = t.astype("float64")
    T = np.identity(3, dtype=t.dtype)
    T[:2, 2] = t
    return T


def tr2pos2(T):
    """
    Extract translation from SE(2) matrix

    :param x: SE(2) transform matrix
    :type x: ndarray(3,3)
    :return: translation elements of SE(2) matrix
    :rtype: ndarray(2)

    - ``t = tr2pos2(T)`` is the translational part of the SE(3) matrix ``T`` as a
      2-element NumPy array.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        >>> tr2pos2(T)

    :seealso: :func:`pos2tr2` :func:`transl2`
    """
    return T[:2, 2]


def pos2tr2(x, y=None):
    """
    Create a translational SE(2) matrix

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :return: SE(2) matrix
    :rtype: ndarray(3,3)

    - ``T = pos2tr2([X, Y])`` is an SE(2) homogeneous transform (3x3)
      representing a pure translation.
    - ``T = pos2tr2( V )`` as above but the translation is given by a 2-element
      list, dict, or a numpy array, row or column vector.


    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> pos2tr2(3, 4)
        >>> pos2tr2([3, 4])
        >>> pos2tr2(np.array([3, 4]))

    :seealso: :func:`tr2pos2` :func:`transl2`
    """
    if smb.isscalar(x) and smb.isscalar(y):
        # (x, y) -> SE(2)
        t = np.r_[x, y]
    elif smb.isvector(x, 2):
        # R2 -> SE(2)
        t = cast(NDArray, smb.getvector(x, 2))
    else:
        raise ValueError("bad argument")

    if t.dtype != "O":
        t = t.astype("float64")
    T = np.identity(3, dtype=t.dtype)
    T[:2, 2] = t
    return T


def ishom2(T: Any, check: bool = False, tol: float = 20) -> bool:  # TypeGuard(SE2):
    """
    Test if matrix belongs to SE(2)

    :param T: SE(2) matrix to test
    :type T: ndarray(3,3)
    :param check: check validity of rotation submatrix
    :type check: bool
    :param tol: Tolerance in units of eps for zero-rotation case, defaults to 20
    :type: float
    :return: whether matrix is an SE(2) homogeneous transformation matrix
    :rtype: bool

    - ``ishom2(T)`` is True if the argument ``T`` is of dimension 3x3
    - ``ishom2(T, check=True)`` as above, but also checks orthogonality of the
      rotation sub-matrix and validitity of the bottom row.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        >>> ishom2(T)
        >>> T = np.array([[1, 1, 3], [0, 1, 4], [0, 0, 1]]) # invalid SE(2)
        >>> ishom2(T)  # a quick check says it is an SE(2)
        >>> ishom2(T, check=True) # but if we check more carefully...
        >>> R = np.array([[1, 0], [0, 1]])
        >>> ishom2(R)

    :seealso: isR, isrot2, ishom, isvec
    """
    return (
        isinstance(T, np.ndarray)
        and T.shape == (3, 3)
        and (
            not check
            or (
                smb.isR(T[:2, :2], tol=tol)
                and T[2, 0] == 0
                and T[2, 1] == 0
                and T[2, 2] == 1
            )
        )
    )


def isrot2(R: Any, check: bool = False, tol: float = 20) -> bool:  # TypeGuard(SO2):
    """
    Test if matrix belongs to SO(2)

    :param R: SO(2) matrix to test
    :type R: ndarray(3,3)
    :param check: check validity of rotation submatrix
    :type check: bool
    :param tol: Tolerance in units of eps for zero-rotation case, defaults to 20
    :type: float
    :return: whether matrix is an SO(2) rotation matrix
    :rtype: bool

    - ``isrot2(R)`` is True if the argument ``R`` is of dimension 2x2
    - ``isrot2(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> import numpy as np
        >>> T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        >>> isrot2(T)
        >>> R = np.array([[1, 0], [0, 1]])
        >>> isrot2(R)
        >>> R = np.array([[1, 1], [0, 1]])  # invalid SO(2)
        >>> isrot2(R)  # a quick check says it is an SO(2)
        >>> isrot2(R, check=True)  # but if we check more carefully...

    :seealso: isR, ishom2, isrot
    """
    return (
        isinstance(R, np.ndarray)
        and R.shape == (2, 2)
        and (not check or smb.isR(R, tol=tol))
    )


# ---------------------------------------------------------------------------------------#


def trinv2(T: SE2Array) -> SE2Array:
    r"""
    Invert an SE(2) matrix

    :param T: SE(2) matrix
    :type T: ndarray(3,3)
    :return: inverse of SE(2) matrix
    :rtype: ndarray(3,3)
    :raises ValueError: bad arguments

    Computes an efficient inverse of an SE(2) matrix:

    :math:`\begin{pmatrix} {\bf R} & t \\ 0\,0 & 1 \end{pmatrix}^{-1} =  \begin{pmatrix} {\bf R}^T & -{\bf R}^T t \\ 0\, 0 & 1 \end{pmatrix}`

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = trot2(0.3, t=[4,5])
        >>> trinv2(T)
        >>> T @ trinv2(T)

    :SymPy: supported
    """
    if not ishom2(T):
        raise ValueError("expecting SE(2) matrix")
    # inline this code for speed, don't use tr2rt and rt2tr
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.zeros((3, 3), dtype=T.dtype)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    Ti[2, 2] = 1
    return Ti


@overload  # pragma: no cover
def trlog2(
    T: SO2Array,
    twist: bool = False,
    check: bool = True,
    tol: float = 20,
) -> so2Array:
    ...


@overload  # pragma: no cover
def trlog2(
    T: SE2Array,
    twist: bool = False,
    check: bool = True,
    tol: float = 20,
) -> se2Array:
    ...


@overload  # pragma: no cover
def trlog2(
    T: SO2Array,
    twist: bool = True,
    check: bool = True,
    tol: float = 20,
) -> float:
    ...


@overload  # pragma: no cover
def trlog2(
    T: SE2Array,
    twist: bool = True,
    check: bool = True,
    tol: float = 20,
) -> R3:
    ...


def trlog2(
    T: Union[SO2Array, SE2Array],
    twist: bool = False,
    check: bool = True,
    tol: float = 20,
) -> Union[float, R3, so2Array, se2Array]:
    """
    Logarithm of SO(2) or SE(2) matrix

    :param T: SE(2) or SO(2) matrix
    :type T: ndarray(3,3) or ndarray(2,2)
    :param check: check that matrix is valid
    :type check: bool
    :param twist: return a twist vector instead of matrix [default]
    :type twist: bool
    :param tol: Tolerance in units of eps for zero-rotation case, defaults to 20
    :type: float
    :return: logarithm
    :rtype: ndarray(3,3) or ndarray(3); or ndarray(2,2) or ndarray(1)
    :raises ValueError: bad argument

    An efficient closed-form solution of the matrix logarithm for arguments that
    are SO(2) or SE(2).

    - ``trlog2(R)`` is the logarithm of the passed rotation matrix ``R`` which
      will be 2x2 skew-symmetric matrix.  The equivalent vector from ``vex()``
      is parallel to rotation axis and its norm is the amount of rotation about
      that axis.
    - ``trlog(T)`` is the logarithm of the passed homogeneous transformation
      matrix ``T`` which will be 3x3 augumented skew-symmetric matrix. The
      equivalent vector from ``vexa()`` is the twist vector (6x1) comprising [v
      w].

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> trlog2(trot2(0.3))
        >>> trlog2(trot2(0.3), twist=True)
        >>> trlog2(rot2(0.3))
        >>> trlog2(rot2(0.3), twist=True)

    :seealso: :func:`~trexp`, :func:`~spatialmath.base.transformsNd.vex`,
              :func:`~spatialmath.base.transformsNd.vexa`
    """

    if ishom2(T, check=check, tol=tol):
        # SE(2) matrix

        if smb.iseye(T, tol=tol):
            # is identity matrix
            if twist:
                return np.zeros((3,))
            else:
                return np.zeros((3, 3))
        else:
            st = T[1, 0]
            ct = T[0, 0]
            theta = math.atan(st / ct)
            if abs(theta) < tol * _eps:
                tr = T[:2, 2].flatten()
            else:
                V = np.array([[st, -(1 - ct)], [1 - ct, st]])
                tr = (np.linalg.inv(V) @ T[:2, 2]) * theta
            if twist:
                return np.hstack([tr, theta])
            else:
                return np.block(
                    [[smb.skew(theta), tr[:, np.newaxis]], [np.zeros((1, 3))]]
                )

    elif isrot2(T, check=check, tol=tol):
        # SO(2) rotation matrix
        theta = math.atan(T[1, 0] / T[0, 0])
        if twist:
            return theta
        else:
            return smb.skew(theta)
    else:
        raise ValueError("Expect SO(2) or SE(2) matrix")


# ---------------------------------------------------------------------------------------#
@overload  # pragma: no cover
def trexp2(S: so2Array, theta: Optional[float] = None, check: bool = True) -> SO2Array:
    ...


@overload  # pragma: no cover
def trexp2(S: se2Array, theta: Optional[float] = None, check: bool = True) -> SE2Array:
    ...


def trexp2(
    S: Union[so2Array, se2Array],
    theta: Optional[float] = None,
    check: bool = True,
) -> Union[SO2Array, SE2Array]:
    """
    Exponential of so(2) or se(2) matrix

    :param S: se(2), so(2) matrix or equivalent vector
    :type T: ndarray(3,3) or ndarray(2,2)
    :param theta: motion
    :type theta: float
    :return: matrix exponential in SE(2) or SO(2)
    :rtype: ndarray(3,3) or ndarray(2,2)
    :raises ValueError: bad argument

    An efficient closed-form solution of the matrix exponential for arguments
    that are se(2) or so(2).

    For se(2) the results is an SE(2) homogeneous transformation matrix:

    - ``trexp2(Σ)`` is the matrix exponential of the se(2) element ``Σ`` which is
      a 3x3 augmented skew-symmetric matrix.
    - ``trexp2(Σ, θ)`` as above but for an se(3) motion of Σθ, where ``Σ``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
    - ``trexp2(S)`` is the matrix exponential of the se(2) element ``S`` represented as
      a 3-vector which can be considered a screw motion.
    - ``trexp2(S, θ)`` as above but for an se(2) motion of Sθ, where ``S``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> trexp2(skew(1))
        >>> trexp2(skew(1), 2)  # revolute unit twist
        >>> trexp2(1)
        >>> trexp2(1, 2)  # revolute unit twist

    For so(2) the results is an SO(2) rotation matrix:

    - ``trexp2(Ω)`` is the matrix exponential of the so(3) element ``Ω`` which is a 2x2
      skew-symmetric matrix.
    - ``trexp2(Ω, θ)`` as above but for an so(3) motion of Ωθ, where ``Ω`` is
      unit-norm skew-symmetric matrix representing a rotation axis and a rotation magnitude
      given by ``θ``.
    - ``trexp2(ω)`` is the matrix exponential of the so(2) element ``ω`` expressed as
      a 1-vector.
    - ``trexp2(ω, θ)`` as above but for an so(3) motion of ωθ where ``ω`` is a
      unit-norm vector representing a rotation axis and a rotation magnitude
      given by ``θ``. ``ω`` is expressed as a 1-vector.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> trexp2(skewa([1, 2, 3]))
        >>> trexp2(skewa([1, 0, 0]), 2)  # prismatic unit twist
        >>> trexp2([1, 2, 3])
        >>> trexp2([1, 0, 0], 2)

    :seealso: trlog, trexp2
    """

    if smb.ismatrix(S, (3, 3)) or smb.isvector(S, 3):
        # se(2) case
        if smb.ismatrix(S, (3, 3)):
            # augmentented skew matrix
            if check and not smb.isskewa(S):
                raise ValueError("argument must be a valid se(2) element")
            tw = smb.vexa(cast(se2Array, S))
        else:
            # 3 vector
            tw = smb.getvector(S)

        if smb.iszerovec(tw):
            return np.eye(3)

        if theta is None:
            (tw, theta) = smb.unittwist2_norm(tw)
        elif not smb.isunittwist2(tw):
            raise ValueError("If theta is specified S must be a unit twist")

        t = tw[0:2]
        w = tw[2]

        R = smb.rot2(w * theta)

        skw = smb.skew(w)
        V = (
            np.eye(2) * theta
            + (1.0 - math.cos(theta)) * skw
            + (theta - math.sin(theta)) * skw @ skw
        )

        return smb.rt2tr(R, V @ t)

    elif smb.ismatrix(S, (2, 2)) or smb.isvector(S, 1):
        # so(2) case
        if smb.ismatrix(S, (2, 2)):
            # skew symmetric matrix
            if check and not smb.isskew(S):
                raise ValueError("argument must be a valid so(2) element")
            w = smb.vex(S)
        else:
            # 1 vector
            w = smb.getvector(S)

        if theta is not None:
            if not smb.isunitvec(w):
                raise ValueError("If theta is specified S must be a unit twist")
            w *= theta

        # compute rotation matrix, simpler than Rodrigues for 2D case
        return smb.rot2(w[0])
    else:
        raise ValueError(" First argument must be SO(2), 1-vector, SE(2) or 3-vector")


@overload  # pragma: no cover
def trnorm2(R: SO2Array) -> SO2Array:
    ...


def trnorm2(T: SE2Array) -> SE2Array:
    r"""
    Normalize an SO(2) or SE(2) matrix

    :param T: SE(2) or SO(2) matrix
    :type T: ndarray(3,3) or ndarray(2,2)
    :return: normalized SE(2) or SO(2) matrix
    :rtype: ndarray(3,3) or ndarray(2,2)
    :raises ValueError: bad arguments

    - ``trnorm(R)`` is guaranteed to be a proper orthogonal matrix rotation
      matrix (2,2) which is *close* to the input matrix R (2,2).
    - ``trnorm(T)`` as above but the rotational submatrix of the homogeneous
      transformation T (3,3) is normalised while the translational part is
      unchanged.

    The steps in normalization are:

    #. If :math:`\mathbf{R} = [a, b]`
    #. Form unit vectors :math:`\hat{b}`
    #. Form the orthogonal planar vector :math:`\hat{a} = [\hat{b}_y  -\hat{b}_x]`
    #. Form the normalized SO(2) matrix :math:`\mathbf{R} = [\hat{a}, \hat{b}]`

    .. runblock:: pycon

        >>> from spatialmath.base import trnorm2, trot2
        >>> from numpy import linalg
        >>> T = trot2(45, 'deg', t=[3, 4])
        >>> linalg.det(T[:2,:2]) - 1 # is a valid SO(3)
        >>> T = T @ T @ T @ T @ T @ T @ T @ T @ T @ T @ T @ T @ T
        >>> linalg.det(T[:2,:2]) - 1  # not quite a valid SE(2) anymore
        >>> T = trnorm2(T)
        >>> linalg.det(T[:2,:2]) - 1  # once more a valid SE(2)

    .. note::

        - Only the direction of a-vector (the z-axis) is unchanged.
        - Used to prevent finite word length arithmetic causing transforms to
          become 'unnormalized', ie. determinant :math:`\ne 1`.
    """

    if not ishom2(T) and not isrot2(T):
        raise ValueError("expecting SO(2) or SE(2)")

    a = T[:, 0]
    b = T[:, 1]

    b = unitvec(b)
    # fmt: off
    R = np.array([
        [ b[1], b[0]],
        [-b[0], b[1]]
    ])
    # fmt: on

    if ishom2(T):
        return rt2tr(cast(SO2Array, R), T[:2, 2])
    else:
        return R


@overload  # pragma: no cover
def tr2adjoint2(T: SO2Array) -> R1x1:
    ...


@overload  # pragma: no cover
def tr2adjoint2(T: SE2Array) -> R3x3:
    ...


def tr2adjoint2(T):
    r"""
    Adjoint matrix in 2D

    :param T: SE(2) or SO(2) matrix
    :type T: ndarray(3,3) or ndarray(2,2)
    :return: adjoint matrix
    :rtype: ndarray(3,3) or ndarray(1,1)

    Computes an adjoint matrix that maps the Lie algebra between frames.

    .. math:

        Ad(\mat{T}) \vec{X} X = \vee \left( \mat{T} \skew{\vec{X} \mat{T}^{-1} \right)

    where :math:`\mat{T} \in \SE2`.

    ``tr2jac2(T)`` is an adjoint matrix (6x6) that maps spatial velocity or
    differential motion between frame {B} to frame {A} which are attached to the
    same moving body.  The pose of {B} relative to {A} is represented by the
    homogeneous transform T = :math:`{}^A {\bf T}_B`.

    .. runblock:: pycon

        >>> from spatialmath.base import tr2adjoint2, trot2
        >>> T = trot2(0.3, t=[1,2])
        >>> tr2adjoint2(T)

    :Reference:
        - *Robotics, Vision & Control for Python*, Section 3.1, P. Corke, Springer 2023.
        - `*Lie groups for 2D and 3D Transformations* <http://ethaneade.com/lie.pdf>`_

    :SymPy: supported
    """
    # http://ethaneade.com/lie.pdf
    if T.shape == (2, 2):
        # SO(2) adjoint
        return np.identity(1)
    elif T.shape == (3, 3):
        # SE(2) adjoint
        (R, t) = smb.tr2rt(cast(SE3Array, T))
        # fmt: off
        return np.block([
                [R, np.c_[t[1], -t[0]].T],
                [0, 0,           1]
                ])  # type: ignore
        # fmt: on
    else:
        raise ValueError("bad argument")


def tradjoint2(T):
    """
    Adjoint matrix in 2D (deprecated)

    .. deprecated:: 1.1.16
        Renamed to :func:`tr2adjoint2` for naming consistency.  This alias
        will be removed in a future release.
    """
    warnings.warn(
        "tradjoint2 is deprecated since 1.1.16, use tr2adjoint2 instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return tr2adjoint2(T)


def tr2jac2(T: SE2Array) -> R3x3:
    r"""
    SE(2) Jacobian matrix

    :param T: SE(2) matrix
    :type T: ndarray(3,3)
    :return: Jacobian matrix
    :rtype: ndarray(3,3)

    Computes an Jacobian matrix that maps spatial velocity between two frames defined by
    an SE(2) matrix.

    ``tr2jac2(T)`` is a Jacobian matrix (3x3) that maps spatial velocity or
    differential motion from frame {B} to frame {A} where the pose of {B}
    elative to {A} is represented by the homogeneous transform T = :math:`{}^A {\bf T}_B`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = trot2(0.3, t=[4,5])
        >>> tr2jac2(T)

    :Reference: Robotics, Vision & Control for Python, Section 3.1, P. Corke, Springer 2023.
    :SymPy: supported
    """

    if not ishom2(T):
        raise ValueError("expecting an SE(2) matrix")

    J = np.eye(3, dtype=T.dtype)
    J[:2, :2] = smb.t2r(T)
    return J


@overload
def trinterp2(
    start: Optional[SO2Array], end: SO2Array, s: float, shortest: bool = True
) -> SO2Array:
    ...


@overload
def trinterp2(
    start: Optional[SE2Array], end: SE2Array, s: float, shortest: bool = True
) -> SE2Array:
    ...


def trinterp2(start, end, s, shortest: bool = True):
    """
    Interpolate SE(2) or SO(2) matrices

    :param start: initial SE(2) or SO(2) matrix value when s=0, if None then identity is used
    :type start: ndarray(3,3) or ndarray(2,2) or None
    :param end: final SE(2) or SO(2) matrix, value when s=1
    :type end: ndarray(3,3) or ndarray(2,2)
    :param s: interpolation coefficient, range 0 to 1
    :type s: float
    :param shortest: take the shortest path along the great circle for the rotation
    :type shortest: bool, default to True
    :return: interpolated SE(2) or SO(2) matrix value
    :rtype: ndarray(3,3) or ndarray(2,2)
    :raises ValueError: bad arguments

    - ``trinterp2(None, T, s)`` is an sE(2) matrix interpolated
      between identity when `s=0` and `T`  when `s=1`.
    - ``trinterp2(T0, T1, s)`` as above but interpolated
      between `T0` when `s=0` and `T1` when `s=1`.
    - ``trinterp2(None, R, s)`` is an sO(2) matrix interpolated
      between identity when `s=0` and `R` when `s=1`.
    - ``trinterp2(R0, R1, s)`` as above but interpolated
      between `R0` when `s=0` and `R1` when `s=1`.

    .. note:: Rotation angle is linearly interpolated.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T1 = transl2(1, 2)
        >>> T2 = transl2(3, 4)
        >>> trinterp2(T1, T2, 0)
        >>> trinterp2(T1, T2, 1)
        >>> trinterp2(T1, T2, 0.5)
        >>> trinterp2(None, T2, 0)
        >>> trinterp2(None, T2, 1)
        >>> trinterp2(None, T2, 0.5)

    :seealso: :func:`~spatialmath.base.transforms3d.trinterp`

    """
    if smb.ismatrix(end, (2, 2)):
        # SO(2) case
        if start is None:
            # 	TRINTERP2(T, s)

            th0 = math.atan2(end[1, 0], end[0, 0])

            th = s * th0
        else:
            # 	TRINTERP2(T1, start= s)
            if start.shape != end.shape:
                raise ValueError("start and end matrices must be same shape")

            th0 = math.atan2(start[1, 0], start[0, 0])
            th1 = math.atan2(end[1, 0], end[0, 0])
            if shortest:
                th1 = th0 + smb.wrap_mpi_pi(th1 - th0)

            th = th0 * (1 - s) + s * th1

        return rot2(th)
    elif smb.ismatrix(end, (3, 3)):
        if start is None:
            # 	TRINTERP2(T, s)

            th0 = math.atan2(end[1, 0], end[0, 0])
            p0 = transl2(end)

            th = s * th0
            pr = s * p0
        else:
            # 	TRINTERP2(T0, T1, s)
            if start.shape != end.shape:
                raise ValueError("both matrices must be same shape")

            th0 = math.atan2(start[1, 0], start[0, 0])
            th1 = math.atan2(end[1, 0], end[0, 0])
            if shortest:
                th1 = th0 + smb.wrap_mpi_pi(th1 - th0)

            p0 = transl2(start)
            p1 = transl2(end)

            pr = p0 * (1 - s) + s * p1
            th = th0 * (1 - s) + s * th1

        return smb.rt2tr(rot2(th), pr)
    else:
        raise ValueError("Argument must be SO(2) or SE(2)")


def trprint2(
    T: Union[SO2Array, SE2Array],
    label: str = "",
    file: TextIO = sys.stdout,
    fmt: str = "{:.3g}",
    unit: str = "deg",
) -> str:
    """
    Compact display of SE(2) or SO(2) matrices

    :param T: matrix to format
    :type T: ndarray(3,3) or ndarray(2,2)
    :param label: text label to put at start of line
    :type label: str
    :param file: file to write formatted string to
    :type file: file object
    :param fmt: conversion format for each number
    :type fmt: str
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: formatted string
    :rtype: str

    The matrix is formatted and written to ``file`` and the
    string is returned.  To suppress writing to a file, set ``file=None``.

    - ``trprint2(R)`` displays the SO(2) rotation matrix in a compact
      single-line format and returns the string::

        [LABEL:] θ UNIT

    - ``trprint2(T)`` displays the SE(2) homogoneous transform in a compact
      single-line format and returns the string::

        [LABEL:] [t=X, Y;] θ UNIT

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> T = transl2(1,2) @ trot2(0.3)
        >>> trprint2(T, file=None, label='T')
        >>> trprint2(T, file=None, label='T', fmt='{:8.4g}')


    .. note::

        - Default formatting is for compact display of data
        - For tabular data set ``fmt`` to a fixed width format such as
          ``fmt='{:.3g}'``

    :seealso: trprint
    """

    s = ""

    if label != "":
        s += "{:s}: ".format(label)

    # print the translational part if it exists
    if ishom2(T):
        s += "t = {};".format(_vec2s(fmt, transl2(cast(SE2Array, T))))

    angle = math.atan2(T[1, 0], T[0, 0])
    if unit == "deg":
        angle *= 180.0 / math.pi
        s += " {}°".format(_vec2s(fmt, [angle]))
    else:
        s += " {} rad".format(_vec2s(fmt, [angle]))

    if file:
        print(s, file=file)
    return s


def _vec2s(fmt: str, v: ArrayLikePure, tol: float = 20) -> str:
    """
    Return a string representation for vector using the provided fmt.

    :param fmt: format string for each value in v
    :type fmt: str
    :param tol: Tolerance when checking for near-zero values, in multiples of eps, defaults to 20
    :type tol: float, optional
    :return: string representation for the vector
    :rtype: str

    Return a string representation for vector using the provided fmt, where
    near-zero values are rounded to 0.
    """

    v = [x if np.abs(x) > tol * _eps else 0.0 for x in v]
    return ", ".join([fmt.format(x) for x in v])


def points2tr2(p1: NDArray, p2: NDArray) -> SE2Array:
    """
    SE(2) transform from corresponding points

    :param p1: first set of points
    :type p1: array_like(2,N)
    :param p2: second set of points
    :type p2: array_like(2,N)
    :return: transform from ``p1`` to ``p2``
    :rtype: ndarray(3,3)

    Compute an SE(2) matrix that transforms the point set ``p1`` to ``p2``.
    p1 and p2 must have the same number of columns, and columns correspond
    to the same point.

    :seealso: :func:`ICP2d`
    """

    # first find the centroids of both point clouds
    p1_centroid = np.mean(p1, axis=1)
    p2_centroid = np.mean(p2, axis=1)

    # get the point clouds in reference to their centroids
    p1_centered = p1 - p1_centroid[:, np.newaxis]
    p2_centered = p2 - p2_centroid[:, np.newaxis]

    # compute moment matrix
    M = np.dot(p2_centered, p1_centered.T)

    # get singular value decomposition of the cross covariance matrix, use Umeyama trick
    U, W, VT = np.linalg.svd(M)

    # get rotation between the two point clouds
    s = [1, np.linalg.det(U) * np.linalg.det(VT)]
    R = U @ np.diag(s) @ VT

    # get the translation
    t = p2_centroid - R @ p1_centroid

    return rt2tr(R, t)


def ICP2d(
    reference: Points2,
    source: Points2,
    T: Optional[SE2Array] = None,
    max_iter: int = 20,
    min_delta_err: float = 1e-4,
) -> SE2Array:
    """
    Iterated closest point (ICP) in 2D

    :param reference: points (columns) to which the source points are to be aligned
    :type reference: ndarray(2,N)
    :param source: points (columns) to align to the reference set of points
    :type source: ndarray(2,M)
    :param T: initial pose , defaults to None
    :type T: ndarray(3,3), optional
    :param max_iter: max number of iterations, defaults to 20
    :type max_iter: int, optional
    :param min_delta_err: min_delta_err, defaults to 1e-4
    :type min_delta_err: float, optional
    :return: pose of source point cloud relative to the reference point cloud
    :rtype: SE2Array

    Uses the iterative closest point algorithm to find the transformation that
    transforms the source point cloud to align with the reference point cloud, which
    minimizes the sum of squared errors between nearest neighbors in the two point
    clouds.

    .. note:: Point correspondence is not required and the two point clouds do not have
        to have the same number of points.

    .. warning:: The point cloud argument order is reversed compared to :func:`points2tr`.

    :seealso: :func:`points2tr`
    """

    # https://github.com/ClayFlannigan/icp/blob/master/icp.py
    # https://github.com/1988kramer/intel_dataset/blob/master/scripts/Align2D.py
    # hack below to use points2tr above
    # use ClayFlannigan's improved data association

    from scipy.spatial import KDTree

    def _FindCorrespondences(
        tree, source, reference
    ) -> Tuple[NDArray, NDArray, NDArray]:
        # get distances to nearest neighbors and indices of nearest neighbors
        dist, indices = tree.query(source.T)

        # remove multiple associatons from index list
        # only retain closest associations
        unique = False
        matched_src = source.copy()
        while not unique:
            unique = True
            for i, idxi in enumerate(indices):
                if idxi == -1:
                    continue
                # could do this with np.nonzero
                for j in range(i + 1, len(indices)):
                    if idxi == indices[j]:
                        if dist[i] < dist[j]:
                            indices[j] = -1
                        else:
                            indices[i] = -1
                            break
        # build array of nearest neighbor reference points
        # and remove unmatched source points
        point_list = []
        src_idx = 0
        for idx in indices:
            if idx != -1:
                point_list.append(reference[:, idx])
                src_idx += 1
            else:
                matched_src = np.delete(matched_src, src_idx, axis=1)

        matched_ref = np.array(point_list).T

        return matched_ref, matched_src, indices

    mean_sq_error = 1.0e6  # initialize error as large number
    delta_err = 1.0e6  # change in error (used in stopping condition)
    num_iter = 0  # number of iterations
    if T is None:
        T = np.eye(3)

    ref_kdtree = KDTree(reference.T)

    source_hom = np.vstack((source, np.ones(source.shape[1])))

    # tf_source = source
    tf_source = cast(NDArray, T) @ source_hom
    tf_source = tf_source[:2, :]

    while delta_err > min_delta_err and num_iter < max_iter:
        # find correspondences via nearest-neighbor search
        matched_ref_pts, matched_source, indices = _FindCorrespondences(
            ref_kdtree, tf_source, reference
        )

        # find alignment between source and corresponding reference points via SVD
        # note: svd step doesn't use homogeneous points
        new_T = points2tr2(matched_source, matched_ref_pts)

        # update transformation between point sets
        T = T @ new_T

        # apply transformation to the source points
        tf_source = cast(NDArray, T) @ source_hom
        tf_source = tf_source[:2, :]

        # find mean squared error between transformed source points and reference points
        # TODO: do this with fancy indexing
        new_err = 0
        for i in range(len(indices)):
            if indices[i] != -1:
                diff = tf_source[:, i] - reference[:, indices[i]]
                new_err += np.dot(diff, diff.T)

        new_err /= float(len(matched_ref_pts))

        # update error and calculate delta error
        delta_err = abs(mean_sq_error - new_err)
        mean_sq_error = new_err
        print("ITER", num_iter, delta_err, mean_sq_error)

        num_iter += 1

    return T


if _matplotlib_exists:
    import matplotlib.pyplot as plt

    # from mpl_toolkits.axisartist import Axes
    from matplotlib.axes import Axes

    def trplot2(
        T: Union[SO2Array, SE2Array],
        color: str = "blue",
        frame: Optional[str] = None,
        axislabel: bool = True,
        axissubscript: bool = True,
        textcolor: Optional[Color] = None,
        labels: Tuple[str, str] = ("X", "Y"),
        length: float = 1,
        arrow: bool = True,
        originsize: float = 20,
        rviz: bool = False,
        ax: Optional[Axes] = None,
        block: Optional[bool] = None,
        dims: Optional[ArrayLike] = None,
        wtl: float = 0.2,
        width: float = 1,
        d1: float = 0.1,
        d2: float = 1.15,
        **kwargs,
    ):
        """
        Plot a 2D coordinate frame

        :param T: an SE(3) or SO(3) pose to be displayed as coordinate frame
        :type: ndarray(3,3) or ndarray(2,2)
        :param color: color of the lines defining the frame
        :type color: str
        :param textcolor: color of text labels for the frame, default color of lines above
        :type textcolor: str
        :param frame: label the frame, name is shown below the frame and as subscripts on the frame axis labels
        :type frame: str
        :param axislabel: display labels on axes, default True
        :type axislabel: bool
        :param axissubscript: display subscripts on axis labels, default True
        :type axissubscript: bool
        :param labels: labels for the axes, defaults to X and Y
        :type labels: 2-tuple of strings
        :param length: length of coordinate frame axes, default 1
        :type length: float
        :param arrow: show arrow heads, default True
        :type arrow: bool
        :param ax: the axes to plot into, defaults to current axes
        :type ax: Axes3D reference
        :param block: run the GUI main loop until all windows are closed, default None
        :type block: bool
        :param dims: dimension of plot volume as [xmin, xmax, ymin, ymax]
        :type dims: array_like(4)
        :param wtl: width-to-length ratio for arrows, default 0.2
        :type wtl: float
        :param rviz: show Rviz style arrows, default False
        :type rviz: bool
        :param width: width of lines, default 1
        :type width: float
        :param d1: distance of frame axis label text from origin, default 0.05
        :type d1: float
        :param d2: distance of frame label text from origin, default 1.15
        :type d2: float
        :return: axes containing the frame
        :rtype: AxesSubplot
        :raises ValueError: bad argument

        Adds a 2D coordinate frame represented by the SO(2) or SE(2) matrix to the current axes.

        The appearance of the coordinate frame depends on many parameters:

        - coordinate axes depend on:

            - ``color`` of axes
            - ``width`` of line
            - ``length`` of line
            - ``arrow`` if True [default] draw the axis with an arrow head

        - coordinate axis labels depend on:

            - ``axislabel`` if True [default] label the axis, default labels are X, Y, Z
            - ``labels`` 2-list of alternative axis labels
            - ``textcolor`` which defaults to ``color``
            - ``axissubscript`` if True [default] add the frame label ``frame`` as a subscript
            for each axis label

        - coordinate frame label depends on:

            - `frame` the label placed inside {...} near the origin of the frame

        - a dot at the origin

            - ``originsize`` size of the dot, if zero no dot
            - ``origincolor`` color of the dot, defaults to ``color``
            - If no current figure, one is created
            - If current figure, but no axes, a 3d Axes is created

        Examples::

            trplot2(T, frame='A')
            trplot2(T, frame='A', color='green')
            trplot2(T1, 'labels', 'AB');

        .. plot::

            import matplotlib.pyplot as plt
            from spatialmath.base import trplot2, transl2, trot2
            import math
            fig, ax = plt.subplots(3,3, figsize=(10,10))
            text_opts = dict(bbox=dict(boxstyle="round",
                fc="w",
                alpha=0.9),
                zorder=20,
                family='monospace',
                fontsize=8,
                verticalalignment='top')
            T = transl2(2, 1)@trot2(math.pi/3)
            trplot2(T, ax=ax[0][0], dims=[0,4,0,4])
            ax[0][0].text(0.2, 3.8, "trplot2(T)", **text_opts)
            trplot2(T, ax=ax[0][1], dims=[0,4,0,4], originsize=0)
            ax[0][1].text(0.2, 3.8, "trplot2(T, originsize=0)", **text_opts)
            trplot2(T, ax=ax[0][2], dims=[0,4,0,4], arrow=False)
            ax[0][2].text(0.2, 3.8, "trplot2(T, arrow=False)", **text_opts)
            trplot2(T, ax=ax[1][0], dims=[0,4,0,4], axislabel=False)
            ax[1][0].text(0.2, 3.8, "trplot2(T, axislabel=False)", **text_opts)
            trplot2(T, ax=ax[1][1], dims=[0,4,0,4], width=3)
            ax[1][1].text(0.2, 3.8, "trplot2(T, width=3)", **text_opts)
            trplot2(T, ax=ax[1][2], dims=[0,4,0,4], frame='B')
            ax[1][2].text(0.2, 3.8, "trplot2(T, frame='B')", **text_opts)
            trplot2(T, ax=ax[2][0], dims=[0,4,0,4], color='r', textcolor='k')
            ax[2][0].text(0.2, 3.8, "trplot2(T, color='r',textcolor='k')", **text_opts)
            trplot2(T, ax=ax[2][1], dims=[0,4,0,4], labels=("u", "v"))
            ax[2][1].text(0.2, 3.8, "trplot2(T, labels=('u', 'v'))", **text_opts)
            trplot2(T, ax=ax[2][2], dims=[0,4,0,4], rviz=True)
            ax[2][2].text(0.2, 3.8, "trplot2(T, rviz=True)", **text_opts)


        :SymPy: not supported

        :seealso: :func:`tranimate2` :func:`plotvol2` :func:`axes_logic`
        """

        # TODO
        # animation
        # style='line', 'arrow', 'rviz'

        # check input types
        if isrot2(T, check=True):
            T = smb.r2t(cast(SO2Array, T))
        elif not ishom2(T, check=True):
            raise ValueError("argument is not valid SE(2) matrix")

        ax = smb.axes_logic(ax, 2)

        try:
            if not ax.get_xlabel():
                ax.set_xlabel(labels[0])
            if not ax.get_ylabel():
                ax.set_ylabel(labels[1])
        except AttributeError:
            pass  # if axes are an Animate object

        if not hasattr(ax, "_plotvol"):
            ax.set_aspect("equal")

        if dims is not None:
            ax.axis(smb.expand_dims(dims))
        elif not hasattr(ax, "_plotvol"):
            ax.autoscale(enable=True, axis="both")

        # create unit vectors in homogeneous form
        o = T @ np.array([0, 0, 1])
        x = T @ np.array([length, 0, 1])
        y = T @ np.array([0, length, 1])

        # draw the axes

        if rviz:
            ax.plot([o[0], x[0]], [o[1], x[1]], color="red", linewidth=5 * width)
            ax.plot([o[0], y[0]], [o[1], y[1]], color="lime", linewidth=5 * width)
        elif arrow:
            ax.quiver(
                o[0],
                o[1],
                x[0] - o[0],
                x[1] - o[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                linewidth=width,
                facecolor=color,
                edgecolor=color,
            )
            ax.quiver(
                o[0],
                o[1],
                y[0] - o[0],
                y[1] - o[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                linewidth=width,
                facecolor=color,
                edgecolor=color,
            )
        else:
            ax.plot([o[0], x[0]], [o[1], x[1]], color=color, linewidth=width)
            ax.plot([o[0], y[0]], [o[1], y[1]], color=color, linewidth=width)

        if originsize > 0:
            ax.scatter(x=[o[0], x[0], y[0]], y=[o[1], x[1], y[1]], s=[originsize, 0, 0])

        # label the frame
        if frame:
            if textcolor is not None:
                color = textcolor

            o1 = T @ np.array([-d1, -d1, 1])
            ax.text(
                o1[0],
                o1[1],
                r"$\{" + frame + r"\}$",
                color=color,
                verticalalignment="top",
                horizontalalignment="left",
            )

        if axislabel:
            if textcolor is not None:
                color = textcolor
            # add the labels to each axis
            x = (x - o) * d2 + o
            y = (y - o) * d2 + o

            if frame is None or not axissubscript:
                format = "${:s}$"
            else:
                format = "${:s}_{{{:s}}}$"

            ax.text(
                x[0],
                x[1],
                format.format(labels[0], frame),
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.text(
                y[0],
                y[1],
                format.format(labels[1], frame),
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
            )

        if block is not None:
            # calling this at all, causes FuncAnimation to fail so when invoked from tranimate2 skip this bit
            plt.show(block=block)
        return ax

    def tranimate2(T: Union[SO2Array, SE2Array], **kwargs):
        """
        Animate a 2D coordinate frame

        :param T: an SE(2) or SO(2) pose to be displayed as coordinate frame
        :type: ndarray(3,3) or ndarray(2,2)
        :param nframes: number of steps in the animation [defaault 100]
        :type nframes: int
        :param repeat: animate in endless loop [default False]
        :type repeat: bool
        :param interval: number of milliseconds between frames [default 50]
        :type interval: int
        :param movie: name of file to write MP4 movie into
        :type movie: str

        Animates a 2D coordinate frame moving from the world frame to a frame represented by the SO(2) or SE(2) matrix to the current axes.

        - If no current figure, one is created
        - If current figure, but no axes, a 3d Axes is created


        Examples:

                tranimate2(transl(1,2)@trot2(1), frame='A', arrow=False, dims=[0, 5])
                tranimate2(transl(1,2)@trot2(1), frame='A', arrow=False, dims=[0, 5], movie='spin.mp4')
        """
        dims = kwargs.pop("dims", None)
        ax = kwargs.pop("ax", None)
        anim = smb.animate.Animate2(dims=dims, axes=ax, **kwargs)
        anim.trplot2(T, **kwargs)
        return anim.run(**kwargs)


if __name__ == "__main__":  # pragma: no cover
    import pathlib
    import matplotlib.pyplot as plt

    # trplot2( transl2(1,2), frame='A', rviz=True, width=1)
    # trplot2( transl2(3,1), color='red', arrow=True, width=3, frame='B')
    # trplot2( transl2(4, 3)@trot2(math.pi/3), color='green', frame='c')
    # plt.grid(True)

    # fig, ax = plt.subplots(3,3, figsize=(10,10))
    # text_opts = dict(bbox=dict(boxstyle="round",
    #     fc="w",
    #     alpha=0.9),
    #     zorder=20,
    #     family='monospace',
    #     fontsize=8,
    #     verticalalignment='top')
    # T = transl2(2, 1)@trot2(math.pi/3)
    # trplot2(T, ax=ax[0][0], dims=[0,4,0,4])
    # ax[0][0].text(0.2, 3.8, "trplot2(T)", **text_opts)

    # trplot2(T, ax=ax[0][1], dims=[0,4,0,4], originsize=0)
    # ax[0][1].text(0.2, 3.8, "trplot2(T, originsize=0)", **text_opts)

    # trplot2(T, ax=ax[0][2], dims=[0,4,0,4], arrow=False)
    # ax[0][2].text(0.2, 3.8, "trplot2(T, arrow=False)", **text_opts)

    # trplot2(T, ax=ax[1][0], dims=[0,4,0,4], axislabel=False)
    # ax[1][0].text(0.2, 3.8, "trplot2(T, axislabel=False)", **text_opts)

    # trplot2(T, ax=ax[1][1], dims=[0,4,0,4], width=3)
    # ax[1][1].text(0.2, 3.8, "trplot2(T, width=3)", **text_opts)

    # trplot2(T, ax=ax[1][2], dims=[0,4,0,4], frame='B')
    # ax[1][2].text(0.2, 3.8, "trplot2(T, frame='B')", **text_opts)

    # trplot2(T, ax=ax[2][0], dims=[0,4,0,4], color='r', textcolor='k')
    # ax[2][0].text(0.2, 3.8, "trplot2(T, color='r',\n        textcolor='k')", **text_opts)

    # trplot2(T, ax=ax[2][1], dims=[0,4,0,4], labels=("u", "v"))
    # ax[2][1].text(0.2, 3.8, "trplot2(T, labels=('u', 'v'))", **text_opts)

    # trplot2(T, ax=ax[2][2], dims=[0,4,0,4], rviz=True)
    # ax[2][2].text(0.2, 3.8, "trplot2(T, rviz=True)", **text_opts)

    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_transforms2d.py"
        ).read()
    )  # pylint: disable=exec-used

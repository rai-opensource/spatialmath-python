"""
Classes to abstract quaternions and unit-quaternions.

To use::

    from spatialmath.quaternion import *
    T = UnitQuaternion.Rx(0.3)

    import spatialmath as sm
    T = sm.UnitQuaternion.Rx(0.3)

 .. inheritance-diagram:: spatialmath.quaternion
    :top-classes: collections.UserList
    :parts: 1
"""
# pylint: disable=invalid-name
from __future__ import annotations
import math
import numpy as np
from typing import Any
import spatialmath.base as smb
from spatialmath.pose3d import SO3, SE3
from spatialmath.baseposelist import BasePoseList
from spatialmath.base.types import *

_eps = np.finfo(np.float64).eps


class Quaternion(BasePoseList):
    r"""
    Quaternion class

    A quaternion can be considered an ordered pair :math:`(s, \vec{v})`
    where :math:`s \in \mathbb{R}` is the *scalar* part and :math:`\vec{v} = (v_x, v_y, v_z) \in \mathbb{R}^3`
    is the *vector* part and is often written as

    .. math:: \q = s \langle v_x, v_y, v_z \rangle

    .. inheritance-diagram:: spatialmath.quaternion.Quaternion
       :top-classes: collections.UserList
       :parts: 1
    """

    def __init__(self, s: Any = None, v=None, check: Optional[bool] = True):
        r"""
        Construct a new quaternion

        :param s: scalar part
        :type s: float or ndarray(N)
        :param v: vector part
        :type v: ndarray(3), ndarray(Nx3)

        - ``Quaternion()`` constructs a zero quaternion
        - ``Quaternion(s, v)`` construct a new quaternion from the scalar ``s``
          and the vector ``v``
        - ``Quaternion(q)`` construct a new quaternion from the 4-vector
          ``q = [s, v]``
        - ``Quaternion([q1, q2 .. qN])`` construct a new quaternion with ``N``
          values where each element is a 4-vector
        - ``Quaternion([Q1, Q2 .. QN])`` construct a new quaternion with ``N``
          values where each element is a Quaternion instance
        - ``Quaternion(M)`` construct a new quaternion with ``N`` values where
          ``Q`` is a 4xN NumPy array.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion()
            >>> Quaternion(1, [2,3,4])
            >>> Quaternion([1,2,3,4])
            >>> q=Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            >>> len(q)
            >>> print(q)

        """
        super().__init__()

        if s is None and smb.isvector(v, 4):
            v, s = (s, v)

        if v is None:
            # single argument
            if super().arghandler(s, check=False):
                return

            elif smb.isvector(s, 4):
                self.data = [smb.getvector(s)]

        elif smb.isscalar(s) and smb.isvector(v, 3):
            # Quaternion(s, v)
            self.data = [np.r_[s, smb.getvector(v)]]

        elif (
            smb.isvector(s) and smb.ismatrix(v, (None, 3)) and s.shape[0] == v.shape[0]
        ):
            # Quaternion(s, v) where s and v are arrays
            self.data = [np.r_[_s, _v] for _s, _v in zip(s, v)]
        else:
            raise ValueError("bad argument to Quaternion constructor")

    @classmethod
    def Pure(cls, v: ArrayLike3) -> Quaternion:
        r"""
        Construct a pure quaternion from a vector

        :param v: vector
        :type v: 3-element array_like

        ``Quaternion.Pure(v)`` is a Quaternion with a zero scalar part and the
        vector part set to ``v``,
        ie. :math:`q = 0 \langle v_x, v_y, v_z \rangle`

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> print(Quaternion.Pure([1,2,3]))
        """
        return cls(s=0, v=smb.getvector(v, 3))

    @staticmethod
    def _identity():
        return np.zeros((4,))

    @property
    def shape(self) -> Tuple[int]:
        """
        Shape of the object's interal matrix representation

        :return: (4,)
        :rtype: tuple
        """
        return (4,)

    @staticmethod
    def isvalid(x: ArrayLike4) -> bool:
        """
        Test if vector is valid quaternion

        :param x: vector to test
        :type x: numpy.ndarray
        :arg check: explicitly check vector is unit length [default True]
        :type check: bool
        :return: True if the matrix has shape (4,).
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> import numpy as np
            >>> Quaternion.isvalid(np.r_[1, 0, 0, 0])
            >>> Quaternion.isvalid(np.r_[1, 2, 3, 4])
        """
        return x.shape == (4,)

    @property
    def s(self) -> float:
        """
        Scalar part of quaternion

        :return: scalar part of quaternion
        :rtype: float or numpy.ndarray

        ``q.s`` is the scalar part.  If `len(q)` is:

            - 1, return a scalar float
            - N>1, return a NumPy array shape=(N,) is returned.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]).s
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).s

        """
        if len(self) == 1:
            return self._A[0]
        else:
            return np.array([q.s for q in self])

    @property
    def v(self) -> R3:
        """
        Vector part of quaternion

        :return: vector part of quaternion
        :rtype: NumPy ndarray

        ``q.v`` is the vector part.  If `len(q)` is:

            - 1, return a NumPy array shape=(3,)
            - N>1, return a NumPy array shape=(N,3).

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]).v
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).v

        """
        if len(self) == 1:
            return self._A[1:4]
        else:
            return np.array([q.v for q in self])

    @property
    def vec(self) -> R4:
        """
        Quaternion as a vector

        :return: quaternion expressed as a 4-vector
        :rtype: numpy ndarray, shape=(4,)

        ``q.vec`` is the quaternion as a vector.  If `len(q)` is:

            - 1, return a NumPy array shape=(4,)
            - N>1, return a NumPy array shape=(N,4).

        The quaternion coefficients are in the order (s, vx, vy, vz), ie. with
        the scalar (real part) first.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]).vec
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).vec
        """
        if len(self) == 1:
            return self._A
        else:
            return np.array([q._A for q in self])

    @property
    def vec_xyzs(self) -> R4:
        """
        Quaternion as a vector

        :return: quaternion expressed as a 4-vector
        :rtype: numpy ndarray, shape=(4,)

        ``q.vec_xyzs`` is the quaternion as a vector.  If `len(q)` is:

            - 1, return a NumPy array shape=(4,)
            - N>1, return a NumPy array shape=(N,4).

        The quaternion coefficients are in the order (vx, vy, vz, s), ie. with
        the scalar (real part) last. This is useful when exporting to other
        packages like three.js or pybullet.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]).vec_xyzs
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).vec_xyzs
        """
        if len(self) == 1:
            return np.roll(self._A, -1)
        else:
            return np.array([np.roll(q._A, -1) for q in self])

    @property
    def matrix(self) -> R4x4:
        """
        Matrix equivalent of quaternion

        :rtype: Numpy array, shape=(4,4)

        ``q.matrix`` is a 4x4 matrix which encodes the arithmetic rules of Hamilton multiplication.
        This matrix, multiplied by the 4-vector equivalent of a second quaternion, results in the 4-vector
        equivalent of the Hamilton product.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]).matrix
            >>> Quaternion([1,2,3,4]) * Quaternion([5,6,7,8])   # Hamilton product
            >>> Quaternion([1,2,3,4]).matrix @ Quaternion([5,6,7,8]).vec  # matrix-vector product

        :seealso: :func:`~spatialmath.base.quaternions.qmatrix`
        """

        return smb.qmatrix(self._A)

    def conj(self) -> Quaternion:
        r"""
        Conjugate of quaternion

        :rtype: Quaternion instance

        ``q.conj()`` is the quaternion ``q`` with the vector part negated, ie.
        :math:`q = s \langle -v_x, -v_y, -v_z \rangle`

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> print(Quaternion.Pure([1,2,3]).conj())

        :seealso: :func:`~spatialmath.base.quaternions.qconj`
        """

        return self.__class__([smb.qconj(q._A) for q in self])

    def norm(self) -> float:
        r"""
        Norm of quaternion

        :rtype: float

        ``q.norm()`` is the norm or length of the quaternion
        :math:`\sqrt{s^2 + v_x^2 + v_y^2 + v_z^2}`


        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]).norm()
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).norm()

        :seealso: :func:`~spatialmath.base.quaternions.qnorm`
        """
        if len(self) == 1:
            return smb.qnorm(self._A)
        else:
            return np.array([smb.qnorm(q._A) for q in self])

    def unit(self) -> UnitQuaternion:
        r"""
        Unit quaternion

        :rtype: UnitQuaternion instance

        ``q.unit()`` is the quaternion ``q`` normalized to have a unit length.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q = Quaternion([1,2,3,4])
            >>> print(q)
            >>> print(q.unit())
            >>> print(Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).unit())

        Note that the return type is different, a ``UnitQuaternion``, which is
        distinguished by the use of double angle brackets to delimit the
        vector part.

        :seealso: :func:`~spatialmath.base.quaternions.qnorm`
        """
        return UnitQuaternion([smb.qunit(q._A) for q in self], norm=False)

    def log(self) -> Quaternion:
        r"""
        Logarithm of quaternion

        :rtype: Quaternion instance

        ``q.log()`` is the logarithm of the quaternion ``q``, ie.

        .. math::

             \ln \| q \|,  \langle \frac{\vec{v}}{\| \vec{v} \|} \cos^{-1} \frac{s}{\| q \|} \rangle

        For a ``UnitQuaternion`` the logarithm is a pure quaternion whose vector
        part :math:`\vec{v}` and :math:`\vec{v}/2` is a Euler vector: parallel
        to the axis of rotation and whose norm is the magnitude of rotation.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion, UnitQuaternion
            >>> from math import pi
            >>> q = Quaternion([1, 2, 3, 4])
            >>> print(q.log())
            >>> q = UnitQuaternion.Rx(pi / 2)
            >>> print(q.log())

        :reference: `Wikipedia <https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions>`_

        :seealso: :meth:`Quaternion.exp` :meth:`Quaternion.log` :meth:`UnitQuaternion.angvec`
        """
        norm = self.norm()
        s = np.log(norm)
        if len(self) == 1:
            if smb.iszerovec(self._A[1:4]):
                v = np.zeros((3,))
            else:
                v = math.acos(np.clip(self._A[0] / norm, -1, 1)) * smb.unitvec(
                    self._A[1:4]
                )
            return Quaternion(s=s, v=v)
        else:
            v = [
                np.zeros((3,))
                if smb.iszerovec(A[1:4])
                else math.acos(np.clip(A[0] / n, -1, 1)) * smb.unitvec(A[1:4])
                for A, n in zip(self._A, norm)
            ]
            return Quaternion(s=s, v=np.array(v))

    def exp(self, tol: float = 20) -> Quaternion:
        r"""
        Exponential of quaternion

        :param tol: Tolerance when checking for pure quaternion, in multiples of eps, defaults to 20
        :type tol: float, optional
        :rtype: Quaternion instance

        ``q.exp()`` is the exponential of the quaternion ``q``, ie.

        .. math::

             e^s \cos \| v \|,  \langle e^s \frac{\vec{v}}{\| \vec{v} \|} \sin \| \vec{v} \| \rangle

        For a pure quaternion with vector value :math:`\vec{v}` the the result
        is a unit quaternion equivalent to a rotation defined by
        :math:`2\vec{v}` intepretted as an Euler vector, that is, parallel to
        the axis of rotation and whose norm is the magnitude of rotation.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> from math import pi
            >>> q = Quaternion([1, 2, 3, 4])
            >>> print(q.exp())
            >>> q = Quaternion.Pure([pi / 4, 0, 0])
            >>> print(q.exp())  # result is a UnitQuaternion
            >>> print(q.exp().angvec())

        :reference: `Wikipedia <https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions>`_

        :seealso: :meth:`Quaternion.log` :meth:`UnitQuaternion.log` :meth:`UnitQuaternion.AngVec` :meth:`UnitQuaternion.EulerVec`
        """
        exp_s = math.exp(self.s)
        norm_v = smb.norm(self.v)
        s = exp_s * math.cos(norm_v)
        if smb.iszerovec(self.v, tol * _eps):
            # result will be a unit quaternion
            v = np.zeros((3,))
        else:
            v = exp_s * self.v / norm_v * math.sin(norm_v)
        if abs(self.s) < tol * _eps:
            # result will be a unit quaternion
            return UnitQuaternion(s=s, v=v)
        else:
            return Quaternion(s=s, v=v)

    def inner(self, other) -> float:
        """
        Inner product of quaternions

        :rtype: float

        ``q1.inner(q2)`` is the dot product of the equivalent vectors,
        ie. ``numpy.dot(q1.vec, q2.vec)``.
        The value of ``q.inner(q)`` is the same as ``q.norm ** 2``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> import numpy as np
            >>> Quaternion([1,2,3,4]).inner(Quaternion([5,6,7,8]))
            >>> np.dot([1,2,3,4], [5,6,7,8])

        :seealso: :func:`~spatialmath.base.quaternions.qinner`
        """

        assert isinstance(
            other, Quaternion
        ), "operands to inner must be Quaternion subclass"
        return self.binop(other, smb.qinner, list1=False)

    # -------------------------------------------- operators

    def __eq__(
        left, right: Quaternion
    ) -> bool:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``==`` operator

        :return: Equality of two operands
        :rtype: bool or list of bool

        ``q1 == q2`` is True if ``q1` is elementwise-equal to ``q2``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q1 = Quaternion([1,2,3,4])
            >>> q2 = Quaternion([5,6,7,8])
            >>> q1 == q1
            >>> q1 == q2
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) == q1
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) == q2
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) == Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])

        :seealso: :func:`__ne__` :func:`~spatialmath.base.quaternions.qisequal`
        """
        assert isinstance(left, type(right)), "operands to == are of different types"
        return left.binop(right, smb.qisequal, list1=False)

    def __ne__(
        left, right: Quaternion
    ) -> bool:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``!=`` operator

        :rtype: bool

        ``q1 != q2`` is True if ``q` is elementwise not equal to ``q2``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q1 = Quaternion([1,2,3,4])
            >>> q2 = Quaternion([5,6,7,8])
            >>> q1 != q1
            >>> q1 != q2
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) != q1
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) != q2

        :seealso: :func:`__ne__` :func:`~spatialmath.base.quaternions.qisequal`
        """
        assert isinstance(left, type(right)), "operands to == are of different types"
        return left.binop(right, lambda x, y: not smb.qisequal(x, y), list1=False)

    def __mul__(
        left, right: Quaternion
    ) -> Quaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :return: product
        :rtype: Quaternion
        :raises: ValueError

        - ``q1 * q2`` is the Hamilton product of two quaternions
        - ``q * s`` is the scalar product, where ``s`` is a scalar

        ==============   ==============   ==============  ================
                   Multiplicands                   Product
        -------------------------------   --------------------------------
            left             right            type           result
        ==============   ==============   ==============  ================
        Quaternion       Quaternion       Quaternion      Hamilton product
        Quaternion       UnitQuaternion   Quaternion      Hamilton product
        Quaternion       scalar           Quaternion      scalar product
        ==============   ==============   ==============  ================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left * right``
         1      N       N    ``prod[i] = left * right[i]``
         N      1       N    ``prod[i] = left[i] * right``
         N      N       N    ``prod[i] = left[i] * right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]) * Quaternion([5,6,7,8])
            >>> Quaternion([1,2,3,4]) * 2
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) * 2
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) * Quaternion([1,2,3,4])
            >>> Quaternion([1,2,3,4]) * Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) * Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])

        :seealso: :func:`__rmul__` :func:`__imul__` :func:`~spatialmath.base.quaternions.qqmul`
        """
        if isinstance(right, left.__class__):
            # quaternion * [unit]quaternion case
            return Quaternion(left.binop(right, smb.qqmul))

        elif smb.isscalar(right):
            # quaternion * scalar case
            # print('scalar * quat')
            return Quaternion([right * q._A for q in left])

        else:
            raise ValueError("operands to * are of different types")

    def __rmul__(
        right, left: Quaternion
    ) -> Quaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :return: product
        :rtype: Quaternion
        :raises: ValueError

        ``s * q`` is the scalar product, where ``s`` is a scalar.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> 2 * Quaternion([1,2,3,4])
            >>> 2 * Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])

        :seealso: :func:`__mul__`
        """
        # scalar * quaternion case
        return Quaternion([left * q._A for q in right])

    def __imul__(
        left, right: Quaternion
    ) -> bool:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*=`` operator

        :return: product
        :rtype: Quaternion
        :raises: ValueError

        ``q1 *= q2`` sets ``q1 := q1 * q2``
        ``q1 *= s`` sets ``q1 := q1 * s`` where ``s`` is a scalar

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q = Quaternion([1,2,3,4])
            >>> q *= Quaternion([5,6,7,8])
            >>> print(q)
            >>> q *= 2
            >>> print(q)

        :seealso: :func:`__mul__`
        """
        return left.__mul__(right)

    def __imatmul__(
        left, right: Quaternion
    ) -> Quaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``@=`` operator

        :return: product
        :rtype: Quaternion
        :raises: ValueError

        ``q1 @= q2`` sets ``q1 := qnorm(q1 * q2)``. Only meaningful for
        ``UnitQuaternion``, which is the only subclass defining ``__matmul__``
        (with normalization) that this delegates to; on a plain ``Quaternion``
        this raises the same ``TypeError`` that ``q1 @ q2`` would.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion
            >>> q = UnitQuaternion.Eul([0.1, 0.2, 0.3])
            >>> q @= UnitQuaternion.Eul([0.3, 0.4, 0.5])
            >>> print(q)


        :seealso: :func:`__matmul__`
        """
        # NOT left.__matmul__(right): Quaternion itself has no __matmul__
        # (only UnitQuaternion defines one), and calling the dunder
        # directly as a plain attribute skips Python's normal operator
        # fallback, raising a confusing AttributeError instead of the
        # TypeError that `left @ right` raises consistently.
        return left @ right

    def __pow__(self, n: int) -> Quaternion:
        """
        Overloaded ``**`` operator

        :rtype: Quaternion instance

        ``q ** N`` computes the product of ``q`` with itself ``N-1`` times, where ``N`` must be
        an integer.  If ``N`` < 0 the result is conjugated.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> print(Quaternion([1,2,3,4]) ** 2)
            >>> print(Quaternion([1,2,3,4]) ** -1)
            >>> print(Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) ** 2)

        :seealso: :func:`~spatialmath.base.quaternions.qpow`
        """
        return self.__class__([smb.qpow(q._A, n) for q in self])

    def __ipow__(self, n: int) -> Quaternion:
        """
        Overloaded ``=**`` operator

        :rtype: Quaternion instance

        ``q **= N`` computes the product of ``q`` with itself ``N-1`` times, where ``N`` must be
        an integer.  If ``N``<0 the result is conjugated.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q = Quaternion([1,2,3,4])
            >>> q **= 2
            >>> q
            >>> q = Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            >>> q **= 2
            >>> q


        :seealso: :func:`__pow__`
        """
        return self.__pow__(n)

    def __truediv__(self, other: Quaternion):
        return NotImplemented  # Quaternion division not supported

    def __add__(
        left, right: Quaternion
    ) -> Quaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``+`` operator

        :return: sum
        :rtype: Quaternion, UnitQuaternion
        :raises: ValueError

        ==============   ==============   ==============  ===================
                   Operands                            Sum
        -------------------------------   -----------------------------------
            left             right            type           result
        ==============   ==============   ==============  ===================
        Quaternion       Quaternion       Quaternion      elementwise sum
        Quaternion       UnitQuaternion   Quaternion      elementwise sum
        Quaternion       scalar           Quaternion      add to each element
        UnitQuaternion   Quaternion       Quaternion      elementwise sum
        UnitQuaternion   UnitQuaternion   Quaternion      elementwise sum
        UnitQuaternion   scalar           Quaternion      add to each element
        ==============   ==============   ==============  ===================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``sum = left + right``
         1      N       N    ``sum[i] = left + right[i]``
         N      1       N    ``sum[i] = left[i] + right``
         N      N       N    ``sum[i] = left[i] + right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        A scalar of length N is a list, tuple or numpy array.
        A 3-vector of length N is a 3xN numpy array, where each column is a 3-vector.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]) + Quaternion([5,6,7,8])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) + Quaternion([1,2,3,4])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) + Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
        """
        # results is not in the group, return an array, not a class
        assert isinstance(left, type(right)), "operands to + are of different types"
        return Quaternion(left.binop(right, lambda x, y: x + y))

    def __sub__(
        left, right: Quaternion
    ) -> Quaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``-`` operator

        :return: difference
        :rtype: Quaternion, UnitQuaternion
        :raises: ValueError

        ==============   ==============   ==============  ==========================
                   Operands                          Difference
        -------------------------------   ------------------------------------------
            left             right            type           result
        ==============   ==============   ==============  ==========================
        Quaternion       Quaternion       Quaternion      elementwise sum
        Quaternion       UnitQuaternion   Quaternion      elementwise sum
        Quaternion       scalar           Quaternion      subtract from each element
        UnitQuaternion   Quaternion       Quaternion      elementwise sum
        UnitQuaternion   UnitQuaternion   Quaternion      elementwise sum
        UnitQuaternion   scalar           Quaternion      subtract from each element
        ==============   ==============   ==============  ==========================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``diff = left - right``
         1      N       N    ``diff[i] = left - right[i]``
         N      1       N    ``diff[i] = left[i] - right``
         N      N       N    ``diff[i] = left[i] - right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        A scalar of length N is a list, tuple or numpy array.
        A 3-vector of length N is a 3xN numpy array, where each column is a 3-vector.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> Quaternion([1,2,3,4]) - Quaternion([5,6,7,8])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) - Quaternion([1,2,3,4])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) - Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])

        """
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        assert isinstance(left, type(right)), "operands to - are of different types"
        return Quaternion(left.binop(right, lambda x, y: x - y))

    def __neg__(self) -> Quaternion:
        r"""
        Overloaded unary ``-`` operator

        :rtype: Quaternion or UnitQuaternion

        ``-q`` is a quaternion with all its components negated.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> -Quaternion([1,2,3,4])
            >>> -Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
        """

        return UnitQuaternion(
            [-x for x in self.data]
        )  # pylint: disable=invalid-unary-operand-type

    def __repr__(self) -> str:
        """
        Readable representation of pose (superclass method)

        :return: readable representation of the pose as a list of arrays
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q = Quaternion([1,2,3,4])
            >>> q
        """
        name = type(self).__name__
        if len(self) == 0:
            return name + "([])"
        elif len(self) == 1:
            # need to indent subsequent lines of the native repr string by 4 spaces
            return name + "(" + self._A.__repr__() + ")"
        else:
            # format this as a list of ndarrays
            return (
                name
                + "([\n  "
                + ",\n  ".join([v.__repr__() for v in self.data])
                + " ])"
            )

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython (superclass method)

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        Example::

            In [1]: x

        """
        print(self.__str__())

    def __str__(self) -> str:
        """
        Pretty string representation of quaternion

        :return: readable representation of quaternion
        :rtype: str

        Format the quaternion elements into a single line format.  For example::

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Quaternion
            >>> q = Quaternion([1,2,3,4])
            >>> print(x)
            >> q = UnitQuaternion.Rx(0.3)

        Note that unit quaternions are denoted by different delimiters for
        the vector part.

        :seealso: :func:`~spatialmath.base.quaternions.qnorm`
        """
        if isinstance(self, UnitQuaternion):
            delim = ("<<", ">>")
        else:
            delim = ("<", ">")
        return "\n".join([smb.q2str(q, delim=delim) for q in self.data])


# ========================================================================= #


class UnitQuaternion(Quaternion):
    r"""
    Unit quaternion class

    A unit quaternion can be considered an ordered pair :math:`(s, \vec{v})`
    where :math:`s \in \mathbb{R}` is the *scalar* part and :math:`\vec{v} = (v_x, v_y, v_z) \in \mathbb{R}^3`
    is the *vector* part and is often written as

    .. math:: \q = s \langle v_x, v_y, v_z \rangle

    and subject to a unit-length constraint :math:`s^2+v_x^2+v_y^2+v_z^2 = 1`.

    A unit-quaternion can be considered as a rotation :math:`\theta` about the
    vector :math:`\vec{v}`, so the unit quaternion can also be
    written as

    .. math:: \q = \cos \frac{\theta}{2} \sin \frac{\theta}{2} <v_x v_y v_z>

    The quaternion :math:`\q` and :math:`-\q` represent the equivalent rotation, and this is referred to
    as a double mapping.

    .. inheritance-diagram:: spatialmath.quaternion.UnitQuaternion
        :top-classes: collections.UserList
        :parts: 1

    The ``UnitQuaternion`` class inherits many methods from the ``Quaternion`` class

    """

    def __init__(
        self,
        s: Any = None,
        v=None,
        norm: Optional[bool] = True,
        check: Optional[bool] = True,
    ):
        """
        Construct a UnitQuaternion instance

        :param norm: explicitly normalize the quaternion [default True]
        :type norm: bool
        :param check: explicitly check validity of argument [default True]
        :type check: bool
        :return: unit-quaternion
        :rtype: UnitQuaternion instance
        :raises: ValueError

        - ``UnitQuaternion()`` constructs the identity quaternion 1<0,0,0>
        - ``UnitQuaternion(s, v)`` constructs a unit quaternion with specified
          real ``s`` and ``v`` vector parts. ``v`` is a 3-vector given as a
          list, tuple, or ndarray(3). If ``norm`` is True the resulting
          quaternion is normalized.
        - ``UnitQuaternion(v)`` constructs a unit quaternion with specified
          elements from ``v`` which is a 4-vector given as a list, tuple, or ndarray(4). Also known
          as the Euler parameters.
        - ``UnitQuaternion(M)`` construct a new unit quaternion with ``N`` values where ``Q`` is a Nx4 NumPy array
          whose rows are the quaternion in vector form
        - ``UnitQuaternion(R)`` constructs a unit quaternion from an SO(3)
          rotation matrix given as a ndarray(3,3). If ``check`` is True
          test the rotation submatrix for orthogonality.
        - ``UnitQuaternion(X)`` constructs a unit quaternion from the rotational
          part of ``X`` which is an SO3 or SE3 instance.  If len(X) > 1 then
          the resulting unit quaternion is of the same length.
        - ``UnitQuaternion([q1, q2 .. qN])`` construct a new unit quaternion with ``N`` values where each element is a 4-vector
        - ``UnitQuaternion([Q1, Q2 .. QN])`` construct a new unit quaternion with ``N`` values where each element is a UnitQuaternion instance
        - ``UnitQuaternion([X1, X2 .. XN])`` construct a new unit quaternion with ``N`` values where each element is an SO3 or SE3 instance

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q = UQ()
            >>> q         # repr()
            >>> print(q)  # str()

        """
        super().__init__()

        # handle: UnitQuaternion(v)`` constructs a unit quaternion with specified elements
        # from ``v`` which is a 4-vector given as a list, tuple, or ndarray(4)
        if s is None and smb.isvector(v, 4):
            v, s = (s, v)

        if v is None:
            # single argument
            if super().arghandler(s, check=check):
                # create unit quaternion
                self.data = [smb.qunit(q) for q in self.data]

            elif isinstance(s, np.ndarray):
                # passed a NumPy array, it could be:
                #  an SO(3) or SE(3) matrix
                #  a quaternion as a 1D array
                #  an array of quaternions as an nx4 array

                if s.shape == (3, 3):
                    if smb.isrot(s, check=check):
                        # UnitQuaternion(R) R is 3x3 rotation matrix
                        self.data = [smb.r2q(s)]
                    else:
                        raise ValueError(
                            "invalid rotation matrix provided to UnitQuaternion constructor"
                        )
                elif s.shape == (4,):
                    # passed a 4-vector
                    if norm:
                        self.data = [smb.qunit(s)]
                    else:
                        self.data = [s]
                elif s.ndim == 2 and s.shape[1] == 4:
                    if norm:
                        self.data = [smb.qunit(x) for x in s]
                    else:
                        # self.data = [smb.qpositive(x) for x in s]
                        self.data = [x for x in s]
                else:
                    raise ValueError("array could not be interpreted as UnitQuaternion")

            elif isinstance(s, SO3):
                # UnitQuaternion(x) x is SO3 or SE3 (since SE3 is subclass of SO3)
                self.data = [smb.r2q(x.R) for x in s]

            elif isinstance(s[0], SO3):
                # list of SO3 or SE3
                self.data = [smb.r2q(x.R) for x in s]

            else:
                raise ValueError("bad argument to UnitQuaternion constructor")

        elif smb.isscalar(s) and smb.isvector(v, 3):
            # UnitQuaternion(s, v)   s is scalar, v is 3-vector
            q = np.r_[s, smb.getvector(v)]
            if norm:
                q = smb.qunit(q)
            self.data = [q]

        else:
            raise ValueError("bad argument to UnitQuaternion constructor")

    @staticmethod
    def _identity():
        return smb.qeye()

    @staticmethod
    def isvalid(x: ArrayLike, check: Optional[bool] = True) -> bool:
        """
        Test if vector is valid unit quaternion

        :param x: vector to test
        :type x: numpy.ndarray
        :arg check: explicitly check vector is unit length [default True]
        :type check: bool
        :return: True if the matrix has shape (4,).
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion
            >>> import numpy as np
            >>> UnitQuaternion.isvalid(np.r_[1, 0, 0, 0])
            >>> UnitQuaternion.isvalid(np.r_[1, 2, 3, 4])
        """
        return x.shape == (4,) and (not check or smb.isunitvec(x))

    @property
    def R(self) -> SO3Array:
        """
        Unit quaternion as a rotation matrix

        :return: equivalent rotational matrix
        :rtype: ndarray(3,3)

        ``q.R`` returns the rotation matrix which describes the equivalent rotation. If ``len(x)`` is:

            - 1, return an ndarray with shape=(3,3)
            - N>1, return ndarray with shape=(N,3,3)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q = UQ.Rx(0.3)
            >>> q.R
            >>> q = UQ.Rx([0.3, 0.4])
            >>> q.R

        .. warning:: The i'th rotation matrix is ``x[i,:,:]`` or simply
            ``x[i]``. This is different to the MATLAB version where the i'th
            rotation matrix is ``x(:,:,i)``.
        """
        if len(self) > 1:
            return np.array([smb.q2r(q) for q in self.data])
        else:
            return smb.q2r(self._A)

    @property
    def vec3(self) -> R3:
        r"""
        Unit quaternion unique vector part

        :return: vector part of unit quaternion
        :rtype: numpy array, shape=(3,)

        ``q.vec3`` is the vector part of a unit quaternion.  If ``q`` has a negative scalar
        part we take the vector part of ``-q``, since  ``q`` and ``-q`` represent the
        same rotation.

        This vector part is a minimal unique representation of the unit quaternion and can be used in
        optimization procedures such as bundle adjustment.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q = UQ.Rz(-4)
            >>> print(q)
            >>> q.vec3
            >>> q2 = UQ.Vec3(q.vec3)
            >>> print(q2)
            >>> q == q2

        :seealso: :meth:`UnitQuaternion.Vec3`
        """
        return smb.q2v(self._A)

    # -------------------------------------------- constructor variants
    @classmethod
    def Rx(cls, angles: ArrayLike, unit: Optional[str] = "rad") -> UnitQuaternion:
        """
        Construct a UnitQuaternion object representing rotation about the X-axis

        :arg θ: rotation angle
        :type θ: array_like
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        - ``UnitQuaternion(θ)`` constructs a unit quaternion representing a
          rotation of ``θ`` radians about the X-axis.
        - ``UnitQuaternion(θ, 'deg')`` constructs a unit quaternion representing a
          rotation of ``θ`` degrees about the X-axis.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rx(0.3))
            >>> print(UQ.Rx([0, 0.3, 0.6]))
        """
        angles = smb.getunit(angles, unit)
        return cls(
            [np.r_[math.cos(a / 2), math.sin(a / 2), 0, 0] for a in angles], check=False
        )

    @classmethod
    def Ry(cls, angles: ArrayLike, unit: Optional[str] = "rad") -> UnitQuaternion:
        """
        Construct a UnitQuaternion object representing rotation about the Y-axis

        :arg θ: rotation angle
        :type θ: array_like
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        - ``UnitQuaternion(θ)`` constructs a unit quaternion representing a
          rotation of ``θ`` radians about the Y-axis.
        - ``UnitQuaternion(θ, 'deg')`` constructs a unit quaternion representing a
          rotation of ``θ`` degrees about the Y-axis.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Ry(0.3))
            >>> print(UQ.Ry([0, 0.3, 0.6]))
        """
        angles = smb.getunit(angles, unit)
        return cls(
            [np.r_[math.cos(a / 2), 0, math.sin(a / 2), 0] for a in angles], check=False
        )

    @classmethod
    def Rz(cls, angles: ArrayLike, unit: Optional[str] = "rad") -> UnitQuaternion:
        """
        Construct a UnitQuaternion object representing rotation about the Z-axis

        :arg θ: rotation angle
        :type θ: array_like
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        - ``UnitQuaternion(θ)`` constructs a unit quaternion representing a
          rotation of ``θ`` radians about the Z-axis.
        - ``UnitQuaternion(θ, 'deg')`` constructs a unit quaternion representing a
          rotation of ``θ`` degrees about the Z-axis.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rz(0.3))
            >>> print(UQ.Rz([0, 0.3, 0.6]))
        """
        angles = smb.getunit(angles, unit)
        return cls(
            [np.r_[math.cos(a / 2), 0, 0, math.sin(a / 2)] for a in angles], check=False
        )

    @classmethod
    def Rand(
        cls, N: int = 1, *, theta_range: Optional[ArrayLike2] = None, unit: str = "rad"
    ) -> UnitQuaternion:
        """
        Construct a new random unit quaternion

        :param N: number of random rotations
        :type N: int
        :param theta_range: angular magnitude range [min,max], defaults to None -> [0,pi].
        :type xrange: 2-element sequence, optional
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: random unit-quaternion
        :rtype: UnitQuaternion instance

        - ``UnitQuaternion.Rand()`` is a uniformly distributed random unit quaternion value.
        - ``SO3.Rand(N)`` is a unit quaternion instance containing a sequence of N random unit quaternion
          values.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rand())
            >>> print(UQ.Rand(3))

        :seealso: :meth:`UnitQuaternion.Rand`
        """
        return cls(
            [smb.qrand(theta_range=theta_range, unit=unit) for i in range(0, N)],
            check=False,
        )

    @classmethod
    def Eul(cls, *angles: List[float], unit: Optional[str] = "rad") -> UnitQuaternion:
        r"""
        Construct a new unit quaternion from Euler angles

        :param 𝚪: 3-vector of Euler angles
        :type 𝚪: 3 floats, array_like(3) or ndarray(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        - ``UnitQuaternion.Eul(𝚪)`` is a unit quaternion that describes the 3D
          rotation defined by a 3-vector of Euler angles :math:`\Gamma = (\phi,
          \theta, \psi)` which correspond to consecutive rotations about the Z,
          Y, Z axes respectively.

        - ``UnitQuaternion.Eul(φ, θ, ψ)`` as above but the angles are provided
          as three scalars.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Eul([0.1, 0.2, 0.3]))

        :seealso: :meth:`UnitQuaternion.RPY` :meth:`SE3.eul` :meth:`SE3.Eul` :meth:`~spatialmath.base.transforms3d.eul2r`
        """
        if len(angles) == 1:
            angles = angles[0]

        if smb.isvector(angles, 3):
            return cls(smb.r2q(smb.eul2r(angles, unit=unit)), check=False)
        else:
            return cls([smb.r2q(smb.eul2r(a, unit=unit)) for a in angles], check=False)

    @classmethod
    def RPY(
        cls,
        *angles,
        order: Optional[str] = "zyx",
        unit: Optional[str] = "rad",
    ) -> UnitQuaternion:
        r"""
        Construct a new unit quaternion from roll-pitch-yaw angles

        :param 𝚪: 3-vector of roll-pitch-yaw angles
        :type 𝚪: 3 floats, array_like(3) or ndarray(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type unit: str
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        - ``UnitQuaternion.RPY(𝚪)`` is a unit quaternion that describes the 3D
          rotation defined by a  3-vector of roll, pitch, yaw angles
          :math:`\Gamma = (r, p, y)` which correspond to successive rotations
          about the axes specified by ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch
              about the new y-axis, then by roll about the new x-axis.
              Convention for a mobile robot with x-axis forward and y-axis
              sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the
              new y-axis, then by roll about the new z-axis. Convention for a
              robot gripper with z-axis forward and y-axis between the gripper
              fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the
              new x-axis, then by roll about the new z-axis. Convention for a
              camera with z-axis parallel to the optic axis and x-axis parallel
              to the pixel rows.


        - ``UnitQuaternion.RPY(⍺, β, 𝛾)`` as above but the angles are provided
          as three scalars.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.RPY([0.1, 0.2, 0.3]))

        :seealso: :meth:`UnitQuaternion.Eul` :meth:`SE3.rpy` :meth:`SE3.RPY` :func:`~spatialmath.base.transforms3d.rpy2r`
        """
        if len(angles) == 1:
            angles = angles[0]

        if smb.isvector(angles, 3):
            return cls(smb.r2q(smb.rpy2r(angles, unit=unit, order=order)), check=False)
        else:
            return cls(
                [smb.r2q(smb.rpy2r(a, unit=unit, order=order)) for a in angles],
                check=False,
            )

    @classmethod
    def OA(cls, o: ArrayLike3, a: ArrayLike3) -> UnitQuaternion:
        """
        Construct a new unit quaternion from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type a: array_like
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        ``UnitQuaternion.OA(O, A)`` is a unit quaternion that describes the 3D rotation
        defined in terms of vectors parallel to the Y- and Z-axes of its reference
        frame, respectively :math:`\vec{o}` and :math:`\vec{a}`.  In robotics these axes
        are respectively called the orientation and approach vectors defined such that
        :math:`\mat{R}=[\vec{n}, \vec{o}, \vec{a}]` and :math:`\vec{n} = \vec{o} \times \vec{a}`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.OA([0,0,-1], [0,1,0]))

        .. note::

            - Only the ``A`` vector is guaranteed to have the same direction in the resulting
              rotation matrix
            - ``O`` and ``A`` do not have to be unit-length, they are normalized
            - ``O`` and ``A` do not have to be orthogonal, so long as they are not parallel

        :seealso: :func:`~spatialmath.base.transforms3d.oa2r`
        """
        return cls(smb.r2q(smb.oa2r(o, a)), check=False)

    @classmethod
    def AngVec(
        cls, theta: float, v: ArrayLike3, *, unit: Optional[str] = "rad"
    ) -> UnitQuaternion:
        r"""
        Construct a new unit quaternion from rotation angle and axis

        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        ``UnitQuaternion.AngVec(θ, v)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of ``θ`` about the 3-vector ``v``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.AngVec(0, [1,0,0]))
            >>> print(UQ.AngVec(90, [1,0,0], unit='deg'))

        .. note:: :math:`\theta = 0` the result in an identity quaternion, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :meth:`UnitQuaternion.angvec` :meth:`UnitQuaternion.exp` :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        v = smb.getvector(v, 3)
        theta = smb.getunit(theta, unit, vector=False)
        return cls(
            s=math.cos(theta / 2), v=math.sin(theta / 2) * v, norm=False, check=False
        )

    @classmethod
    def EulerVec(cls, w: ArrayLike3) -> UnitQuaternion:
        r"""
        Construct a new unit quaternion from an Euler rotation vector

        :param ω: rotation axis
        :type ω: 3-element array_like
        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        ``UnitQuaternion.EulerVec(ω)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of :math:`\theta = \lVert \omega \rVert` about the
        unit 3-vector :math:`\omega / \lVert \omega \rVert`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.EulerVec([0.5,0,0]))

        .. note:: :math:`\theta \eq 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :meth:`SE3.angvec` :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        assert smb.isvector(w, 3), "w must be a 3-vector"
        w = smb.getvector(w)
        theta = smb.norm(w)
        s = math.cos(theta / 2)
        v = math.sin(theta / 2) * smb.unitvec(w)
        return cls(s=s, v=v, check=False)

    @classmethod
    def Vec3(cls, vec: ArrayLike3) -> UnitQuaternion:
        r"""
        Construct a new unit quaternion from its vector part

        :param vec: vector part of unit quaternion
        :type vec: 3-element array_like

        ``UnitQuaternion.Vec(v)`` is a new unit quaternion with the specified vector part
        and the scalar part is

        .. math:: s = \sqrt{1 - v_x^2 - v_y^2 - v_z^2}

        The unit quaternion will always have a positive scalar part.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q = UQ.Rz(-4)
            >>> print(q)
            >>> q.vec3
            >>> q2 = UQ.Vec3(q.vec3)
            >>> print(q2)
            >>> q == q2

        :seealso: :meth:`UnitQuaternion.vec3`
        """
        return cls(smb.v2q(vec))

    def inv(self) -> UnitQuaternion:
        """
        Inverse of unit quaternion

        :return: unit-quaternion
        :rtype: UnitQuaternion instance

        ``q.inv()`` is the inverse of the unit-quaternion.  This is a group operation
        and the product of the unit-quaternion and its inverse is the identity quaternion.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rx(0.3).inv())
            >>> print(UQ.Rx(0.3).inv() * UQ.Rx(0.3))
            >>> print(UQ.Rx([0.3, 0.6]).inv())

        :seealso: :func:`~spatialmath.base.quaternions.qinv`
        """
        return UnitQuaternion([smb.qconj(q._A) for q in self])

    @staticmethod
    def qvmul(qv1: ArrayLike3, qv2: ArrayLike3) -> R3:
        """
        Multiply unit quaternions defined by unique vector parts

        :param qv1: vector representation of first multiplicand
        :type qv1: ndarray(3)
        :param qv1: vector representation of second multiplicand
        :type qv1: ndarray(3)

        ``UnitQuaternion(qv1, qv2)`` is the Hamilton product of two unit quaternions
        represented in minimal vector form.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q1 = UQ.Rx(0.3)
            >>> q2 = UQ.Ry(-0.3)
            >>> qv1 = q1.vec3
            >>> qv1
            >>> qv2 = q2.vec3
            >>> qv = UQ.qvmul(qv1, qv2)
            >>> qv
            >>> print(UQ.Vec3(qv))
            >>> print(UQ.Rx(0.3) * UQ.Ry(-0.3))

        :seealso: :meth:`UnitQuaternion.vec3` :meth:`UnitQuaternion.Vec3`
        """
        return smb.vvmul(qv1, qv2)

    def dot(self, omega: ArrayLike3) -> R4:
        """
        Rate of change of a unit quaternion in world frame

        :param ω: angular velocity in world frame
        :type ω: 3-element array_like
        :return: rate of change of unit quaternion
        :rtype: ndarray(4)

        ``q.dot(ω)`` is the rate of change of the elements of the unit quaternion ``q``
        which represents the orientation of a body frame with angular velocity ``ω`` in
        the world frame.

        :seealso: :func:`~spatialmath.base.quaternions.qdot`
        """
        return smb.qdot(self._A, omega)

    def dotb(self, omega: ArrayLike3) -> R4:
        """
        Rate of change of a unit quaternion in body frame

        :param ω: angular velocity in body frame
        :type ω: 3-element array_like
        :return: rate of change of unit quaternion
        :rtype: ndarray(4)

        ``q.dotb(ω)`` is the rate of change of the elements of the unit quaternion ``q``
        which represents the orientation of a body frame with angular velocity ``ω`` in
        the body frame.

        :seealso: :func:`~spatialmath.base.quaternions.qdotb`
        """
        return smb.qdotb(self._A, omega)

    # def mean(self, tol: float = 20) -> SO3:
    #     """Mean of a set of rotations

    #     :param tol: iteration tolerance in units of eps, defaults to 20
    #     :type tol: float, optional
    #     :return: the mean rotation
    #     :rtype: :class:`UnitQuaternion` instance.

    #     Computes the Karcher mean of the set of rotations within the unit quaternion instance.

    #     :references:
    #         - `**Hartley, Trumpf** - "Rotation Averaging" - IJCV 2011 <https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf>`_
    #         - `Karcher mean <https://en.wikipedia.org/wiki/Karcher_mean`_
    #     """

    #     R_mean = self.SO3().mean(tol=tol)
    #     return R_mean.UnitQuaternion()

    def __mul__(
        left, right: UnitQuaternion
    ) -> UnitQuaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Multiply unit quaternion

        :return: product
        :rtype: Quaternion, UnitQuaternion
        :raises: ValueError

        ==============   ==============   ==============  ================
                   Multiplicands                   Product
        -------------------------------   --------------------------------
            left             right            type           result
        ==============   ==============   ==============  ================
        UnitQuaternion   Quaternion       Quaternion      Hamilton product
        UnitQuaternion   UnitQuaternion   UnitQuaternion  Hamilton product
        UnitQuaternion   scalar           Quaternion      scalar product
        UnitQuaternion   3-vector         3-vector        vector rotation
        UnitQuaternion   3xN array        3xN array       vector rotations
        ==============   ==============   ==============  ================

        Any other input combinations result in a ValueError.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rx(0.3) * UQ.Rx(0.4))
            >>> print(UQ.Rx(0.3) * 2)
            >>> print(UQ.Rx(0.3) * [1, 2, 3])

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left * right``
         1      N       N    ``prod[i] = left * right[i]``
         N      1       N    ``prod[i] = left[i] * right``
         N      N       N    ``prod[i] = left[i] * right[i]``
         N      M       n/a    ``ValueError``
        ====   =====   ====  ================================

        A scalar of length N is a list, tuple or numpy array.
        A 3-vector of length N is a 3xN numpy array, where each column is
        a 3-vector.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rx(0.3) * UQ.Rx(0.4))
            >>> q = UQ.Rx(0.3)
            >>> q *= UQ.Rx(0.4)
            >>> print(q)
            >>> print(UQ.Rx(0.3) * UQ.Rx([0.4, 0.6]))
            >>> print(UQ.Rx([0.3, 0.6]) * UQ.Rx(0.3))
            >>> print(UQ.Rx([0.3, 0.6]) * UQ.Rx([0.3, 0.6]))

        :seealso: :meth:`Quaternion.__mul__`
        """
        if isinstance(left, right.__class__):
            # quaternion * quaternion case (same class)
            return right.__class__(left.binop(right, smb.qqmul))

        elif smb.isscalar(right):
            # quaternion * scalar case
            # print('scalar * quat')
            return Quaternion([right * q._A for q in left])

        elif isinstance(right, (list, tuple, np.ndarray)):
            # unit quaternion * vector
            # print('*: pose x array')
            if smb.isvector(right, 3):
                v = smb.getvector(right)
                if len(left) == 1:
                    # pose x vector
                    # print('*: pose x vector')
                    return smb.qvmul(left._A, smb.getvector(right, 3))

                elif len(left) > 1 and smb.isvector(right, 3):
                    # pose array x vector
                    # print('*: pose array x vector')
                    return np.array([smb.qvmul(x, v) for x in left._A]).T

            elif (
                len(left) == 1 and isinstance(right, np.ndarray) and right.shape[0] == 3
            ):
                # pose x stack of vectors
                return np.array([smb.qvmul(left._A, x) for x in right.T]).T
            else:
                raise ValueError("bad operands")
        else:
            raise ValueError("UnitQuaternion: operands to * are of different types")

    def __imul__(
        left, right: UnitQuaternion
    ) -> UnitQuaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Multiply unit quaternion in place

        :return: product
        :rtype: UnitQuaternion, Quaternion
        :raises: ValueError

        Multiplies a quaternion in place. If the right operand is a list,
        the result will be a list.

        Example::

            >>> q = UQ.Rx(0.3)
            >>> q *= UQ.Rx(0.3)
            >>> q

        :seealso: :func:`__mul__`

        """
        return left.__mul__(right)

    def __truediv__(
        left, right: UnitQuaternion
    ) -> UnitQuaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``/`` operator

        :rtype: Quaternion or UnitQuaternion

        - ``q1 / q2`` is equivalent to ``q1 * q1.inv()``.
        - ``q / s`` performs elementwise division of the elements of ``q`` by
          ``s``. This is not a group operation so the result will be a
          Quaternion.

        ==============   ==============   ==============  ===========================
                   Multiplicands                   Quotient
        -------------------------------   -------------------------------------------
            left             right            type           result
        ==============   ==============   ==============  ===========================
        UnitQuaternion   UnitQuaternion   UnitQuaternion  Hamilton product by inverse
        UnitQuaternion   scalar           Quaternion      element-wise division
        ==============   ==============   ==============  ===========================

        Any other input combinations result in a ValueError.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rx(0.3) / UQ.Rx(0.3))
            >>> print(UQ.Rx(0.3) / 2)

        For pose composition either or both operands may hold more than one value which
        results in the composition holding more than one value according to:

        =========   ==========   ====  =====================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  =====================================
         1          1             1    ``quo = left * right.inv()``
         1          M             M    ``quo[i] = left * right[i].inv()``
         N          1             M    ``quo[i] = left[i] * right.inv()``
         M          M             M    ``quo[i] = left[i] * right[i].inv()``
        =========   ==========   ====  =====================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> print(UQ.Rx(0.3) / UQ.Rx(0.3))
            >>> print(UQ.Rx([0.3, 0.6]) / UQ.Rx(0.3))
            >>> print(UQ.Rx(0.3) / UQ.Rx([0.3, 0.6]))
            >>> print(UQ.Rx([0.3, 0.6]) / UQ.Rx([0.3, 0.6]))

        """
        if isinstance(left, right.__class__):
            return UnitQuaternion(
                left.binop(right, lambda x, y: smb.qqmul(x, smb.qconj(y)))
            )
        elif smb.isscalar(right):
            return Quaternion(left.binop(right, lambda x, y: x / y))
        else:
            raise ValueError("bad operands")

    def __eq__(
        left, right: UnitQuaternion
    ) -> bool:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``==`` operator

        :rtype: bool

        ``q1 == q2`` is True if ``q1`` is elementwise equal to ``q2`` and accounts for the
        double mapping. Supports broadcasting.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q1 = UQ.Rx(0.3)
            >>> q2 = UQ.Ry(0.3)
            >>> q1 == q1
            >>> q1 == (-q1)
            >>> q1 == q2
            >>> UQ([q1, q2]) == q1
            >>> UQ([q1, q2]) == q2
            >>> UQ([q1, q2]) == UQ([q1, q2])

        :seealso: :func:`__ne__` :func:`~spatialmath.base.quaternions.qisequal`
        """
        return left.binop(
            right, lambda x, y: smb.qisequal(x, y, unitq=True), list1=False
        )

    def __ne__(
        left, right: UnitQuaternion
    ) -> bool:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``!=`` operator

        :rtype: bool

        ``q1 != q2`` is True if ``q1`` is elementwise not equal to ``q2`` and accounts for the
        double mapping. Supports broadcasting.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q1 = UQ.Rx(0.3)
            >>> q2 = UQ.Ry(0.3)
            >>> q1 != q1
            >>> q1 != (-q1)
            >>> q1 != q2
            >>> UQ([q1, q2]) == q1
            >>> UQ([q1, q2]) == q2
            >>> UQ([q1, q2]) == UQ([q1, q2])

        :seealso: :func:`__eq__` :func:`~spatialmath.base.quaternions.qisequal`
        """
        return left.binop(
            right, lambda x, y: not smb.qisequal(x, y, unitq=True), list1=False
        )

    def __matmul__(
        left, right: UnitQuaternion
    ) -> UnitQuaternion:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded @ operator

        :return: product :rtype: UnitQuaternion

        - ``q1 @ q2`` is the Hamilton product of ``q1`` and ``q2``, both unit
          quaternions, followed by explicit normalization.

        .. note:: This operator is functionally equivalent to ``*`` but is more
            costly.  It is useful for cases where a pose is incrementally update
            over many cycles.
        """
        return left.__class__(
            left.binop(right, lambda x, y: smb.qunit(smb.qqmul(x, y)))
        )

    def interp(
        self, end: UnitQuaternion, s: float = 0, shortest: Optional[bool] = False
    ) -> UnitQuaternion:
        """
        Interpolate between two unit quaternions

        :param end: final unit quaternion
        :type end: UnitQuaternion
        :param shortest: Take the shortest path along the great circle
        :param s: interpolation coefficient, range 0 to 1, or number of steps
        :type s: array_like or int
        :return: interpolated unit quaternion
        :rtype: UnitQuaternion instance

        - ``q0.interp(q1, s)`` is a unit quaternion that is interpolated between
          ``q0`` when s=0 and ``q1`` when s=1. Spherical linear interpolation
          (slerp) is used.  If ``s`` is an ndarray(n) then the result will be
          a UnitQuaternion with n values.

        - ``q0.interp(q1, N)`` interpolate between ``q0`` and ``q1`` in ``N``
          steps.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q1 = UQ.Rx(0.3); q2 = UQ.Rz(-0.4)
            >>> print(q1)
            >>> print(q2)
            >>> q1.interp(q2, 0)    # this is q1
            >>> q1.interp(q2, 1,)   # this is q2
            >>> q1.interp(q2, 0.5)  # this is in between
            >>> q = q1.interp(q2, 11)  # in 11 steps
            >>> len(q)
            >>> q[0]                # this is q1
            >>> q[5]                # this is in between

        .. note:: values of ``s`` are silently clipped to the range [0, 1]

        :seealso: :func:`~spatialmath.base.quaternions.qslerp`
        """
        # TODO allow self to have len() > 1

        if isinstance(s, int) and s > 1:
            s = np.linspace(0, 1, s)
        else:
            s = smb.getvector(s)
            s = np.clip(s, 0, 1)  # enforce valid values

        # 2 quaternion form
        if not isinstance(end, UnitQuaternion):
            raise TypeError("end argument must be a UnitQuaternion")
        q1 = self.vec
        q2 = end.vec
        dot = smb.qinner(q1, q2)

        # If the dot product is negative, the quaternions
        # have opposite handed-ness and slerp won't take
        # the shorter path. Fix by reversing one quaternion.
        if shortest:
            if dot < 0:
                q1 = -q1
                dot = -dot

        # shouldn't be needed by handle numerical errors: -eps, 1+eps cases
        dot = np.clip(dot, -1, 1)  # Clip within domain of acos()

        theta_0 = math.acos(dot)  # theta_0 = angle between input vectors

        qi = []
        for sk in s:
            theta = theta_0 * sk  # theta = angle between v0 and result

            s1 = float(math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0))
            s2 = math.sin(theta) / math.sin(theta_0)
            out = (q1 * s1) + (q2 * s2)
            qi.append(out)

        return UnitQuaternion(qi)

    def interp1(self, s: float = 0, shortest: Optional[bool] = False) -> UnitQuaternion:
        """
        Interpolate a unit quaternion

        :param shortest: Take the shortest path along the great circle
        :param s: interpolation coefficient, range 0 to 1, or number of steps
        :type s: array_like or int
        :return: interpolated unit quaternion
        :rtype: UnitQuaternion instance

        - ``q.interp1(s)`` is a unit quaternion that is interpolated between
          identity when s=0 and ``q`` when s=1. Spherical linear interpolation
          (slerp) is used.  If ``s`` is an ndarray(n) then the result will be
          a UnitQuaternion with n values.

        - ``q.interp1(N)`` interpolate between identity and ``q1`` in ``N``
          steps.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> q = UQ.Rx(0.3)
            >>> print(q)
            >>> q.interp1(0)    # this is identity
            >>> q.interp1(1)    # this is q
            >>> q.interp1(0.5)  # this is in between
            >>> qi = q.interp1(11)  # in 11 steps
            >>> len(qi)
            >>> qi[0]                # this is q1
            >>> qi[5]                # this is in between

        .. note:: values of ``s`` are silently clipped to the range [0, 1]

        :seealso: :func:`~spatialmath.base.quaternions.qslerp`
        """
        # TODO allow self to have len() > 1

        if isinstance(s, int) and s > 1:
            s = np.linspace(0, 1, s)
        else:
            s = smb.getvector(s)
            s = np.clip(s, 0, 1)  # enforce valid values

        q = self.vec
        dot = q[0]  # s

        # If the dot product is negative, the quaternions
        # have opposite handed-ness and slerp won't take
        # the shorter path. Fix by reversing one quaternion.
        if shortest:
            if dot < 0:
                q = -q
                dot = -dot

        # shouldn't be needed by handle numerical errors: -eps, 1+eps cases
        dot = np.clip(dot, -1, 1)  # Clip within domain of acos()

        theta_0 = math.acos(dot)  # theta_0 = angle between input vectors

        qi = []
        for sk in s:
            theta = theta_0 * sk  # theta = angle between v0 and result

            s1 = float(math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0))
            s2 = math.sin(theta) / math.sin(theta_0)
            out = np.r_[s1, 0, 0, 0] + (q * s2)
            qi.append(out)

        return UnitQuaternion(qi)

    def increment(self, w: ArrayLike3, normalize: Optional[bool] = False) -> None:
        """
        Quaternion incremental update

        :param w: angular displacement, Euler vector
        :type w: array_like(3)
        :param normalize: normalize the result, defaults to False
        :type normalize: bool, optional

        .. note:: The object state is updated
        """

        # is (v, theta) or None
        try:
            v, theta = smb.unitvec_norm(w)
        except ValueError:
            # zero update
            return

        ds = math.cos(theta / 2)
        dv = math.sin(theta / 2) * v

        updated = smb.qqmul(self.A, np.r_[ds, dv])
        if normalize:
            updated = smb.qunit(updated)
        self.data = [updated]

    def plot(self, *args: List, **kwargs):
        """
        Plot unit quaternion as a coordinate frame

        :param `**kwargs`: plotting options

        - ``q.plot()`` displays the orientation ``q`` as a coordinate frame in 3D.
          There are many options, see the links below.

        Example::

            >>> q = UQ.Rx(0.3)
            >>> q.plot(frame='A', color='green')

        :seealso: :func:`~spatialmath.base.transforms3d.trplot`
        """
        smb.trplot(smb.q2r(self._A), *args, **kwargs)

    def animate(self, *args: List, **kwargs):
        """
        Plot unit quaternion as an animated coordinate frame

        :param start: initial pose, defaults to null/identity
        :type start: UnitQuaternion
        :param `**kwargs`: plotting options

        - ``q.animate()`` displays the orientation ``q`` as a coordinate frame moving
          from the origin in either 3D.  There are
          many options, see the links below.
        - ``q.animate(*args, start=q1)`` displays the orientation ``q`` as a coordinate
          frame moving from orientation ``q11``, in 3D.  There are
          many options, see the links below.

        Example::

            >>> X = UQ.Rx(0.3)
            >>> X.animate(frame='A', color='green')
            >>> X.animate(start=UQ.Ry(0.2))

        :see :func:`~spatialmath.base.transforms3d.tranimate` :func:`~spatialmath.base.transforms3d.trplot`
        """
        if len(self) > 1:
            return smb.tranimate([smb.q2r(q) for q in self.data], *args, **kwargs)
        else:
            return smb.tranimate(smb.q2r(self._A), *args, **kwargs)

    def rpy(
        self, unit: Optional[str] = "rad", order: Optional[str] = "zyx"
    ) -> Union[R3, RNx3]:
        """
        Unit quaternion as roll-pitch-yaw angles

        :param order: angle sequence order, default to 'zyx'
        :type order: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of roll-pitch-yaw angles
        :rtype: ndarray(3) or ndarray(n,3)

        ``q.rpy`` is the roll-pitch-yaw angle representation of the 3D rotation.  The angles are
        a 3-vector :math:`(r, p, y)` which correspond to successive rotations about the axes
        specified by ``order``:

            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        If ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(N,3)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> UQ.Rx(0.3).rpy()
            >>> UQ.Rz([0.2, 0.3]).rpy()

        :seealso: :meth:`SE3.RPY` :func:`~spatialmath.base.transforms3d.tr2rpy`
        """
        if len(self) == 1:
            return smb.tr2rpy(self.R, unit=unit, order=order)
        else:
            return np.array([smb.tr2rpy(q.R, unit=unit, order=order) for q in self])

    def eul(self, unit: Optional[str] = "rad") -> Union[R3, RNx3]:
        r"""
        Unit quaternion as Euler angles

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of Euler angles
        :rtype: ndarray(3)

        ``q.eul`` is the Euler angle representation of the rotation.  Euler angles are
        a 3-vector :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        If ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(N,3)

        - ndarray with shape=(3,), if len(R) == 1
        - ndarray with shape=(N,3), if len(R) = N > 1

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> UQ.Rz(0.3).eul()
            >>> UQ.Ry([0.3, 0.4]).eul()

        :seealso: :meth:`SE3.Eul` :func:`~spatialmath.base.transforms3d.tr2eul`
        """
        if len(self) == 1:
            return smb.tr2eul(self.R, unit=unit)
        else:
            return np.array([smb.tr2eul(q.R, unit=unit) for q in self])

    def angvec(self, unit: Optional[str] = "rad") -> Tuple[float, R3]:
        r"""
        Unit quaternion as angle and rotation vector

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param check: check that rotation matrix is valid
        :type check: bool
        :return: :math:`(\theta, {\bf v})`
        :rtype: float, ndarray(3)

        ``q.angvec()`` is a tuple :math:`(\theta, v)` containing the rotation
        angle and a rotation axis which is equivalent to the rotation of
        the unit quaternion ``q``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> UQ.Rz(0.3).angvec()

        :seealso: :meth:`Quaternion.AngVec` :meth:`UnitQuaternion.log` :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        return smb.tr2angvec(self.R, unit=unit)

    # def log(self):
    #     r"""
    #     Logarithm of unit quaternion

    #     :rtype: Quaternion instance

    #     ``q.log()`` is the logarithm of the unit quaternion ``q``, ie.

    #     .. math::

    #          0  \langle \frac{\mathb{v}}{\| \mathbf{v} \|} \acos s \rangle

    #     Example:

    #     .. runblock:: pycon

    #         >>> from spatialmath import UnitQuaternion
    #         >>> q = UnitQuaternion.Rx(0.3)
    #         >>> print(q.log())

    #     :reference: `Wikipedia <https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions>`_

    #     :seealso: :meth:`Quaternion.Quaternion.log`, `~spatialmath.quaternion.Quaternion.exp`
    #     """
    #     return Quaternion(s=0, v=math.acos(self.s) * smb.unitvec(self.v))

    def angdist(self, other: UnitQuaternion, metric: Optional[int] = 3) -> float:
        r"""
        Angular distance metric between unit quaternions

        :param other: second unit quaternion
        :type other: UnitQuaternion instance
        :param metric: metric, default is 3
        :type metric: int
        :raises TypeError: if other is not a UnitQuaternion
        :return: angle in radians
        :rtype: float

        ``q1.angdist(q2)`` is the geodesic norm, or geodesic distance between two
        unit quaternions.  We can consider it as the angle between two quaternions.

        Several metrics are supported:

        ======   ===============================================================
        Metric   Details
        ======   ===============================================================
        0        :math:`1 - | \q_1 \bullet \q_2 | \in [0, 1]`
        1        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        2        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        3        :math:`2 \tan^{-1} \| \q_1 \pm \q_2\| / \|\q_1 \mp \q_2\| \in [0, \pi/2]`
        4        :math:`\cos^{-1} \left( 2 (\q_1 \bullet \q_2)^2 - 1\right) \in [0, 1]`
        ======   ===============================================================

        Metric 3 computes the sum and difference of the quaternions and uses
        the largest value in the denominator.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion
            >>> q1 = UnitQuaternion.Rx(0.3)
            >>> q2 = UnitQuaternion.Ry(0.3)
            >>> print(q1.angdist(q1))
            >>> print(q1.angdist(q2))

        .. note::
            - metrics 1, 2, 4 can throw ValueError "math domain error" due to
              numeric errors which push the argument of ``acos()`` marginally
              outside its domain [0, 1].
            - metrics 2 and 3 are equivalent, but 3 is more robust
            - SMTB-MATLAB uses metric 3 for UnitQuaternion.angle()
            - MATLAB's quaternion.dist() uses metric 4

        """
        if not isinstance(other, UnitQuaternion):
            raise TypeError("bad operand")

        if metric == 0:
            measure = lambda p, q: 1 - abs(np.dot(p, q))
        elif metric == 1:
            measure = lambda p, q: math.acos(min(1.0, abs(np.dot(p, q))))
        elif metric == 2:
            measure = lambda p, q: math.acos(min(1.0, abs(np.dot(p, q))))
        elif metric == 3:

            def metric3(p, q):
                x = smb.norm(p - q)
                y = smb.norm(p + q)
                if x >= y:
                    return 2 * math.atan(y / x)
                else:
                    return 2 * math.atan(x / y)

            measure = metric3
        elif metric == 4:
            measure = lambda p, q: math.acos(min(1.0, 2 * np.dot(p, q) ** 2 - 1))

        ad = self.binop(other, measure)
        if len(ad) == 1:
            return ad[0]
        else:
            return np.array(ad)

    def SO3(self) -> SO3:
        """
        Unit quaternion as SO3 instance

        :return: an SO(3) representation
        :rtype: SO3 instance

        ``q.SO3()`` is an ``SO3`` instance representing the same rotation
        as the unit quaternion ``q``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> UQ.Rz(0.3).SO3()

        """
        return SO3(self.R, check=False)

    def SE3(self) -> SE3:
        """
        Unit quaternion as SE3 instance

        :return: an SE(3) representation
        :rtype: SE3 instance

        ``q.SE3()`` is an ``SE3`` instance representing the same rotation
        as the unit quaternion ``q`` and with zero translation.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion as UQ
            >>> UQ.Rz(0.3).SE3()

        """
        return SE3(smb.r2t(self.R), check=False)


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    a = UnitQuaternion([0, 1, 0, 0])

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute()
            / "tests"
            / "test_quaternion.py"
        ).read()
    )  # pylint: disable=exec-used

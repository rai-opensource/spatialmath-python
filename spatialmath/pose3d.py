# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
Classes to abstract 3D pose and orientation using matrices in SE(3) and SO(3)

To use::

    from spatialmath.pose3d import *
    T = SE3.Rx(0.3)

    import spatialmath as sm
    T = sm.SE3.Rx(0.3)


 .. inheritance-diagram:: spatialmath.pose3d
    :top-classes: collections.UserList
    :parts: 1

.. image:: ../figs/pose-values.png
"""
from __future__ import annotations

# pylint: disable=invalid-name

import numpy as np

import spatialmath.base as smb
from spatialmath.base.types import *
from spatialmath.base.vectors import orthogonalize
from spatialmath.baseposematrix import BasePoseMatrix
from spatialmath.pose2d import SE2

from spatialmath.twist import Twist3

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from spatialmath.quaternion import UnitQuaternion

# ============================== SO3 =====================================#


class SO3(BasePoseMatrix):
    """
       SO(3) matrix class

       This subclass represents rotations in 3D space.  Internally it is a 3x3
       orthogonal matrix belonging to the group SO(3).

    .. inheritance-diagram:: spatialmath.pose3d.SO3
       :top-classes: collections.UserList
       :parts: 1
    """

    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, arg: SO3, *, check=True):
        ...

    @overload
    def __init__(self, arg: SE3, *, check=True):
        ...

    @overload
    def __init__(self, arg: SO3Array, *, check=True):
        ...

    @overload
    def __init__(self, arg: List[SO3Array], *, check=True):
        ...

    @overload
    def __init__(self, arg: List[Union[SO3, SO3Array]], *, check=True):
        ...

    def __init__(self, arg=None, *, check=True):
        """
        Construct new SO(3) object

        :rtype: SO3 instance

        There are multiple call signatures:

        - ``SO3()`` is an ``SO3`` instance with one value -- a 3x3 identity
          matrix which corresponds to a null rotation
        - ``SO3(R)`` is an ``SO3`` instance with with the value ``R`` which is a
          3x3 numpy array representing an SO(3) rotation matrix.  If ``check``
          is ``True`` check the matrix belongs to SO(3).
        - ``SO3([R1, R2, ... RN])`` is an ``SO3`` instance wwith ``N`` values
          given by the elements ``Ri`` each of which is a 3x3 NumPy array
          representing an SO(3) matrix. If ``check`` is ``True`` check the
          matrix belongs to SO(3).
        - ``SO3([X1, X2, ... XN])`` is an ``SO3`` instance with ``N`` values
          given by the elements ``Xi`` each of which is an SO3 or SE3 instance.

        :SymPy: supported
        """
        super().__init__()

        if isinstance(arg, SE3):
            self.data = [smb.t2r(x) for x in arg.data]

        elif not super().arghandler(arg, check=check):
            raise ValueError("bad argument to constructor")

    @staticmethod
    def _identity() -> R3x3:
        return np.eye(3)

    # ------------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the object's interal matrix representation

        :return: (3,3)
        :rtype: tuple

        Each value within the ``SO3`` instance is a NumPy array of this shape.
        """
        return (3, 3)

    @property
    def R(self) -> SO3Array:
        """
        SO(3) or SE(3) as rotation matrix

        :return: rotational component
        :rtype: ndarray(3,3)

        ``x.R`` is the rotation matrix component of ``x`` as an array with
        shape (3,3). If ``len(x) > 1``, return an array with shape=(N,3,3).

        .. warning:: The i'th rotation matrix is ``x[i,:,:]`` or simply
            ``x[i]``. This is different to the MATLAB version where the i'th
            rotation matrix is ``x(:,:,i)``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> x = SO3.Rx(0.3)
            >>> x.R

        :SymPy: supported
        """
        if len(self) == 1:
            return self.A[:3, :3]  # type: ignore
        else:
            return np.array([x[:3, :3] for x in self.A])  # type: ignore

    @property
    def n(self) -> R3:
        """
        Normal vector of SO(3) or SE(3)

        :return: normal vector
        :rtype: ndarray(3)

        This is the first column of the rotation submatrix, sometimes called the
        *normal vector*.  It is parallel to the x-axis of the frame defined by
        this pose.
        """
        if len(self) != 1:
            raise ValueError("can only determine n-vector for singleton pose")
        return self.A[:3, 0]  # type: ignore

    @property
    def o(self) -> R3:
        """
        Orientation vector of SO(3) or SE(3)

        :return: orientation vector
        :rtype: ndarray(3)

        This is the second column of the rotation submatrix, sometimes called
        the *orientation vector*.  It is parallel to the y-axis of the frame
        defined by this pose.
        """
        if len(self) != 1:
            raise ValueError("can only determine o-vector for singleton pose")
        return self.A[:3, 1]  # type: ignore

    @property
    def a(self) -> R3:
        """
        Approach vector of SO(3) or SE(3)

        :return: approach vector
        :rtype: ndarray(3)

        This is the third column of the rotation submatrix, sometimes called the
        *approach vector*.  It is parallel to the z-axis of the frame defined by
        this pose.
        """
        if len(self) != 1:
            raise ValueError("can only determine a-vector for singleton pose")
        return self.A[:3, 2]  # type: ignore

    # ------------------------------------------------------------------------ #

    def inv(self) -> Self:
        """
        Inverse of SO(3)

        :return: inverse
        :rtype: SO2 instance

        Efficiently compute the inverse of each of the SO(3) values taking into
        account the matrix structure.  For an SO(3) matrix the inverse is the
        transpose.
        """
        if len(self) == 1:
            return SO3(self.A.T, check=False)  # type: ignore
        else:
            return SO3([x.T for x in self.A], check=False)

    def eul(self, unit: str = "rad", flip: bool = False) -> Union[R3, RNx3]:
        r"""
        SO(3) or SE(3) as Euler angles

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of Euler angles
        :rtype: ndarray(3,), ndarray(n,3)

        ``x.eul`` is the Euler angle representation of the rotation.  Euler angles are
        a 3-vector :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        If ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(3,N)

        :seealso: :func:`~spatialmath.pose3d.SE3.Eul`, :func:`~spatialmath.base.transforms3d.tr2eul`
        :SymPy: not supported
        """
        if len(self) == 1:
            return smb.tr2eul(self.A, unit=unit, flip=flip)  # type: ignore
        else:
            return np.array([base.tr2eul(x, unit=unit, flip=flip) for x in self.A])

    def rpy(self, unit: str = "rad", order: str = "zyx") -> Union[R3, RNx3]:
        """
        SO(3) or SE(3) as roll-pitch-yaw angles

        :param order: angle sequence order, default to 'zyx'
        :type order: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of roll-pitch-yaw angles
        :rtype: ndarray(3,), ndarray(n,3)

        ``x.rpy`` is the roll-pitch-yaw angle representation of the rotation.  The angles are
        a 3-vector :math:`(r, p, y)` which correspond to successive rotations about the axes
        specified by ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        If `len(x)` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(3,N)

        :seealso: :func:`~spatialmath.pose3d.SE3.RPY`, :func:`~spatialmath.base.transforms3d.tr2rpy`
        :SymPy: not supported
        """
        if len(self) == 1:
            return smb.tr2rpy(self.A, unit=unit, order=order)  # type: ignore
        else:
            return np.array([smb.tr2rpy(x, unit=unit, order=order) for x in self.A])

    def angvec(self, unit: str = "rad") -> Tuple[float, R3]:
        r"""
        SO(3) or SE(3) as angle and rotation vector

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: :math:`(\theta, \hat{\bf v})`
        :rtype: float or ndarray(3)

        ``x.angvec()`` is a tuple :math:`(\theta, v)` containing the rotation
        angle and a rotation axis.

        By default the angle is in radians but can be changed setting `unit='deg'`.

        .. note::

            - If the input is SE(3) the translation component is ignored.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> R = SO3.Rx(0.3)
            >>> R.angvec()

        :seealso: :meth:`eulervec` :meth:`AngVec` :meth:`~spatialmath.quaternion.UnitQuaternion.angvec` :meth:`~spatialmath.quaternion.AngVec`, :func:`~angvec2r`
        """
        return smb.tr2angvec(self.R, unit=unit)

    def eulervec(self) -> R3:
        r"""
        SO(3) or SE(3) as Euler vector (exponential coordinates)

        :return: :math:`\theta \hat{\bf v}`
        :rtype: ndarray(3)

        ``x.eulervec()`` is the Euler vector (or exponential coordinates) which
        is related to angle-axis notation and is the product of the rotation
        angle and the rotation axis.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> R = SO3.Rx(0.3)
            >>> R.eulervec()

        .. note::

            - If the input is SE(3) the translation component is ignored.

        :seealso: :meth:`angvec` :func:`~angvec2r`
        """
        theta, v = smb.tr2angvec(self.R)
        return theta * v

    # ------------------------------------------------------------------------ #

    @staticmethod
    def isvalid(x: NDArray, check: bool = True) -> bool:
        """
        Test if matrix is valid SO(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: ``True`` if the matrix is a valid element of SO(3), ie. it is a 3x3
            orthonormal matrix with determinant of +1.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return smb.isrot(x, check=True)

    # ---------------- variant constructors ---------------------------------- #

    @classmethod
    def Rx(cls, theta: float, unit: str = "rad") -> Self:
        """
        Construct a new SO(3) from X-axis rotation

        :param θ: rotation angle about the X-axis
        :type θ: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SE3.Rx(θ)`` is an SO(3) rotation of ``θ`` radians about the x-axis
        - ``SE3.Rx(θ, "deg")`` as above but ``θ`` is in degrees

        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> import numpy as np
            >>> x = SO3.Rx(np.linspace(0, np.pi, 20))
            >>> len(x)
            >>> x[7]

        """
        return cls([smb.rotx(x, unit=unit) for x in smb.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit: str = "rad") -> Self:
        """
        Construct a new SO(3) from Y-axis rotation

        :param θ: rotation angle about Y-axis
        :type θ: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.Ry(θ)`` is an SO(3) rotation of ``θ`` radians about the y-axis
        - ``SO3.Ry(θ, "deg")`` as above but ``θ`` is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined by consecutive
        elements.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> import numpy as np
            >>> x = SO3.Ry(np.linspace(0, np.pi, 20))
            >>> len(x)
            >>> x[7]

        """
        return cls([smb.roty(x, unit=unit) for x in smb.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit: str = "rad") -> Self:
        """
        Construct a new SO(3) from Z-axis rotation

        :param θ: rotation angle about Z-axis
        :type θ: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.Rz(θ)`` is an SO(3) rotation of ``θ`` radians about the z-axis
        - ``SO3.Rz(θ, "deg")`` as above but ``θ`` is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined by consecutive
        elements.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> import numpy as np
            >>> x = SO3.Rz(np.linspace(0, np.pi, 20))
            >>> len(x)
            >>> x[7]

        """
        return cls([smb.rotz(x, unit=unit) for x in smb.getvector(theta)], check=False)

    @classmethod
    def Rand(
        cls, N: int = 1, *, theta_range: Optional[ArrayLike2] = None, unit: str = "rad"
    ) -> Self:
        """
        Construct a new SO(3) from random rotation

        :param N: number of random rotations
        :type N: int
        :param theta_range: angular magnitude range [min,max], defaults to None.
        :type xrange: 2-element sequence, optional
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation matrix
        :rtype: SO3 instance

        - ``SO3.Rand()`` is a random SO(3) rotation.
        - ``SO3.Rand(N)`` is a sequence of N random rotations.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> x = SO3.Rand()
            >>> x

        :seealso: :func:`spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls(
            [
                smb.q2r(smb.qrand(theta_range=theta_range, unit=unit))
                for _ in range(0, N)
            ],
            check=False,
        )

    @overload
    @classmethod
    def Eul(cls, *angles: float, unit: str = "rad") -> Self:
        ...

    @overload
    @classmethod
    def Eul(cls, *angles: Union[ArrayLike3, RNx3], unit: str = "rad") -> Self:
        ...

    @classmethod
    def Eul(cls, *angles, unit: str = "rad") -> Self:
        r"""
        Construct a new SO(3) from Euler angles

        :param 𝚪: Euler angles
        :type 𝚪: 3 floats, array_like(3) or ndarray(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.Eul(𝚪)`` is an SO(3) rotation defined by a 3-vector of Euler
          angles :math:`\Gamma = (\phi, \theta, \psi)` which correspond to
          consecutive rotations about the Z, Y, Z axes respectively. If ``𝚪``
          is an Nx3 matrix then the result is a sequence of rotations each
          defined by Euler angles corresponding to the rows of ``angles``.

        ``SO3.Eul(φ, θ, ψ)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> SO3.Eul(0.1, 0.2, 0.3)
            >>> SO3.Eul([0.1, 0.2, 0.3])
            >>> SO3.Eul(10, 20, 30, unit="deg")

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`~spatialmath.base.transforms3d.eul2r`
        """
        if len(angles) == 1:
            angles = angles[0]

        if smb.isvector(angles, 3):
            return cls(smb.eul2r(angles, unit=unit), check=False)
        else:
            return cls([smb.eul2r(a, unit=unit) for a in angles], check=False)

    @overload
    @classmethod
    def RPY(
        cls,
        *angles: float,
        unit: str = "rad",
        order="zyx",
    ) -> Self:
        ...

    @overload
    @classmethod
    def RPY(
        cls, *angles: Union[ArrayLike3, RNx3], unit: str = "rad", order="zyx"
    ) -> Self:
        ...

    @classmethod
    def RPY(cls, *angles, unit="rad", order="zyx"):
        r"""
        Construct a new SO(3) from roll-pitch-yaw angles

        :param angles: roll-pitch-yaw angles
        :type angles: array_like(3), array_like(n,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.RPY(angles)`` is an SO(3) rotation defined by a 3-vector of
          roll, pitch, yaw angles :math:`(\alpha, \beta, \gamma)`. If ``angles``
          is an Nx3 matrix then the result is a sequence of rotations each
          defined by RPY angles corresponding to the rows of angles. The angles
          correspond to successive rotations about the axes specified by
          ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        - ``SO3.RPY(⍺, β, 𝛾)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> SO3.RPY(0.1, 0.2, 0.3)
            >>> SO3.RPY([0.1, 0.2, 0.3])
            >>> SO3.RPY(0.1, 0.2, 0.3, order='xyz')
            >>> SO3.RPY(10, 20, 30, unit="deg")


        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        if len(angles) == 1:
            angles = angles[0]

        # angles = base.getmatrix(angles, (None, 3))
        # return cls(base.rpy2r(angles, order=order, unit=unit), check=False)

        if smb.isvector(angles, 3):
            return cls(smb.rpy2r(angles, unit=unit, order=order), check=False)
        else:
            return cls(
                [smb.rpy2r(a, unit=unit, order=order) for a in angles], check=False
            )

    @classmethod
    def OA(cls, o: ArrayLike3, a: ArrayLike3) -> Self:
        r"""
        Construct a new SO(3) from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type o: array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.OA(O, A)`` is an SO(3) rotation defined in terms of vectors parallel to
        the Y- and Z-axes of its reference frame.  In robotics these axes are
        respectively called the *orientation* and *approach* vectors defined such that
        :math:`\mat{R}=[\vec{n}, \vec{o}, \vec{a}]` and :math:`\vec{n} = \vec{o} \times \vec{a}`.

        .. note::

            - Only the ``A`` vector is guaranteed to have the same direction in the resulting
              rotation matrix
            - ``O`` and ``A`` do not have to be unit-length, they are normalized
            - ``O`` and ``A`` do not have to be orthogonal, so long as they are not parallel

        :seealso: :func:`~base.transforms3d.oa2r`
        """
        return cls(smb.oa2r(o, a), check=False)

    @classmethod
    def TwoVectors(
        cls,
        x: Optional[Union[str, ArrayLike3]] = None,
        y: Optional[Union[str, ArrayLike3]] = None,
        z: Optional[Union[str, ArrayLike3]] = None,
    ) -> Self:
        """
        Construct a new SO(3) from any two vectors

        :param x: new x-axis, defaults to None
        :type x: str, array_like(3), optional
        :param y: new y-axis, defaults to None
        :type y: str, array_like(3), optional
        :param z: new z-axis, defaults to None
        :type z: str, array_like(3), optional

        Create a rotation by defining the direction of two of the new
        axes in terms of the old axes.  Axes are denoted by strings ``"x"``,
        ``"y"``, ``"z"``, ``"-x"``, ``"-y"``, ``"-z"``.

        The directions can also be specified by 3-element vectors. If the vectors are not orthogonal,
        they will orthogonalized w.r.t. the first available dimension. I.e. if x is available, it will be
        normalized and the remaining vector will be orthogonalized w.r.t. x, else, y will be normalized
        and z will be orthogonalized w.r.t. y.

        To create a rotation where the new frame has its x-axis in -z-direction
        of the previous frame, and its z-axis in the x-direction of the previous
        frame is::

            >>> SO3.TwoVectors(x='-z', z='x')
        """

        def vval(v):
            if isinstance(v, str):
                sign = 1
                if v[0] == "-":
                    sign = -1
                    v = v[1:]  # skip sign char
                elif v[0] == "+":
                    v = v[1:]  # skip sign char
                if v[0] == "x":
                    v = [sign, 0, 0]
                elif v[0] == "y":
                    v = [0, sign, 0]
                elif v[0] == "z":
                    v = [0, 0, sign]
                return np.r_[v]
            else:
                return smb.unitvec(smb.getvector(v, 3))

        if x is not None and y is not None and z is not None:
            raise ValueError(
                "Only two vectors should be provided. Please set one to None."
            )

        elif x is not None and y is not None and z is None:
            # z = x x y
            x = vval(x)
            y = vval(y)
            # Orthogonalizes y w.r.t. x
            y = orthogonalize(y, x, normalize=True)
            z = np.cross(x, y)

        elif x is None and y is not None and z is not None:
            # x = y x z
            y = vval(y)
            z = vval(z)
            # Orthogonalizes z w.r.t. y
            z = orthogonalize(z, y, normalize=True)
            x = np.cross(y, z)

        elif x is not None and y is None and z is not None:
            # y = z x x
            z = vval(z)
            x = vval(x)
            # Orthogonalizes z w.r.t. x
            z = orthogonalize(z, x, normalize=True)
            y = np.cross(z, x)

        else:
            raise ValueError(
                "Insufficient number of vectors. Please provide exactly two vectors."
            )

        return cls(np.c_[x, y, z], check=True)

    @classmethod
    def RotatedVector(cls, v1: ArrayLike3, v2: ArrayLike3, tol=20) -> Self:
        """
        Construct a new SO(3) from a vector and its rotated image

        :param v1: initial vector
        :type v1: array_like(3)
        :param v2: vector after rotation
        :type v2: array_like(3)
        :param tol: tolerance for singularity in units of eps, defaults to 20
        :type tol: float
        :return: SO(3) rotation
        :rtype: :class:`SO3` instance

        ``SO3.RotatedVector(v1, v2)`` is an SO(3) rotation defined in terms of
        two vectors. The rotation takes vector ``v1`` to ``v2``.

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> v1 = [1, 2, 3]
            >>> v2 = SO3.Eul(0.3, 0.4, 0.5) * v1
            >>> print(v2)
            >>> R = SO3.RotatedVector(v1, v2)
            >>> print(R)
            >>> print(R * v1)

        .. note:: The vectors do not have to be unit-length.
        """
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        v1 = smb.unitvec(v1)
        v2 = smb.unitvec(v2)
        v = smb.cross(v1, v2)
        s = smb.norm(v)
        if abs(s) < tol * np.finfo(float).eps:
            c = np.dot(v1, v2)
            if c > 0:
                # v1 and v2 already (anti)parallel in the same direction
                return cls(np.eye(3), check=False)
            # v1 and v2 point in opposite directions -- the formula below
            # is singular here too (it divides by s**2), but unlike the
            # c > 0 case the answer isn't identity: any 180 degree
            # rotation about an axis perpendicular to v1 takes v1 to v2.
            # Closed form for a 180 degree rotation about unit axis u:
            # R = 2*u*u^T - I (Rodrigues at theta=pi, sin=0, cos=-1).
            # Pick u by crossing v1 with whichever world axis it's least
            # aligned with, so the cross product is never itself
            # degenerate.
            axis = np.zeros(3)
            axis[np.argmin(np.abs(v1))] = 1.0
            u = smb.unitvec(smb.cross(v1, axis))
            R = 2 * np.outer(u, u) - np.eye(3)
            return cls(R, check=False)
        else:
            c = np.dot(v1, v2)
            V = smb.skew(v)
            R = np.eye(3) + V + V @ V * (1 - c) / (s**2)
            return cls(R, check=False)

    @classmethod
    def AngleAxis(cls, theta: float, v: ArrayLike3, *, unit: str = "rad") -> Self:
        r"""
        Construct a new SO(3) rotation matrix from rotation angle and axis

        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.AngleAxis(theta, V)`` is an SO(3) rotation defined by
        a rotation of ``THETA`` about the vector ``V``.

        .. note:: :math:`\theta \eq 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`spatialmath.base.transforms3d.angvec2r`
        """
        return cls(smb.angvec2r(theta, v, unit=unit), check=False)

    @classmethod
    def AngVec(cls, theta, v, *, unit="rad") -> Self:
        r"""
        Construct a new SO(3) rotation matrix from rotation angle and axis

        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.AngVec(theta, V)`` is an SO(3) rotation defined by
        a rotation of ``THETA`` about the vector ``V``.

        .. deprecated:: 0.9.8
            Use :meth:`AngleAxis` instead.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`spatialmath.base.transforms3d.angvec2r`
        """
        return cls(smb.angvec2r(theta, v, unit=unit), check=False)

    @classmethod
    def EulerVec(cls, w) -> Self:
        r"""
        Construct a new SO(3) rotation matrix from an Euler rotation vector

        :param ω: rotation axis
        :type ω: 3-element array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.EulerVec(ω)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of :math:`\theta = \lVert \omega \rVert` about the
        unit 3-vector :math:`\omega / \lVert \omega \rVert`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> SO3.EulerVec([0.5,0,0])

        .. note:: :math:`\theta \eq 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        assert smb.isvector(w, 3), "w must be a 3-vector"
        w = smb.getvector(w)
        theta = smb.norm(w)
        return cls(smb.angvec2r(theta, w), check=False)

    @classmethod
    def Exp(
        cls,
        S: Union[R3, RNx3],
        check: bool = True,
        so3: bool = True,
    ) -> Self:
        r"""
        Create an SO(3) rotation matrix from so(3)

        :param S: Lie algebra so(3)
        :type S: ndarray(3,3), ndarray(n,3)
        :param check: check that passed matrix is valid so(3), default True
        :bool check: bool, optional
        :param so3: the input is interpretted as an so(3) matrix not a stack of three twists, default True
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.Exp(S)`` is an SO(3) rotation defined by its Lie algebra
          which is a 3x3 so(3) matrix (skew symmetric)
        - ``SO3.Exp(t)`` is an SO(3) rotation defined by a 3-element twist
          vector (the unique elements of the so(3) skew-symmetric matrix)
        - ``SO3.Exp(T)`` is a sequence of SO(3) rotations defined by an Nx3 matrix
          of twist vectors, one per row.

        .. note::

            - if :math:`\theta \eq 0` the result in an identity matrix
            - an input 3x3 matrix is ambiguous, it could be the first or third case above.  In this
              case the parameter `so3` is the decider.

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if smb.ismatrix(S, (-1, 3)) and not so3:
            return cls([smb.trexp(s, check=check) for s in S], check=False)
        else:
            return cls(smb.trexp(cast(R3, S), check=check), check=False)

    def UnitQuaternion(self) -> UnitQuaternion:
        """
        SO3 as a unit quaternion instance

        :return: a unit quaternion representation
        :rtype: UnitQuaternion instance

        ``R.UnitQuaternion()`` is an ``UnitQuaternion`` instance representing the same rotation
        as the SO3 rotation ``R``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> SO3.Rz(0.3).UnitQuaternion()

        """
        # Function level import to avoid circular dependencies
        from spatialmath import UnitQuaternion

        return UnitQuaternion(smb.r2q(self.R), check=False)

    def angdist(self, other: SO3, metric: int = 6) -> Union[float, ndarray]:
        r"""
        Angular distance metric between rotations

        :param other: second rotation
        :type other: SO3 instance
        :param metric: metric, default is 6
        :type metric: int
        :raises TypeError: if other is not an SO3
        :return: angle in radians
        :rtype: float or ndarray

        ``R1.angdist(R2)`` is the geodesic norm, or geodesic distance between two
        rotations.

        Several metrics are supported, the first 5 are computed after conversion
        to unit quaternions.

        ======   ===============================================================
        Metric   Details
        ======   ===============================================================
        0        :math:`1 - | \q_1 \bullet \q_2 | \in [0, 1]`
        1        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        2        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        3        :math:`2 \tan^{-1} \| \q_1 - \q_2\| / \|\q_1 + \q_2\| \in [0, \pi/2]`
        4        :math:`\cos^{-1} \left( 2 (\q_1 \bullet \q_2)^2 - 1\right) \in [0, 1]`
        5        :math:`\|I - \mat{R}_1 \mat{R}_2^T\| \in [0, 2]`
        6        :math:`\|\log \mat{R}_1 \mat{R}_2^T\| \in [0, \pi]`
        ======   ===============================================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> R1 = SO3.Rx(0.3)
            >>> R2 = SO3.Ry(0.3)
            >>> print(R1.angdist(R1))
            >>> print(R1.angdist(R2))

        .. note::
            - metrics 1, 2, 4 can throw ValueError "math domain error" due to
              numeric errors which push the argument of ``acos()`` marginally
              outside its domain [0, 1].
            - metrics 2 and 3 are equivalent, but 3 is more robust

        :seealso: :func:`UnitQuaternion.angdist`
        """

        if metric < 5:
            from spatialmath.quaternion import UnitQuaternion

            return UnitQuaternion(self).angdist(UnitQuaternion(other), metric=metric)

        elif metric == 5:
            op = lambda R1, R2: np.linalg.norm(np.eye(3) - R1 @ R2.T)
        elif metric == 6:
            op = lambda R1, R2: smb.norm(smb.trlog(R1 @ R2.T, twist=True))
        else:
            raise ValueError("unknown metric")

        ad = self._op2(other, op)
        if isinstance(ad, list):
            return np.array(ad)
        else:
            return ad

    def mean(self, tol: float = 20) -> SO3:
        """Mean of a set of SO(3) values

        :param tol: iteration tolerance in units of eps, defaults to 20
        :type tol: float, optional
        :return: the mean rotation
        :rtype: :class:`SO3` instance.

        Computes the Karcher mean of the set of SO(3) rotations within the :class:`SO3` instance.

        :references:
            - `**Hartley, Trumpf** - "Rotation Averaging" - IJCV 2011 <https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf>`_, Algorithm 1, page 15.
            - `Karcher mean <https://en.wikipedia.org/wiki/Karcher_mean>`_

        :seealso: :class:`SE3.mean`
        """

        eta = tol * np.finfo(float).eps
        R_mean = self[0]  # initial guess
        while True:
            r = np.dstack((R_mean.inv() * self).log()).mean(axis=2)
            if np.linalg.norm(r) < eta:
                return R_mean
            R_mean = R_mean @ self.Exp(r)  # update estimate and normalize


# ============================== SE3 =====================================#


class SE3(SO3):
    """
       SE(3) matrix class

       This subclass represents rigid-body motion in 3D space.  Internally it is a
       4x4 homogeneous transformation matrix belonging to the group SE(3).

    .. inheritance-diagram:: spatialmath.pose3d.SE3
       :top-classes: collections.UserList
       :parts: 1
    """

    @overload
    def __init__(self):  # identity
        ...

    @overload
    def __init__(self, x: Union[SE3, SO3, SE2], *, check=True):  # copy/promote
        ...

    @overload
    def __init__(self, x: List[SE3], *, check=True):  # import list of SE3
        ...

    @overload
    def __init__(self, x: float, y: float, z: float, *, check=True):  # pure translation
        ...

    @overload
    def __init__(self, x: ArrayLike3, *, check=True):  # pure translation
        ...

    @overload
    def __init__(self, x: SE3Array, *, check=True):  # import native array
        ...

    @overload
    def __init__(self, x: List[SE3Array], *, check=True):  # import native arrays
        ...

    def __init__(self, x=None, y=None, z=None, *, check=True):
        """
        Construct new SE(3) object

        :rtype: SE3 instance

        There are multiple call signatures that return an ``SE3`` instance
        with one or more values.

        - ``SE3()`` null motion, value is the identity matrix.
        - ``SE3(x, y, z)`` is a pure translation of (x,y,z)
        - ``SE3(T)``  where ``T`` is a 4x4 Numpy  array representing an SE(3)
          matrix.  If ``check`` is ``True`` check the matrix belongs to SE(3).

        - ``SE3([T1, T2, ... TN])`` has ``N`` values
          given by the elements ``Ti`` each of which is a 4x4 NumPy array
          representing an SE(3) matrix. If ``check`` is ``True`` check the
          matrix belongs to SE(3).

        - ``SE3(X)`` where ``X`` is:
          - ``SE3`` is a copy of ``X``
          - ``SO3`` is the rotation of ``X`` with zero translation
          - ``SE2`` is the z-axis rotation and x- and y-axis translation of ``X``

        - ``SE3([X1, X2, ... XN])`` has ``N`` values
          given by the elements ``Xi`` each of which is an SE3 instance.

        :SymPy: supported
        """
        if y is None and z is None:
            # just one argument passed

            if super().arghandler(x, check=check):
                return
            elif isinstance(x, SO3):
                self.data = [smb.r2t(_x) for _x in x.data]
            elif isinstance(x, SE2):  # type(x).__name__ == "SE2":
                self.data = x.SE3().data
            elif smb.isvector(x, 3):
                # SE3( [x, y, z] )
                self.data = [smb.transl(x)]
            elif isinstance(x, np.ndarray) and x.shape[1] == 3:
                # SE3( Nx3 )
                self.data = [smb.transl(T) for T in x]

            else:
                raise ValueError("bad argument to constructor")

        elif y is not None and z is not None:
            # SE3(x, y, z)
            self.data = [smb.transl(x, y, z)]

        else:
            raise ValueError("Invalid arguments. See documentation for correct format.")

    @staticmethod
    def _identity() -> NDArray:
        return np.eye(4)

    # ------------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the object's internal matrix representation

        :return: (4,4)
        :rtype: tuple

        Each value within the ``SE3`` instance is a NumPy array of this shape.
        """
        return (4, 4)

    @SO3.R.setter
    def R(self, r: SO3Array) -> None:
        if len(self) > 1:
            raise ValueError("can only assign rotation to length 1 object")
        so3 = SO3(r)
        self.A[:3, :3] = so3.R

    @property
    def t(self) -> R3:
        """
        Translational component of SE(3)

        :return: translational component of SE(3)
        :rtype: numpy.ndarray

        ``x.t`` is the translational component of ``x`` as an array with
        shape (3,). If ``len(x) > 1``, return an array with shape=(N,3).

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3(1,2,3)
            >>> x.t
            >>> x = SE3([ SE3(1,2,3), SE3(4,5,6)])
            >>> x.t

        :SymPy: supported
        """
        if len(self) == 1:
            return self.A[:3, 3]
        else:
            return np.array([x[:3, 3] for x in self.A])

    @t.setter
    def t(self, v: ArrayLike3):
        if len(self) > 1:
            raise ValueError("can only assign translation to length 1 object")
        v = smb.getvector(v, 3)
        self.A[:3, 3] = v

    @property
    def x(self) -> float:
        """
        First element of translational component of SE(3)

        :return: first element of translational component of SE(3)
        :rtype: float

        If ``len(v) > 1``, return an array with shape=(N,).

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> v = SE3(1,2,3)
            >>> v.x
            >>> v = SE3([ SE3(1,2,3), SE3(4,5,6)])
            >>> v.x

        :SymPy: supported
        """
        if len(self) == 1:
            return self.A[0, 3]
        else:
            return np.array([v[0, 3] for v in self.A])

    @x.setter
    def x(self, x: float):
        if len(self) > 1:
            raise ValueError("can only assign elements to length 1 object")
        self.A[0, 3] = x

    @property
    def y(self) -> float:
        """
        Second element of translational component of SE(3)

        :return: second element of translational component of SE(3)
        :rtype: float

        If ``len(v) > 1``, return an array with shape=(N,).

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> v = SE3(1,2,3)
            >>> v.y
            >>> v = SE3([ SE3(1,2,3), SE3(4,5,6)])
            >>> v.y

        :SymPy: supported
        """
        if len(self) == 1:
            return self.A[1, 3]
        else:
            return np.array([v[1, 3] for v in self.A])

    @y.setter
    def y(self, y: float):
        if len(self) > 1:
            raise ValueError("can only assign elements to length 1 object")
        self.A[1, 3] = y

    @property
    def z(self) -> float:
        """
        Third element of translational component of SE(3)

        :return: third element of translational component of SE(3)
        :rtype: float

        If ``len(v) > 1``, return an array with shape=(N,).

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> v = SE3(1,2,3)
            >>> v.z
            >>> v = SE3([ SE3(1,2,3), SE3(4,5,6)])
            >>> v.z

        :SymPy: supported
        """
        if len(self) == 1:
            return self.A[2, 3]
        else:
            return np.array([v[2, 3] for v in self.A])

    @z.setter
    def z(self, z: float):
        if len(self) > 1:
            raise ValueError("can only assign elements to length 1 object")
        self.A[2, 3] = z

    # ------------------------------------------------------------------------ #

    def inv(self) -> SE3:
        r"""
        Inverse of SE(3)

        :return: inverse
        :rtype: SE3 instance

        Efficiently compute the inverse of each of the SE(3) values taking into
        account the matrix structure.

        .. math::

            T = \left[ \begin{array}{cc} \mat{R} & \vec{t} \\ 0 & 1 \end{array} \right],
            \mat{T}^{-1} = \left[ \begin{array}{cc} \mat{R}^T & -\mat{R}^T \vec{t} \\ 0 & 1 \end{array} \right]`

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3(1,2,3)
            >>> x.inv()


        :seealso: :func:`~spatialmath.base.transforms3d.trinv`

        :SymPy: supported
        """
        if len(self) == 1:
            return SE3(smb.trinv(self.A), check=False)
        else:
            return SE3([smb.trinv(x) for x in self.A], check=False)

    def yaw_SE2(self, order: str = "zyx") -> SE2:
        """
        Create SE(2) from SE(3) yaw angle.

        :param order: angle sequence order, default to 'zyx'
        :type order: str
        :return: SE(2) with same rotation as the yaw angle using the roll-pitch-yaw convention,
            and translation along the roll-pitch axes.
        :rtype: SE2 instance

        Roll-pitch-yaw corresponds to successive rotations about the axes specified by ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        """
        if len(self) == 1:
            if order == "zyx":
                return SE2(self.x, self.y, self.rpy(order=order)[2])
            elif order == "xyz":
                return SE2(self.z, self.y, self.rpy(order=order)[2])
            elif order == "yxz":
                return SE2(self.z, self.x, self.rpy(order=order)[2])
        else:
            return SE2([e.yaw_SE2() for e in self])

    def delta(self, X2: Optional[SE3] = None) -> R6:
        r"""
        Infinitesimal difference of SE(3) values

        :return: differential motion vector
        :rtype: ndarray(6)

        ``X1.delta(X2)`` is the differential motion (6x1) corresponding to
        infinitesimal motion (in the ``X1`` frame) from pose ``X1`` to ``X2``.

        The vector :math:`d = [\delta_x, \delta_y, \delta_z, \theta_x, \theta_y, \theta_z]`
        represents infinitesimal translation and rotation.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x1 = SE3.Rx(0.3)
            >>> x2 = SE3.Rx(0.3001)
            >>> x1.delta(x2)

        .. note::

            - the displacement is only an approximation to the motion, and assumes
              that ``X1`` ~ ``X2``.
            - can be considered as an approximation to the effect of spatial velocity over a
              a time interval, ie. the average spatial velocity multiplied by time.

        :Reference: Robotics, Vision & Control for Python, Section 3.1, P. Corke, Springer 2023.

        :seealso: :func:`~spatialmath.base.transforms3d.tr2delta`
        """
        if X2 is None:
            return smb.tr2delta(self.A)
        else:
            return smb.tr2delta(self.A, X2.A)

    def rtvec(self) -> Tuple[R3, R3]:
        """
        Convert to OpenCV-style rotation and translation vectors

        :return: rotation and translation vectors
        :rtype: ndarray(3), ndarray(3)

        Many OpenCV functions accept pose as two 3-vectors: a rotation vector using
        exponential coordinates and a translation vector.  This method combines them
        into an SE(3) instance.

        :seealso: :meth:`rtvec`
        """
        return SO3(self).log(twist=True), self.t

    def Ad(self) -> R6x6:
        r"""
        Adjoint of SE(3)

        :return: adjoint matrix
        :rtype: ndarray(6,6)

        ``SE3.Ad`` is the 6x6 adjoint matrix

        If spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        and the SE(3) represents the pose of {B} relative to {A},
        ie. :math:`{}^A {\bf T}_B`, and the adjoint is :math:`\mathbf{A}` then
        :math:`{}^{A}\!\nu = \mathbf{A} {}^{B}\!\nu`.

        .. warning:: Do not use this method to map velocities
            between robot base and end-effector frames - use ``jacob()``.

        .. note:: Use this method to map velocities between two frames on
            the same rigid-body.

        :Reference: Robotics, Vision & Control for Python, Section 3.1, P. Corke, Springer 2023.
        :seealso: SE3.jacob, Twist.ad, :func:`~spatialmath.base.tr2jac`
        :SymPy: supported
        """
        return smb.tr2adjoint(self.A)

    def jacob(self) -> R6x6:
        r"""
        Velocity transform for SE(3)

        :return: Jacobian matrix
        :rtype: ndarray(6,6)

        ``SE3.jacob()`` is the 6x6 Jacobian that maps spatial velocity or
        differential motion from frame {B} to frame {A} where the pose of {B}
        relative to {A} is represented by the homogeneous transform T =
        :math:`{}^A {\bf T}_B`.

        .. note::
            - To map from frame {A} to frame {B} use the transpose of this matrix.
            - Use this method to map velocities between the robot end-effector frame
              and the base frames.

        .. warning:: Do not use this method to map velocities between two frames
            on the same rigid-body.

        :seealso: SE3.Ad, Twist.ad, :func:`~spatialmath.base.tr2jac`
        :Reference: Robotics, Vision & Control for Python, Section 3.1, P. Corke, Springer 2023.
        :SymPy: supported
        """
        return smb.tr2jac(self.A)

    def twist(self) -> Twist3:
        """
        SE(3) as twist

        :return: equivalent rigid-body motion as a twist vector
        :rtype: Twist3 instance

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3(1,2,3)
            >>> x.twist()

        :seealso: :func:`spatialmath.twist.Twist3`
        """
        return Twist3(self.log(twist=True))

    # ------------------------------------------------------------------------ #

    @staticmethod
    def isvalid(x: NDArray, check: bool = True) -> bool:
        """
        Test if matrix is a valid SE(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: ``True`` if the matrix is 4x4 and a valid element of SE(3), ie. it
                 is a valid homogeneous transformation matrix.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transforms3d.ishom`
        """
        return smb.ishom(x, check=check)

    # ---------------- variant constructors ---------------------------------- #

    @classmethod
    def Rx(
        cls,
        theta: ArrayLike,
        unit: str = "rad",
        t: Optional[ArrayLike3] = None,
    ) -> SE3:
        """
        Create anSE(3) pure rotation about the X-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param t: translation, optional
        :type t: 3-element array-like
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Rx(θ)`` is an SE(3) rotation of θ radians about the x-axis
        - ``SE3.Rx(θ, "deg")`` as above but θ is in degrees
        - ``SE3.Rx(θ, t=T)`` as above but also sets the translational component

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        .. note:: The translation option only works for the scalar θ case.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Rx(0.3)
            >>> SE3.Rx([0.3, 0.4])

        :seealso: :func:`~spatialmath.base.transforms3d.trotx`
        :SymPy: supported
        """
        return cls(
            [smb.trotx(x, t=t, unit=unit) for x in smb.getvector(theta)],
            check=False,
        )

    @classmethod
    def Ry(
        cls,
        theta: ArrayLike,
        unit: str = "rad",
        t: Optional[ArrayLike3] = None,
    ) -> SE3:
        """
        Create an SE(3) pure rotation about the Y-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param t: translation, optional
        :type t: 3-element array-like
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Ry(θ)`` is an SO(3) rotation of θ radians about the y-axis
        - ``SE3.Ry(θ, "deg")`` as above but θ is in degrees
        - ``SE3.Ry(θ, t=T)`` as above but also sets the translational component

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        .. note:: The translation option only works for the scalar θ case.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Ry(0.3)
            >>> SE3.Ry([0.3, 0.4])

        :seealso: :func:`~spatialmath.base.transforms3d.troty`
        :SymPy: supported
        """
        return cls(
            [smb.troty(x, t=t, unit=unit) for x in smb.getvector(theta)],
            check=False,
        )

    @classmethod
    def Rz(
        cls,
        theta: ArrayLike,
        unit: str = "rad",
        t: Optional[ArrayLike3] = None,
    ) -> SE3:
        """
        Create an SE(3) pure rotation about the Z-axis

        :param θ: rotation angle about Z-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param t: translation, optional
        :type t: 3-element array-like
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Rz(θ)`` is an SO(3) rotation of θ radians about the z-axis
        - ``SE3.Rz(θ, "deg")`` as above but θ is in degrees
        - ``SE3.Rz(θ, t=T)`` as above but also sets the translational component

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        .. note:: The translation option only works for the scalar θ case.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Rz(0.3)
            >>> SE3.Rz([0.3, 0.4])

        :seealso: :func:`~spatialmath.base.transforms3d.trotz`
        :SymPy: supported
        """
        return cls(
            [smb.trotz(x, t=t, unit=unit) for x in smb.getvector(theta)],
            check=False,
        )

    @classmethod
    def Rand(
        cls,
        N: int = 1,
        xrange: Optional[ArrayLike2] = (-1, 1),
        yrange: Optional[ArrayLike2] = (-1, 1),
        zrange: Optional[ArrayLike2] = (-1, 1),
        theta_range: Optional[ArrayLike2] = None,
        unit: str = "rad",
    ) -> SE3:  # pylint: disable=arguments-differ
        """
        Create a random SE(3)

        :param xrange: x-axis range [min,max], defaults to [-1, 1]
        :type xrange: 2-element sequence, optional
        :param yrange: y-axis range [min,max], defaults to [-1, 1]
        :type yrange: 2-element sequence, optional
        :param zrange: z-axis range [min,max], defaults to [-1, 1]
        :type zrange: 2-element sequence, optional
        :param theta_range: angular magnitude range [min,max], defaults to None -> [0,pi].
        :type xrange: 2-element sequence, optional
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param N: number of random transforms
        :type N: int
        :return: SE(3) matrix
        :rtype: SE3 instance

        Return an SE3 instance with random rotation and translation.

        - ``SE3.Rand()`` is a random SE(3) translation.
        - ``SE3.Rand(N)`` is an SE3 object containing a sequence of N random
          poses.


        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Rand(2)

        :seealso: :func:`~spatialmath.quaternions.UnitQuaternion.Rand`
        """
        X = np.random.uniform(
            low=xrange[0], high=xrange[1], size=N
        )  # random values in the range
        Y = np.random.uniform(
            low=yrange[0], high=yrange[1], size=N
        )  # random values in the range
        Z = np.random.uniform(
            low=zrange[0], high=zrange[1], size=N
        )  # random values in the range
        R = SO3.Rand(N=N, theta_range=theta_range, unit=unit)
        return cls(
            [smb.transl(x, y, z) @ smb.r2t(r.A) for (x, y, z, r) in zip(X, Y, Z, R)],
            check=False,
        )

    @overload
    def Eul(cls, phi: float, theta: float, psi: float, unit: str = "rad") -> SE3:
        ...

    @overload
    def Eul(cls, angles: ArrayLike3, unit: str = "rad") -> SE3:
        ...

    @classmethod
    def Eul(cls, *angles, unit="rad") -> SE3:
        r"""
        Create an SE(3) pure rotation from Euler angles

        :param 𝚪: Euler angles
        :type 𝚪: 3 floats, array_like(3) or ndarray(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Eul(𝚪)`` is an SE(3) rotation defined by a 3-vector of Euler
          angles :math:`\Gamma=(\phi, \theta, \psi)` which correspond to
          consecutive rotations about the Z, Y, Z axes respectively.

        If ``𝚪`` is an Nx3 matrix then the result is a sequence of
        rotations each defined by Euler angles corresponding to the rows of
        ``𝚪``.

        - ``SE3.Eul(φ, θ, ψ)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Eul(0.1, 0.2, 0.3)
            >>> SE3.Eul([0.1, 0.2, 0.3])
            >>> SE3.Eul(10, 20, 30, unit="deg")

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.base.transforms3d.eul2r`
        :SymPy: supported
        """
        if len(angles) == 1:
            angles = angles[0]
        if smb.isvector(angles, 3):
            return cls(smb.eul2tr(angles, unit=unit), check=False)
        else:
            return cls([smb.eul2tr(a, unit=unit) for a in angles], check=False)

    @overload
    def RPY(cls, roll: float, pitch: float, yaw: float, unit: str = "rad") -> SE3:
        ...

    @overload
    def RPY(cls, angles: ArrayLike3, unit: str = "rad") -> SE3:
        ...

    @classmethod
    def RPY(cls, *angles, unit="rad", order="zyx") -> SE3:
        r"""
        Create an SE(3) pure rotation from roll-pitch-yaw angles

        :param 𝚪: roll-pitch-yaw angles
        :type 𝚪: 3 floats, array_like(3) or ndarray(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.RPY(𝚪)`` is an SE(3) rotation defined by a 3-vector of roll,
          pitch, yaw angles :math:`\Gamma=(r, p, y)` which correspond to
          successive rotations about the axes specified by ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  This is the **convention** for a mobile robot with x-axis forward
              and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. This is the **convention** for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. This is the **convention** for a camera with z-axis parallel
              to the optical axis and x-axis parallel to the pixel rows.

        If ``𝚪`` is an Nx3 matrix then the result is a sequence of rotations each defined by RPY angles
        corresponding to the rows of ``𝚪``.

        - ``SE3.RPY(⍺, β, 𝛾)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.RPY(0.1, 0.2, 0.3)
            >>> SE3.RPY([0.1, 0.2, 0.3])
            >>> SE3.RPY(0.1, 0.2, 0.3, order='xyz')
            >>> SE3.RPY(10, 20, 30, unit='deg')

        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.base.transforms3d.rpy2r`
        :SymPy: supported
        """
        if len(angles) == 1:
            angles = angles[0]

        if smb.isvector(angles, 3):
            return cls(smb.rpy2tr(angles, order=order, unit=unit), check=False)
        else:
            return cls(
                [smb.rpy2tr(a, order=order, unit=unit) for a in angles], check=False
            )

    @classmethod
    def OA(cls, o: ArrayLike3, a: ArrayLike3) -> SE3:
        r"""
        Create an SE(3) pure rotation from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like(3)
        :param a: 3-vector parallel to the Z-axis
        :type a: array_like(3)
        :return: SE(3) matrix
        :rtype: SE3 instance

        ``SE3.OA(o, a)`` is an SE(3) rotation defined in terms of vectors ``o``
        and ``a`` respectively parallel to the Y- and Z-axes of its reference
        frame.  In robotics these axes are respectively called the *orientation*
        and *approach* vectors defined such that :math:`\mathbf{R} = [n, o, a]`
        and :math:`n = o \times a`.

        .. note::

            - The ``a`` vector is the only guaranteed to have the same direction in the resulting
              rotation matrix
            - ``o`` and ``a`` do not have to be unit-length, they are normalized
            - ``o`` and ``a`` do not have to be orthogonal, so long as they are not parallel
              ``o`` is adjusted to be orthogonal to ``a``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.OA([1, 0, 0], [0, 0, -1])

        :seealso: :func:`~spatialmath.base.transforms3d.oa2r`
        """
        return cls(smb.oa2tr(o, a), check=False)

    @classmethod
    def AngleAxis(
        cls, theta: float, v: ArrayLike3, *, unit: Optional[str] = "rad"
    ) -> SE3:
        r"""
        Create an SE(3) pure rotation matrix from rotation angle and axis

        :param θ: rotation
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis
        :type v: array_like(3)
        :return: SE(3) matrix
        :rtype: SE3 instance

        ``SE3.AngleAxis(θ, v)`` is an SE(3) rotation defined by
        a rotation of ``θ`` about the vector ``v``.

        .. math::

            \mbox{if}\,\, \theta \left\{ \begin{array}{ll}
                = 0 & \mbox{return identity matrix}\\
                \ne 0 & \mbox{v must have a finite length}
                \end{array}
                \right.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`~spatialmath.pose3d.SE3.EulerVec`, :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        return cls(smb.angvec2tr(theta, v, unit=unit), check=False)

    @classmethod
    def AngVec(cls, theta: float, v: ArrayLike3, *, unit: str = "rad") -> SE3:
        r"""
        Create an SE(3) pure rotation matrix from rotation angle and axis

        :param θ: rotation
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis
        :type v: array_like(3)
        :return: SE(3) matrix
        :rtype: SE3 instance

        ``SE3.AngVec(θ, v)`` is an SE(3) rotation defined by
        a rotation of ``θ`` about the vector ``v``.

        .. deprecated:: 0.9.8
            Use :meth:`AngleAxis` instead.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`~spatialmath.pose3d.SE3.EulerVec`, :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        return cls(smb.angvec2tr(theta, v, unit=unit), check=False)

    @classmethod
    def EulerVec(cls, w: ArrayLike3) -> SE3:
        r"""
        Construct a new SE(3) pure rotation matrix from an Euler rotation vector

        :param ω: rotation axis
        :type ω: array_like(3)
        :return: SE(3) rotation
        :rtype: SE3 instance

        ``SE3.EulerVec(ω)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of :math:`\theta = \lVert \omega \rVert` about the
        unit 3-vector :math:`\omega / \lVert \omega \rVert`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.EulerVec([0.5,0,0])

        .. note:: :math:`\theta = 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.AngVec`, :func:`~spatialmath.base.transforms3d.angvec2tr`
        """
        assert smb.isvector(w, 3), "w must be a 3-vector"
        w = smb.getvector(w)
        theta = smb.norm(w)
        return cls(smb.angvec2tr(theta, w), check=False)

    @classmethod
    def Exp(cls, S: Union[R6, R4x4], check: bool = True) -> SE3:
        """
        Create an SE(3) matrix from se(3)

        :param S: Lie algebra se(3) matrix
        :type S: ndarray(6), ndarray(4,4)
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Exp(S)`` is an SE(3) rotation defined by its Lie algebra
          which is a 4x4 se(3) matrix (skew symmetric)
        - ``SE3.Exp(t)`` is an SE(3) rotation defined by a 6-element twist
          vector (the unique elements of the se(3) skew-symmetric matrix)

        :seealso: :func:`~spatialmath.base.transforms3d.trexp`, :func:`~spatialmath.base.transformsNd.skew`
        """
        if smb.isvector(S, 6):
            return cls(smb.trexp(smb.getvector(S)), check=False)
        else:
            return cls(smb.trexp(S), check=False)

    @classmethod
    def RTvec(cls, rvec: ArrayLike3, tvec: ArrayLike3) -> Self:
        """
        Construct a new SE(3) from OpenCV-style rotation and translation vectors

        :param rvec: rotation as exponential coordinates
        :type rvec: ArrayLike3
        :param tvec: translation vector
        :type tvec: ArrayLike3
        :return: An SE(3) instance
        :rtype: SE3 instance

        Many OpenCV functions (such as pose estimation) return pose as two 3-vectors: a
        rotation vector using exponential coordinates and a translation vector.  This
        method combines them into an SE(3) instance.

        :seealso: :meth:`rtvec`
        """
        return SE3.Rt(smb.trexp(rvec), tvec)

    @classmethod
    def Delta(cls, d: ArrayLike6) -> SE3:
        r"""
        Create SE(3) from differential motion

        :param d: differential motion
        :type d: array_like(6)
        :return: SE(3) matrix
        :rtype: SE3 instance

        ``SE3.Delta2tr(d)`` is an SE(3) representing differential
        motion :math:`d = [\delta_x, \delta_y, \delta_z, \theta_x, \theta_y, \theta_z]`.

        :Reference: Robotics, Vision & Control for Python, Section 3.1, P. Corke, Springer 2023.

        :seealso: :meth:`~delta` :func:`~spatialmath.base.transform3d.delta2tr`
        :SymPy: supported
        """
        return cls(smb.trnorm(smb.delta2tr(d)))

    @overload
    def Trans(cls, x: float, y: float, z: float) -> SE3:
        ...

    @overload
    def Trans(cls, xyz: ArrayLike3) -> SE3:
        ...

    @classmethod
    def Trans(cls, x, y=None, z=None) -> SE3:
        """
        Create SE(3) from translation vector

        :param x: x-coordinate or translation vector
        :type x: float or array_like(3)
        :param y: y-coordinate, defaults to None
        :type y: float, optional
        :param z: z-coordinate, defaults to None
        :type z: float, optional
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Trans(x, y, z)`` is an SE(3) representing pure translation.

        - ``SE3.Trans([x, y, z])`` as above, but translation is given as an
          array.

        - ``SE3.Trans(t)`` where ``t`` is Nx3 then create an SE3 object with
          N elements whose translation is defined by the rows of ``t``.

        """
        if y is None and z is None:
            # single passed value, assume is 3-vector or Nx3
            t = smb.getmatrix(x, (None, 3))
            return cls([smb.transl(_t) for _t in t], check=False)
        else:
            return cls(np.array([x, y, z]))

    @classmethod
    def Tx(cls, x: float) -> SE3:
        """
        Create an SE(3) translation along the X-axis

        :param x: translation distance along the X-axis
        :type x: float
        :return: SE(3) matrix
        :rtype: SE3 instance

        `SE3.Tx(x)` is an SE(3) translation of ``x`` along the x-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Tx(2)
            >>> SE3.Tx([2,3])


        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([smb.transl(_x, 0, 0) for _x in smb.getvector(x)], check=False)

    @classmethod
    def Ty(cls, y: float) -> SE3:
        """
        Create an SE(3) translation along the Y-axis

        :param y: translation distance along the Y-axis
        :type y: float
        :return: SE(3) matrix
        :rtype: SE3 instance

        `SE3.Ty(y) is an SE(3) translation of ``y`` along the y-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Ty(2)
            >>> SE3.Ty([2,3])


        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([smb.transl(0, _y, 0) for _y in smb.getvector(y)], check=False)

    @classmethod
    def Tz(cls, z: float) -> SE3:
        """
        Create an SE(3) translation along the Z-axis

        :param z: translation distance along the Z-axis
        :type z: float
        :return: SE(3) matrix
        :rtype: SE3 instance

        `SE3.Tz(z)` is an SE(3) translation of ``z`` along the z-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Tz(2)
            >>> SE3.Tz([2,3])

        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([smb.transl(0, 0, _z) for _z in smb.getvector(z)], check=False)

    @classmethod
    def Rt(
        cls,
        R: Union[SO3, SO3Array],
        t: Optional[ArrayLike3] = None,
        check: bool = True,
    ) -> SE3:
        """
        Create an SE(3) from rotation and translation

        :param R: rotation
        :type R: SO3 or ndarray(3,3)
        :param t: translation
        :type t: array_like(3)
        :param check: check rotation validity, defaults to True
        :type check: bool, optional
        :raises ValueError: bad rotation matrix
        :return: SE(3) matrix
        :rtype: SE3 instance
        """
        if isinstance(R, SO3):
            R = R.A
        elif smb.isrot(R, check=check):
            pass
        else:
            raise ValueError("expecting SO3 or rotation matrix")

        if t is None:
            t = np.zeros((3,))
        return cls(smb.rt2tr(R, t, check=check), check=check)

    @classmethod
    def CopyFrom(cls, T: SE3Array, check: bool = True) -> SE3:
        """
        Create an SE(3) from a 4x4 numpy array that is passed by value.

        :param T: homogeneous transformation
        :type T: ndarray(4, 4)
        :param check: check rotation validity, defaults to True
        :type check: bool, optional
        :raises ValueError: bad rotation matrix, bad transformation matrix
        :return: SE(3) matrix representing that transformation
        :rtype: SE3 instance
        """
        if T is None:
            raise ValueError("Transformation matrix must not be None")
        return cls(np.copy(T), check=check)

    def angdist(self, other: SE3, metric: int = 6) -> float:
        r"""
        Angular distance metric between poses

        :param other: second rotation
        :type other: SE3 instance
        :param metric: metric, default is 6
        :type metric: int
        :raises TypeError: if other is not an SE3
        :return: angle in radians
        :rtype: float or ndarray

        ``T1.angdist(T2)`` is the geodesic norm, or geodesic distance between the
        rotational parts of the two poses.

        Several metrics are supported, the first 5 are computed after conversion
        to unit quaternions.

        ======   ===============================================================
        Metric   Details
        ======   ===============================================================
        0        :math:`1 - | \q_1 \bullet \q_2 | \in [0, 1]`
        1        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        2        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        3        :math:`2 \tan^{-1} \| \q_1 - \q_2\| / \|\q_1 + \q_2\| \in [0, \pi/2]`
        4        :math:`\cos^{-1} \left( 2 (\q_1 \bullet \q_2)^2 - 1\right) \in [0, 1]`
        5        :math:`\|I - \mat{R}_1 \mat{R}_2^T\| \in [0, 2]`
        6        :math:`\|\log \mat{R}_1 \mat{R}_2^T\| \in [0, \pi]`
        ======   ===============================================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> T1 = SE3.Rx(0.3)
            >>> T2 = SE3.Ry(0.3)
            >>> print(T1.angdist(T1))
            >>> print(T1.angdist(T2))

        .. note::
            - metrics 1, 2, 4 can throw ValueError "math domain error" due to
              numeric errors which push the argument of ``acos()`` marginally
              outside its domain [0, 1].
            - metrics 2 and 3 are equivalent, but 3 is more robust

        :seealso: :func:`UnitQuaternion.angdist`
        """

        if metric < 5:
            from spatialmath.quaternion import UnitQuaternion

            return UnitQuaternion(self).angdist(UnitQuaternion(other), metric=metric)

        elif metric == 5:
            op = lambda T1, T2: np.linalg.norm(np.eye(3) - T1[:3, :3] @ T2[:3, :3].T)
        elif metric == 6:
            op = lambda T1, T2: smb.norm(
                smb.trlog(T1[:3, :3] @ T2[:3, :3].T, twist=True)
            )
        else:
            raise ValueError("unknown metric")

        ad = self._op2(other, op)
        if isinstance(ad, list):
            return np.array(ad)
        else:
            return ad

    def mean(self, tol: float = 20) -> SE3:
        """Mean of a set of SE(3) values

        :param tol: iteration tolerance in units of eps, defaults to 20
        :type tol: float, optional
        :return: the mean SE(3) pose
        :rtype: :class:`SE3` instance.

        Computes the mean of all the SE(3) values within the :class:`SE3` instance.  Rotations are
        averaged using the Karcher mean, and translations are averaged using the
        arithmetic mean.

        :references:
            - `**Hartley, Trumpf** - "Rotation Averaging" - IJCV 2011 <https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf>`_, Algorithm 1, page 15.
            - `Karcher mean <https://en.wikipedia.org/wiki/Karcher_mean>`_

        :seealso:  :meth:`SO3.mean`
        """
        R_mean = SO3(self).mean(tol)
        t_mean = self.t.mean(axis=0)
        return SE3.Rt(R_mean, t_mean)

    # @classmethod
    # def SO3(cls, R, t=None, check=True):
    #     if isinstance(R, SO3):
    #         R = R.A
    #     elif base.isrot(R, check=check):
    #         pass
    #     else:
    #         raise ValueError('expecting SO3 or rotation matrix')
    #     if t is None:
    #         return cls(base.r2t(R))
    #     else:
    #         return cls(base.rt2tr(R, t))


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_pose3d.py"
        ).read()
    )  # pylint: disable=exec-used

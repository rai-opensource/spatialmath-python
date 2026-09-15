#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes to abstract 2D pose and orientation using matrices in SE(2) and SO(2)

To use::

    from spatialmath.pose2d import *
    T = SE2(1, 2, 0.3)

    import spatialmath as sm
    T = sm.SE2.Rx(1, 2, 0.3)


 .. inheritance-diagram:: spatialmath.pose3d
    :top-classes: collections.UserList
    :parts: 1
"""

# pylint: disable=invalid-name

import math
import numpy as np

import spatialmath.base as smb
from spatialmath.baseposematrix import BasePoseMatrix

# ============================== SO2 =====================================#


class SO2(BasePoseMatrix):
    """
    SO(2) matrix class

    This subclass represents rotations in 2D space.  Internally it is a 2x2 orthogonal matrix belonging
    to the group SO(2).

    .. inheritance-diagram:: spatialmath.pose2d.SO2
       :top-classes: collections.UserList
       :parts: 1
    """

    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects

    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg=None, *, unit="rad", check=True):
        """
        Construct new SO(2) object

        :param unit: angular units 'deg' or 'rad' [default] if applicable
        :type unit: str, optional
        :param check: check for valid SO(2) elements if applicable, default to True
        :type check: bool
        :return: SO(2) rotation
        :rtype: SO2 instance

        - ``SO2()`` is an SO2 instance representing a null rotation -- the identity matrix.
        - ``SO2(θ)`` is an SO2 instance representing a rotation by ``θ`` radians.  If ``θ`` is array_like
          `[θ1, θ2, ... θN]` then an SO2 instance containing a sequence of N rotations.
        - ``SO2(θ, unit='deg')`` is an SO2 instance representing a rotation by ``θ`` degrees.  If ``θ`` is array_like
          `[θ1, θ2, ... θN]` then an SO2 instance containing a sequence of N rotations.
        - ``SO2(R)`` is an SO2 instance with rotation described by the SO(2) matrix R which is a 2x2 numpy array.  If ``check``
          is ``True`` check the matrix belongs to SO(2).
        - ``SO2([R1, R2, ... RN])`` is an SO2 instance containing a sequence of N rotations, each described by an SO(2) matrix
          Ri which is a 2x2 numpy array. If ``check`` is ``True`` then check each matrix belongs to SO(2).
        - ``SO2([X1, X2, ... XN])`` is an SO2 instance containing a sequence of N rotations, where each Xi is an SO2 instance.

        """
        super().__init__()

        if isinstance(arg, SE2):
            self.data = [smb.t2r(x) for x in arg.data]

        elif super().arghandler(arg, check=check):
            return

        elif smb.isscalar(arg):
            self.data = [smb.rot2(arg, unit=unit)]

        elif smb.isvector(arg):
            self.data = [smb.rot2(x, unit=unit) for x in smb.getvector(arg)]

        else:
            raise ValueError("bad argument to constructor")

    @staticmethod
    def _identity():
        return np.eye(2)

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (2,2)
        :rtype: tuple
        """
        return (2, 2)

    @classmethod
    def Rand(cls, N=1, arange=(0, 2 * math.pi), unit="rad"):
        r"""
        Construct new SO(2) with random rotation

        :param arange: rotation range, defaults to :math:`[0, 2\pi)`.
        :type arange: 2-element array-like, optional
        :param unit: angular units as 'deg or 'rad' [default]
        :type unit: str, optional
        :param N: number of random rotations, defaults to 1
        :type N: int
        :return: SO(2) rotation matrix
        :rtype: SO2 instance

        - ``SO2.Rand()`` is a random SO(2) rotation.
        - ``SO2.Rand([-90, 90], unit='deg')`` is a random SO(2) rotation between
          -90 and +90 degrees.
        - ``SO2.Rand(N)`` is a sequence of N random rotations.

        Rotations are uniform over the specified interval.

        """
        rand = np.random.uniform(
            low=arange[0], high=arange[1], size=N
        )  # random values in the range
        return cls([smb.rot2(x) for x in smb.getunit(rand, unit)])

    @classmethod
    def Exp(cls, S, check=True):
        """
        Construct new SO(2) rotation matrix from so(2) Lie algebra

        :param S: element of Lie algebra so(2)
        :type S: numpy ndarray
        :param check: check that passed matrix is valid so(2), default True
        :type check: bool
        :return: SO(2) rotation matrix
        :rtype: SO2 instance

        - ``SO2.Exp(S)`` is an SO(2) rotation defined by its Lie algebra
          which is a 2x2 so(2) matrix (skew symmetric)

        :seealso: :func:`spatialmath.base.transforms2d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if isinstance(S, (list, tuple)):
            return cls([smb.trexp2(s, check=check) for s in S])
        else:
            return cls(smb.trexp2(S, check=check), check=False)

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is valid SO(2)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: True if the matrix is a valid element of SO(2), ie. it is a 2x2
            orthonormal matrix with determinant of +1.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return not check or smb.isrot2(x, check=True)

    def inv(self):
        """
        Inverse of SO(2)

        :return: inverse rotation
        :rtype: SO2 instance

        - ``x.inv()`` is the inverse of `x`.

        Notes:

            - for elements of SO(2) this is the transpose.
            - if `x` contains a sequence, returns an `SO2` with a sequence of inverses
        """
        if len(self) == 1:
            return SO2(self.A.T)
        else:
            return SO2([x.T for x in self.A])

    @property
    def R(self):
        """
        SO(2) or SE(2) as rotation matrix

        :return: rotational component
        :rtype: numpy.ndarray, shape=(2,2)

        ``x.R`` returns the rotation matrix, when `x` is `SO2` or `SE2`. If `len(x)` is:

        - 1, return an ndarray with shape=(2,2)
        - N>1, return ndarray with shape=(N,2,2)
        """
        return self.A[:2, :2]

    def theta(self, unit="rad"):
        """
        SO(2) as a rotation angle

        :param unit: angular units 'deg' or 'rad' [default]
        :type unit: str, optional
        :return: rotation angle
        :rtype: float or list

        ``x.theta`` is the rotation angle such that `x` is `SO2(x.theta)`.

        """
        if unit == "deg":
            conv = 180.0 / math.pi
        else:
            conv = 1.0

        if len(self) == 1:
            return conv * math.atan2(self.A[1, 0], self.A[0, 0])
        else:
            return [conv * math.atan2(x.A[1, 0], x.A[0, 0]) for x in self]

    def SE2(self):
        """
        Create SE(2) from SO(2)

        :return: SE(2) with same rotation but zero translation
        :rtype: SE2 instance

        """
        return SE2(smb.rt2tr(self.A, [0, 0]))


# ============================== SE2 =====================================#


class SE2(SO2):
    """
       SE(2) matrix class

       This subclass represents rigid-body motion (pose) in 2D space.  Internally
       it is a 3x3 homogeneous transformation matrix belonging to the group SE(2).

    .. inheritance-diagram:: spatialmath.pose2d.SE2
       :top-classes: collections.UserList
       :parts: 1
    """

    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, x=None, y=None, theta=None, *, unit="rad", check=True):
        """
        Construct new SE(2) object

        :param unit: angular units 'deg' or 'rad' [default] if applicable
        :type unit: str, optional
        :param check: check for valid SE(2) elements if applicable, default to True
        :type check: bool
        :return: SE(2) matrix
        :rtype: SE2 instance

        - ``SE2()`` is an SE2 instance representing a null motion -- the
          identity matrix
        - ``SE2(θ)`` is an SE2 instance representing a pure rotation of
          ``θ`` radians
        - ``SE2(θ, unit='deg')`` as above but ``θ`` in degrees
        - ``SE2(x, y)`` is an SE2 instance representing a pure translation of
          (``x``, ``y``)
        - ``SE2(t)`` is an SE2 instance representing a pure translation of
          (``x``, ``y``) where``t``=[x,y] is a 2-element array_like
        - ``SE2(x, y, θ)`` is an SE2 instance representing a translation of
          (``x``, ``y``) and a rotation of ``θ`` radians
        - ``SE2(x, y, θ, unit='deg')`` as above but ``θ`` in degrees
        - ``SE2(t)`` where ``t``=[x,y] is a 2-element array_like, is an SE2
          instance representing a pure translation of (``x``, ``y``)
        - ``SE2(q)`` where ``q``=[x,y,θ] is a 3-element array_like, is an SE2
          instance representing a translation of (``x``, ``y``) and a rotation
          of ``θ`` radians
        - ``SE2(t, unit='deg')`` as above but ``θ`` in degrees
        - ``SE2(T)`` is an SE2 instance with rigid-body motion described by the
          SE(2) matrix T which is a 3x3 numpy array.  If ``check`` is ``True``
          check the matrix belongs to SE(2).
        - ``SE2([T1, T2, ... TN])`` is an SE2 instance containing a sequence of
          N rigid-body motions, each described by an SE(2) matrix Ti which is a
          3x3 numpy array. If ``check`` is ``True`` then check each matrix
          belongs to SE(2).
        - ``SE2([X1, X2, ... XN])`` is an SE2 instance containing a sequence of
          N rigid-body motions, where each Xi is an SE2 instance.

        """
        if y is None and theta is None:
            # just one argument passed

            if super().arghandler(x, check=check):
                return

            if isinstance(x, SO2):
                self.data = [smb.r2t(_x) for _x in x.data]

            elif smb.isscalar(x):
                self.data = [smb.trot2(x, unit=unit)]
            elif len(x) == 2:
                # SE2([x,y])
                self.data = [smb.transl2(x)]
            elif len(x) == 3:
                # SE2([x,y,theta])
                self.data = [smb.trot2(x[2], t=x[:2], unit=unit)]

            else:
                raise ValueError("bad argument to constructor")

        elif x is not None:
            if y is not None and theta is None:
                # SE2(x, y)
                self.data = [smb.transl2(x, y)]

            elif y is not None and theta is not None:
                # SE2(x, y, theta)
                self.data = [smb.trot2(theta, t=[x, y], unit=unit)]

        else:
            raise ValueError("bad arguments to constructor")

    @staticmethod
    def _identity():
        return np.eye(3)

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (3,3)
        :rtype: tuple
        """
        return (3, 3)

    @classmethod
    def Rand(
        cls, N=1, xrange=(-1, 1), yrange=(-1, 1), arange=(0, 2 * math.pi), unit="rad"
    ):  # pylint: disable=arguments-differ
        r"""
        Construct a new random SE(2)

        :param xrange: x-axis range [min,max], defaults to [-1, 1]
        :type xrange: 2-element sequence, optional
        :param yrange: y-axis range [min,max], defaults to [-1, 1]
        :type yrange: 2-element sequence, optional
        :param arange: angle range [min,max], defaults to :math:`[0, 2\pi)`
        :type arange: 2-element sequence, optional
        :param N: number of random rotations, defaults to 1
        :type N: int
        :param unit: angular units 'deg' or 'rad' [default] if applicable
        :type unit: str, optional
        :return: homogeneous rigid-body transformation matrix
        :rtype: SE2 instance

        Return an SE2 instance with random rotation and translation.

        - ``SE2.Rand()`` is a random SE(2) rotation.
        - ``SE2.Rand(N)`` is an SE2 object containing a sequence of N random
          poses.

        Example, create random ten vehicles in the xy-plane::

            >>> x = SE3.Rand(N=10, xrange=[-2,2], yrange=[-2,2])
            >>> len(x)
            10

        """
        x = np.random.uniform(
            low=xrange[0], high=xrange[1], size=N
        )  # random values in the range
        y = np.random.uniform(
            low=yrange[0], high=yrange[1], size=N
        )  # random values in the range
        theta = np.random.uniform(
            low=arange[0], high=arange[1], size=N
        )  # random values in the range
        return cls(
            [
                smb.trot2(t, t=[x, y])
                for (t, x, y) in zip(x, y, smb.getunit(theta, unit))
            ]
        )

    @classmethod
    def Exp(cls, S, check=True):  # pylint: disable=arguments-differ
        """
        Construct a new SE(2) from se(2) Lie algebra

        :param S: element of Lie algebra se(2)
        :type S: numpy ndarray
        :param check: check that passed matrix is valid se(2), default True
        :type check: bool
        :return: homogeneous transform matrix
        :rtype: SE2 instance

        - ``SE2.Exp(S)`` is an SE(2) rotation defined by its Lie algebra
          which is a 3x3 se(2) matrix (skew symmetric)
        - ``SE2.Exp(t)`` is an SE(2) rotation defined by a 3-element twist
          vector array_like (the unique elements of the se(2) skew-symmetric matrix)
        - ``SE2.Exp(T)`` is a sequence of SE(2) rigid-body motions defined by an Nx3 matrix of twist vectors, one per row.

        Note:

        - an input 3x3 matrix is ambiguous, it could be the first or third case above. In this case the argument ``se2`` is the decider.

        :seealso: :func:`spatialmath.base.transforms2d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if isinstance(S, (list, tuple)):
            return cls([smb.trexp2(s) for s in S])
        else:
            return cls(smb.trexp2(S), check=False)

    @classmethod
    def Rot(cls, theta, unit="rad"):
        """
        Create an SE(2) rotation

        :param theta: rotation angle in radians
        :type theta: float
        :param unit: angular units: "rad" [default] or "deg"
        :type unit: str
        :return: SE(2) matrix
        :rtype: SE2 instance

        `SE2.Rot(theta)` is an SE(2) rotation of ``theta``

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2
            >>> SE2.Rot(0.3)
            >>> SE2.Rot([0.2, 0.3])


        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls(
            [smb.trot2(_th, unit=unit) for _th in smb.getvector(theta)],
            check=False,
        )

    @classmethod
    def Tx(cls, x):
        """
        Create an SE(2) translation along the X-axis

        :param x: translation distance along the X-axis
        :type x: float
        :return: SE(2) matrix
        :rtype: SE2 instance

        `SE2.Tx(x)` is an SE(2) translation of ``x`` along the x-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2
            >>> SE2.Tx(2)
            >>> SE2.Tx([2,3])


        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([smb.transl2(_x, 0) for _x in smb.getvector(x)], check=False)

    @classmethod
    def Ty(cls, y):
        """
        Create an SE(2) translation along the Y-axis

        :param y: translation distance along the Y-axis
        :type y: float
        :return: SE(2) matrix
        :rtype: SE2 instance

        `SE2.Ty(y) is an SE(2) translation of ``y`` along the y-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2
            >>> SE2.Ty(2)
            >>> SE2.Ty([2,3])

        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([smb.transl2(0, _y) for _y in smb.getvector(y)], check=False)

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is valid SE(2)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true if the matrix is a valid element of SE(2), ie. it is a
                 3x3 homogeneous rigid-body transformation matrix.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform2d.ishom`
        """
        return not check or smb.ishom2(x, check=True)

    @property
    def t(self):
        """
        Translational component of SE(2)

        :param self: SE(2)
        :type self: SE2 instance
        :return: translational component
        :rtype: numpy.ndarray

        ``x.t`` is the translational vector component.  If ``len(x)`` is:

        - 1, return an ndarray with shape=(2,)
        - N>1, return an ndarray with shape=(N,2)
        """
        if len(self) == 1:
            return self.A[:2, 2]
        else:
            return np.array([x[:2, 2] for x in self.A])

    @property
    def x(self):
        """
        First element of the translational component of SE(2)

        :param self: SE(2)
        :type self: SE2 instance
        :return: translational component
        :rtype: float

        ``v.x`` is the first element of the translational vector component.  If ``len(x)`` is:

        - 1, return an float
        - N>1, return an ndarray with shape=(N,)
        """
        if len(self) == 1:
            return self.A[0, 2]
        else:
            return np.array([v[0, 2] for v in self.A])

    @property
    def y(self):
        """
        Second element of the translational component of SE(2)

        :param self: SE(2)
        :type self: SE2 instance
        :return: translational component
        :rtype: float

        ``v.y`` is the second element of the translational vector component.  If ``len(x)`` is:

        - 1, return an float
        - N>1, return an ndarray with shape=(N,)
        """
        if len(self) == 1:
            return self.A[1, 2]
        else:
            return np.array([v[1, 2] for v in self.A])

    def xyt(self):
        r"""
        SE(2) as a configuration vector

        :return: An array :math:`[x, y, \theta]` :rtype: numpy.ndarray

        ``x.xyt`` is the rigidbody motion in minimal form as a translation and
        rotation expressed in vector form as :math:`[x, y, \theta]`.  If
        ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return an ndarray with shape=(N,3)
        """
        if len(self) == 1:
            return smb.tr2xyt(self.A)
        else:
            return [smb.tr2xyt(x) for x in self.A]

    def inv(self):
        r"""
        Inverse of SE(2)

        :param self: pose
        :type self: SE2 instance
        :return: inverse
        :rtype: SE2

        Notes:

            - for elements of SE(2) this takes into account the matrix structure :math:`T = \left[ \begin{array}{cc} R & t \\ 0 & 1 \end{array} \right], T^{-1} = \left[ \begin{array}{cc} R^T & -R^T t \\ 0 & 1 \end{array} \right]`
            - if `x` contains a sequence, returns an `SE2` with a sequence of inverses

        """
        if len(self) == 1:
            return SE2(smb.rt2tr(self.R.T, -self.R.T @ self.t), check=False)
        else:
            return SE2([smb.rt2tr(x.R.T, -x.R.T @ x.t) for x in self], check=False)

    def SE3(self, z=0):
        """
        Create SE(3) from SE(2)

        :param z: default z coordinate, defaults to 0
        :type z: float
        :return: SE(2) with same rotation but zero translation
        :rtype: SE2 instance

        "Lifts" 2D rigid-body motion to 3D, rotation in the xy-plane (about the z-axis) and
        z-coordinate is settable.

        """
        from spatialmath.pose3d import SE3

        def lift3(x):
            y = np.eye(4)
            y[:2, :2] = x.A[:2, :2]
            y[:2, 3] = x.A[:2, 2]
            y[2, 3] = z
            return y

        return SE3([lift3(x) for x in self])

    def Twist2(self):
        from spatialmath.twist import Twist2

        return Twist2(self.log(twist=True))


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_pose2d.py"
        ).read()
    )  # pylint: disable=exec-used

# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

import numpy as np
import spatialmath.base as smb
from spatialmath.baseposelist import BasePoseList
from spatialmath.geom3d import Line3


class BaseTwist(BasePoseList):
    """
    Superclass for 3D and 2D twist objects

    Subclasses are:

    - ``Twist3`` representing rigid-body motion in 3D as a 6-vector
    - ``Twist2`` representing rigid-body motion in 2D as a 3-vector

    A twist is the unique elements of the logarithm of the corresponding SE(N)
    matrix.

    Arithmetic operators are overloaded but the operation they perform depend
    on the types of the operands.  For example:

    - ``*`` will compose two instances of the same subclass, and the result will be
      an instance of the same subclass, since this is a group operator.

    These classes all inherit from ``UserList`` which enables them to
    represent a sequence of values, ie. a ``Twist3`` instance can contain
    a sequence of twists.  Most of the Python ``list`` operators
    are applicable:

    .. runblock:: pycon
        >>> from spatialmath import Twist3
        >>> x = Twist3()  # new instance with zero value
        >>> len(x)     # it is a sequence of one value
        >>> x.append(x)  # append to itself
        >>> len(x)       # it is a sequence of two values
        >>> x[1]         # the element has a 4x4 matrix value
        >>> x[1] = SE3.Rx(0.3).Twist3()  # set an elements of the sequence
        >>> x.reverse()         # reverse the elements in the sequence
        >>> del x[1]            # delete an element

    :References:

        - "Mechanics, planning and control"
          Park & Lynch, Cambridge, 2016.

    This class is subclassed for the 3D and 2D cases

    .. inheritance-diagram:: spatialmath.twist.Twist3 spatialmath.twist.Twist2
       :top-classes: collections.UserList
       :parts: 2

    """

    def __init__(self):
        super().__init__()  # enable UserList superpowers

    @property
    def S(self):
        """
        Twist as a vector (superclass property)

        :return: Twist vector
        :rtype: ndarray(N)

        - ``X.S`` is a 3-vector if X is a ``Twist2`` instance, and a 6-vector if
          X is a ``Twist3`` instance.

        .. note::

            - the vector is the unique elements of the se(N) representation.
            - the vector is sometimes referred to as the twist coordinate vector.
            - if ``len(X)`` > 1 then return a list of vectors.
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    @property
    def isprismatic(self):
        r"""
        Test for prismatic twist (superclass property)

        :return: Whether twist is purely prismatic
        :rtype: bool

        A prismatic twist has :math:`\vec{\omega} = 0`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> x = Twist3.UnitPrismatic([1,2,3])
            >>> x.isprismatic
            >>> x = Twist3.UnitRevolute([1,2,3], [4,5,6])
            >>> x.isprismatic

        """
        if len(self) == 1:
            return smb.iszerovec(self.w)
        else:
            return [smb.iszerovec(x.w) for x in self.data]

    @property
    def isrevolute(self):
        r"""
        Test for revolute twist (superclass property)

        :return: Whether twist is purely revolute
        :rtype: bool

        A revolute twist has :math:`\vec{v} = 0`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> x = Twist3.UnitPrismatic([1,2,3])
            >>> x.isrevolute
            >>> x = Twist3.UnitRevolute([1,2,3], [0,0,0])
            >>> x.isrevolute

        """
        if len(self) == 1:
            return smb.iszerovec(self.v)
        else:
            return [smb.iszerovec(x.v) for x in self.data]

    @property
    def isunit(self):
        r"""
        Test for unit twist (superclass property)

        :return: Whether twist is a unit-twist
        :rtype: bool

        A unit twist is one with a norm of 1, ie. :math:`\| S \| = 1`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S = Twist3([1,2,3,4,5,6])
            >>> S.isunit
            >>> S = Twist3.UnitRevolute([1,2,3], [4,5,6])
            >>> S.isunit
        """
        if len(self) == 1:
            return smb.isunitvec(self.S)
        else:
            return [smb.isunitvec(x) for x in self.data]

    @property
    def theta(self):
        """
        Twist angle (superclass method)

        :return: magnitude of rotation (1x1) about the twist axis in radians
        :rtype: float
        """
        if self.N == 2:
            return abs(self.w)
        else:
            return smb.norm(np.array(self.w))

    def inv(self):
        """
        Inverse of Twist (superclass method)

        :return: inverse
        :rtype: Twist instance

        Compute the inverse of each of the values within the twist instance.
        The inverse is the negative of the twist vector.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S = Twist3(SE3.Rand())
            >>> S
            >>> S.inv()
            >>> S * S.inv()
        """
        return self.__class__([-t for t in self.data])

    def prod(self):
        r"""
        Product of twists (superclass method)

        :return: Product of elements
        :rtype: Twist2 or Twist3

        For a twist instance with N values return the matrix product of those
        elements :math:`\prod_i=0^{N-1} S_i`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S = Twist3.Rx([0.2, 0.3, 0.4])
            >>> len(S)
            >>> S.prod()
            >>> Twist3.Rx(0.9)
        """
        if self.N == 2:
            log = smb.trlog2
            exp = smb.trexp2
        else:
            log = smb.trlog
            exp = smb.trexp

        twprod = exp(self.data[0])
        for tw in self.data[1:]:
            twprod = twprod @ exp(tw)
        return self.__class__(log(twprod))

    def __eq__(left, right):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``==`` operator (superclass method)

        :return: Equality of two operands
        :rtype: bool or list of bool

        ``S1 == S2`` is True if ``S1` is elementwise equal to ``S2``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S1 = Twist3([1,2,3,4,5,6])
            >>> S2 = Twist3([1,2,3,4,5,6])
            >>> S1 == S2
            >>> S2 = Twist3([1,2,3,4,5,7])
            >>> S1 == S2

        :seealso: :func:`__ne__`
        """
        if type(left) != type(right):
            raise TypeError("operands to == are of different types")
        return left.binop(right, lambda x, y: all(x == y), list1=False)

    def __ne__(left, right):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``!=`` operator (superclass method)

        :rtype: bool

        ``S1 == S2`` is True if ``S1`` is not elementwise equal to ``S2``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S1 = Twist3([1,2,3,4,5,6])
            >>> S2 = Twist3([1,2,3,4,5,6])
            >>> S1 != S2
            >>> S2 = Twist3([1,2,3,4,5,7])
            >>> S1 != S2

        :seealso: :func:`__ne__`
        """
        if type(left) != type(right):
            raise TypeError("operands to != are of different types")
        return left.binop(right, lambda x, y: not all(x == y), list1=False)

    def __truediv__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        if smb.isscalar(right):
            return left.__class__(left.S / right)
        else:
            raise ValueError("Twist /, incorrect right operand")


# ======================================================================== #


class Twist3(BaseTwist):
    r"""
    3D twist class

    A Twist class holds the parameters of a twist, a representation of a
    3D rigid body transformation which is the unique elements of the Lie
    algebra se(3) of the corresponding SE(3) matrix.

    :References:
        - Robotics, Vision & Control for Python, Section 2.3.2.3, P. Corke, Springer 2023.
        - Modern Robotics, Lynch & Park, Cambridge 2017

    .. note:: Compared to Lynch & Park this module implements twist vectors
        with the translational components first, followed by rotational
        components, ie. :math:`[\omega, \vec{v}]`.

    """

    def __init__(self, arg=None, w=None, check=True):
        """
        Construct a new 3D twist object

        - ``Twist3()`` is a Twist3 instance representing null motion -- the
          identity twist
        - ``Twist3(S)`` is a Twist3 instance from an array-like (6,)
        - ``Twist3(v, w)`` is a Twist3 instance from a moment ``v`` (3,) and
          direction ``w`` (3,)
        - ``Twist3([S1, S2, ... SN])`` where each ``Si`` is a numpy array (6,)
        - ``Twist3(X)`` is a Twist3 instance with the same value as ``X``, ie.
          a copy
        - ``Twist3([X1, X2, ... XN])`` where each Xi is a Twist3 instance, is a
          Twist3 instance containing N motions

        """
        from spatialmath.pose3d import SE3

        super().__init__()

        if w is None:
            # zero or one arguments passed
            if super().arghandler(arg, check=check):
                return
            elif isinstance(arg, SE3):
                self.data = [arg.twist().A]

        elif w is not None and smb.isvector(w, 3) and smb.isvector(arg, 3):
            # Twist(v, w)
            self.data = [np.r_[arg, w]]
            return

        else:
            raise ValueError("bad value to Twist constructor")

    # ------------------------ SMUserList required ---------------------------#

    @staticmethod
    def _identity():
        return np.zeros((6,))

    def _import(self, value, check=True):
        if isinstance(value, np.ndarray) and self.isvalid(value, check=check):
            if value.shape == (4, 4):
                # it's an se(3)
                return smb.vexa(value)
            elif value.shape == (6,):
                # it's a twist vector
                return value
        elif smb.ishom(value, check=check):
            return smb.trlog(value, twist=True, check=False)
        raise TypeError("bad type passed")

    @staticmethod
    def isvalid(v, check=True):
        """
        Test if matrix is valid twist

        :param x: array to test
        :type x: ndarray
        :return: Whether the value is a 6-vector or a valid 4x4 se(3) element
        :rtype: bool

        A twist can be represented by a 6-vector or a 4x4 skew symmetric matrix,
        for example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> from spatialmath.base import skewa
            >>> import numpy as np
            >>> Twist3.isvalid([1, 2, 3, 4, 5, 6])
            >>> a = skewa([1, 2, 3, 4, 5, 6])
            >>> a
            >>> Twist3.isvalid(a)
            >>> Twist3.isvalid(np.random.rand(4,4))
        """
        if smb.isvector(v, 6):
            return True
        elif smb.ismatrix(v, (4, 4)):
            # maybe be an se(3)
            if not smb.iszerovec(v.diagonal()):  # check diagonal is zero
                return False
            if not smb.iszerovec(v[3, :]):  # check bottom row is zero
                return False
            if check and not smb.isskew(v[:3, :3]):
                # top left 3x3 is skew symmetric
                return False
            return True
        return False

    # ------------------------ properties ---------------------------#

    @property
    def shape(self):
        """
        Shape of the object's internal array representation

        :return: (6,)
        :rtype: tuple
        """
        return (6,)

    @property
    def N(self):
        """
        Dimension of the object's group

        :return: dimension
        :rtype: int

        Dimension of the group is 3 for ``Twist3`` and corresponds to the
        dimension of the space (3D in this case) to which these
        rigid-body motions apply.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> x = Twist3()
            >>> x.N
        """
        return 3

    @property
    def v(self):
        """
        Moment vector of twist

        :return: Moment vector
        :rtype: ndarray(3)

        ``X.v`` is a 3-vector representing the moment vector of the twist.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> t = Twist3([1, 2, 3, 4, 5, 6])
            >>> t.v
        """
        return self.data[0][:3]

    @property
    def w(self):
        """
        Direction vector of twist

        :return: Direction vector
        :rtype: ndarray(3)

        ``X.w`` is a 3-vector representing the direction vector of the twist.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> t = Twist3([1, 2, 3, 4, 5, 6])
            >>> t.w

        """
        return self.data[0][3:6]

    # -------------------- variant constructors ----------------------------#

    @classmethod
    def UnitRevolute(cls, a, q, pitch=None):
        """
        Construct a new 3D rotational unit twist

        :param a: Twist axis or line of action
        :type a: array_like(3)
        :param q: Point on the line of action
        :type q: array_like(3)
        :param p: pitch, defaults to None
        :type p: float, optional
        :return: a rotational or helical twist
        :rtype: Twist instance

        A revolute twist with a line of action in the z-direction and passing
        through (1, 2, 0) would be:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.UnitRevolute([0, 0, 1], [1, 2, 0])

        """
        w = smb.unitvec(smb.getvector(a, 3))
        v = -np.cross(w, smb.getvector(q, 3))
        if pitch is not None:
            v = v + pitch * w
        return cls(v, w)

    @classmethod
    def UnitPrismatic(cls, a):
        """
        Construct a new 3D unit prismatic twist

        :param a: Twist axis or line of action
        :type a: array_like(3)
        :return: a prismatic twist
        :rtype: Twist instance

        A prismatic twist with a line of action in the z-direction would be:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.UnitPrismatic([0, 0, 1])

        """
        w = np.r_[0, 0, 0]
        v = smb.unitvec(smb.getvector(a, 3))

        return cls(v, w)

    @classmethod
    def Rx(cls, theta, unit="rad"):
        """
        Create a new 3D twist for pure rotation about the X-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3D twist vector
        :rtype: Twist3 instance

        - ``Twist3.Rx(θ)`` is an SE(3) rotation of θ radians about the x-axis
        - ``Twist3.Rx(θ, "deg")`` as above but θ is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Rx(0.3)
            >>> Twist3.Rx([0.3, 0.4])

        :seealso: :func:`~spatialmath.smb.transforms3d.trotx`
        :SymPy: supported
        """
        return cls([np.r_[0, 0, 0, x, 0, 0] for x in smb.getunit(theta, unit=unit)])

    @classmethod
    def Ry(cls, theta, unit="rad", t=None):
        """
        Create a new 3D twist for pure rotation about the Y-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3D twist vector
        :rtype: Twist3 instance

        - ``Twist3.Ry(θ)`` is an SO(3) rotation of θ radians about the y-axis
        - ``Twist3.Ry(θ, "deg")`` as above but θ is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Ry(0.3)
            >>> Twist3.Ry([0.3, 0.4])

        :seealso: :func:`~spatialmath.smb.transforms3d.troty`
        :SymPy: supported
        """
        return cls([np.r_[0, 0, 0, 0, x, 0] for x in smb.getunit(theta, unit=unit)])

    @classmethod
    def Rz(cls, theta, unit="rad", t=None):
        """
        Create a new 3D twist for pure rotation about the Z-axis

        :param θ: rotation angle about Z-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3D twist vector
        :rtype: Twist3 instance

        - ``Twist3.Rz(θ)`` is an SO(3) rotation of θ radians about the z-axis
        - ``Twist3.Rz(θ, "deg")`` as above but θ is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Rz(0.3)
            >>> Twist3.Rz([0.3, 0.4])

        :seealso: :func:`~spatialmath.smb.transforms3d.trotz`
        :SymPy: supported
        """
        return cls([np.r_[0, 0, 0, 0, 0, x] for x in smb.getunit(theta, unit=unit)])

    @classmethod
    def RPY(cls, *pos, **kwargs):
        r"""
        Create a new 3D twist from roll-pitch-yaw angles

        :param 𝚪: roll-pitch-yaw angles
        :type 𝚪: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: 3D twist vector
        :rtype: Twist3 instance

        - ``Twist3.RPY(𝚪)`` is a 3D rotation defined by a 3-vector of roll,
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

        - ``Twist3.RPY(⍺, β, 𝛾)`` as above but the angles are provided as three
          scalars.

        Foo bar!

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.RPY(0.1, 0.2, 0.3)
            >>> Twist3.RPY([0.1, 0.2, 0.3])
            >>> Twist3.RPY(0.1, 0.2, 0.3, order='xyz')
            >>> Twist3.RPY(10, 20, 30, unit='deg')

        :seealso: :meth:`~spatialmath.SE3.RPY`
        :SymPy: supported
        """
        from spatialmath.pose3d import SE3

        T = SE3.RPY(*pos, **kwargs)
        return cls(T)

    @classmethod
    def Tx(cls, x):
        """
        Create a new 3D twist for pure translation along the X-axis

        :param x: translation distance along the X-axis
        :type x: float
        :return: 3D twist vector
        :rtype: Twist3 instance

        ``Twist3.Tx(x)`` is an se(3) translation of ``x`` along the x-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Tx(2)
            >>> Twist3.Tx([2,3])


        :seealso: :func:`~spatialmath.smb.transforms3d.transl`
        :SymPy: supported
        """
        return cls([np.r_[_x, 0, 0, 0, 0, 0] for _x in smb.getvector(x)], check=False)

    @classmethod
    def Ty(cls, y):
        """
        Create a new 3D twist for pure translation along the Y-axis

        :param y: translation distance along the Y-axis
        :type y: float
        :return: 3D twist vector
        :rtype: Twist3 instance

        ``Twist3.Ty(y)`` is an se(3) translation of ``y`` along the y-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Ty(2)
            >>> Twist3.Ty([2, 3])


        :seealso: :func:`~spatialmath.smb.transforms3d.transl`
        :SymPy: supported
        """
        return cls([np.r_[0, _y, 0, 0, 0, 0] for _y in smb.getvector(y)], check=False)

    @classmethod
    def Tz(cls, z):
        """
        Create a new 3D twist for pure translation along the Z-axis

        :param z: translation distance along the Z-axis
        :type z: float
        :return: 3D twist vector
        :rtype: Twist3 instance

        ``Twist3.Tz(z)`` is an se(3) translation of ``z`` along the z-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Tz(2)
            >>> Twist3.Tz([2, 3])

        :seealso: :func:`~spatialmath.smb.transforms3d.transl`
        :SymPy: supported
        """
        return cls([np.r_[0, 0, _z, 0, 0, 0] for _z in smb.getvector(z)], check=False)

    @classmethod
    def Rand(
        cls, *, xrange=(-1, 1), yrange=(-1, 1), zrange=(-1, 1), N=1
    ):  # pylint: disable=arguments-differ
        """
        Create a new random 3D twist

        :param xrange: x-axis range [min,max], defaults to [-1, 1]
        :type xrange: 2-element sequence, optional
        :param yrange: y-axis range [min,max], defaults to [-1, 1]
        :type yrange: 2-element sequence, optional
        :param zrange: z-axis range [min,max], defaults to [-1, 1]
        :type zrange: 2-element sequence, optional
        :param N: number of random transforms
        :type N: int
        :return: SE(3) matrix
        :rtype: SE3 instance

        Return an SE3 instance with random rotation and translation.

        - ``SE3.Rand()`` is a random SE(3) translation.
        - ``SE3.Rand(N=N)`` is an SE3 object containing a sequence of N random
          poses.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> Twist3.Rand(N=2)

        :seealso: :func:`~spatialmath.quaternions.UnitQuaternion.Rand`
        """
        from spatialmath.pose3d import SO3

        X = np.random.uniform(
            low=xrange[0], high=xrange[1], size=N
        )  # random values in the range
        Y = np.random.uniform(
            low=yrange[0], high=yrange[1], size=N
        )  # random values in the range
        Z = np.random.uniform(
            low=yrange[0], high=zrange[1], size=N
        )  # random values in the range
        R = SO3.Rand(N=N)

        def _twist(x, y, z, r):
            T = smb.transl(x, y, z) @ smb.r2t(r.A)
            return smb.trlog(T, twist=True)

        return cls(
            [_twist(x, y, z, r) for (x, y, z, r) in zip(X, Y, Z, R)], check=False
        )

    # -------------------------  methods -------------------------------#

    def printline(self, **kwargs):
        return self.SE3().printline(**kwargs)

    def unit(self):
        """
        Unit twist

        - ``S.unit()`` is a Twist2 objec3 representing a unit twist aligned with the
          Twist ``S``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3, Twist3
            >>> T = SE3(1, 2, 3) * SE3.Rx(0.3)
            >>> S = Twist3(T)
            >>> S.unit()
        """
        if smb.iszerovec(self.w):
            # rotational twist
            return Twist3(self.S / smb.norm(S.w))
        else:
            # prismatic twist
            return Twist3(smb.unitvec(self.v), [0, 0, 0])

    def ad(self):
        """
        Logarithm of adjoint of 3D twist

        :return: logarithm of adjoint matrix
        :rtype: ndarray(6,6)

        ``S.ad()`` is the 6x6 logarithm of the adjoint matrix of the
        corresponding homogeneous transformation.

        For a twist representing motion from frame {B} to {A}, the adjoint will
        transform a twist relative to frame {A} to one relative to frame {B}.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S = Twist3.Rx(0.3)
            >>> S.ad()

        .. note:: An alternative approach to computing the adjoint is to exponentiate this 6x6
            matrix.

        :seealso: :func:`Twist3.Ad`
        """
        return np.block(
            [
                [smb.skew(self.w), smb.skew(self.v)],
                [np.zeros((3, 3)), smb.skew(self.w)],
            ]
        )

    def Ad(self):
        """
        Adjoint of 3D twist

        :return: adjoint matrix
        :rtype: ndarray(6,6)

        ``S.Ad()`` is the 6x6 adjoint matrix of the corresponding
        homogeneous transformation.

        For a twist representing motion from frame {B} to {A}, the adjoint will
        transform a twist relative to frame {A} to one relative to frame {B}.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S = Twist3.Rx(0.3)
            >>> S.Ad()

        .. note:: Equivalent to, but faster than, ``self.SE3().Ad()`` -- computes
            the adjoint directly from the twist's exponential without
            constructing an intermediate ``SE3`` instance.

        :seealso: :func:`Twist3.ad`, :func:`Twist3.SE3`, :func:`Twist3.exp`
        """
        return smb.tr2adjoint(smb.trexp(self.S, check=False))

    def skewa(self):
        """
        Convert 3D twist to se(3)

        :return: An se(3) matrix
        :rtype: ndarray(4,4)

        ``X.skewa()`` is the twist as a 4x4 augmented skew-symmetric matrix
        belonging to the group se(3). This is the Lie algebra of the
        corresponding SE(3) element.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3, base
            >>> S = Twist3.Rx(0.3)
            >>> se = S.skewa()
            >>> se
            >>> base.trexp(se)
        """
        if len(self) == 1:
            return smb.skewa(self.S)
        else:
            return [smb.skewa(x.S) for x in self]

    @property
    def pitch(self):
        """
        Pitch of a 3D twist

        :return: the pitch of the twist
        :rtype: float

        ``X.pitch()`` is the pitch of the twist as a scalar in units of distance
        per radian.

        If we consider the twist as a screw, this is the distance of
        translation along the screw axis for a one radian rotation about the
        screw axis.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3, Twist3
            >>> T = SE3(1, 2, 3) * SE3.Rx(0.3)
            >>> S = Twist3(T)
            >>> S.pitch

        """
        return np.dot(self.w, self.v)

    def line(self):
        """
        Line of action of 3D twist as a Plucker line

        :return: the 3D line of action
        :rtype: Line instance

        ``X.line()`` is a Plucker object representing the line of the twist axis.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3, Twist3
            >>> T = SE3(1, 2, 3) * SE3.Rx(0.3)
            >>> S = Twist3(T)
            >>> S.line()
        """
        return Line3([Line3(-tw.v + tw.pitch * tw.w, tw.w) for tw in self])

    @property
    def pole(self):
        """
        Pole of a 3D twist

        :return: the pole of the twist
        :rtype: ndarray(3)

        ``X.pole()`` is a point on the twist axis. For a pure translation
        this point is at infinity.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3, Twist3
            >>> T = SE3(1, 2, 3) * SE3.Rx(0.3)
            >>> S = Twist3(T)
            >>> S.pole
        """
        return np.cross(self.w, self.v) / self.theta

    def SE3(self, theta=1, unit="rad"):
        """
        Convert 3D twist to SE(3) matrix

        :return: an SE(3) representation
        :rtype: SE3 instance

        ``S.SE3()`` is an SE3 object representing the homogeneous transformation
        equivalent to the Twist3. This is the exponentiation of the twist vector.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist3
            >>> S = Twist3.Rx(0.3)
            >>> S.SE3()

        :seealso: :func:`Twist3.exp`
        """
        from spatialmath.pose3d import SE3

        theta = smb.getunit(theta, unit)

        if len(theta) == 1:
            # theta is a scalar
            return SE3(smb.trexp(self.S * theta))
        else:
            # theta is a vector
            if len(self) == 1:
                return SE3([smb.trexp(self.S * t) for t in theta])
            elif len(self) == len(theta):
                return SE3([smb.trexp(S * t) for S, t in zip(self.data, theta)])
            else:
                raise ValueError("length of twist and theta not consistent")

    def exp(self, theta=1, unit="rad"):
        """
        Exponentiate a 3D twist

        :param theta: rotation magnitude, defaults to None
        :type theta: float, optional
        :param units: rotational units, defaults to 'rad'
        :type units: str, optional
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``X.exp()`` is the homogeneous transformation equivalent to the twist,
          :math:`e^{[S]}`
        - ``X.exp(θ) as above but with a rotation of ``θ`` about the twist axis,
          :math:`e^{\theta[S]}`

        If ``len(X)==1`` and ``len(θ)==N`` then the resulting SE3 object has
        ``N`` values equivalent to the twist :math:`e^{\theta_i[S]}`.

        If ``len(X)==N`` and ``len(θ)==1`` then the resulting SE3 object has
        ``N`` values equivalent to the twist :math:`e^{\theta[S_i]}`.

        If ``len(X)==N`` and ``len(θ)==N`` then the resulting SE3 object has
        ``N`` values equivalent to the twist :math:`e^{\theta_i[S_i]}`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3, Twist3
            >>> T = SE3(1, 2, 3) * SE3.Rx(0.3)
            >>> S = Twist3(T)
            >>> S.exp(0)
            >>> S.exp(1)

        .. note::

            - For the second form, the twist must, if rotational, have a unit
              rotational component.

        :seealso: :func:`spatialmath.smb.trexp`
        """
        from spatialmath.pose3d import SE3

        theta = smb.getunit(theta, unit)

        if len(self) == 1:
            return SE3([smb.trexp(self.S * t) for t in theta], check=False)
        elif len(self) == len(theta):
            return SE3([smb.trexp(s * t) for s, t in zip(self.S, theta)], check=False)
        else:
            raise ValueError("length mismatch")

    # ------------------------- arithmetic -------------------------------#

    def __mul__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        Twist composition or scaling:

        - ``X * Y`` compounds the twists ``X`` and ``Y``
        - ``X * s`` performs elementwise multiplication of the elements of ``X`` by ``s``
        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``

        ========  ====================  ===================  ========================
             Multiplicands                   Product
        ------------------------------  ---------------------------------------------
           left      right                 type                 operation
        ========  ====================  ===================  ========================
        Twist3    Twist3                Twist3               product of exponentials
        Twist3    scalar                Twist3               element-wise product
        scalar    Twist3                Twist3               element-wise product
        Twist3    SE3                   Twist3               exponential x SE3
        ========  ====================  ===================  ========================

        .. note::

            #. scalar x Twist is handled by ``__rmul__``
            #. scalar multiplication is commutative but the result is not a group
               operation so the result will be a matrix
            #. Any other input combinations result in a ValueError.

        For pose composition the ``left`` and ``right`` operands may be a sequence

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left * right``
         1          M             M    ``prod[i] = left * right[i]``
         N          1             M    ``prod[i] = left[i] * right``
         M          M             M    ``prod[i] = left[i] * right[i]``
        =========   ==========   ====  ================================

        """
        from spatialmath.pose3d import SE3

        # TODO TW * T compounds a twist with an SE2/3 transformation

        if isinstance(right, Twist3):
            # twist composition -> Twist
            return Twist3(
                left.binop(
                    right,
                    lambda x, y: smb.trlog(smb.trexp(x) @ smb.trexp(y), twist=True),
                )
            )
        elif isinstance(right, SE3):
            # twist * SE3 -> SE3
            return SE3(left.binop(right, lambda x, y: smb.trexp(x) @ y), check=False)
        elif smb.isscalar(right):
            # return Twist(left.S * right)
            return Twist3(left.binop(right, lambda x, y: x * y))
        else:
            raise ValueError("twist *, incorrect right operand")

    def __rmul__(
        right, left
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :arg right: right multiplicand
        :arg left: left multiplicand
        :return: product
        :raises: NotImplemented

        Left-multiplication by a scalar

        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``
        """
        if smb.isscalar(left):
            return Twist3(right.S * left)
        else:
            raise ValueError("Twist3 *, incorrect left operand")

    def __str__(self):
        """
        Pretty string representation of 3D twist

        :return: readable representation of the twist
        :rtype: str

        Convert the twist's value to an array of numbers.

        Example:

        .. runblock: pycon

            >>> from spatialmath import Twist3
            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> print(x)
        """
        return "\n".join(
            [
                "({:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g})".format(
                    *list(smb.removesmall(tw.S))
                )
                for tw in self
            ]
        )

    def __repr__(self):
        """
        Readable representation of 3D twist

        :return: readable representation of a twist as a list of arrays
        :rtype: str

        Example:

        .. runblock: pycon

            >>> from spatialmath import Twist3
            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> x
            >>> a.append(a)
            >>> a

        """
        if len(self) == 0:
            return "Twist([])"
        elif len(self) == 1:
            return "Twist3([{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}])".format(
                *list(self.S)
            )
        else:
            return (
                "Twist3([\n"
                + ",\n".join(
                    [
                        "  [{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}]".format(
                            *list(tw)
                        )
                        for tw in self.data
                    ]
                )
                + "\n])"
            )

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        """
        if len(self) == 1:
            p.text(str(self))
        else:
            for i, x in enumerate(self):
                if i > 0:
                    p.break_()
                p.text(f"{i:3d}: {str(x)}")


# ======================================================================== #


class Twist2(BaseTwist):
    def __init__(self, arg=None, w=None, check=True):
        r"""
            Construct a new 2D Twist object

            :type a: 2-element array-like
            :return: 2D prismatic twist
            :rtype: Twist2 instance

            - ``Twist2(R)`` is a 2D Twist object representing the SO(2) rotation expressed as
              a 2x2 matrix.
            - ``Twist2(T)`` is a 2D Twist object representing the SE(2) rigid-body motion expressed as
              a 3x3 matrix.
            - ``Twist2(X)`` if X is an SO2 instance then create a 2D Twist object representing the SO(2) rotation,
              and if X is an SE2 instance then create a 2D Twist object representing the SE(2) motion
            - ``Twist2(V)`` is a  2D Twist object specified directly by a 3-element array-like comprising the
              moment vector (1 element) and direction vector (2 elements).

        :References:
            - Robotics, Vision & Control for Python, Section 2.2.2.4, P. Corke, Springer 2023.
            - Modern Robotics, Lynch & Park, Cambridge 2017

        .. note:: Compared to Lynch & Park this module implements twist vectors
            with the translational components first, followed by rotational
            components, ie. :math:`[\omega, \vec{v}]`.
        """
        from spatialmath.pose2d import SE2

        super().__init__()

        if w is None:
            # zero or one arguments passed
            if super().arghandler(arg, convertfrom=(SE2,), check=check):
                return

        elif w is not None and smb.isscalar(w) and smb.isvector(arg, 2):
            # Twist(v, w)
            self.data = [np.r_[arg, w]]
            return

        raise ValueError("bad twist value")

    # ------------------------ SMUserList required ---------------------------#
    @staticmethod
    def _identity():
        return np.zeros((3,))

    @property
    def shape(self):
        """
        Shape of the object's interal array representation

        :return: (3,)
        :rtype: tuple
        """
        return (3,)

    def _import(self, value, check=True):
        if isinstance(value, np.ndarray) and self.isvalid(value, check=check):
            if value.shape == (3, 3):
                # it's an se(2)
                return smb.vexa(value)
            elif value.shape == (3,):
                # it's a twist vector
                return value
        elif smb.ishom2(value, check=check):
            return smb.trlog2(value, twist=True, check=False)
        raise TypeError("bad type passed")

    @staticmethod
    def isvalid(v, check=True):
        """
        Test if matrix is valid twist

        :param x: array to test
        :type x: ndarray
        :return: Whether the value is a 3-vector or a valid 3x3 se(2) element
        :rtype: bool

        A twist can be represented by a 6-vector or a 4x4 skew symmetric matrix,
        for example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2, base
            >>> import numpy as np
            >>> Twist2.isvalid([1, 2, 3])
            >>> a = base.skewa([1, 2, 3])
            >>> a
            >>> Twist2.isvalid(a)
            >>> Twist2.isvalid(np.random.rand(3,3))
        """
        if smb.isvector(v, 3):
            return True
        elif smb.ismatrix(v, (3, 3)):
            # maybe be an se(2)
            if not smb.iszerovec(v.diagonal()):  # check diagonal is zero
                return False
            if not smb.iszerovec(v[2, :]):  # check bottom row is zero
                return False
            if check and not smb.isskew(v[:2, :2]):
                # top left 2x2 is skew symmetric
                return False
            return True
        return False

    # -------------------- variant constructors ----------------------------#

    @classmethod
    def UnitRevolute(cls, q):
        """
        Construct a new 2D revolute unit twist

        :param q: Point on the line of action
        :type q: array_like(2)
        :return: 2D prismatic twist
        :rtype: Twist2 instance

        - ``Twist2.UnitRevolute(q)`` is a 2D Twist object representing rotation about the 2D point ``q``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> Twist2.UnitRevolute([0, 1])

        :seealso: :meth:`Twist2.UnitPrismatic`
        """

        q = smb.getvector(q, 2)
        v = -np.cross(np.r_[0.0, 0.0, 1.0], np.r_[q, 0.0])
        return cls(v[:2], 1)

    @classmethod
    def UnitPrismatic(cls, a):
        """
        Construct a new 2D primsmatic unit twist

        :param a: Displacment
        :type a: array-like(2)
        :return: 2D prismatic twist
        :rtype: Twist2 instance

        - ``Twist2.UnitPrismatic(a)`` is a 2D Twist object representing 2D-translation in the direction ``a``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> Twist2.UnitPrismatic([1, 2])

        :seealso: :meth:`Twist2.UnitRevolute`
        """
        w = 0
        v = smb.unitvec(smb.getvector(a, 2))
        return cls(v, w)

    # ------------------------ properties ---------------------------#

    @property
    def N(self):
        """
        Dimension of the object's group

        :return: dimension
        :rtype: int

        Dimension of the group is 2 for ``Twist2`` and corresponds to the
        dimension of the space (2D in this case) to which these
        rigid-body motions apply.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> x = Twist2()
            >>> x.N
        """
        return 2

    @property
    def v(self):
        """
        Moment vector of twist

        :return: Moment vector
        :rtype: ndarray(2)

        ``X.v`` is a 2-vector representing the moment vector of the twist.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> t = Twist2([1, 2, 3])
            >>> t.v

        """
        return self.data[0][:2]

    @property
    def w(self):
        """
        Direction vector of twist

        :return: Direction vector
        :rtype: float

        ``X.w`` is a scalar representing the direction "vector" of the twist.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> t = Twist2([1, 2, 3])
            >>> t.w

        """
        return self.data[0][2]

    @property
    def pole(self):
        """
        Pole of a 2D twist

        :return: the pole of the twist
        :rtype: ndarray(2)

        ``X.pole()`` is a point on the twist axis. For a pure translation
        this point is at infinity.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2, Twist2
            >>> T = SE2(1, 2, 0.3)
            >>> S = Twist2(T)
            >>> S.pole()

        """
        p = np.cross(np.r_[0, 0, self.w], np.r_[self.v, 0]) / self.theta
        return p[:2]

    # -------------------------  methods -------------------------------#

    def printline(self, **kwargs):
        return self.SE2().printline(**kwargs)

    def SE2(self, theta=1, unit="rad"):
        """
        Convert 2D twist to SE(2) matrix

        :return: an SE(2) representation
        :rtype: SE3 instance

        ``S.SE2()`` is an SE2 object representing the homogeneous transformation
        equivalent to the Twist2. This is the exponentiation of the twist vector.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> S = Twist2.UnitPrismatic([1,2])
            >>> S.SE2()

        :seealso: :func:`Twist3.exp`
        """
        from spatialmath.pose2d import SE2

        if unit != "rad" and self.isprismatic:
            print("Twist3.exp: using degree mode for a prismatic twist")

        theta = smb.getunit(theta, unit)

        if len(theta) == 1:
            return SE2(smb.trexp2(self.S * theta))
        else:
            return SE2([smb.trexp2(self.S * t) for t in theta])

    def skewa(self):
        """
        Convert 2D twist to se(2)

        :return: An se(2) matrix
        :rtype: ndarray(3,3)

        ``X.skewa()`` is the twist as a 3x3 augmented skew-symmetric matrix
        belonging to the group se(2). This is the Lie algebra of the
        corresponding SE(2) element.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2, base
            >>> S = Twist2([1,2,3])
            >>> se = S.skewa()
            >>> se
            >>> base.trexp2(se)
        """
        if len(self) == 1:
            return smb.skewa(self.S)
        else:
            return [smb.skewa(x.S) for x in self]

    def exp(self, theta=1, unit="rad"):
        r"""
        Exponentiate a 2D twist

        :param theta: rotation magnitude, defaults to None
        :type theta: float, optional
        :param unit: rotational units, defaults to 'rad'
        :type unit: str, optional
        :return: SE(2) matrix
        :rtype: SE2 instance

        - ``X.exp()`` is the homogeneous transformation equivalent to the twist,
          :math:`e^{[S]}`
        - ``X.exp(θ) as above but with a rotation of ``θ`` about the twist axis,
          :math:`e^{\theta[S]}`

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2, Twist2
            >>> T = SE2(1, 2, 0.3)
            >>> S = Twist2(T)
            >>> S.exp(0)
            >>> S.exp(1)

        .. note::

            - For the second form, the twist must, if rotational, have a unit
              rotational component.

        :seealso: :func:`spatialmath.smb.trexp2`
        """
        from spatialmath.pose2d import SE2

        theta = smb.getunit(theta, unit)

        if len(self) == 1:
            return SE2([smb.trexp2(self.S * t) for t in theta], check=False)
        elif len(self) == len(theta):
            return SE2([smb.trexp2(s * t) for s, t in zip(self.S, theta)], check=False)
        else:
            raise ValueError("length mismatch")

    def unit(self):
        """
        Unit twist

        - ``S.unit()`` is a Twist2 object representing a unit twist aligned with the
          Twist ``S``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2, Twist2
            >>> T = SE2(1, 2, 0.3)
            >>> S = Twist2(T)
            >>> S.unit()
        """
        if smb.iszerovec(self.w):
            # rotational twist
            return Twist2(self.S / smb.norm(S.w))
        else:
            # prismatic twist
            return Twist2(smb.unitvec(self.v), [0, 0, 0])

    @property
    def ad(self):
        """
        Twist2.ad Logarithm of adjoint

        - ``S.ad()`` is the logarithm of the adjoint matrix of the corresponding
          homogeneous transformation.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2, Twist2
            >>> T = SE2(1, 2, 0.3)
            >>> S = Twist2(T)
            >>> S.unit()

        :seealso: SE3.Ad.
        """
        return np.array(
            [
                [smb.skew(self.w), smb.skew(self.v)],
                [np.zeros((3, 3)), smb.skew(self.w)],
            ]
        )

    @classmethod
    def Tx(cls, x):
        """
        Create a new 2D twist for pure translation along the X-axis

        :param x: translation distance along the X-axis
        :type x: float
        :return: 2D twist vector
        :rtype: Twist2 instance

        `Twist2.Tx(x)` is an se(2) translation of ``x`` along the x-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> Twist2.Tx(2)
            >>> Twist2.Tx([2,3])


        :seealso: :func:`~spatialmath.smb.transforms2d.transl2`
        :SymPy: supported
        """
        return cls([np.r_[_x, 0, 0] for _x in smb.getvector(x)], check=False)

    @classmethod
    def Ty(cls, y):
        """
        Create a new 2D twist for pure translation along the Y-axis

        :param y: translation distance along the Y-axis
        :type y: float
        :return: 2D twist vector
        :rtype: Twist2 instance

        `Twist2.Ty(y) is an se(2) translation of ``y`` along the y-axis

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Twist2
            >>> Twist2.Ty(2)
            >>> Twist2.Ty([2, 3])


        :seealso: :func:`~spatialmath.smb.transforms2d.transl2`
        :SymPy: supported
        """
        return cls([np.r_[0, _y, 0] for _y in smb.getvector(y)], check=False)

    def __mul__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        - ``X * Y`` compounds the twists ``X`` and ``Y``
        - ``X * s`` performs elementwise multiplication of the elements of ``X`` by ``s``
        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``

        ========  ====================  ===================  ========================
             Multiplicands                   Product
        ------------------------------  ---------------------------------------------
           left      right                 type                 operation
        ========  ====================  ===================  ========================
        Twist2    Twist2                Twist2               product of exponentials
        Twist2    scalar                Twist2               element-wise product
        scalar    Twist2                Twist2               element-wise product
        Twist2    SE2                   Twist2               exponential x SE2
        ========  ====================  ===================  ========================

        .. note::

            #. scalar x Twist is handled by ``__rmul__``
            #. scalar multiplication is commutative but the result is not a group
               operation so the result will be a matrix
            #. Any other input combinations result in a ValueError.

        For pose composition the ``left`` and ``right`` operands may be a sequence

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left * right``
         1          M             M    ``prod[i] = left * right[i]``
         N          1             M    ``prod[i] = left[i] * right``
         M          M             M    ``prod[i] = left[i] * right[i]``
        =========   ==========   ====  ================================
        """
        from spatialmath.pose2d import SE2

        if isinstance(right, Twist2):
            # twist composition -> Twist
            return Twist2(
                left.binop(
                    right,
                    lambda x, y: smb.trlog2(smb.trexp2(x) @ smb.trexp2(y), twist=True),
                )
            )
        elif isinstance(right, SE2):
            # twist * SE2 -> SE2
            return SE2(left.binop(right, lambda x, y: smb.trexp2(x) @ y), check=False)
        elif smb.isscalar(right):
            # return Twist(left.S * right)
            return Twist2(left.binop(right, lambda x, y: x * y))
        else:
            raise ValueError("Twist2 *, incorrect right operand")

    def __rmul(self, left):
        if smb.isscalar(left):
            return Twist2(self.S * left)
        else:
            raise ValueError("twist *, incorrect left operand")

    def __str__(self):
        """
        Pretty string representation of 2D twist

        :return: readable representation of the twist
        :rtype: str

        Convert the twist's value to an array of numbers.

        Example:

        .. runblock: pycon

            >>> x = Twist2([1,2,3])
            >>> print(x)
        """
        return "\n".join(["({:.5g} {:.5g}; {:.5g})".format(*list(tw.S)) for tw in self])

    def __repr__(self):
        """
        Readable representation of 2D twist

        :return: readable representation of a twist as a list of arrays
        :rtype: str

        Example:

        .. runblock: pycon

            >>> from spatialmath import Twist2
            >>> x = Twist2([1,2,3])
            >>> x
            >>> a.append(a)
            >>> a

        """

        if len(self) == 1:
            return "Twist2([{:.5g}, {:.5g}, {:.5g}])".format(*list(self.S))
        else:
            return (
                "Twist2([\n"
                + ",\n".join(
                    ["  [{:.5g}, {:.5g}, {:.5g}]".format(*list(tw.S)) for tw in self]
                )
                + "\n])"
            )

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        """
        if len(self) == 1:
            p.text(str(self))
        else:
            for i, x in enumerate(self):
                if i > 0:
                    p.break_()
                p.text(f"{i:3d}: {str(x)}")


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_twist.py"
        ).read()
    )  # pylint: disable=exec-used

# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
A set of cooperating classes to support Featherstone's spatial vector formalism


.. inheritance-diagram:: spatialmath.spatialvector
   :top-classes: collections.UserList
   :parts: 1

.. note:: Compared to Featherstone's papers these spatial vectors have the
    translational components first, followed by rotational components.
"""

from abc import abstractmethod
import numpy as np
from spatialmath.baseposelist import BasePoseList
from spatialmath import base
from spatialmath.pose3d import SE3
from spatialmath.twist import Twist3


class SpatialVector(BasePoseList):
    """
    Spatial 6-vector abstract superclass

    This class has two abstract subclasses, which each have concrete subclasses.
    Key characteristics:

    - 6D vectors that represent velocity, acceleration, momentum and force of
      bodies in 3D.
    - inherit list-like properties from ``SMUserList`` class
    - support operators:

    ========   ===========================================================
    Operator   Operation
    ========   ===========================================================
    ``+``      addition of spatial vectors of the same subclass
    ``-``      subtraction of spatial vectors of the same subclass
    ``-``      unary minus
    ``*``      see table below
    ``^``      cross product x or x*
    ========   ===========================================================


    Certain subtypes can be multiplied

    ===================   ====================  ===================  =========================
                Multiplicands                   Product
    ------------------------------------------  ----------------------------------------------
    left                  right                 type                 operation
    ===================   ====================  ===================  =========================
    SE3, Twist3           SpatialVelocity       SpatialVelocity      adjoint product
    SE3, Twist3           SpatialAcceleration   SpatialAcceleration  adjoint product
    SE3, Twist3           SpatialMomentum       SpatialMomentum      adjoint transpose product
    SE3, Twist3           SpatialForce          SpatialForce         adjoint transpose product
    SpatialAcceleration   SpatialInertia        SpatialForce         matrix-vector product**
    SpatialVelocity       SpatialInertia        SpatialMomentum      matrix-vector product**
    ===================   ====================  ===================  =========================

    ** indicates commutative operator.

    .. inheritance-diagram:: spatialmath.spatialvector.SpatialVelocity spatialmath.spatialvector.SpatialAcceleration spatialmath.spatialvector.SpatialForce spatialmath.spatialvector.SpatialMomentum
       :top-classes: spatialmath.spatialvector.SpatialVector
       :parts: 1

    **References:**

    - "Robot Dynamics Algorithms", R. Featherstone, volume 22,
      Springer International Series in Engineering and Computer Science,
      Springer, 1987.
    - "A beginner's guide to 6-d vectors (part 1)", R. Featherstone,
      IEEE Robotics Automation Magazine, 17(3):83-94, Sep. 2010.
    - `Online notes <http://users.cecs.anu.edu.au/~roy/spatial>`_
      Methods:

    :seealso: :func:`~spatialmath.spatialvector.SpatialM6`, :func:`~spatialmath.spatialvector.SpatialF6`, :func:`~spatialmath.spatialvector.SpatialVelocity`, :func:`~spatialmath.spatialvector.SpatialAcceleration`, :func:`~spatialmath.spatialvector.SpatialForce`, :func:`~spatialmath.spatialvector.SpatialMomentum`.
    """

    def __init__(self, value):
        """
        Create a new spatial vector (abstract superclass)

        :param value: Value of the

        - ``SpatialVector(vec)`` is a spatial vector constructed from the 6-element array-like ``vec``
        - ``SpatialVector([V1, V2, ... VN])`` is a spatial vector array with N elements, constructed from the 6-element
          array-like values ``Vi``
        - ``SpatialVector(A)`` is a spatial vector array with N elements, constructed from the columns of the 6xN
          array ``A``.

        """
        # print('spatialVec6 init')
        super().__init__()

        if base.isvector(value, 6):
            self.data = [np.array(value)]
        elif base.isvector(value, 3):
            self.data = [np.r_[value, 0, 0, 0]]
        elif isinstance(value, SpatialVector):
            self.data = [value.A]
        elif base.ismatrix(value, (6, None)):
            self.data = [x for x in value.T]
        elif not super().arghandler(value):
            raise ValueError("bad argument to constructor")

        # elif isinstance(value, list):
        #     assert all(map(lambda x: base.isvector(x, 6), value)), 'all elements of list must have valid shape and value for the class'
        #     self.data = [np.array(x) for x in value]
        # else:
        #     raise ValueError('bad arguments to constructor')

    @staticmethod
    def _identity():
        return np.zeros((6,))

    def isvalid(self, x, check):
        """
        Test if vector is valid spatial vector

        :param x: vector to test
        :type x: numpy.ndarray
        :arg check: ignored
        :type check: bool
        :return: True if the matrix has shape (6,).
        :rtype: bool
        """
        return x.shape == self.shape

    def _import(self, value, check=True):
        if isinstance(value, np.ndarray) and self.isvalid(value, check=check):
            return value
        raise TypeError("bad type passed")

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (6,)
        :rtype: tuple
        """
        return (6,)

    def __getitem__(self, i):
        return self.__class__(self.data[i])

    # ------------------------------------------------------------------------ #

    def __repr__(self):
        """

        :return:
        SpatialVec6.display Display parameters

        V.display() displays the spatial vector parameters in compact single line format.
        If V is an array of spatial vector objects it displays one per line.

        Notes:

         - This method is invoked implicitly at the command line when the result
           of an expression is a serial vector subclass object and the command has
           no trailing semicolon.
        """
        return self.__str__()

    def __str__(self):
        """
        Pretty string representation (superclass method)

        :return: readable representation of the spatial vector
        :rtype: str

        - ``s = str(v)`` is a string showing spatial vector parameters in a
        compact single line format.

        If V is an array of spatial vector objects return a string with one
        line per element.
        """
        typ = type(self).__name__
        return "\n".join(
            [
                "{:s}[{:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g}]".format(typ, *list(x))
                for x in self.data
            ]
        )

    def __neg__(self):
        """
        Overloaded unary ``-`` operator (superclass method)

        :return: negative of spatial vector
        :rtype: SpatialVector subclass instance

        ``-v`` is a spatial vector of the same type as ``v`` whose value is
        the element-wise negative of ``v``.

        :seealso: :func:`__sub__`
        """

        # for i=1:numel(obj)
        # y(i) = obj.new(-obj(i).vw);

        return self.__class__([-x for x in self.data])

    def __add__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator (superclass method)

        :return: sum of spatial vectors
        :rtype: SpatialVector subclass instance
        :raises TypeError: attempting to add SpatialVectors of different subclass
        :raises ValueErrror: attempting to add SpatialVectors with different numbers of values

        ``v1 + v2`` is a spatial vector of the same type as ``v1`` and ``v2`` whose value is
        the element-wise sum of ``v1`` and ``v2``.  If both are arrays of spatial vectors V1 (1xN) and
        V2 (1xN) the result is an array (1xN).

        :seealso: :func:`__sub__`
        """

        # TODO broadcasting with binop
        if type(left) != type(right):
            raise TypeError("can only add spatial vectors of same type")
        if len(left) != len(right):
            raise ValueError("can only add equal length arrays of spatial vectors")

        return left.__class__([x + y for x, y in zip(left.data, right.data)])

    def __sub__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``-`` operator (superclass method)

        :return: difference of spatial vectors
        :rtype: SpatialVector subclass instance
        :raises TypeError: attempting to subtract SpatialVectors of different subclass
        :raises ValueErrror: attempting to subtract SpatialVectors with different numbers of values

        ``v1 - v2`` is a spatial vector of the same type as ``v1`` and ``v2``
        whose value is the element-wise difference of ``v1`` and ``v2``.  If
        both are arrays of spatial vectors V1 (1xN) and V2 (1xN) the result is
        an array (1xN).

        :seealso: :func:`__add__`, :func:`__neg__`
        """
        if type(left) != type(right):
            raise TypeError("can only add spatial vectors of same type")
        if len(left) != len(right):
            raise ValueError("can only add equal length arrays of spatial vectors")

        return left.__class__([x - y for x, y in zip(left.data, right.data)])

    def __rmul__(
        right, left
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator (superclass method)

        :return: transformed spatial vectors
        :rtype: SpatialVector subclass instance
        :raises TypeError: for incompatible left operand

        ``X * S`` transforms the spatial vector ``S`` by the relative pose ``X``
        which may be either an ``SE3`` or ``Twist3`` instance.  The spatial
        vector is premultiplied by the adjoint of ``X`` or adjoint transpose
        of ``X`` depending on the SpatialVector subclass of ``S``.

        ===========  ====================  ===================  =========================
                   Multiplicands                   Product
        -------------------------------   -----------------------------------
        left         right                type                 operation
        ===========  ====================  ===================  =========================
        SE3, Twist3  SpatialVelocity       SpatialVelocity      adjoint product
        SE3, Twist3  SpatialAcceleration   SpatialAcceleration  adjoint product
        SE3, Twist3  SpatialMomentum       SpatialMomentum      adjoint transpose product
        SE3, Twist3  SpatialForce          SpatialForce         adjoint transpose product
        ===========  ====================  ===================  =========================
        """
        if isinstance(left, (SE3, Twist3)):
            X = left.Ad()
            if isinstance(right, SpatialM6):
                return right.__class__(X @ right.A)
            else:
                # Force/momentum are dual to motion and transform by the
                # coadjoint (inverse-transpose of the adjoint), not the
                # adjoint transpose. These are only equal for a pure rotation.
                return right.__class__(np.linalg.inv(X).T @ right.A)
        else:
            raise TypeError("left operand of * must be SE3 or Twist3")


# ------------------------------------------------------------------------- #


class SpatialM6(SpatialVector):
    """
    Spatial 6-vector abstract motion superclass

    Abstract superclass that represents the vector space for spatial motion.

    :seealso: :func:`~spatialmath.spatialvector.SpatialVelocity`, :func:`~spatialmath.spatialvector.SpatialAcceleration`
    """

    @abstractmethod
    def __init__(self, value):
        super().__init__(value)

    def cross(self, other):
        r"""
        Spatial vector cross product

        :param other: spatial motion vector
        :type other: SpatialM6 instance
        :return: cross product of spatial vectors
        :rtype: SpatialF6 instance if ``other`` is SpatialF6 instance
        :rtype: SpatialM6 instance if ``other`` is SpatialM6 instance

        ``v1.cross(v2)`` is a spatial vector cross product whose result depends
        on the SpatialVector subclass of ``v2``:

        - if :math:`\vec{m} \in \mat{M}^6` is a spatial motion vector fixed in a
          body with velocity :math:`\vec{v}` then
          :math:`\dvec{m} = \vec{v} \times \vec{m}` or the ``crm()`` function.

        - if :math:`\vec{f} \in \mat{F}^6` is a spatial force vector fixed in a
          body with velocity :math:`\vec{v}` then
          :math:`\dvec{f} = \vec{v} \times^* \vec{f}` or the ``crm()`` function.
        """

        # v = obj.vw;
        # # vcross = [ skew(w) skew(v); zeros(3,3) skew(w) ]

        v = self.A
        vcross = np.array(
            [
                [0, -v[5], v[4], 0, -v[2], v[1]],
                [v[5], 0, -v[3], v[2], 0, -v[0]],
                [-v[4], v[3], 0, -v[1], v[0], 0],
                [0, 0, 0, 0, -v[5], v[4]],
                [0, 0, 0, v[5], 0, -v[3]],
                [0, 0, 0, -v[4], v[3], 0],
            ]
        )
        if isinstance(other, SpatialVelocity):
            return SpatialAcceleration(vcross @ other.A)  # x operator (crm)
        elif isinstance(other, SpatialF6):
            return SpatialForce(-vcross.T @ other.A)  # x* operator (crf)
        else:
            raise TypeError("type mismatch")


# ------------------------------------------------------------------------- #


class SpatialF6(SpatialVector):
    """
    Spatial 6-vector abstract force superclass

    Abstract superclass that represents the vector space for spatial force.

    :seealso: :func:`~spatialmath.spatialvector.SpatialForce`, :func:`~spatialmath.spatialvector.SpatialMomentum`.
    """

    @abstractmethod
    def __init__(self, value):
        super().__init__(value)

    def dot(self, value):
        return np.dot(self.A, base.getvector(value, 6))


# ------------------------------------------------------------------------- #


class SpatialVelocity(SpatialM6):
    """
    Spatial velocity class

    Concrete subclass of SpatialM6 that represents the
    translational and rotational velocity of a rigid-body moving in 3D space.

    .. inheritance-diagram:: spatialmath.spatialvector.SpatialVelocity
       :top-classes: collections.UserList
       :parts: 1

    :seealso: :func:`~spatialmath.spatialvector.SpatialM6`, :func:`~spatialmath.spatialvector.SpatialAcceleration`

    """

    def __init__(self, value=None):
        super().__init__(value)

    # def cross(self, other):
    #     r"""
    #     Spatial vector cross product

    #     :param other: spatial velocity vector
    #     :type other: SpatialVelocity or SpatialMomentum instance
    #     :return: cross product of spatial vectors
    #     :rtype: SpatialAcceleration instance if ``other`` is SpatialVelocity instance
    #     :rtype: SpatialMomentum instance if ``other`` is SpatialForce instance

    #     - ``v1.cross(v2)`` is spatial acceleration given spatial velocities
    #       ``v1`` and ``v2`` or :math:`\vec{v}_1 \times \vec{v}_2`
    #     - ``v1.cross(m2)`` is spatial force given spatial velocity
    #       ``v1`` and spatial momentum ``m2`` or :math:`\vec{v}_1 \times^* \vec{m}_2`

    #     :seealso: :func:`~spatialmath.spatialvector.SpatialM6`, :func:`~spatialmath.spatialvector.SpatialVelocity.__xor__`
    #     """
    #     if not len(self) == 1 or not len(other) == 1:
    #         raise ValueError("can only perform cross product on single-valued spatial vectors")
    #     return SpatialAcceleration(super().cross(other))

    def __matmul__(self, other):
        r"""
        Overloaded ``@`` operator (superclass method)

        :param other: spatial velocity vector
        :type other: SpatialVelocity or SpatialMomentum instance
        :return: cross product of spatial vectors
        :rtype: SpatialAcceleration instance if ``other`` is SpatialVelocity instance
        :rtype: SpatialMomentum instance if ``other`` is SpatialForce instance

        This operator implements the spatial vector cross product.

        - ``v1 @v2`` is spatial acceleration given spatial velocities
          ``v1`` and ``v2`` or :math:`\vec{v}_1 \times \vec{v}_2`
        - ``v1 @ m2`` is spatial force given spatial velocity
          ``v1`` and spatial momentum ``m2`` or :math:`\vec{v}_1 \times^* \vec{m}_2`

        .. note:: The ``@`` operator was chosen because it has high precendence
            and is somewhat invocative of multiplication.

        :seealso: :func:`~spatialmath.spatialvector.SpatialVelocity.cross`
        """
        return self.cross(other)


# ------------------------------------------------------------------------- #


class SpatialAcceleration(SpatialM6):
    """
    Spatial acceleration class

    Concrete subclass of SpatialM6 that represents the
    translational and rotational acceleration of a rigid-body moving in 3D space.

    .. inheritance-diagram:: spatialmath.spatialvector.SpatialAcceleration
       :top-classes: collections.UserList
       :parts: 1

    :seealso: :func:`~spatialmath.spatialvector.SpatialM6`, :func:`~spatialmath.spatialvector.SpatialVelocity`

    """

    def __init__(self, value=None):
        super().__init__(value)


# ------------------------------------------------------------------------- #


class SpatialForce(SpatialF6):
    """
    Spatial force class

    Concrete subclass of SpatialF6 and represents the
    translational and rotational forces and torques acting on a rigid-body in 3D space.

    .. inheritance-diagram:: spatialmath.spatialvector.SpatialForce
       :top-classes: collections.UserList
       :parts: 1

    :seealso: :func:`~spatialmath.spatialvector.SpatialF6`, :func:`~spatialmath.spatialvector.SpatialMomentum`
    """

    def __init__(self, value=None):
        super().__init__(value)

    # n = SpatialForce(val);

    def __rmul__(
        right, left
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        # Twist * SpatialForce -> SpatialForce
        # A force is dual to motion, so it transforms by the coadjoint
        # (inverse-transpose of the adjoint), not the adjoint transpose.
        return SpatialForce(np.linalg.inv(left.Ad()).T @ right.A)


# ------------------------------------------------------------------------- #


class SpatialMomentum(SpatialF6):

    """
    Spatial momentum class

    Concrete subclass of SpatialF6 and represents the
    translational and rotational momentum of a rigid-body in 3D space.

    .. inheritance-diagram:: spatialmath.spatialvector.SpatialMomentum
       :top-classes: collections.UserList
       :parts: 1

    :seealso: :func:`~spatialmath.spatialvector.SpatialF6`,  :func:`~spatialmath.spatialvector.SpatialForce`
    """

    def __init__(self, value=None):
        super().__init__(value)


# ------------------------------------------------------------------------- #


class SpatialInertia(BasePoseList):
    """
    Spatial inertia class

    Spatial inertia of a body in 3D space.

    ========   ===========================================================
    Operator   Operation
    ========   ===========================================================
    ``+``      addition of spatial inertias of joined bodies
    ``*``      acceleration x inertia is force
    ========   ===========================================================

    :seealso: :func:`~spatialmath.spatialvector.SpatialM6`, :func:`~spatialmath.spatialvector.SpatialF6`, :func:`~spatialmath.spatialvector.SpatialVelocity`, :func:`~spatialmath.spatialvector.SpatialAcceleration`, :func:`~spatialmath.spatialvector.SpatialForce`, :func:`~spatialmath.spatialvector.SpatialMomentum`.

    """

    def __init__(self, m=None, r=None, I=None):
        """
        Create a new spatial inertia

        :param m: mass
        :type m: float
        :param r: centre of mass relative to link frame
        :type r: 3-element array_like
        :param I: inertia about the centre of mass, axes aligned with link frame
        :type I: numpy.array, shape=(6,6)

        - ``SpatialInertia(m, r, I)`` is a spatial inertia object for a rigid-body
          with mass ``m``, centre of mass at ``r`` relative to the link frame, and an
          inertia matrix ``I`` (3x3) about the centre of mass.

        - ``SpatialInertia(I)`` is a spatial inertia object with a value equal
          to ``I`` (6x6).

        :SymPy: supported
        """
        super().__init__()

        if m is None and r is None and I is None:
            # no arguments
            I = SpatialInertia._identity()
        elif m is not None and r is None and I is None and base.ismatrix(m, (6, 6)):
            I = base.getmatrix(m, (6, 6))
        elif m is not None and r is not None:
            r = base.getvector(r, 3)
            if I is None:
                I = np.zeros((3, 3))
            else:
                I = base.getmatrix(I, (3, 3))
            C = base.skew(r)
            M = np.diag((m,) * 3)  # sym friendly
            I = np.block([[M, m * C.T], [m * C, I + m * C @ C.T]])
        else:
            raise ValueError("bad values")

        self.data = [I]

    @staticmethod
    def _identity():
        return np.zeros((6, 6))

    def isvalid(self, x, check):
        """
        Test if matrix is valid spatial inertia

        :param x: matrix to test
        :type x: numpy.ndarray
        :arg check: ignored
        :type check: bool
        :return: True if the matrix has shape (6,6).
        :rtype: bool
        """
        return self.shape == x.shape

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (6,6)
        :rtype: tuple
        """
        return (6, 6)

    def __getitem__(self, i):
        return SpatialInertia(self.data[i])

    def __repr__(self):
        """
        Convert to string

        s = SI.char() is a string showing spatial inertia parameters in a
        compact format.
        If SI is an array of spatial inertia objects return a string with the
        inertia values in a vertical list.

        See also SpatialInertia.display.
        """
        return self.__str__()

    def __str__(self):
        return str(self.A)

    def __add__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Spatial inertia addition
        :param left:
        :param right:
        :return:
        :raises TypeError: attempting to add invalid type to SpatialInertia

        - ``SI1 + SI2`` is the SpatialInertia of a composite body when bodies with
           SpatialInertia ``SI1`` and ``SI2`` are connected.
        """
        if not isinstance(right, SpatialInertia):
            raise TypeError("can only add spatial inertia to spatial inertia")
        return SpatialInertia(left.A + right.A)

    def __mul__(
        left, right
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator (superclass method)

        :param other: spatial acceleration vector
        :type other: SpatialAcceleration instance
        :return: force
        :rtype: SpatialForce instance if ``other`` is SpatialAcceleration instance
        :rtype: SpatialMomentum instance if ``other`` is SpatialVelocity instance

        - ``I * a`` is the SpatialForce required for a body with SpatialInertia ``I`` to accelerate with
          the SpatialAcceleration ``a``.
        - ``I * v`` is the SpatialMomemtum of a body with SpatialInertia ``I`` and SpatialVelocity ``v``.
        """

        if isinstance(right, SpatialAcceleration):
            return SpatialForce(left.A @ right.A)  # F = ma
        elif isinstance(right, SpatialVelocity):
            # crf(v(i).vw)*model.I(i).I*v(i).vw;
            # v = Wrench( a.cross() * I.I * a.vw );
            return SpatialMomentum(left.A @ right.A)  # M = mv
        else:
            raise TypeError("bad postmultiply operands for Inertia *")

    def __rmul__(
        right, left
    ):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator (superclass method)

        :param other: spatial acceleration vector
        :type other: SpatialAcceleration instance
        :return: force
        :rtype: SpatialForce instance if ``other`` is SpatialAcceleration instance
        :rtype: SpatialMomentum instance if ``other`` is SpatialVelocity instance

        - ``a * I`` is the SpatialForce required for a body with SpatialInertia ``I`` to accelerate with
          the SpatialAcceleration ``a``.
        - ``v * I`` is the SpatialMomemtum of a body with SpatialInertia ``I`` and SpatialVelocity ``v``.
        """
        return right.__mul__(left)


if __name__ == "__main__":
    import numpy.testing as nt
    import pathlib

    v = SpatialVelocity()
    print(v)
    print(len(v))
    v.append(v)
    print(v)
    print(len(v))

    v = SpatialVelocity(np.r_[1, 2, 3, 4, 5, 6])
    print(v)
    v = SpatialVelocity(np.r_[1, 2, 3])
    print(v)

    a = v + v
    print(a)

    vj = SpatialVelocity()

    x = vj @ vj
    print(x)

    # I = SpatialInertia()
    # print(I)
    # print(len(I))
    # I.append(I)
    # print(I)
    # print(len(I))

    # z = SpatialForce([1,2,3,4,5,6])
    # print(z)
    # z = SpatialMomentum([1,2,3,4,5,6])
    # print(z)

    v = SpatialVelocity()
    a = SpatialAcceleration()
    I = SpatialInertia()
    x = I * v
    print(I * v)
    print(I * a)

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute()
            / "tests"
            / "test_spatialvector.py"
        ).read()
    )  # pylint: disable=exec-used

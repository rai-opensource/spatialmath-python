# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE
from __future__ import annotations

import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import spatialmath.base as base
from spatialmath.base.types import *
from spatialmath.baseposelist import BasePoseList
import warnings

_eps = np.finfo(np.float64).eps

# ======================================================================== #


class Plane3:
    r"""
    Create a plane object from linear coefficients

    :param c: Plane coefficients
    :type c: array_like(4)
    :return: a Plane object
    :rtype: Plane

    Planes are represented by the 4-vector :math:`[a, b, c, d]` which describes
    the plane :math:`\pi: ax + by + cz + d=0`.
    """

    def __init__(self, c: ArrayLike4):
        self.plane = base.getvector(c, 4)

    # point and normal
    @classmethod
    def PointNormal(cls, p: ArrayLike3, n: ArrayLike3) -> Self:
        """
        Create a plane object from point and normal

        :param p: Point in the plane
        :type p: array_like(3)
        :param n: Normal vector to the plane
        :type n: array_like(3)
        :return: a Plane object
        :rtype: Plane

        :seealso: :meth:`ThreePoints` :meth:`LinePoint`
        """
        n = base.getvector(n, 3)  # normal to the plane
        p = base.getvector(p, 3)  # point on the plane
        return cls(np.r_[n, -np.dot(n, p)])

    # point and normal
    @classmethod
    def ThreePoints(cls, p: R3x3) -> Self:
        """
        Create a plane object from three points

        :param p: Three points in the plane
        :type p: ndarray(3,3)
        :return: a Plane object
        :rtype: Plane

        The points in ``p`` are arranged as columns.

        :seealso: :meth:`PointNormal`  :meth:`LinePoint`
        """

        p = base.getmatrix(p, (3, 3))
        v1 = p[:, 0]
        v2 = p[:, 1]
        v3 = p[:, 2]

        # compute a normal
        n = np.cross(v2 - v1, v3 - v1)

        return cls(np.r_[n, -np.dot(n, v1)])

    @classmethod
    def LinePoint(cls, l: Line3, p: ArrayLike3) -> Self:
        """
        Create a plane object from a line and point

        :param l: 3D line
        :type l: Line3
        :param p: Points in the plane
        :type p: ndarray(3)
        :return: a Plane object
        :rtype: Plane

        :seealso: :meth:`PointNormal`  :meth:`ThreePoints`
        """
        n = np.cross(l.w, p)
        d = np.dot(l.v, p)

        return cls(np.r_[n, d])

    @classmethod
    def TwoLines(cls, l1: Line3, l2: Line3) -> Self:
        """
        Create a plane object from two line

        :param l1: 3D line
        :type l1: Line3
        :param l2: 3D line
        :type l2: Line3
        :return: a Plane object
        :rtype: Plane

        .. warning:: This algorithm fails if the lines are parallel.

        :seealso: :meth:`LinePoint` :meth:`PointNormal`  :meth:`ThreePoints`
        """
        n = np.cross(l1.w, l2.w)
        d = np.dot(l1.v, l2.w)

        return cls(np.r_[n, d])

    @staticmethod
    def intersection(pi1: Plane3, pi2: Plane3, pi3: Plane3) -> R3:
        """
        Intersection point of three planes

        :param pi1: plane 1
        :type pi1: Plane
        :param pi2: plane 2
        :type pi2: Plane
        :param pi3: plane 3
        :type pi3: Plane
        :return: coordinates of intersection point
        :rtype: ndarray(3)

        This static method computes the intersection point of the three planes
        given as arguments.

        .. warning:: This algorithm fails if the planes do not intersect, or
            intersect along a line.

        :seealso: :meth:`Plane`
        """
        A = np.vstack([pi1.n, pi2.n, pi3.n])
        b = np.array([pi1.d, pi2.d, pi3.d])
        return np.linalg.det(A) @ b

    @property
    def n(self) -> R3:
        r"""
        Normal to the plane

        :return: Normal to the plane
        :rtype: ndarray(3)

        For a plane :math:`\pi: ax + by + cz + d=0` this is the vector
        :math:`[a,b,c]`.

        :seealso: :meth:`d`
        """
        # normal
        return self.plane[:3]

    @property
    def d(self) -> float:
        r"""
        Plane offset

        :return: Offset of the plane
        :rtype: float

        For a plane :math:`\pi: ax + by + cz + d=0` this is the scalar
        :math:`d`.


        :seealso: :meth:`n`
        """
        return self.plane[3]

    def contains(self, p: ArrayLike3, tol: float = 20) -> bool:
        """
        Test if point in plane

        :param p: A 3D point
        :type p: array_like(3)
        :param tol: tolerance in units of eps, defaults to 20
        :type tol: float
        :return: if the point is in the plane
        :rtype: bool
        """
        return abs(np.dot(self.n, p) - self.d) < tol * _eps

    def plot(
        self,
        bounds: Optional[ArrayLike] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ):
        """
        Plot plane

        :param bounds: bounds of plot volume, defaults to None
        :type bounds: array_like(2|4|6), optional
        :param ax: 3D axes to plot into, defaults to None
        :type ax: Axes, optional
        :param kwargs: optional arguments passed to ``plot_surface``

        The ``bounds`` of the 3D plot volume is [xmin, xmax, ymin, ymax, zmin, zmax]
        and a 3D plot is created if not already existing.  If ``bounds`` is not
        provided it is taken from current 3D axes.

        The plane is drawn using ``plot_surface``.

        :seealso: :func:`axes_logic`
        """
        ax = base.axes_logic(ax, 3)
        if bounds is None:
            bounds = np.r_[ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]

        # X, Y = np.meshgrid(bounds[0: 2], bounds[2: 4])
        # Z = -(X * self.plane[0] + Y * self.plane[1] + self.plane[3]) / self.plane[2]

        X, Y = np.meshgrid(
            np.linspace(bounds[0], bounds[1], 50), np.linspace(bounds[2], bounds[3], 50)
        )
        Z = -(X * self.plane[0] + Y * self.plane[1] + self.plane[3]) / self.plane[2]
        Z[Z < bounds[4]] = np.nan
        Z[Z > bounds[5]] = np.nan
        ax.plot_surface(X, Y, Z, **kwargs)

    def __str__(self) -> str:
        """
        Convert plane to string representation

        :return: Compact string representation of plane
        :rtype: str
        """
        return str(self.plane)

    def __repr__(self) -> str:
        """
        Display parameters of plane

        :return: Compact string representation of plane
        :rtype: str
        """
        return str(self)


# ======================================================================== #


class Line3(BasePoseList):
    __array_ufunc__ = None  # allow pose matrices operators with NumPy values

    @overload
    def __init__(self, v: ArrayLike3, w: ArrayLike3):
        ...

    @overload
    def __init__(self, v: ArrayLike6):
        ...

    def __init__(self, v=None, w=None, check=True):
        """
        Create a Line3 object

        :param v: Plucker coordinate vector, or Plucker moment vector
        :type v: array_like(6) or array_like(3)
        :param w: Plucker direction vector, optional
        :type w: array_like(3), optional
        :param check: check that the parameters are valid, defaults to True
        :type check: bool
        :raises ValueError: bad arguments
        :return: 3D line
        :rtype: ``Line3`` instance

        A representation of a 3D line using Plucker coordinates.

        - ``Line3(p)`` creates a 3D line from a Plucker coordinate vector ``p=[v, w]``
           where ``v`` (3,) is the moment and ``w`` (3,) is the line direction.

        - ``Line3(v, w)`` as above but the components ``v`` and ``w`` are
          provided separately.

        - ``Line3(L)`` creates a copy of the ``Line3`` object ``L``.

        :notes:

            - The ``Line3`` object inherits from ``collections.UserList`` and has list-like
              behaviours.
            - A single ``Line3`` object contains a 1D-array of Plucker coordinates.
            - The elements of the array are guaranteed to be Plucker coordinates.
            - The number of elements is given by ``len(L)``
            - The elements can be accessed using index and slice notation, eg. ``L[1]`` or
              ``L[2:3]``
            - The ``Line3`` instance can be used as an iterator in a for loop or list comprehension.
            - Some methods support operations on the internal list.

        :seealso: :meth:`Join` :meth:`TwoPlanes` :meth:`PointDir`
        """
        from spatialmath.pose3d import SE3

        super().__init__()  # enable list powers

        if w is None:
            # zero or one arguments passed
            if super().arghandler(v, convertfrom=(SE3,)):
                if check and not base.iszero(np.dot(self.v, self.w)):
                    raise ValueError("invalid Plucker coordinates")
                return

        if base.isvector(v, 3) and base.isvector(w, 3):
            if check and not base.iszero(np.dot(v, w)):
                raise ValueError("invalid Plucker coordinates")
            self.data = [np.r_[v, w]]

        else:
            raise ValueError("invalid argument to Line3 constructor")

        # needed to allow __rmul__ to work if left multiplied by ndarray
        # self.__array_priority__ = 100

    @property
    def shape(self) -> Tuple[int]:
        return (6,)

    @staticmethod
    def _identity() -> R6:
        return np.zeros((6,))

    @staticmethod
    def isvalid(x: NDArray, check: bool = False) -> bool:
        return x.shape == (6,)

    @classmethod
    def Join(cls, P: ArrayLike3, Q: ArrayLike3) -> Self:
        """
        Create 3D line from two 3D points

        :param P: First 3D point
        :type P: array_like(3)
        :param Q: Second 3D point
        :type Q: array_like(3)
        :return: 3D line
        :rtype: ``Line3`` instance

        ``Line3.Join(P, Q)`` create a ``Line3`` object that represents
        the line joining the 3D points ``P`` (3,) and ``Q`` (3,). The direction
        is from ``Q`` to ``P``.

        :seealso: :meth:`IntersectingPlanes` :meth:`PointDir`
        """
        P = base.getvector(P, 3)
        Q = base.getvector(Q, 3)
        # compute direction and moment
        w = P - Q
        v = np.cross(w, P)
        return cls(np.r_[v, w])

    @classmethod
    def TwoPlanes(cls, pi1: Plane3, pi2: Plane3) -> Self:
        r"""
        Create 3D line from intersection of two planes

        :param pi1: First plane
        :type pi1: array_like(4), or ``Plane``
        :param pi2: Second plane
        :type pi2: array_like(4), or ``Plane``
        :return: 3D line
        :rtype: ``Line3`` instance

        ``L = Line3.TwoPlanes(π1, π2)`` is a ``Line3`` object that represents
        the line formed by the intersection of two planes ``π1`` and ``π3``.

        Planes are represented by the 4-vector :math:`[a, b, c, d]` which describes
        the plane :math:`\pi: ax + by + cz + d=0`.

        :seealso: :meth:`Join` :meth:`PointDir`
        """

        # TODO inefficient to create 2 temporary planes

        if not isinstance(pi1, Plane3):
            pi1 = Plane3(base.getvector(pi1, 4))
        if not isinstance(pi2, Plane3):
            pi2 = Plane3(base.getvector(pi2, 4))

        w = np.cross(pi1.n, pi2.n)
        v = pi2.d * pi1.n - pi1.d * pi2.n
        return cls(np.r_[v, w])

    @classmethod
    def IntersectingPlanes(cls, pi1: Plane3, pi2: Plane3) -> Self:
        warnings.warn("use TwoPlanes method instead", DeprecationWarning)
        return cls.TwoPlanes(pi1, pi2)

    @classmethod
    def PointDir(cls, point: ArrayLike3, dir: ArrayLike3) -> Self:
        """
        Create 3D line from a point and direction

        :param point: A 3D point
        :type point: array_like(3)
        :param dir: Direction vector
        :type dir: array_like(3)
        :return: 3D line
        :rtype: ``Line3`` instance

        ``Line3.PointDir(P, W)`` is a `Line3`` object that represents the
        line containing the point ``P`` and parallel to the direction vector ``W``.

        :seealso: :meth:`Join` :meth:`IntersectingPlanes`
        """

        p = base.getvector(point, 3)
        w = base.getvector(dir, 3)
        v = np.cross(w, p)
        return cls(np.r_[v, w])

    def append(self, x: Line3):
        """
        Append a line

        :param x: line object
        :type x: Line3
        :raises ValueError: Attempt to append a non Plucker object
        :return: Line3 object with new line appended
        :rtype: Line3 instance

        """
        # print('in append method')
        if not type(self) == type(x):
            raise ValueError("can only append Line3 object")
        if len(x) > 1:
            raise ValueError("cant append a Line3 sequence - use extend")
        super().append(x.A)

    @property
    def A(self) -> R6:
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        # print('getitem', i, 'class', self.__class__)
        return self.__class__(self.data[i])

    @property
    def v(self) -> R3:
        r"""
        Moment vector

        :return: the moment vector
        :rtype: ndarray(3)

        The line is represented by a vector :math:`(\vec{v}, \vec{w}) \in \mathbb{R}^6`.

        :seealso: :meth:`w`
        """
        return self.data[0][0:3]

    @property
    def w(self) -> R3:
        r"""
        Direction vector

        :return: the direction vector
        :rtype: ndarray(3)

        The line is represented by a vector :math:`(\vec{v}, \vec{w}) \in \mathbb{R}^6`.

        :seealso: :meth:`v` :meth:`uw`
        """
        return self.data[0][3:6]

    @property
    def uw(self) -> R3:
        r"""
        Line direction as a unit vector

        :return: Line direction as a unit vector
        :rtype: ndarray(3,)

        ``line.uw`` is a unit-vector parallel to the line.

        The line is represented by a vector :math:`(\vec{v}, \vec{w}) \in \mathbb{R}^6`.

        :seealso: :meth:`w`
        """
        return base.unitvec(self.w)

    @property
    def vec(self) -> R6:
        r"""
        Line as a Plucker coordinate vector

        :return: Plucker coordinate vector
        :rtype: ndarray(6,)

        ``line.vec`` is the  Plucker coordinate vector :math:`(\vec{v}, \vec{w}) \in \mathbb{R}^6`.

        """
        return np.r_[self.v, self.w]

    def skew(self) -> R4x4:
        r"""
        Line as a Plucker skew-symmetric matrix

        :return: Skew-symmetric matrix form of Plucker coordinates
        :rtype: ndarray(4,4)

        ``line.skew()`` is the Plucker matrix, a 4x4 skew-symmetric matrix
        representation of the line whose six unique elements are the Plucker coordinates
        of the line.

        .. math::

            \sk{L} = \begin{bmatrix} 0 & v_z & -v_y & \omega_x \\
                -v_z & 0 & v_x & \omega_y \\
                v_y & -v_x & 0 & \omega_z \\
                -\omega_x & -\omega_y & -\omega_z & 0 \end{bmatrix}

        .. note::

            - For two homogeneous points P and Q on the line, :math:`PQ^T-QP^T` is also
              skew symmetric.
            - The projection of Plucker line by a perspective camera is a homogeneous
              line (3x1) given by :math:`\vee C M C^T` where :math:`C \in \mathbf{R}^{3
              \times 4}` is the camera matrix.
        """

        v = self.v
        w = self.w

        # the following matrix is at odds with H&Z pg. 72
        return np.array(
            [
                [0, v[2], -v[1], w[0]],
                [-v[2], 0, v[0], w[1]],
                [v[1], -v[0], 0, w[2]],
                [-w[0], -w[1], -w[2], 0],
            ]  # type: ignore
        )

    @property
    def pp(self) -> R3:
        """
        Principal point of the 3D line

        :return: Principal point of the line
        :rtype: ndarray(3)

        ``line.pp`` is the point on the line that is closest to the origin.

        Notes:

         - Same as Plucker.point(0)

        :seealso: :meth:`ppd` :meth`point`
        """
        return np.cross(self.v, self.w) / np.dot(self.w, self.w)

    @property
    def ppd(self) -> float:
        """
        Distance from principal point to the origin

        :return: Distance from principal point to the origin
        :rtype: float

        ``line.ppd`` is the distance from the principal point to the origin.
        This is the smallest distance of any point on the line
        to the origin.

        :seealso: :meth:`pp`
        """
        return math.sqrt(np.dot(self.v, self.v) / np.dot(self.w, self.w))

    def point(self, lam: Union[float, ArrayLike]) -> Points3:
        r"""
        Generate point on line

        :param lam: Scalar distance from principal point
        :type lam: float
        :return: Distance from principal point to the origin
        :rtype: float

        ``line.point(λ)`` is a point on the line, where ``λ`` is the parametric
        distance along the line from the principal point of the line such
        that :math:`P = P_p + \lambda \hat{d}` and :math:`\hat{d}` is the line
        direction given by ``line.uw``.

        :seealso: :meth:`pp` :meth:`closest` :meth:`uw` :meth:`lam`
        """
        lam = base.getvector(lam, out="row")
        return cast(Points3, self.pp.reshape((3, 1)) + self.uw.reshape((3, 1)) * lam)

    def lam(self, point: ArrayLike3) -> float:
        r"""
        Parametric distance from principal point

        :param point: 3D point
        :type point: array_like(3)
        :return: parametric distance λ
        :rtype: float

        ``line.lam(P)`` is the value of :math:`\lambda` such that
        :math:`Q = P_p + \lambda \hat{d}` is closest to ``P``.

        :seealso: :meth:`point`
        """
        return np.dot(base.getvector(point, 3, out="row") - self.pp, self.uw)

    # ------------------------------------------------------------------------- #
    #  TESTS ON PLUCKER OBJECTS
    # ------------------------------------------------------------------------- #

    def contains(
        self, x: Union[R3, Points3], tol: float = 20
    ) -> Union[bool, List[bool]]:
        """
        Test if points are on the line

        :param x: 3D point
        :type x: 3-element array_like, or ndarray(3,N)
        :param tol: Tolerance in units of eps, defaults to 20
        :type tol: float, optional
        :raises ValueError: Bad argument
        :return: Whether point is on the line
        :rtype: bool or numpy.ndarray(N) of bool

        ``line.contains(X)`` is true if the point ``X`` lies on the line defined by
        the Line3 object self.

        If ``X`` is an array with 3 rows, the test is performed on every column and
        an array of booleans is returned.
        """
        if base.isvector(x, 3):
            x = cast(R3, base.getvector(x))
            return bool(np.linalg.norm(np.cross(x - self.pp, self.w)) < tol * _eps)
        elif base.ismatrix(x, (3, None)):
            return [
                bool(np.linalg.norm(np.cross(p - self.pp, self.w)) < tol * _eps)
                for p in x.T
            ]
        else:
            raise ValueError("bad argument")

    def isequal(
        l1, l2: Line3, tol: float = 20  # type: ignore
    ) -> bool:  # pylint: disable=no-self-argument
        """
        Test if two lines are equivalent

        :param l2: Second line
        :type l2: ``Line3``
        :param tol: Tolerance in multiples of eps, defaults to 20
        :type tol: float, optional
        :return: lines are equivalent
        :rtype: bool

        ``L1 == L2`` is True if the ``Line3`` objects describe the same line in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if their coordinate vectors are different.

        :seealso: :meth:`__eq__`
        """
        return bool(
            abs(1 - np.dot(base.unitvec(l1.vec), base.unitvec(l2.vec))) < tol * _eps
        )

    def isparallel(
        l1, l2: Line3, tol: float = 20  # type: ignore
    ) -> bool:  # pylint: disable=no-self-argument
        """
        Test if lines are parallel

        :param l2: Second line
        :type l2: ``Line3``
        :param tol: Tolerance in multiples of eps, defaults to 20
        :type tol: float, optional
        :return: lines are parallel
        :rtype: bool

        ``l1.isparallel(l2)`` is true if the two lines are parallel.

        ``l1 | l2`` as above but in binary operator form

        :seealso: :meth:`__or__` :meth:`intersects`
        """
        return bool(np.linalg.norm(np.cross(l1.w, l2.w)) < tol * _eps)

    def isintersecting(
        l1, l2: Line3, tol: float = 20  # type: ignore
    ) -> bool:  # pylint: disable=no-self-argument
        """
        Test if lines are intersecting

        :param l2: Second line
        :type l2: Line3
        :param tol: Tolerance in multiples of eps, defaults to 20
        :type tol: float, optional
        :return: lines intersect
        :rtype: bool

        ``l1.isintersecting(l2)`` is true if the two lines intersect.

        .. note:: Is ``False`` if the lines are equivalent since they would intersect at
              an infinite number of points.

        :seealso: :meth:`__xor__` :meth:`intersects` :meth:`isparallel`
        """
        return not l1.isparallel(l2, tol=tol) and bool(abs(l1 * l2) < tol * _eps)

    def __eq__(l1, l2: Line3) -> bool:  # type: ignore pylint: disable=no-self-argument
        """
        Test if two lines are equivalent

        :param l2: Second line
        :type l2: ``Line3``
        :return: lines are equivalent
        :rtype: bool

        ``L1 == L2`` is True if the ``Line3`` objects describe the same line in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if their coordinate vectors are different.

        .. note:: There is a hardwired tolerance of 10eps.

        :seealso: :meth:`isequal` :meth:`__ne__`
        """
        return l1.isequal(l2)

    def __ne__(l1, l2: Line3) -> bool:  # type:ignore pylint: disable=no-self-argument
        """
        Test if two lines are not equivalent

        :param l2: Second line
        :type l2: ``Line3``
        :return: lines are not equivalent
        :rtype: bool

        ``L1 != L2`` is True if the Line3 objects describe different lines in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if their coordinate vectors are different.

        .. note:: There is a hardwired tolerance of 10eps.

        :seealso: :meth:`__ne__`
        """
        return not l1.isequal(l2)

    def __or__(l1, l2: Line3) -> bool:  # type:ignore pylint: disable=no-self-argument
        """
        Overloaded ``|`` operator tests for parallelism

        :param l2: Second line
        :type l2: ``Line3``
        :return: lines are parallel
        :rtype: bool

        ``l1 | l2`` is an operator which is true if the two lines are parallel.

        .. note:: The ``|`` operator has low precendence.

        .. note:: There is a hardwired tolerance of 10eps.

        :seealso: :meth:`isparallel` :meth:`__xor__`
        """
        return l1.isparallel(l2)

    def __xor__(l1, l2: Line3) -> bool:  # type:ignore pylint: disable=no-self-argument
        """
        Overloaded ``^`` operator tests for intersection

        :param l2: Second line
        :type l2: Line3
        :return: lines intersect
        :rtype: bool

        ``l1 ^ l2`` is an operator which is true if the two lines intersect.

        .. note::

            - The ``^`` operator has low precendence.
            - Is ``False`` if the lines are equivalent since they would intersect at
              an infinite number of points.

        .. note:: There is a hardwired tolerance of 10eps.

        :seealso: :meth:`intersects` :meth:`isparallel` :meth:`isintersecting`
        """
        return l1.isintersecting(l2)

    # ------------------------------------------------------------------------- #
    #  PLUCKER LINE DISTANCE AND INTERSECTION
    # ------------------------------------------------------------------------- #

    def intersects(
        l1, l2: Line3  # type:ignore
    ) -> Union[R3, None]:  # pylint: disable=no-self-argument
        """
        Intersection point of two lines

        :param l2: Second line
        :type l2: ``Line3``
        :return: 3D intersection point
        :rtype: ndarray(3) or None

        ``l1.intersects(l2)`` is the point of intersection of the two lines, or
        ``None`` if the lines do not intersect or are equivalent.

        :seealso: :meth:`commonperp :meth:`eq` :meth:`__xor__`
        """
        if l1 ^ l2:
            # lines do intersect
            return -(
                np.dot(l1.v, l2.w) * np.eye(3, 3)
                + l1.w.reshape((3, 1)) @ l2.v.reshape((1, 3))
                - l2.w.reshape((3, 1)) @ l1.v.reshape((1, 3))
            ) * base.unitvec(np.cross(l1.w, l2.w))
        else:
            # lines don't intersect
            return None

    def distance(
        l1, l2: Line3, tol: float = 20  # type:ignore
    ) -> float:  # pylint: disable=no-self-argument
        """
        Minimum distance between lines

        :param l2: Second line
        :type l2: ``Line3``
        :param tol: Tolerance in multiples of eps, defaults to 20
        :type tol: float, optional
        :return: Closest distance between lines
        :rtype: float

        ``l1.distance(l2)`` is the minimum distance between two lines.

        .. note:: Works for parallel, skew and intersecting lines.

        :seealso: :meth:`closest_to_line`
        """
        if l1 | l2:
            # lines are parallel
            l = np.cross(
                l1.w, l1.v - l2.v * np.dot(l1.w, l2.w) / dot(l2.w, l2.w)
            ) / np.linalg.norm(l1.w)
        else:
            # lines are not parallel
            if abs(l1 * l2) < tol * _eps:
                # lines intersect at a point
                l = 0
            else:
                # lines don't intersect, find closest distance
                l = abs(l1 * l2) / np.linalg.norm(np.cross(l1.w, l2.w)) ** 2
        return l

    def closest_to_line(
        l1, l2: Line3  # type:ignore
    ) -> Tuple[Points3, Rn]:  # pylint: disable=no-self-argument
        """
        Closest point between lines

        :param l2: second line
        :type l2: Line3
        :return: nearest points and distance between lines at those points
        :rtype: ndarray(3,N), ndarray(N)

        There are four cases:

        * ``len(self) == len(other) == 1`` find the point on the first line closest to the second line, as well
          as the minimum distance between the lines.
        * ``len(self) == 1, len(other) == N`` find the point of intersection between the first
          line and the ``N`` other lines, returning ``N`` intersection points and distances.
        * ``len(self) == N, len(other) == 1`` find the point of intersection between the ``N`` first
          lines and the other line, returning ``N`` intersection points and distances.
        * ``len(self) == N, len(other) == M`` for each  of the ``N`` first
          lines find the closest intersection with each of the ``M`` other lines, returning ``N``
          intersection points and distances.

        ** this last one should be an option, default behavior would be to
        test self[i] against line[i]
        ** maybe different function

        For two sets of lines, of equal size, return an array of closest points
        and distances.

        Example::

            .. runblock:: pycon

                >>> from spatialmath import Line3
                >>> line1 = Line3.Join([1, 1, 0], [1, 1, 1])
                >>> line2 = Line3.Join([0, 0, 0], [2, 3, 5])
                >>> line1.closest_to_line(line2)

        :reference: `Plucker coordinates <https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf>`_


        :seealso: :meth:`distance`
        """
        # point on line closest to another line
        # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
        # but (20) (21) is the negative of correct answer

        points = []
        dists = []

        def intersection(line1, line2):
            with np.errstate(divide="ignore", invalid="ignore"):
                # compute the distance between all pairs of lines
                v1 = line1.v
                w1 = line1.w
                v2 = line2.v
                w2 = line2.w

                p1 = (
                    np.cross(v1, np.cross(w2, np.cross(w1, w2)))
                    - np.dot(v2, np.cross(w1, w2)) * w1
                ) / np.sum(np.cross(w1, w2) ** 2)
                p2 = (
                    np.cross(-v2, np.cross(w1, np.cross(w1, w2)))
                    + np.dot(v1, np.cross(w1, w2)) * w2
                ) / np.sum(np.cross(w1, w2) ** 2)

            return p1, np.linalg.norm(p1 - p2)

        if len(l1) == len(l2):
            # two sets of lines of equal length
            for line1, line2 in zip(l1, l2):
                point, dist = intersection(line1, line2)
                points.append(point)
                dists.append(dist)

        elif len(l1) == 1 and len(l2) > 1:
            for line in l2:
                point, dist = intersection(l1, line)
                points.append(point)
                dists.append(dist)

        elif len(l1) > 1 and len(l2) == 1:
            for line in l1:
                point, dist = intersection(line, l2)
                points.append(point)
                dists.append(dist)

        if len(points) == 1:
            # 1D case for self or line
            return points[0], dists[0]
        else:
            return np.array(points).T, np.array(dists)

    def closest_to_point(self, x: ArrayLike3) -> Tuple[R3, float]:
        """
        Point on line closest to given point

        :param x: An arbitrary 3D point
        :type x: array_like(3)
        :return: Point on the line and distance to line
        :rtype: ndarray(3), float

        Find the point on the line closest to ``x`` as well as the distance
        at that closest point.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Line3
            >>> line1 = Line3.Join([0, 0, 0], [2, 2, 3])
            >>> line1.closest_to_point([1, 1, 1])

        :seealso: meth:`point`
        """
        # http://www.ahinson.com/algorithms_general/Sections/Geometry/PluckerLine.pdf
        # has different equation for moment, the negative

        x = base.getvector(x, 3)

        lam = np.dot(x - self.pp, self.uw)
        p = self.point(lam).flatten()  # is the closest point on the line
        d = np.linalg.norm(x - p)

        return p, d

    def commonperp(
        l1, l2: Line3
    ) -> Line3:  # type:ignore pylint: disable=no-self-argument
        """
        Common perpendicular to two lines

        :param l2: Second line
        :type l2: Line3
        :return: Perpendicular line
        :rtype: Line3 instance or None

        ``l1.commonperp(l2)`` is the common perpendicular line between the two lines.
        Returns ``None`` if the lines are parallel.

        :seealso: :meth:`intersect`
        """
        if l1 | l2:
            # no common perpendicular if lines are parallel
            return None
        else:
            # lines are skew or intersecting
            w = np.cross(l1.w, l2.w)
            v = (
                np.cross(l1.v, l2.w)
                - np.cross(l2.v, l1.w)
                + (l1 * l2) * np.dot(l1.w, l2.w) * base.unitvec(np.cross(l1.w, l2.w))
            )

        return l1.__class__(v, w)

    def __mul__(
        left, right: Line3
    ) -> float:  # type:ignore pylint: disable=no-self-argument
        r"""
        Reciprocal product

        :param left: Left operand
        :type left: Line3
        :param right: Right operand
        :type right: Line3
        :return: reciprocal product
        :rtype: float

        ``left * right`` is the scalar reciprocal product :math:`\hat{w}_L \dot m_R + \hat{w}_R \dot m_R`.

        .. note::

            - Multiplication or composition of lines is not defined.
            - Pre-multiplication by an SE3 object is supported, see ``__rmul__``.

        :seealso: :meth:`__rmul__`
        """
        if isinstance(right, Line3):
            # reciprocal product
            return np.dot(left.uw, right.v) + np.dot(right.uw, left.v)
        else:
            raise ValueError("bad arguments")

    def __rmul__(
        right, left: SE3
    ) -> Line3:  # type:ignore pylint: disable=no-self-argument
        """
        Rigid-body transformation of 3D line

        :param left: Rigid-body transform
        :type left: SE3
        :param right: 3D line
        :type right: Line
        :return: transformed 3D line
        :rtype: Line3 instance

        ``T * line`` is the line transformed by the rigid body transformation ``T``.

        :seealso: :meth:`__mul__`
        """
        from spatialmath.pose3d import SE3

        if isinstance(left, SE3):
            A = left.inv().Ad()
            return right.__class__(A @ right.vec)  # premultiply by SE3.Ad
        else:
            raise ValueError("can only premultiply Line3 by SE3")

    # ------------------------------------------------------------------------- #
    #  PLUCKER LINE DISTANCE AND INTERSECTION
    # ------------------------------------------------------------------------- #

    def intersect_plane(
        self, plane: Union[ArrayLike4, Plane3], tol: float = 20
    ) -> Tuple[R3, float]:
        r"""
        Line intersection with a plane

        :param plane: A plane
        :type plane: array_like(4) or Plane3
        :param tol: Tolerance in multiples of eps, defaults to 20
        :type tol: float, optional
        :return: Intersection point, λ
        :rtype: ndarray(3), float

        - ``P, λ = line.intersect_plane(plane)`` is the point where the line
          intersects the plane, and the corresponding λ value.
          Return None, None if no intersection.

        The plane can be specified as:

         - a 4-vector :math:`[a, b, c, d]` which describes the plane :math:`\pi: ax + by + cz + d=0`.
         - a ``Plane`` object

         The return value is a named tuple with elements:

            - ``.p`` for the point on the line as a numpy.ndarray, shape=(3,)
            - ``.lam`` the `lambda` value for the point on the line.

        :sealso: :meth:`point` :class:`Plane`
        """

        # Line U, V
        # Plane N n
        # (VxN-nU:U.N)
        # Note that this is in homogeneous coordinates.
        #    intersection of plane (n,p) with the line (v,p)
        #    returns point and line parameter
        if not isinstance(plane, Plane3):
            plane = Plane3(base.getvector(plane, 4))

        den = np.dot(self.w, plane.n)

        if abs(den) > (tol * _eps):
            # P = -(np.cross(line.v, plane.n) + plane.d * line.w) / den
            p = (np.cross(self.v, plane.n) - plane.d * self.w) / den

            t = self.lam(p)
            return namedtuple("intersect_plane", "p lam")(p, t)
        else:
            return None

    def intersect_volume(self, bounds: ArrayLike6) -> Tuple[Points3, Rn]:
        """
        Line intersection with a volume

        :param bounds: Bounds of an axis-aligned rectangular cuboid
        :type plane: array_like(6)
        :return: Intersection point, λ value
        :rtype: ndarray(3,N), ndarray(N)

        ``P, λ = line.intersect_volume(bounds)`` is a matrix (3xN) with columns
        that indicate where the line intersects the faces of the volume and
        the corresponding λ values.

        The volume is specified by ``bounds`` = [xmin xmax ymin ymax zmin zmax].

        The number of
        columns N is either:

        - 0, when the line is outside the plot volume or,
        - 2 when the line pierces the bounding volume.


        See also :meth:`plot` :meth:`point`
        """

        intersections = []

        # reshape, top row is minimum, bottom row is maximum
        bounds23 = bounds.reshape((3, 2))

        for face in range(0, 6):
            # for each face of the bounding volume
            #  x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax

            # planes are:
            #  0 normal in x direction, xmin
            #  1 normal in x direction, xmax
            #  2 normal in y direction, ymin
            #  3 normal in y direction, ymax
            #  4 normal in z direction, zmin
            #  5 normal in z direction, zmax

            i = face // 2  # 0, 1, 2
            I = np.eye(3, 3)
            p = [0, 0, 0]
            p[i] = bounds[face]
            plane = Plane3.PointNormal(n=I[:, i], p=p)

            # find where line pierces the plane
            try:
                p, lam = self.intersect_plane(plane)
            except TypeError:
                continue  # no intersection with this plane

            # print('face %d: n=(%f, %f, %f)' % (face, plane.n[0], plane.n[1], plane.n[2]))
            # print('       : p=(%f, %f, %f)  ' % (p[0], p[1], p[2]))

            # print('face', face, ' point ', p, ' plane ', plane)
            # print('lamda', lam, self.point(lam))
            # find if intersection point is within the cube face
            #  test x,y,z simultaneously
            k = (p >= bounds23[:, 0]) & (p <= bounds23[:, 1])
            k = np.delete(k, i)  # remove the boolean corresponding to current face
            if all(k):
                # if within bounds, add
                intersections.append(lam)

        #                     print('  HIT');

        # put them in ascending order
        intersections.sort()
        p = self.point(intersections)

        return namedtuple("intersect_volume", "p lam")(p, intersections)

    # ------------------------------------------------------------------------- #
    #  PLOT AND DISPLAY
    # ------------------------------------------------------------------------- #

    def plot(
        self,
        *pos,
        bounds: Optional[ArrayLike] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> List[plt.Artist]:
        """
         Plot a line

        :param bounds: Bounds of an axis-aligned rectangular cuboid as [xmin xmax ymin ymax zmin zmax], optional
        :type plane: 6-element array_like
        :param kwargs: Extra arguments passed to `Line2D <https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        :type kwargs: dict
        :return: Plotted line
        :rtype: Matplotlib artists

        - ``line.plot(bounds)`` adds a line segment to the current axes, and the handle of the line is returned.
          The line segment is defined by the intersection of the line and the given rectangular cuboid.
          If the line does not intersect the plotting volume None is returned.

        - ``line.plot()`` as above but the bounds are taken from the axis limits of the current axes.

        The line color or style is specified by:

            - a  MATLAB-style linestyle like ``"k--"``
            - additional arguments passed to `Line2D <https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_

        :seealso: :meth:`intersect_volume`
        """
        if ax is None:
            ax = plt.gca()

        print(ax)
        if bounds is None:
            bounds = np.r_[ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        else:
            bounds = base.getvector(bounds, 6)
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:4])
            ax.set_zlim(bounds[4:6])

        lines = []
        for line in self:
            P, lam = line.intersect_volume(bounds)

            if len(lam) > 0:
                l = ax.plot(
                    tuple(P[0, :]), tuple(P[1, :]), tuple(P[2, :]), *pos, **kwargs
                )
                lines.append(l)
        return lines

    def __str__(self) -> str:
        """
        Convert Line3 to a string

        :return: String representation of line parameters
        :rtype: str

        ``str(line)`` is a string showing Plucker parameters in a compact single
        line format like::

            { 0 0 0; -1 -2 -3}

        where the first three numbers are the moment, and the last three are the
        direction vector.

        For a multi-valued ``Line3``, one line per value in ``Line3``.

        """

        return "\n".join(
            [
                "{{ {:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g}}}".format(
                    *list(base.removesmall(x.vec))
                )
                for x in self
            ]
        )

    def __repr__(self) -> str:
        """
        Display Line3

        :return: String representation of line parameters
        :rtype: str

        Displays the line parameters in compact single line format.

        For a multi-valued ``Line3``, one line per value in ``Line3``.
        """

        if len(self) == 1:
            return "Line3([{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}])".format(
                *list(self.A)
            )
        else:
            return (
                "Line3([\n"
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

        Example::

            In [1]: x

        """
        if len(self) == 1:
            p.text(str(self))
        else:
            for i, x in enumerate(self):
                if i > 0:
                    p.break_()
                p.text(f"{i:3d}: {str(x)}")

    #         function z = side(self1, pl2)
    #             Plucker.side Plucker side operator
    #
    #             # X = SIDE(P1, P2) is the side operator which is zero whenever
    #             # the lines P1 and P2 intersect or are parallel.
    #
    #             # See also Plucker.or.
    #
    #             if ~isa(self2, 'Plucker')
    #                 error('SMTB:Plucker:badarg', 'both arguments to | must be Plucker objects');
    #             end
    #             L1 = pl1.line(); L2 = pl2.line();
    #
    #             z = L1([1 5 2 6 3 4]) * L2([5 1 6 2 4 3])';
    #         end

    def side(self, other: Line3) -> float:
        """
        Plucker side operator

        :param other: second line
        :type other: Line3
        :return: permuted dot product
        :rtype: float

        This permuted dot product operator is zero whenever the lines intersect or are parallel.
        """
        if not isinstance(other, Line3):
            raise ValueError("argument must be a Line3")

        return np.dot(self.A[[0, 4, 1, 5, 2, 3]], other.A[4, 0, 5, 1, 3, 2])

    # Static factory methods for constructors from exotic representations


class Plucker(Line3):
    def __init__(self, v=None, w=None):
        import warnings

        warnings.warn("use Line class instead", DeprecationWarning)
        super().__init__(v, w)


if __name__ == "__main__":  # pragma: no cover
    import pathlib
    import os.path

    # L = Line3.TwoPoints((1,2,0), (1,2,1))
    # print(L)
    # print(L.intersect_plane([0, 0, 1, 0]))

    # z = np.eye(6) * L

    # L2 = SE3(2, 1, 10) * L
    # print(L2)
    # print(L2.intersect_plane([0, 0, 1, 0]))

    # print('rx')
    # L2 = SE3.Rx(np.pi/4) * L
    # print(L2)
    # print(L2.intersect_plane([0, 0, 1, 0]))

    # print('ry')
    # L2 = SE3.Ry(np.pi/4) * L
    # print(L2)
    # print(L2.intersect_plane([0, 0, 1, 0]))

    # print('rz')
    # L2 = SE3.Rz(np.pi/4) * L
    # print(L2)
    # print(L2.intersect_plane([0, 0, 1, 0]))

    # base.plotvol3(10)
    # S = Twist3.UnitRevolute([0, 0, 1], [2, 3, 2], 0.5);
    # L = S.line()
    # L.plot('k:', linewidth=2)

    # a = Plane3([0.1, -1, -1, 2])
    # base.plotvol3(5)
    # a.plot(color='r', alpha=0.3)
    # plt.show(block=True)

    # a = SE3.Exp([2,0,0,0,0,0])

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_geom3d.py"
        ).read()
    )  # pylint: disable=exec-used

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 09:42:30 2020

@author: corkep
"""
from __future__ import annotations

from functools import reduce
import warnings
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
import numpy as np

from spatialmath import SE2
import spatialmath.base as smb
from spatialmath.base import plot_ellipse
from spatialmath.base.types import (
    Points2,
    Optional,
    ArrayLike,
    ArrayLike2,
    ArrayLike3,
    NDArray,
    Union,
    List,
    Tuple,
    R2,
    R3,
    R4,
    Iterator,
    Tuple,
    Self,
    cast,
)

_eps = np.finfo(np.float64).eps


class Line2:
    """
    Class to represent 2D lines

    The internal representation is in homogeneous format

    .. math::

            ax + by + c = 0
    """

    def __init__(self, line: ArrayLike3):
        self.line = smb.getvector(line, 3)

    @classmethod
    def Join(cls, p1: ArrayLike2, p2: ArrayLike2) -> Self:
        """
        Create 2D line from two points

        :param p1: point on the line
        :type p1: array_like(2) or array_like(3)
        :param p2: another point on the line
        :type p2: array_like(2) or array_like(3)

        The points can be given in Euclidean or homogeneous form.
        """

        p1 = smb.getvector(p1)
        if len(p1) == 2:
            p1 = np.r_[p1, 1]
        p2 = smb.getvector(p2)
        if len(p2) == 2:
            p2 = np.r_[p2, 1]

        return cls(np.cross(p1, p2))

    @classmethod
    def TwoPoints(cls, p1: ArrayLike2, p2: ArrayLike2) -> Self:
        warnings.warn("use Join method instead", DeprecationWarning)
        return cls.Join(p1, p2)

    @classmethod
    def General(cls, m, c) -> Self:
        """
        Create line from general line

        :param m: line gradient
        :type m: float
        :param c: line intercept
        :type c: float
        :return: a 2D line
        :rtype: a Line2 instance

        Creates a line from the parameters of the general line :math:`y = mx + c`.

        .. note:: A vertical line cannot be represented.
        """
        return cls([m, -1, c])

    def general(self) -> Tuple[float, float]:
        r"""
        Parameters of general line

        :return: parameters of general line (m, c)
        :rtype: ndarray(2)

        Return the parameters of a general line :math:`y = mx + c`.
        """
        return -self.line[[0, 2]] / self.line[1]

    def __str__(self) -> str:
        return f"Line2: {self.line}"

    def plot(self, **kwargs) -> None:
        """
        Plot the line using matplotlib

        :param kwargs: arguments passed to Matplotlib ``pyplot.plot``
        """
        smb.plot_homline(self.line, **kwargs)

    def intersect(self, other: Line2, tol: float = 20) -> R3:
        """
        Intersection with line

        :param other: another 2D line
        :type other: Line2
        :param tol: tolerance in units of eps, defaults to 20
        :type tol: float
        :return: intersection point in homogeneous form
        :rtype: ndarray(3)

        If the lines are parallel then the third element of the returned
        homogeneous point will be zero (an ideal point).
        """
        # return intersection of 2 lines
        # return mindist and points if no intersect
        c = np.cross(self.line, other.line)
        return abs(c[2]) > tol * _eps

    def contains(self, p: ArrayLike2, tol: float = 20) -> bool:
        """
        Test if point is in line

        :param p1: point to test
        :type p1: array_like(2) or array_like(3)
        :param tol: tolerance in units of eps, defaults to 20
        :type tol: float
        :return: True if point lies in the line
        :rtype: bool
        """
        p = smb.getvector(p)
        if len(p) == 2:
            p = np.r_[p, 1]
        return abs(np.dot(self.line, p)) < tol * _eps

    # variant that gives lambda

    def intersect_segment(
        self, p1: ArrayLike2, p2: ArrayLike2, tol: float = 20
    ) -> bool:
        """
        Test for line intersecting line segment

        :param p1: start of line segment
        :type p1: array_like(2) or array_like(3)
        :param p2: end of line segment
        :type p2: array_like(2) or array_like(3)
        :param tol: tolerance in units of eps, defaults to 20
        :type tol: float
        :return: True if they intersect
        :rtype: bool

        Tests whether the line intersects the line segment defined by endpoints
        ``p1`` and ``p2`` which are given in Euclidean or homogeneous form.
        """
        p1 = smb.getvector(p1)
        if len(p1) == 2:
            p1 = np.r_[p1, 1]
        p2 = smb.getvector(p2)
        if len(p2) == 2:
            p2 = np.r_[p2, 1]

        z1 = np.dot(self.line, p1)
        z2 = np.dot(self.line, p2)

        if np.sign(z1) != np.sign(z2):
            return True
        if self.contains(p1, tol=tol) or self.contains(p2, tol=tol):
            return True
        return False

    # these should have same names as for 3d case
    def distance_line_line(self):
        pass

    def distance_line_point(self):
        pass

    def points_join(self):
        pass

    def intersect_polygon___line(self):
        pass

    def contains_polygon_point(self):
        pass


class LineSegment2(Line2):
    # line segment class that subclass
    # has hom line + 2 values of lambda
    pass


class Polygon2:
    """
    Class to represent 2D (planar) polygons

    .. note:: Uses Matplotlib primitives to perform transformations and
        intersections.
    """

    def __init__(self, vertices: Optional[Points2] = None, close: bool = True):
        """
        Create planar polygon from vertices

        :param vertices: vertices of polygon, defaults to None
        :type vertices: ndarray(2, N), optional
        :param close: closes the polygon, replicates the first vertex, defaults to True
        :type closed: bool, optional

        Create a polygon from a set of points provided as columns of the 2D
        array ``vertices``.
        A closed polygon is created so the last vertex should not equal the
        first.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])

        .. warning:: The points must be sequential around the perimeter and
            counter clockwise, otherwise moments will be negative.

        .. note:: The polygon is represented by a Matplotlib ``Path``
        """

        if isinstance(vertices, (list, tuple)):
            vertices = np.array(vertices).T
        elif isinstance(vertices, np.ndarray):
            if vertices.shape[0] != 2:
                raise ValueError("ndarray must be 2xN")
        elif vertices is None:
            return
        else:
            raise TypeError("expecting list of 2-tuples or ndarray(2,N)")

        # replicate the first vertex to make it closed.
        # setting closed=False and codes=None leads to a different
        # path which gives incorrect intersection results
        if close:
            vertices = np.hstack((vertices, vertices[:, 0:1]))

        self.path = Path(vertices.T, closed=True)
        self.path0 = self.path

    def __str__(self) -> str:
        """
        Polygon to string

        :return: brief summary of polygon
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> import numpy as np
            >>> p = Polygon2(np.array([[1, 3, 2], [2, 2, 4]]))
            >>> print(p)
        """
        return f"Polygon2 with {len(self.path)} vertices"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        """
        Number of vertices in polygon

        :return: number of vertices
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> len(p)

        """
        return len(self.path) - 1

    def moment(self, p: int, q: int) -> float:
        r"""
        Moments of polygon

        :param p: moment order x
        :type p: int
        :param q: moment order y
        :type q: int

        Returns the pq'th moment of the polygon

        .. math::

            M(p, q) = \sum_{i=0}^{n-1} x_i^p y_i^q

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.moment(0, 0)  # area
            >>> p.moment(3, 0)

        Note is negative for clockwise perimeter.
        """

        def combin(n, r):
            # compute number of combinations of size r from set n
            def prod(values):
                try:
                    return reduce(lambda x, y: x * y, values)
                except TypeError:
                    return 1

            return prod(range(n - r + 1, n + 1)) / prod(range(1, r + 1))

        vertices = self.vertices(unique=True)  # type: ignore
        x = vertices[0, :]
        y = vertices[1, :]

        m = 0.0
        n = len(x)
        for l in range(n):
            l1 = (l - 1) % n
            dxl = x[l] - x[l1]
            dyl = y[l] - y[l1]
            Al = x[l] * dyl - y[l] * dxl

            s = 0.0
            for i in range(p + 1):
                for j in range(q + 1):
                    s += (
                        (-1) ** (i + j)
                        * combin(p, i)
                        * combin(q, j)
                        / (i + j + 1)
                        * x[l] ** (p - i)
                        * y[l] ** (q - j)
                        * dxl**i
                        * dyl**j
                    )
            m += Al * s

        return m / (p + q + 2)

    def area(self) -> float:
        """
        Area of polygon

        :return: area
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.area()

        :seealso: :meth:`moment`
        """
        return abs(self.moment(0, 0))

    def centroid(self) -> R2:
        """
        Centroid of polygon

        :return: centroid
        :rtype: ndarray(2)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.centroid()

        :seealso: :meth:`moment`
        """
        return np.r_[self.moment(1, 0), self.moment(0, 1)] / self.moment(0, 0)

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> None:
        """
        Plot polygon

        :param ax: axes in which to draw the polygon, defaults to None
        :type ax: Axes, optional
        :param kwargs: options passed to Matplotlib ``Patch``

        A Matplotlib Patch is created with the passed options ``**kwargs`` and
        added to the axes.

        Examples::

            >>> from spatialmath.base import plotvol2, plot_polygon
            >>> plotvol2(5)
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.plot(fill=False)
            >>> p.plot(facecolor="g", edgecolor="none")  # green filled triangle

        .. plot::

            from spatialmath import Polygon2
            from spatialmath.base import plotvol2
            p = Polygon2([(1, 2), (3, 2), (2, 4)])
            plotvol2(5)
            p.plot(fill=False)

        .. plot::

            from spatialmath import Polygon2
            from spatialmath.base import plotvol2
            p = Polygon2([(1, 2), (3, 2), (2, 4)])
            plotvol2(5)
            p.plot(facecolor="g", edgecolor="none")  # green filled triangle


        :seealso: :meth:`animate` :func:`matplotlib.PathPatch`
        """
        self.patch = PathPatch(self.path, **kwargs)
        ax = smb.axes_logic(ax, 2)
        ax.add_patch(self.patch)
        plt.draw()
        self.kwargs = kwargs
        self.ax = ax

    def animate(self, T, **kwargs) -> None:
        """
        Animate a polygon

        :param T: new pose of Polygon
        :type T: SE2
        :param kwargs: options passed to Matplotlib ``Patch``

        The plotted polygon is moved to the pose given by ``T``. The pose is
        always with respect to the initial vertices when the polygon was
        constructed.  The vertices of the polygon will be updated to reflect
        what is plotted.

        If the polygon has already plotted, it will keep the same graphical
        attributes.  If new attributes are given they will replace those
        given at construction time.

        :seealso: :meth:`plot`
        """
        # get the path

        if self.patch is not None:
            self.patch.remove()
        self.path = self.path0.transformed(Affine2D(T.A))
        if len(kwargs) > 0:
            self.args = kwargs
        self.patch = PathPatch(self.path, **self.kwargs)
        self.ax.add_patch(self.patch)

    def contains(self, p: ArrayLike2, radius: float = 0.0) -> Union[bool, List[bool]]:
        """
        Test if point is inside polygon

        :param p: point
        :type p: array_like(2)
        :param radius: Add an additional margin to the polygon boundary, defaults to 0.0
        :type radius: float, optional
        :return: True if point is contained by polygon
        :rtype: bool

        ``radius`` can be used to inflate the polygon, or if negative, to
        deflated it.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.contains([0, 0])
            >>> p.contains([2, 3])

        .. warning:: Returns True if the point is on the edge of the polygon
            but False if the point is one of the vertices.

        .. warning:: For a polygon with clockwise ordering of vertices the
            sign of ``radius`` is flipped.

        :seealso: :func:`matplotlib.contains_point`
        """
        # note the sign of radius is negated if the polygon is drawn clockwise
        # https://stackoverflow.com/questions/45957229/matplotlib-path-contains-points-radius-parameter-defined-inconsistently
        # edges are included but the corners are not

        if isinstance(p, (list, tuple)) or (isinstance(p, np.ndarray) and p.ndim == 1):
            return self.path.contains_point(tuple(p), radius=radius)
        else:
            return self.path.contains_points(p.T, radius=radius)

    def bbox(self) -> R4:
        """
        Bounding box of polygon

        :return: bounding box as [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.bbox()
        """
        return np.array(self.path.get_extents()).ravel(order="C")

    def radius(self) -> float:
        """
        Radius of smallest enclosing circle

        :return: radius
        :rtype: float

        This is the radius of the smalleset circle, centred at the centroid,
        that encloses all vertices.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.radius()

        """
        c = self.centroid()
        dmax = -np.inf
        for vertex in self.path.vertices:
            d = smb.norm(vertex - c)
            dmax = max(dmax, d)
        return dmax

    def intersects(
        self, other: Union[Polygon2, Line2, List[Polygon2], List[Line2]]
    ) -> bool:
        """
        Test for intersection

        :param other: object to test for intersection
        :type other: Polygon2 or Line2 or list(Polygon2) or list(Line2)
        :return: True if the polygon intersects ``other``
        :rtype: bool
        :raises ValueError:

        Returns true if the polygon intersects the the given polygon or 2D
        line. If ``other`` is a list, test against all in the list and return on the
        first intersection.
        """
        if isinstance(other, Polygon2):
            # polygon-polygon intersection is done by matplotlib
            return self.path.intersects_path(other.path, filled=True)
        elif isinstance(other, Line2):
            # polygon-line intersection
            for p1, p2 in self.edges():  # type: ignore
                # test each edge segment against the line
                if other.intersect_segment(p1, p2):
                    return True
            return False
        elif smb.islistof(other, Polygon2):
            for polygon in cast(List[Polygon2], other):
                if self.path.intersects_path(polygon.path, filled=True):
                    return True
            return False
        elif smb.islistof(other, Line2):
            for line in cast(List[Line2], other):
                for p1, p2 in self.edges():
                    # test each edge segment against the line
                    if line.intersect_segment(p1, p2):
                        return True
            return False
        else:
            raise ValueError("bad type for other")

    def transformed(self, T: SE2) -> Self:
        """
        A transformed copy of polygon

        :param T: planar transformation
        :type T: SE2
        :return: transformed polygon
        :rtype: Polygon2

        Returns a new polgyon whose vertices have been transformed by ``T``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2, SE2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4)])
            >>> p.vertices()
            >>> p.transformed(SE2(10, 0, 0)).vertices() # shift by x+10

        """
        new = Polygon2()
        new.path = self.path.transformed(Affine2D(T.A))
        return new

    def vertices(self, unique: bool = True) -> Points2:
        """
        Vertices of polygon

        :param unique: return only the unique vertices , defaults to True
        :type unique: bool, optional
        :return: vertices
        :rtype: ndarray(2,n)

        Returns the set of vertices.  The polygon is always closed, that is, the first
        and last vertices are the same.  The ``unique`` option does not include the last
        vertex.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Polygon2
            >>> p = Polygon2([(1, 2), (3, 2), (2, 4), (1,2)])
            >>> p.vertices()
            >>> p.vertices(unique=True)
        """
        vertices = self.path.vertices.T
        if unique:
            vertices = vertices[:, :-1]

        return vertices

    def edges(self) -> Iterator:
        """
        Iterate over polygon edge segments

        Creates an iterator that returns pairs of points representing the
        end points of each segment.
        """
        vertices = self.vertices(unique=True)

        n = len(self)
        for i in range(n):
            yield (vertices[:, i], vertices[:, (i + 1) % n])


class Ellipse:
    def __init__(
        self,
        radii: Optional[ArrayLike2] = None,
        E: Optional[NDArray] = None,
        centre: ArrayLike2 = (0, 0),
        theta: Optional[float] = None,
    ):
        r"""
        Create an ellipse

        :param radii: radii of ellipse, defaults to None
        :type radii: arraylike(2), optional
        :param E: 2x2 matrix describing ellipse, defaults to None
        :type E: ndarray(2,2), optional
        :param centre: centre of ellipse, defaults to (0, 0)
        :type centre: arraylike(2), optional
        :param theta: orientation of ellipse, defaults to None
        :type theta: float, optional
        :raises ValueError: bad parameters

        The ellipse shape can be specified by ``radii`` and ``theta`` or by a
        symmetric 2x2 matrix ``E``.

        Internally the ellipse is represented by a symmetric matrix :math:`\mat{E} \in \mathbb{R}^{2\times 2}`
        and its centre coordinate :math:`\vec{x}_0 \in \mathbb{R}^2` such that

        .. math::

            (\vec{x} - \vec{x}_0)^{\top} \mat{E} \, (\vec{x} - \vec{x}_0) = 1

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> import numpy as np
            >>> Ellipse(radii=(1,2), theta=0)
            >>> Ellipse(E=np.array([[1, 1], [1, 2]]))

        """
        if E is not None:
            if not smb.ismatrix(E, (2, 2)):
                raise ValueError("matrix must be 2x2")
            if not np.allclose(E, E.T):
                raise ValueError("matrix must be symmetric")
            if np.linalg.det(E) <= 0:
                raise ValueError("determinant of E must be > 0 for an ellipse")
            self._E = E
        elif radii is not None:
            M = np.array(
                [[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]]
            )
            self._E = M.T @ np.diag([radii[0] ** (-2), radii[1] ** (-2)]) @ M
        else:
            raise ValueError("must specify radii or E")

        self._centre = centre

    @classmethod
    def Polynomial(cls, e: ArrayLike, p: Optional[ArrayLike2] = None) -> Self:
        r"""
        Create an ellipse from polynomial

        :param e: polynomial coeffients :math:`e` or :math:`\eta`
        :type e: arraylike(4) or arraylike(5)
        :param p: point to set scale
        :type p: array_like(2), optional
        :return: an ellipse instance
        :rtype: Ellipse

        An ellipse can be specified by a polynomial :math:`\vec{e} \in \mathbb{R}^6`

        .. math::

            e_0 x^2 + e_1 y^2 + e_2 xy + e_3 x + e_4 y + e_5 = 0

        or :math:`\vec{\epsilon} \in \mathbb{R}^5` where the leading coefficient is
        implicitly one

        .. math::

            x^2 + \epsilon_1 y^2 + \epsilon_2 xy + \epsilon_3 x + \epsilon_4 y + \epsilon_5 = 0

        In this latter case, position, orientation and aspect ratio of the
        ellipse will be correct, but the overall scale of the ellipse is not
        determined.  To correct this, we can pass in a single point ``p`` that
        we know lies on the perimeter of the ellipse.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> Ellipse.Polynomial([0.625, 0.625, 0.75, -6.75, -7.25, 24.625])

        :seealso: :meth:`polynomial`
        """
        e = np.array(e)
        if len(e) == 5:
            e = np.insert(e, 0, 1.0)

        a = e[0]
        b = e[1]
        c = e[2] / 2

        # fmt: off
        E = np.array([
            [a, c],
            [c, b],
        ])
        # fmt: on

        # solve for the centre
        centre = np.linalg.lstsq(-2 * E, e[3:5], rcond=None)[0]

        if p is not None:
            # point was passed in, use this to set the scale
            p = smb.getvector(p, 2) - centre
            s = p @ E @ p
            E /= s

        return cls(E=E, centre=centre)

    @classmethod
    def FromPoints(cls, p) -> Self:
        """
        Create an equivalent ellipse from a set of interior points

        :param p: a set of 2D interior points
        :type p: ndarray(2,N)
        :return: an ellipse instance
        :rtype: Ellipse

        Computes the ellipse that has the same inertia as the set of points.

        :seealso: :meth:`FromPerimeter`
        """
        # compute the moments
        m00 = smb.mpq_point(p, 0, 0)
        m10 = smb.mpq_point(p, 1, 0)
        m01 = smb.mpq_point(p, 0, 1)
        xc = np.c_[m10, m01] / m00

        # compute the central second moments
        x0 = p - xc.T
        u20 = smb.mpq_point(x0, 2, 0)
        u02 = smb.mpq_point(x0, 0, 2)
        u11 = smb.mpq_point(x0, 1, 1)

        # compute inertia tensor and ellipse matrix
        J = np.array([[u20, u11], [u11, u02]])
        E = m00 / 4 * np.linalg.inv(J)
        centre = xc.flatten()

        return cls(E=E, centre=centre)

    @classmethod
    def FromPerimeter(cls, p: Points2) -> Self:
        """
        Create an ellipse that fits a set of perimeter points

        :param p: a set of 2D perimeter points
        :type p: ndarray(2,N)
        :return: an ellipse instance
        :rtype: Ellipse

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> import numpy as np
            >>> eref = Ellipse(radii=(1, 2), theta=np.pi / 4, centre=[3, 4])
            >>> perim = eref.points()
            >>> print(perim.shape)
            >>> Ellipse.FromPerimeter(perim)

        :seealso: :meth:`points`
        """
        A = []
        b = []
        for x, y in p.T:
            A.append([y**2, x * y, x, y, 1])
            b.append(-(x**2))
        # solve for polynomial coefficients eta such that
        #  x^2 + eta[0] y^2 + eta[1] xy + eta[2] x + eta[3] y + eta[4] = 0
        e = np.linalg.lstsq(A, b, rcond=None)[0]

        # create ellipse from the polynomial, using one point to set scale
        return cls.Polynomial(e, p[:, 0])

    def __str__(self) -> str:
        return f"Ellipse(radii={self.radii}, centre={self.centre}, theta={self.theta})"

    def __repr__(self) -> str:
        return f"Ellipse(radii={self.radii}, centre={self.centre}, theta={self.theta})"

    @property
    def E(self):
        r"""
        Return ellipse matrix

        :return: ellipse matrix
        :rtype: ndarray(2,2)

        The symmetric matrix :math:`\mat{E} \in \mathbb{R}^{2\times 2}` determines the radii and
        the orientation of the ellipse

        .. math::

            (\vec{x} - \vec{x}_0)^{\top} \mat{E} \, (\vec{x} - \vec{x}_0) = 1

        :seealso: :meth:`centre` :meth:`theta` :meth:`radii`

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.E
        """
        # return 2x2 ellipse matrix
        return self._E

    @property
    def centre(self) -> R2:
        """
        Return ellipse centre

        :return: centre of the ellipse
        :rtype: ndarray(2)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.centre

        :seealso: :meth:`radii` :meth:`theta` :meth:`E`
        """
        # return centre
        return self._centre

    @property
    def radii(self) -> R2:
        """
        Return radii of the ellipse

        :return: radii of the ellipse
        :rtype: ndarray(2)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.radii

        :seealso: :meth:`centre` :meth:`theta` :meth:`E`
        """
        return np.linalg.eigvals(self.E) ** (-0.5)

    @property
    def theta(self) -> float:
        """
        Return orientation of ellipse

        :return: orientation in radians, in the interval [-pi, pi)
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.theta

        :seealso: :meth:`centre` :meth:`radii` :meth:`E`
        """
        e, x = np.linalg.eigh(self.E)
        # major axis is second column
        return np.arctan(x[1, 1] / x[0, 1])

    @property
    def area(self) -> float:
        """
        Area of ellipse

        :return: area
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.area
        """
        return np.pi / np.sqrt(np.linalg.det(self.E))

    @property
    def polynomial(self):
        r"""
        Return ellipse as a polynomial

        :return: polynomial
        :rtype: ndarray(6)

        An ellipse can be described by :math:`\vec{e} \in \mathbb{R}^6` which are the
        coefficents of a quadratic in :math:`x` and :math:`y`

        .. math::

            e_0 x^2 + e_1 y^2 + e_2 xy + e_3 x + e_4 y + e_5 = 0

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.polynomial

        :seealso: :meth:`Polynomial`
        """
        a = self._E[0, 0]
        b = self._E[1, 1]
        c = self._E[0, 1]
        x_0, y_0 = self._centre

        return np.array(
            [
                a,
                b,
                2 * c,
                -2 * a * x_0 - 2 * c * y_0,
                -2 * b * y_0 - 2 * c * x_0,
                a * x_0**2 + b * y_0**2 + 2 * c * x_0 * y_0,
            ]
        )

    def plot(self, **kwargs) -> None:
        """
        Plot ellipse

        :param kwargs: arguments passed to :func:`~spatialmath.base.graphics.plot_ellipse`
        :return: list of artists
        :rtype: _type_

        Example::

            >>> from spatialmath import Ellipse
            >>> from spatialmath.base import plotvol2
            >>> plotvol2(5)
            >>> e = Ellipse(E=np.array([[1, 1], [1, 2]]))
            >>> e.plot()
            >>> e.plot(filled=True, color='r')


        .. plot::

            from spatialmath import Ellipse
            from spatialmath.base import plotvol2
            ax = plotvol2(5)
            e = Ellipse(E=np.array([[1, 1], [1, 2]]))
            e.plot()
            ax.grid()

        .. plot::

            from spatialmath import Ellipse
            from spatialmath.base import plotvol2
            ax = plotvol2(5)
            e = Ellipse(E=np.array([[1, 1], [1, 2]]))
            e.plot(filled=True, color='r')
            ax.grid()

        :seealso: :func:`~spatialmath.base.graphics.plot_ellipse`
        """
        return plot_ellipse(self._E, centre=self._centre, **kwargs)

    def contains(self, p):
        """
        Test if points are contained by ellipse

        :param p: point or points to test
        :type p: arraylike(2), ndarray(2,N)
        :return: true if point is contained within ellipse
        :rtype: bool or list(bool)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.contains((3,4))
            >>> e.contains((0,0))

        """
        inside = []
        p = smb.getmatrix(p, (2, None))
        for x in p.T:
            x -= self._centre
            inside.append(np.linalg.norm(x.T @ self._E @ x) <= 1)

        if len(inside) == 1:
            return inside[0]
        else:
            return inside

    def points(self, resolution=20) -> Points2:
        """
        Generate perimeter points

        :param resolution: number of points on circumferance, defaults to 20
        :type resolution: int, optional
        :return: set of perimeter points
        :rtype: Points2

        Return a set of ``resolution`` points on the perimeter of the ellipse.  The perimeter
        set is not closed, that is, last point != first point.


        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.points()[:,:5]  # first 5 points

        :seealso: :meth:`polygon` :func:`~spatialmath.base.graphics.ellipse`
        """
        return smb.ellipse(self.E, self.centre, resolution=resolution)

    def polygon(self, resolution=10) -> Polygon2:
        """
        Approximate with a polygon

        :param resolution: number of polygon vertices, defaults to 20
        :type resolution: int, optional
        :return: a polygon approximating the ellipse
        :rtype: :class:`Polygon2` instance

        Return a polygon instance with ``resolution`` vertices.  A :class:`Polygon2`` can be
        used for intersection testing with lines or other polygons.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import Ellipse
            >>> e = Ellipse(radii=(1,2), centre=(3,4), theta=0.5)
            >>> e.polygon()

        :seealso: :meth:`points`
        """
        return Polygon2(smb.ellipse(self.E, self.centre, resolution=resolution - 1))


if __name__ == "__main__":
    pass
    # print(Ellipse((500, 500), (100, 200)))
    # p = Polygon2([(1, 2), (3, 2), (2, 4)])
    # p.transformed(SE2(0, 0, np.pi / 2)).vertices()

    # a = Line2.TwoPoints((1, 2), (7, 5))
    # print(a)

    # p = Polygon2(np.array([[4, 4, 6, 6], [2, 1, 1, 2]]))
    # base.plotvol2([8])
    # p.plot(color="b", alpha=0.3)
    # for theta in np.linspace(0, 2 * np.pi, 100):
    #     p.animate(SE2(0, 0, theta))
    #     plt.show()
    #     plt.pause(0.05)

    # print(p)
    # p.plot(alpha=0.5, color='b')
    # print(p.contains([5.,5.]))
    # print(p.contains([5,1.5]))
    # print(p.contains([4, 2.1]))

    # print(p.vertices())
    # print(p.area())
    # print(p.centroid())
    # print(p.bbox())
    # print(p.radius())
    # print(p.vertices(closed=True))

    # for e in p.edges():
    #     print(e)

    # p2 = p.transformed(SE2(-5, -1.5, 0))
    # print(p2.vertices())
    # print(p2.area())

    # p2.plot(alpha=0.5, facecolor='r')

    # p.move(SE2(0, 0, 0.7))
    # plt.show(block=True)

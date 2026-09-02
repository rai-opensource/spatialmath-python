# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE
from __future__ import annotations

import numpy as np

# try:  # pragma: no cover
#     # print('Using SymPy')
#     from sympy.core.singleton import S

#     _symbolics = True

# except ImportError:  # pragma: no cover
#     _symbolics = False

import spatialmath.base as smb
from spatialmath.base.types import *
from spatialmath.baseposelist import BasePoseList

_eps = np.finfo(np.float64).eps

# colored printing of matrices to the terminal
#   colored package has much finer control than colorama, but the latter is available by default with anaconda
try:
    from colored import fg, bg, attr

    _colored = True
    # print('using colored output')
except ImportError:
    # print('colored not found')
    _colored = False
except AttributeError:
    # print('colored failed to load, can happen from MATLAB import')
    _colored = False

try:
    from ansitable import ANSIMatrix

    _ANSIMatrix = True
    # print('using colored output')
except ImportError:
    # print('colored not found')
    _ANSIMatrix = False


class BasePoseMatrix(BasePoseList):
    """
    Superclass for SO(N) and SE(N) objects

    Subclasses are:

    - ``SO2`` representing elements of SO(2) which describe rotations in 2D
    - ``SE2`` representing elements of SE(2) which describe rigid-body motion in 2D
    - ``SO3`` representing elements of SO(3) which describe rotations in 3D
    - ``SE3`` representing elements of SE(3) which describe rigid-body motion in 3D

    Arithmetic operators are overloaded but the operation they perform depend
    on the types of the operands.  For example:

    - ``*`` will compose two instances of the same subclass, and the result will
      be an instance of the same subclass, since this is a group operator.
    - ``+`` will add two instances of the same subclass, and the result will be
      a matrix, not an instance of the same subclass, since addition is not a group operator.

    These classes all inherit from ``UserList`` which enables them to
    represent a sequence of values, ie. an ``SE3`` instance can contain
    a sequence of SE(3) values.  Most of the Python ``list`` operators
    are applicable::

        >>> x = SE3()  # new instance with identity matrix value
        >>> len(x)     # it is a sequence of one value
        1
        >>> x.append(x)  # append to itself
        >>> len(x)       # it is a sequence of two values
        2
        >>> x[1]         # the element has a 4x4 matrix value
        SE3([
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
            [0., 0., 0., 1.]]) ])
        >>> x[1] = SE3.Rx(0.3)  # set an elements of the sequence
        >>> x.reverse()         # reverse the elements in the sequence
        >>> del x[1]            # delete an element

    For console printing colorization is supported if the package ``colored``
    is installed.  Class variables control the colorization and can be assigned
    to at any time.

    ===============  ===================  ============================================
    Variable         Default              Description
    ===============  ===================  ============================================
    _rotcolor        'red'                Foreground color of rotation submatrix
    _transcolor      'blue'               Foreground color of rotation submatrix
    _constcolor      'grey_50'            Foreground color of matrix constant elements
    _bgcolor         None                 Background color of matrix
    _indexcolor      (None, 'yellow_2')   Foreground, background color of index tag
    _format          '{:< 12g}'           Format string for each matrix element
    _suppress_small  True                 Suppress *small* values, set to zero
    _suppress_tol    100                  Threshold for *small* values in eps units
    _ansimatrix      False                Display with matrix brackets
    ===============  ===================  ============================================

    If color is specified as ``None`` it means no colorization is performed.

    For example::

        >> SE3._bgcolor = None
        >> SE3._indexcolor = ('green', None)

    .. note:: The ``_ansimatrix`` option requires that the ``ansitable`` package
        is installed.  It does not currently support colorization of elements.
    """

    _rotcolor = "red"
    _transcolor = "blue"
    _bgcolor = None
    _constcolor = "grey_50"
    _indexcolor = (None, "yellow_2")
    _format = "{:< 9.4g}"
    _suppress_small = True
    _suppress_tol = 100
    _color = _colored
    _ansimatrix = False
    _ansiformatter = None

    __array_ufunc__ = None  # allow pose matrices operators with NumPy values

    def __new__(cls, *args, **kwargs):
        """
        Create the subclass instance (superclass method)

        Create a new instance and call the superclass initializer to enable the
        ``UserList`` capabilities.
        """

        pose = super(BasePoseMatrix, cls).__new__(cls)  # create a new instance
        super().__init__(pose)  # initialize UserList
        return pose

    # ------------------------------------------------------------------------ #

    @property
    def about(self) -> str:
        """
        Succinct summary of object type and length (superclass property)

        :return: succinct summary
        :rtype: str

        Displays the type and the number of elements in compact form, for
        example::

            >>> x = SE3([SE3() for i in range(20)])
            >>> len(x)
            20
            >>> print(x.about)
            SE3[20]
        """
        return "{:s}[{:d}]".format(type(self).__name__, len(self))

    @property
    def N(self) -> int:
        """
        Dimension of the object's group (superclass property)

        :return: dimension
        :rtype: int

        Dimension of the group is 2 for ``SO2`` or ``SE2``, and 3 for ``SO3`` or ``SE3``.
        This corresponds to the dimension of the space, 2D or 3D, to which these
        rotations or rigid-body motions apply.

        Example::

            >>> SE3().N
            3
            >>> SE2().N
            2
        """
        if type(self).__name__ == "SO2" or type(self).__name__ == "SE2":
            return 2
        else:
            return 3

    # ----------------------- tests
    @property
    def isSO(self) -> bool:
        """
        Test if object belongs to SO(n) group (superclass property)

        :param self: object to test
        :type self: SO2, SE2, SO3, SE3 instance
        :return: ``True`` if object is instance of SO2 or SO3
        :rtype: bool
        """
        return type(self).__name__ == "SO2" or type(self).__name__ == "SO3"

    @property
    def isSE(self) -> bool:
        """
        Test if object belongs to SE(n) group (superclass property)

        :param self: object to test
        :type self: SO2, SE2, SO3, SE3 instance
        :return: ``True`` if object is instance of SE2 or SE3
        :rtype: bool
        """
        return type(self).__name__ == "SE2" or type(self).__name__ == "SE3"

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # --------- compatibility methods

    def isrot(self) -> bool:
        """
        Test if object belongs to SO(3) group (superclass method)

        :return: ``True`` if object is instance of SO3
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SO3)``.

        Example::

            >>> x = SO3()
            >>> x.isrot()
            True
            >>> x = SE3()
            >>> x.isrot()
            False
        """
        return type(self).__name__ == "SO3"

    def isrot2(self) -> bool:
        """
        Test if object belongs to SO(2) group (superclass method)

        :return: ``True`` if object is instance of SO2
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SO2)``.

        Example::

            >>> x = SO2()
            >>> x.isrot()
            True
            >>> x = SE2()
            >>> x.isrot()
            False
        """
        return type(self).__name__ == "SO2"

    def ishom(self) -> bool:
        """
        Test if object belongs to SE(3) group (superclass method)

        :return: ``True`` if object is instance of SE3
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SE3)``.

        Example::

            >>> x = SO3()
            >>> x.isrot()
            False
            >>> x = SE3()
            >>> x.isrot()
            True
        """
        return type(self).__name__ == "SE3"

    def ishom2(self) -> bool:
        """
        Test if object belongs to SE(2) group (superclass method)

        :return: ``True`` if object is instance of SE2
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SE2)``.

        Example::

            >>> x = SO2()
            >>> x.isrot()
            False
            >>> x = SE2()
            >>> x.isrot()
            True
        """
        return type(self).__name__ == "SE2"

    # ----------------------- functions

    def det(self) -> Tuple[float, Rn]:
        """
        Determinant of rotational component (superclass method)

        :return: Determinant of rotational component
        :rtype: float or NumPy array

        ``x.det()`` is the determinant of the rotation component of the values
        of ``x``.

        Example::

            >>> x=SE3.Rand()
            >>> x.det()
            1.0000000000000004
            >>> x=SE3.Rand(N=2)
            >>> x.det()
            [0.9999999999999997, 1.0000000000000002]

        :SymPy: not supported
        """
        if type(self).__name__ in ("SO3", "SE3"):
            if len(self) == 1:
                return np.linalg.det(self.A[:3, :3])
            else:
                return [np.linalg.det(T[:3, :3]) for T in self.data]
        elif type(self).__name__ in ("SO2", "SE2"):
            if len(self) == 1:
                return np.linalg.det(self.A[:2, :2])
            else:
                return [np.linalg.det(T[:2, :2]) for T in self.data]

    def log(self, twist: Optional[bool] = False) -> Union[NDArray, List[NDArray]]:
        """
        Logarithm of pose (superclass method)

        :param twist: return twist coordinates vector rather than matrix, defaults to False
        :type twist: bool, optional
        :return: logarithm
        :rtype: ndarray
        :raises: ValueError

        An efficient closed-form solution of the matrix logarithm.  By default the
        result is a structured matrix

        =====  ======  ===============================
        Input         Output
        -----  ---------------------------------------
        Pose   Shape   Structure
        =====  ======  ===============================
        SO2    (2,2)   skew-symmetric SE2    (3,3)   augmented skew-symmetric
        SO3    (3,3)   skew-symmetric SE3    (4,4)   augmented skew-symmetric
        =====  ======  ===============================

        but if ``twist=True`` the result is a vector, the twist coordinates, containing
        the unique elements with translational elements first.  The rotational
        elements are also known as the exponential coordinates of the rotation.

        Example::

        .. runblock:: pycon

            >>> from spatialmath import SO3, SE3
            >>> T = SE3.Rx(0.3)
            >>> T.log()
            >>> T.log(twist=True)
            >>> R = SO3.Rx(0.3)
            >>> R.log(twist=True)  # exponential coordinates

        :seealso: :func:`~spatialmath.base.transforms2d.trlog2` :func:`~spatialmath.base.transforms3d.trlog`

        :SymPy: not supported
        """
        if self.N == 2:
            log = [smb.trlog2(x, twist=twist) for x in self.data]
        else:
            log = [smb.trlog(x, twist=twist) for x in self.data]
        if len(log) == 1:
            return log[0]
        else:
            return log

    def interp(
        self,
        end: Optional[bool] = None,
        s: Union[int, float] = None,
        shortest: bool = True,
    ) -> Self:
        """
        Interpolate between poses (superclass method)

        :param end: final pose
        :type end: same as ``self``
        :param s: interpolation coefficient, range 0 to 1, or number of steps
        :type s: array_like or int
        :param shortest: take the shortest path along the great circle for the rotation
        :type shortest: bool, default to True
        :return: interpolated pose
        :rtype: same as ``self``

        This interpolation is given by a scalar or the number of equally spaced
        steps:

        - ``X.interp(Y, s)`` interpolates pose between X between when s=0
          and Y when s=1.
        - ``X.interp(Y, N)`` interpolates pose between X and Y in ``N`` steps.

        Example:

        .. runblock:: pycon

            >>> x = SE3(-1, -2, 0) * SE3.Rx(-0.3)
            >>> y = SE3(1, 2, 0) * SE3.Rx(0.3)
            >>> x.interp(y, 0)    # this is x
            >>> x.interp(y, 1)    # this is y
            >>> x.interp(y, 0.5)  # this is in between
            >>> z = x.interp(y, 11)  # in 11 steps
            >>> len(z)
            >>> z[0]              # this is x
            >>> z[5]              # this is in between

        .. note::

            - For SO3 and SE3 rotation is interpolated using quaternion spherical linear interpolation (slerp).
            - Values of ``s`` outside the range [0,1] are silently clipped

        :seealso: :func:`interp1`, :func:`~spatialmath.base.transforms3d.trinterp`, :func:`~spatialmath.base.quaternions.qslerp`, :func:`~spatialmath.base.transforms2d.trinterp2`

        :SymPy: not supported
        """

        if isinstance(s, int) and s > 1:
            s = np.linspace(0, 1, s)
        else:
            s = smb.getvector(s)
            s = np.clip(s, 0, 1)

        if len(self) > 1:
            raise ValueError("start pose must be a singleton")

        if end is not None:
            if len(end) > 1:
                raise ValueError("end pose must be a singleton")
            end = end.A

        if self.N == 2:
            # SO(2) or SE(2)
            return self.__class__(
                [
                    smb.trinterp2(start=self.A, end=end, s=_s, shortest=shortest)
                    for _s in s
                ]
            )

        elif self.N == 3:
            # SO(3) or SE(3)
            return self.__class__(
                [
                    smb.trinterp(start=self.A, end=end, s=_s, shortest=shortest)
                    for _s in s
                ]
            )

    def interp1(self, s: float = None) -> Self:
        """
        Interpolate pose (superclass method)

        :param end: final pose
        :type end: same as ``self``
        :param s: interpolation coefficient, range 0 to 1
        :type s: array_like
        :return: interpolated pose
        :rtype: SO2, SE2, SO3, SE3 instance

        - ``X.interp(s)`` interpolates pose between identity when s=0, and X when s=1.

            ======  ======  ===========  ===============================
            len(X)  len(s)  len(result)  Result
            ======  ======  ===========  ===============================
            1       1       1            Y = interp(X, s)
            M       1       M            Y[i] = interp(X[i], s)
            1       M       M            Y[i] = interp(X, s[i])
            ======  ======  ===========  ===============================

        Example::

            >>> x = SE3.Rx(0.3)
            >>> print(x.interp(0))
            SE3(array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]]))
            >>> print(x.interp(1))
            SE3(array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.95533649, -0.29552021,  0.        ],
                       [ 0.        ,  0.29552021,  0.95533649,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]))
            >>> y = x.interp(x, np.linspace(0, 1, 10))
            >>> len(y)
            10
            >>> y[5]
            SE3(array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.98614323, -0.16589613,  0.        ],
                       [ 0.        ,  0.16589613,  0.98614323,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]))

        Notes:

        #. For SO3 and SE3 rotation is interpolated using quaternion spherical linear interpolation (slerp).

        :seealso: :func:`interp`, :func:`~spatialmath.base.transforms3d.trinterp`, :func:`~spatialmath.base.quaternions.qslerp`, :func:`~spatialmath.smb.transforms2d.trinterp2`

        :SymPy: not supported
        """
        s = smb.getvector(s)
        s = np.clip(s, 0, 1)

        if self.N == 2:
            # SO(2) or SE(2)
            if len(s) > 1:
                assert len(self) == 1, "if len(s) > 1, len(X) must == 1"
                return self.__class__([smb.trinterp2(None, self.A, s=_s) for _s in s])
            else:
                return self.__class__(
                    [smb.trinterp2(None, x, s=s[0]) for x in self.data]
                )
        elif self.N == 3:
            # SO(3) or SE(3)
            if len(s) > 1:
                assert len(self) == 1, "if len(s) > 1, len(X) must == 1"
                return self.__class__([smb.trinterp(None, self.A, s=_s) for _s in s])
            else:
                return self.__class__(
                    [smb.trinterp(None, x, s=s[0]) for x in self.data]
                )

    def norm(self) -> Self:
        """
        Normalize pose (superclass method)

        :return: pose
        :rtype: SO2, SE2, SO3, SE3 instance

        - ``X.norm()`` is an equivalent pose object but the rotational matrix
          part of all values has been adjusted to ensure it is a proper orthogonal
          matrix rotation.

        Example::

            >>> x = SE3()
            >>> y = x.norm()
            >>> y
            SE3(array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]]))

        Notes:

        #. Only the direction of A vector (the z-axis) is unchanged.
        #. Used to prevent finite word length arithmetic causing transforms to
           become 'unnormalized'.

        :seealso: :func:`~spatialmath.base.transforms3d.trnorm`, :func:`~spatialmath.base.transforms2d.trnorm2`
        """
        if self.N == 2:
            return self.__class__([smb.trnorm2(x) for x in self.data])
        else:
            return self.__class__([smb.trnorm(x) for x in self.data])

    def simplify(self) -> Self:
        """
        Symbolically simplify matrix values (superclass method)

        :return: pose with symbolic elements
        :rtype: pose instance

        Apply symbolic simplification to every element of every value in the
        pose instance.

        Example::

            >>> a = SE3.Rx(sympy.symbols('theta'))
            >>> b = a * a
            >>> b
            SE3(array([[1, 0, 0, 0.0],
            [0, -sin(theta)**2 + cos(theta)**2, -2*sin(theta)*cos(theta), 0],
            [0, 2*sin(theta)*cos(theta), -sin(theta)**2 + cos(theta)**2, 0],
            [0.0, 0, 0, 1.0]], dtype=object)
            >>> b.simplify()
            SE3(array([[1, 0, 0, 0],
            [0, cos(2*theta), -sin(2*theta), 0],
            [0, sin(2*theta), cos(2*theta), 0],
            [0, 0, 0, 1.00000000000000]], dtype=object))

        .. todo:: No need to simplify the constants in bottom row

        :SymPy: supported
        """

        vf = np.vectorize(smb.sym.simplify)
        return self.__class__([vf(x) for x in self.data], check=False)

    def stack(self) -> NDArray:
        """
        Convert to 3-dimensional matrix

        :return: 3-dimensional NumPy array
        :rtype: ndarray(n,n,m)

        Converts the value to a 3-dimensional NumPy array where the values are
        stacked along the third axis.  The first two dimensions are given by
        ``self.shape``.
        """
        return np.dstack(self.data)

    def conjugation(self, A: NDArray) -> NDArray:
        """
        Matrix conjugation

        :param A: matrix to conjugate
        :type A: ndarray
        :return: conjugated matrix
        :rtype: ndarray

        Compute the conjugation :math:`\mat{X} \mat{A} \mat{X}^{-1}` where :math:`\mat{X}`
        is the current object.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO2
            >>> import numpy as np
            >>> R = SO2(0.5)
            >>> A = np.array([[10, 0], [0, 1]])
            >>> print(R * A * R.inv())
            >>> print(R.conjugation(A))
        """
        if self.isSO:
            return self.A @ A @ self.A.T
        else:
            return self.A @ A @ self.inv().A.T

    # ----------------------- i/o stuff

    def print(self, label: Optional[str] = None, file: Optional[TextIO] = None) -> None:
        """
        Print pose as a matrix (superclass method)

        :param label: label to print before the matrix, defaults to None
        :type label: str, optional
        :param file: file to write to, defaults to None
        :type file: file object, optional

        Print the pose as a matrix, with an optional line beforehand.  By default
        the matrix is printed to stdout.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3().print()
            >>> SE3().print("pose is:")

        :seealso: :meth:`printline` :meth:`strline`
        """
        if label is not None:
            print(label, file=file)
        print(self, file=file)

    def printline(self, *args, **kwargs) -> None:
        r"""
        Print pose in compact single line format (superclass method)

        :param arg: value for orient option, optional
        :type arg: str
        :param label: text label to put at start of line
        :type label: str
        :param fmt: conversion format for each number as used by ``format()``
        :type fmt: str
        :param label: text label to put at start of line
        :type label: str
        :param orient: 3-angle convention to use, optional, ``SO3`` and ``SE3``
                       only
        :type orient: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param file: file to write formatted string to. [default, stdout]
        :type file: file object

        Print pose in a compact single line format. If ``X`` has multiple
        values, print one per line.

        Orientation can be displayed in various formats:

        =============   =================================================
        ``orient``      description
        =============   =================================================
        ``'rpy/zyx'``   roll-pitch-yaw angles in ZYX axis order [default]
        ``'rpy/yxz'``   roll-pitch-yaw angles in YXZ axis order
        ``'rpy/zyx'``   roll-pitch-yaw angles in ZYX axis order
        ``'eul'``       Euler angles in ZYZ axis order
        ``'angvec'``    angle and axis
        =============   =================================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2, SE3
            >>> x = SE3.Rx(0.3)
            >>> x.printline()
            >>> x = SE3.Rx([0.2, 0.3])
            >>> x.printline()
            >>> x.printline(orient="angvec")
            >>> x.printline(orient="angvec", fmt="{:.6f}")
            >>> x = SE2(1, 2, 0.3)
            >>> x.printline()

        .. note::
            - Default formatting is for compact display of data
            - For tabular data set ``fmt`` to a fixed width format such as
              ``fmt='{:.3g}'``

        :seealso: :meth:`strline` :func:`trprint`, :func:`trprint2`
        """
        if self.N == 2:
            for x in self.data:
                smb.trprint2(x, *args, **kwargs)
        else:
            for x in self.data:
                smb.trprint(x, *args, **kwargs)

    def strline(self, *args, **kwargs) -> str:
        """
        Convert pose to compact single line string (superclass method)

        :param label: text label to put at start of line
        :type label: str
        :param fmt: conversion format for each number as used by ``format()``
        :type fmt: str
        :param label: text label to put at start of line
        :type label: str
        :param orient: 3-angle convention to use, optional, ``SO3`` and ``SE3``
                       only
        :type orient: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: pose in string format
        :rtype: str

        Convert pose in a compact single line format. If ``X`` has multiple
        values, the string has one pose per line.

        Orientation can be displayed in various formats:

        =============   =================================================
        ``orient``      description
        =============   =================================================
        ``'rpy/zyx'``   roll-pitch-yaw angles in ZYX axis order [default]
        ``'rpy/yxz'``   roll-pitch-yaw angles in YXZ axis order
        ``'rpy/zyx'``   roll-pitch-yaw angles in ZYX axis order
        ``'eul'``       Euler angles in ZYZ axis order
        ``'angvec'``    angle and axis
        =============   =================================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE2, SE3
            >>> x = SE3.Rx(0.3)
            >>> x.strline()
            >>> x = SE3.Rx([0.2, 0.3])
            >>> x.strline()
            >>> x.strline('angvec')
            >>> x.strline(orient='angvec', fmt="{:.6f}")
            >>> x = SE2(1, 2, 0.3)
            >>> x.strline()

        .. note::
            - Default formatting is for compact display of data
            - For tabular data set ``fmt`` to a fixed width format such as
              ``fmt='{:.3g}'``

        :seealso: :meth:`printline` :func:`trprint`, :func:`trprint2`
        """
        s = ""
        if self.N == 2:
            for x in self.data:
                s += smb.trprint2(x, *args, file=False, **kwargs)
        else:
            for x in self.data:
                s += smb.trprint(x, *args, file=False, **kwargs)
        return s

    def __repr__(self) -> str:
        """
        Readable representation of pose (superclass method)

        :return: readable representation of the pose as a list of arrays
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3.Rx(0.3)
            >>> repr(x)

        """

        # TODO: really should iterate over all the elements, can have a symb
        #       element and ~eps values
        def trim(x):
            if x.dtype == "O":
                return x
            else:
                return smb.removesmall(x)

        name = type(self).__name__
        if len(self) == 0:
            return name + "([])"
        elif len(self) == 1:
            # need to indent subsequent lines of the native repr string by 4 spaces
            return name + "(" + trim(self.A).__repr__().replace("\n", "\n    ") + ")"
        else:
            # format this as a list of ndarrays
            return (
                name
                + "([\n"
                + ",\n".join([trim(v).__repr__() for v in self.data])
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
        # see https://ipython.org/ipython-doc/stable/api/generated/IPython.lib.pretty.html

        if len(self) == 1:
            p.text(str(self))
        else:
            for i, x in enumerate(self):
                p.text(f"{i}:\n{str(x)}")

    def __str__(self) -> str:
        """
        Pretty string representation of pose (superclass method)

        :return: readable representation of the pose
        :rtype: str

        Convert the pose's matrix value to a simple grid of numbers.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3.Rx(0.3)
            >>> print(x)

        Notes:

            - By default, the output is colorised for an ANSI terminal console:

                * red: rotational elements
                * blue: translational elements
                * white: constant elements

        """
        if _ANSIMatrix and self._ansimatrix:
            return self._string_matrix()
        else:
            return self._string_color(color=True)

    def _string_matrix(self) -> str:
        if self._ansiformatter is None:
            self._ansiformatter = ANSIMatrix(style="thick")

        return "\n".join([self._ansiformatter.str(A) for A in self.data])

    def _string_color(self, color: Optional[bool] = False) -> str:
        """
        Pretty print the matrix value

        :param color: colorise the output, defaults to False
        :type color: bool, optional
        :return: multiline matrix representation
        :rtype: str

        Convert a matrix to a simple grid of numbers with optional
        colorization for an ANSI terminal console:

                * red: rotational elements
                * blue: translational elements
                * white: constant elements

        """
        # print('in __str__', _color)

        if self._color:

            def color(c, f):
                if c is None:
                    return ""
                else:
                    return f(c)

            bgcol = color(self._bgcolor, bg)
            trcol = color(self._transcolor, fg) + bgcol
            rotcol = color(self._rotcolor, fg) + bgcol
            constcol = color(self._constcolor, fg) + bgcol
            indexcol = color(self._indexcolor[0], fg) + color(self._indexcolor[1], bg)
            reset = attr(0)
        else:
            bgcol = ""
            trcol = ""
            rotcol = ""
            constcol = ""
            reset = ""

        def mformat(self, X):
            # X is an ndarray value to be display
            # self provides set type for formatting
            out = ""
            n = self.N  # dimension of rotation submatrix
            for rownum, row in enumerate(X):
                rowstr = " "
                # format the columns
                for colnum, element in enumerate(row):
                    if smb.sym.issymbol(element):
                        s = "{:<12s}".format(str(element))
                    else:
                        if (
                            self._suppress_small
                            and abs(element) < self._suppress_tol * _eps
                        ):
                            element = 0
                        s = self._format.format(element)

                    if rownum < n:
                        if colnum < n:
                            # rotation part
                            s = rotcol + bgcol + s + reset
                        else:
                            # translation part
                            s = trcol + bgcol + s + reset
                    else:
                        # bottom row
                        s = constcol + bgcol + s + reset
                    rowstr += " " + s
                out += rowstr + bgcol + "  " + reset + "\n"
            return out

        output_str = ""

        if len(self.data) == 0:
            output_str = "[]"
        elif len(self.data) == 1:
            # single matrix case
            output_str = mformat(self, self.A)
        else:
            # sequence case
            for count, X in enumerate(self.data):
                # add separator lines and the index
                output_str += (
                    indexcol
                    + "[{:d}] =".format(count)
                    + reset
                    + "\n"
                    + mformat(self, X)
                )

        return output_str

    # ----------------------- graphics

    def plot(self, *args, **kwargs) -> None:
        """
        Plot pose object as a coordinate frame (superclass method)

        :param `**kwargs`: plotting options

        - ``X.plot()`` displays the pose ``X`` as a coordinate frame in either
          2D or 3D.  There are many options, see the links below.

        Example::

            >>> X = SE3.Rx(0.3)
            >>> X.plot(frame='A', color='green')

        .. plot::

            from spatialmath import SE3
            X = SE3.Rx(0.3)
            X.plot(frame='A', color='green')

        :seealso: :func:`~spatialmath.base.transforms3d.trplot`, :func:`~spatialmath.base.transforms2d.trplot2`
        """
        if self.N == 2:
            smb.trplot2(self.A, *args, **kwargs)
        else:
            smb.trplot(self.A, *args, **kwargs)

    def animate(self, *args, start=None, **kwargs) -> None:
        """
        Plot pose object as an animated coordinate frame (superclass method)

        :param start: initial pose, defaults to null/identity
        :type start: same as ``self``
        :param `**kwargs`: plotting options

        - ``X.animate()`` displays the pose ``X`` as a coordinate frame moving
          from the origin in either 2D or 3D.  There are many options, see the
          links below.
        - ``X.animate(*args, start=X1)`` displays the pose ``X`` as a coordinate
          frame moving from pose ``X1``, in either 2D or 3D.  There are
          many options, see the links below.

        Example::

            >>> X = SE3.Rx(0.3)
            >>> X.animate(frame='A', color='green')
            >>> X.animate(start=SE3.Ry(0.2))

        :seealso: :func:`~spatialmath.base.transforms3d.tranimate`, :func:`~spatialmath.base.transforms2d.tranimate2`
        """
        if start is not None:
            start = start.A

        if len(self) > 1:
            # trajectory case
            if self.N == 2:
                return smb.tranimate2(self.data, *args, **kwargs)
            else:
                return smb.tranimate(self.data, *args, **kwargs)
        else:
            # singleton case
            if self.N == 2:
                return smb.tranimate2(self.A, start=start, *args, **kwargs)
            else:
                return smb.tranimate(self.A, start=start, *args, **kwargs)

    # ------------------------------------------------------------------------ #
    def prod(self, norm=False, check=True) -> Self:
        r"""
        Product of elements (superclass method)

        :param norm: normalize the product, defaults to False
        :type norm: bool, optional
        :param check: check that computed matrix is valid member of group, default True
        :bool check: bool, optional
        :return: Product of elements
        :rtype: pose instance

        ``x.prod()`` is the product of the values held by ``x``, ie.
        :math:`\prod_i^N T_i`.

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3.Rx([0, 0.1, 0.2, 0.3])
            >>> x.prod()

        .. note:: When compounding many transformations the product may become
            denormalized resulting in a result that is not a proper member of the
            group.  You can either disable membership checking by ``check=False``
            which is risky, or normalize the result by ``norm=True``.
        """
        Tprod = self.__class__._identity()  # identity value
        for T in self.data:
            Tprod = Tprod @ T
        if norm:
            Tprod = smb.trnorm(Tprod)
        return self.__class__(Tprod, check=check)

    def __pow__(self, n: int) -> Self:
        """
        Overloaded ``**`` operator (superclass method)

        :param n: exponent
        :type n: int
        :return: pose to the power ``n``
        :rtype: pose instance

        ``X**n`` raise all values held in `X` to the specified power using repeated
        multiplication.  If ``n`` < 0 then the result is inverted.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> SE3.Rx(0.1) ** 2
            >>> SE3.Rx([0, 0.1]) ** 2

        """

        assert type(n) is int, "exponent must be an int"
        return self.__class__(
            [np.linalg.matrix_power(x, n) for x in self.data], check=False
        )

    # ----------------------- arithmetic

    def __mul__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator (superclass method)

        :return: Product of two operands
        :rtype: Pose instance or NumPy array
        :raises NotImplemented: for incompatible arguments

        Pose composition, scaling or vector transformation:

        - ``X * Y`` compounds the poses ``X`` and ``Y``
        - ``X * s`` performs element-wise multiplication of the elements of ``X`` by ``s``
        - ``s * X`` performs element-wise multiplication of the elements of ``X`` by ``s``
        - ``X * v`` linear transformation of the vector ``v`` where ``v`` is array-like

        ==============   ==============   ===========  ======================
                   Multiplicands                   Product
        -------------------------------   -----------------------------------
            left             right            type           operation
        ==============   ==============   ===========  ======================
        Pose             Pose             Pose         matrix product
        Pose             scalar           NxN matrix   element-wise product
        scalar           Pose             NxN matrix   element-wise product
        Pose             N-vector         N-vector     vector transform
        Pose             NxM matrix       NxM matrix   transform each column
        ==============   ==============   ===========  ======================

        .. note::

            #. Pose is an ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
            #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
            #. Scalar x Pose is handled by __rmul__`
            #. Scalar multiplication is commutative but the result is not a group
               operation so the result will be a matrix
            #. Any other input combinations result in a ValueError.

        For pose composition either or both operands may hold more than one value which
        results in the composition holding more than one value according to:

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left * right``
         1          M             M    ``prod[i] = left * right[i]``
         N          1             M    ``prod[i] = left[i] * right``
         M          M             M    ``prod[i] = left[i] * right[i]``
        =========   ==========   ====  ================================

        Example::

            >>> SE3.Rx(pi/2) * SE3.Ry(pi/2)
            SE3(array([[0., 0., 1., 0.],
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 0., 1.]]))
            >>> SE3.Rx(pi/2) * 2
            array([[ 2.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                   [ 0.0000000e+00,  1.2246468e-16, -2.0000000e+00,  0.0000000e+00],
                   [ 0.0000000e+00,  2.0000000e+00,  1.2246468e-16,  0.0000000e+00],
                   [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  2.0000000e+00]])

        For vector transformation there are three cases:

        =========  ===========  =====  ==========================
              Multiplicands             Product
        ----------------------  ---------------------------------
        len(left)  right.shape  shape  operation
        =========  ===========  =====  ==========================
        1          (N,)         (N,)   vector transformation
        M          (N,)         (N,M)  vector transformations
        1          (N,M)        (N,M)  column transformation
        =========  ===========  =====  ==========================

        .. note::
            - The vector is an array-like, a 1D NumPy array or a list/tuple
            - For the ``SE2`` and ``SE3`` case the vectors are converted to homogeneous
              form, transformed, then converted back to Euclidean form.

        Example::

            >>> SE3.Rx(pi/2) * [0, 1, 0]
            array([0.000000e+00, 6.123234e-17, 1.000000e+00])
            >>> SE3.Rx(pi/2) * np.r_[0, 0, 1]
            array([ 0.000000e+00, -1.000000e+00,  6.123234e-17])
        """
        if type(left) == type(right):
            # print('*: pose x pose')
            return left.__class__(left._op2(right, lambda x, y: x @ y), check=False)

        elif isinstance(right, (list, tuple, np.ndarray)):
            # print('*: pose x array')
            if len(left) == 1:
                if smb.isvector(right, left.N):
                    # pose x vector
                    # print('*: pose x vector')
                    v = smb.getvector(right, out="col")
                    if left.isSE:
                        # SE(n) x vector
                        return smb.h2e(left.A @ smb.e2h(v))
                    else:
                        # SO(n) x vector
                        return left.A @ v
                elif left.isSE and right.shape == left.shape:
                    # SE x conforming matrix
                    return left.A @ right
                else:
                    if left.isSE:
                        # SE(n) x [set of vectors]
                        return smb.h2e(left.A @ smb.e2h(right))
                    else:
                        # SO(n) x [set of vectors]
                        return left.A @ right

            elif len(left) > 1 and smb.isvector(right, left.N):
                # pose array x vector
                # print('*: pose array x vector')
                v = smb.getvector(right)
                if left.isSE:
                    # SE(n) x vector
                    v = smb.e2h(v)
                    return np.array([smb.h2e(x @ v).flatten() for x in left.A]).T
                else:
                    # SO(n) x vector
                    return np.array([(x @ v).flatten() for x in left.A]).T

            elif (
                len(left) == 1
                and isinstance(right, np.ndarray)
                and left.isSO
                and right.shape[0] == left.N
            ):
                # SO(n) x matrix
                return left.A @ right
            elif (
                len(left) == 1
                and isinstance(right, np.ndarray)
                and left.isSE
                and right.shape[0] == left.N
            ):
                # SE(n) x matrix
                return smb.h2e(left.A @ smb.e2h(right))
            elif (
                isinstance(right, np.ndarray)
                and left.isSO
                and right.shape[0] == left.N
                and len(left) == right.shape[1]
            ):
                # SO(n) x matrix
                return np.c_[[x.A @ y for x, y in zip(right, left.T)]].T
            elif (
                isinstance(right, np.ndarray)
                and left.isSE
                and right.shape[0] == left.N
                and len(left) == right.shape[1]
            ):
                # SE(n) x matrix
                return np.c_[
                    [smb.h2e(x.A @ smb.e2h(y)) for x, y in zip(right, left.T)]
                ].T
            else:
                raise ValueError("bad operands")
        elif smb.isscalar(right):
            return left._op2(right, lambda x, y: x * y)
        else:
            return NotImplemented

    def __matmul__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``@`` operator (superclass method)

        :return: Product of two operands with normalization
        :rtype: Pose instance or NumPy array
        :raises ValueError: for incompatible arguments

        - ``X @ Y`` compounds the poses ``X`` and ``Y`` and normalizes the result
        - ``X @= Y`` compounds the poses ``X`` and ``Y``, normalizes the result,
          and places the result in ``X``

        .. note:: This operator is functionally equivalent to ``*`` but is more
            costly.  It is useful for cases where a pose is incrementally
            update over many cycles.

        :seealso: :func:`__mul__`, :func:`~spatialmath.base.trnorm`
        """
        if isinstance(left, right.__class__):
            # print('*: pose x pose')
            return left.__class__(
                left._op2(right, lambda x, y: smb.trnorm(x @ y)), check=False
            )
        else:
            raise TypeError("@ only applies to pose composition")

    def __rmul__(right, left):  # pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator (superclass method)

        :return: Product of two operands
        :rtype: Pose instance or NumPy array
        :raises NotImplemented: for incompatible arguments

        Left-multiplication

        - ``s * X`` where ``s`` is a scalar, performs elementwise multiplication of the
          elements of ``X`` by ``s`` and the result is a NumPy array.
        - ``A * X`` where ``A`` is a conforming matrix, performs matrix multiplication
          of ``A`` and ``X`` and the result is a NumPy array.

        :seealso: :func:`__mul__`
        """
        if isinstance(left, np.ndarray) and left.shape[-1] == right.A.shape[0]:
            # left multiply by conforming matrix
            return left @ right.A
        elif smb.isscalar(left):
            # left multiply by scalar
            return right.__mul__(left)
        else:
            # For other left-operands return ``NotImplemented``.  Other classes
            # such as ``Plucker`` and ``Twist`` implement left-multiplication by
            # an ``SE3`` using their own ``__rmul__`` methods.
            return NotImplemented

    def __imul__(left, right):  # noqa
        """
        Overloaded ``*=`` operator (superclass method)

        :return: Product of two operands
        :rtype: Pose instance or NumPy array
        :raises ValueError: for incompatible arguments

        - ``X *= Y`` compounds the poses ``X`` and ``Y`` and places the result in ``X``
        - ``X *= s`` performs elementwise multiplication of the elements of ``X``
          and ``s`` and places the result in ``X``

        :seealso: ``__mul__``
        """
        return left.__mul__(right)

    def __truediv__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``/`` operator (superclass method)

        :return: Product of right operand and inverse of left operand
        :rtype: pose instance or NumPy array
        :raises ValueError: for incompatible arguments

        Pose composition or scaling:

        - ``X / Y`` compounds the poses ``X`` and ``Y.inv()``
        - ``X / s`` performs elementwise division of the elements of ``X`` by ``s``

        ==============   ==============   ===========  =========================
                   Multiplicands                   Quotient
        -------------------------------   --------------------------------------
            left             right            type           operation
        ==============   ==============   ===========  =========================
        Pose             Pose             Pose         matrix product by inverse
        Pose             scalar           NxN matrix   element-wise division
        ==============   ==============   ===========  =========================

        .. note::

            #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
            #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
            #. Scalar multiplication is not a group operation so the result will
               be a matrix
            #. Any other input combinations result in a ValueError.

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

        """
        if isinstance(left, right.__class__):
            return left.__class__(
                left._op2(right.inv(), lambda x, y: x @ y), check=False
            )
        elif smb.isscalar(right):
            return left._op2(right, lambda x, y: x / y)
        else:
            raise ValueError("bad operands")

    def __itruediv__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``/=`` operator (superclass method)

        :return: Product of right operand and inverse of left operand
        :rtype: Pose instance or NumPy array
        :raises ValueError: for incompatible arguments

        - ``X /= Y`` compounds the poses ``X`` and ``Y.inv()`` and places the result in ``X``
        - ``X /= s`` performs elementwise division of the elements of ``X``
          by ``s`` and places the result in ``X``

        :seealso: ``__truediv__``
        """
        return left.__truediv__(right)

    def __add__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``+`` operator (superclass method)

        :return: Sum of two operands
        :rtype: NumPy array, shape=(N,N)
        :raises ValueError: for incompatible arguments


        Add the elements of two poses.  This is not a group operation so the
        result is a matrix not a pose class.

        - ``X + Y`` is the element-wise sum of the matrix value of ``X`` and ``Y``
        - ``X + s`` is the element-wise sum of the matrix value of ``X`` and scalar ``s``
        - ``s + X`` is the element-wise sum of the scalar ``s`` and the matrix value of ``X``

        ==============   ==============   ===========  ========================
                   Operands                   Sum
        -------------------------------   -------------------------------------
            left             right            type           operation
        ==============   ==============   ===========  ========================
        Pose             Pose             NxN matrix   element-wise matrix sum
        Pose             scalar           NxN matrix   element-wise sum
        scalar           Pose             NxN matrix   element-wise sum
        ==============   ==============   ===========  ========================

        .. note::

            #. Pose is an ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
            #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
            #. scalar + Pose is handled by :meth:`__radd__`
            #. Addition is commutative
            #. Any other input combinations result in a ``ValueError``.

        For pose addition either or both operands may hold more than one value which
        results in the sum holding more than one value according to:

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``sum = left + right``
         1          M             M    ``sum[i] = left + right[i]``
         N          1             M    ``sum[i] = left[i] + right``
         M          M             M    ``sum[i] = left[i] + right[i]``
        =========   ==========   ====  ================================

        """
        # results is not in the group, return an array, not a class
        return left._op2(right, lambda x, y: x + y)

    def __radd__(right, left):  # pylint: disable=no-self-argument
        """
        Overloaded ``+`` operator (superclass method)

        :return: Sum of two operands
        :rtype: NumPy array, shape=(N,N)
        :raises ValueError: for incompatible arguments

        Left-addition by a scalar

        - ``s + X`` performs elementwise addition of the elements of ``X`` and ``s``

        :seealso: :meth:`__add__`
        """
        return right.__add__(left)

    def __iadd__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``+=`` operator (superclass method)

        :return: Sum of two operands
        :rtype: NumPy array, shape=(N,N)
        :raises ValueError: for incompatible arguments

        - ``X += Y`` adds the matrix values of ``X`` and ``Y`` and places the result in ``X``
        - ``X += s`` elementwise addition of the matrix elements of ``X``
          and ``s`` and places the result in ``X``

        :seealso: ``__add__``
        """
        return left.__add__(right)

    def __sub__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``-`` operator (superclass method)

        :return: Difference of two operands
        :rtype: NumPy array, shape=(N,N)
        :raises ValueError: for incompatible arguments


        Subtract elements of two poses.  This is not a group operation so the
        result is a matrix not a pose class.

        - ``X - Y`` is the element-wise difference of the matrix value of ``X`` and ``Y``
        - ``X - s`` is the element-wise difference of the matrix value of ``X`` and the scalar ``s``
        - ``s - X`` is the element-wise difference of the scalar ``s`` and the matrix value of ``X``

        ==============   ==============   ===========  ==============================
                   Operands                   Sum
        -------------------------------   -------------------------------------------
            left             right            type           operation
        ==============   ==============   ===========  ==============================
        Pose             Pose             NxN matrix   element-wise matrix difference
        Pose             scalar           NxN matrix   element-wise sum
        scalar           Pose             NxN matrix   element-wise sum
        ==============   ==============   ===========  ==============================

        .. note::

            #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
            #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
            #. scalar - Pose is handled by :meth:`__rsub__`
            #. Any other input combinations result in a ``ValueError``.

        For pose subtraction either or both operands may hold more than one value which
        results in the difference holding more than one value according to:

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``diff = left - right``
         1          M             M    ``diff[i] = left - right[i]``
         N          1             M    ``diff[i] = left[i] - right``
         M          M             M    ``diff[i] = left[i]  right[i]``
        =========   ==========   ====  ================================
        """

        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        return left._op2(right, lambda x, y: x - y)

    def __rsub__(right, left: Self):  # pylint: disable=no-self-argument
        """
        Overloaded ``-`` operator (superclass method)

        :return: Difference of two operands
        :rtype: NumPy array, shape=(N,N)
        :raises ValueError: for incompatible arguments

        Left-addition by a scalar

        - ``s - X`` performs elementwise addition of the elements of ``X`` and ``s``

        :seealso: :meth:`__sub__`
        """
        return -right.__sub__(left)

    def __isub__(left, right: Self):  # pylint: disable=no-self-argument
        """
        Overloaded ``-=`` operator (superclass method)

        :return: Difference of two operands
        :rtype: NumPy array, shape=(N,N)
        :raises: ValueError

        - ``X -= Y`` is the element-wise difference of the matrix value of ``X``
          and ``Y`` and places the result in ``X``
        - ``X -= s`` is the element-wise difference of the matrix value of ``X``
          and the scalar ``s`` and places the result in ``X``

        :seealso: ``__sub__``
        """
        return left.__sub__(right)

    def __eq__(left, right: Self) -> bool:  # pylint: disable=no-self-argument
        """
        Overloaded ``==`` operator (superclass method)

        :return: Equality of two operands
        :rtype: bool or list of bool

        Test two poses for equality

        ``X == Y`` is true of the poses are of the same type and numerically
        equal.

        If either or both operands may hold more than one value which
        results in the equality test holding more than one value according to:

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``eq = left == right``
         1          M             M    ``eq[i] = left == right[i]``
         N          1             M    ``eq[i] = left[i] == right``
         M          M             M    ``eq[i] = left[i] == right[i]``
        =========   ==========   ====  ================================

        """
        return (
            left._op2(right, lambda x, y: np.allclose(x, y))
            if type(left) == type(right)
            else False
        )

    def __ne__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``!=`` operator (superclass method)

        :return: Inequality of two operands
        :rtype: bool or list of bool

        Test two poses for inequality

        - ``X != Y`` is true of the poses are of the same type but not numerically
          equal.

        If either or both operands may hold more than one value which
        results in the inequality test holding more than one value according to:

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``ne = left != right``
         1          M             M    ``ne[i] = left != right[i]``
         N          1             M    ``ne[i] = left[i] != right``
         M          M             M    ``ne[i] = left[i] != right[i]``
        =========   ==========   ====  ================================

        """
        eq = left == right
        return not eq if isinstance(eq, bool) else [not x for x in eq]

    def _op2(left, right: Self, op: Callable):  # pylint: disable=no-self-argument
        """
        Perform binary operation

        :param left: left side of comparison
        :type self: SO2, SE2, SO3, SE3
        :param right: right side of comparison
        :type self: SO2, SE2, SO3, SE3
        :param op: binary operation
        :type op: callable
        :raises ValueError: arguments are not compatible
        :return: list of matrices
        :rtype: list

        Perform a binary operation on a pair of operands.  If either operand
        contains a sequence the results is a sequence according to this
        truth table.

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``ret = op(left, right)``
         1          M             M    ``ret[i] = op(left, right[i])``
         N          1             M    ``ret[i] = op(left[i], right)``
         M          M             M    ``ret[i] = op(left[i], right[i])``
        =========   ==========   ====  ================================

        """
        if isinstance(right, left.__class__) or isinstance(left, right.__class__):
            # class by class
            if len(left) == 1:
                if len(right) == 1:
                    # print('== 1x1')
                    return op(left.A, right.A)
                else:
                    # print('== 1xN')
                    return [op(left.A, x) for x in right.A]
            else:
                if len(right) == 1:
                    # print('== Nx1')
                    return [op(x, right.A) for x in left.A]
                elif len(left) == len(right):
                    # print('== NxN')
                    return [op(x, y) for (x, y) in zip(left.A, right.A)]
                else:
                    raise ValueError("length of lists to == must be same length")
        elif smb.isscalar(right) or (
            isinstance(right, np.ndarray) and right.shape == left.shape
        ):
            # class by matrix
            if len(left) == 1:
                return op(left.A, right)
            else:
                return [op(x, right) for x in left.A]
        else:
            raise TypeError(
                f"Invalid type ({right.__class__}) for binary operation with {left.__class__}"
            )


if __name__ == "__main__":
    from spatialmath import SO2

    C = SO2(0.5)
    A = np.array([[10, 0], [0, 1]])

    print(C * A)
    print(C * A * C.inv())
    print(C.conjugation(A))

    # x = SE3.Rand(N=6)

    # x.printline(orient="rpy/xyz", fmt="{:8.3g}")

    # d = np.diag([0.25, 0.25, 1])
    # a = SE2()
    # print(a)
    # print(d * a)

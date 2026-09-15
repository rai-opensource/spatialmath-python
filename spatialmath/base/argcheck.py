# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE


"""
Utility functions for testing and converting passed arguments.  Used in all
spatialmath functions and classes to provides for flexibility in argument types
that can be passed.
"""

# pylint: disable=invalid-name

import math
import numpy as np
from collections.abc import Iterable

# from spatialmath.base import symbolic as sym # HACK
from spatialmath.base.symbolic import issymbol, symtype

# valid scalar types
_scalartypes = (int, np.integer, float, np.floating) + symtype

# from typing import Union, List, Tuple, Any, Optional, Type, Callable
# from numpy.typing import DTypeLike
# Array = np.ndarray[Any, np.dtype[np.floating]]
# ArrayLike = Union[float,List[float],Tuple,Array]  # various ways to represent R^3 for input

from spatialmath.base.types import *


def isscalar(x: Any) -> bool:
    """
    Test if argument is a real scalar

    :param x: value to test
    :return: whether value is a scalar
    :rtype: bool

    ``isscalar(x)`` is ``True`` if ``x`` is a Python or numPy int or real float.

    .. runblock:: pycon

        >>> from spatialmath.base import isscalar
        >>> isscalar(1)
        >>> isscalar(1.2)
        >>> isscalar([1])

    """
    return isinstance(x, _scalartypes)


def isinteger(x: Any) -> bool:
    """
    Test if argument is a scalar integer

    :param x: value to test
    :return: whether value is a scalar
    :rtype: bool

    ``isinteger(x)`` is ``True`` if ``x`` is a Python or numPy int or real float.

    .. runblock:: pycon

        >>> from spatialmath.base import isinteger
        >>> isinteger(1)
        >>> isinteger(1.2)

    """
    return isinstance(x, (int, np.integer))


def assertmatrix(
    m: Any, shape: Tuple[Union[int, None], Union[int, None]] = (None, None)
) -> None:
    """
    Assert that argument is a 2D matrix

    :param m: value to test
    :param shape: required shape
    :type shape: 2-tuple
    :raises TypeError: if value is not a real Numpy array
    :raises ValueError: if value is not of the specified shape

    Tests if the argument is a real 2D matrix with a specified shape ``shape``
    but the value ``None`` indicate an unspecified (wildcard, don't care)
    dimension.

    - ``assertsmatrix(A)`` raises an exception if ``m`` is not convertible to
      a 2D array
    - ``assertsmatrix(A, (N,M))`` as above but ``m`` must have shape
      (``N``,``M``)
    - ``assertsmatrix(A, (N,None))`` as above but ``m`` must have ``N`` rows
    - ``assertsmatrix(A, (None,M))`` as above but ``m`` must have ``M`` columns

    :seealso: :func:`ismatrix`
    """

    if not isinstance(m, np.ndarray):
        raise TypeError("input must be a numPy ndarray")
    if m.dtype.kind == "c":
        raise TypeError("input must be a real numPy ndarray")
    if shape is not None:
        if len(shape) != len(m.shape):
            raise ValueError(
                "incorrect scalar of matrix dimensions, expecting {}, got {}".format(
                    shape, m.shape
                )
            )
        if shape[0] is not None and shape[0] > 0 and shape[0] != m.shape[0]:
            raise ValueError(
                "incorrect matrix dimensions, expecting {}, got {}".format(
                    shape, m.shape
                )
            )
        if (
            len(shape) > 1
            and shape[1] is not None
            and shape[1] > 0
            and shape[1] != m.shape[1]
        ):
            raise ValueError(
                "incorrect matrix dimensions, expecting {}, got {}".format(
                    shape, m.shape
                )
            )


def ismatrix(m: Any, shape: Tuple[Union[int, None], Union[int, None]]) -> bool:
    """
    Test if argument is a real 2D matrix

    :param m: value to test
    :param shape: required shape
    :type shape: 2-tuple
    :return: True if value is of specified shape :rtype: bool

    Tests if the argument is a real 2D matrix with a specified shape ``shape``
    but the value ``None`` indicate an unspecified (wildcard, don't care)
    dimension, for example:

    .. runblock:: pycon

        >>> from spatialmath.base import ismatrix
        >>> import numpy as np
        >>> A = np.zeros((2,3))
        >>> ismatrix(A, (2,3))
        >>> ismatrix(A, (None,3))
        >>> ismatrix(A, (2,None))
        >>> ismatrix(A, (2,4))

    .. note:: Unlike ``verifymatrix`` this function: - checks the argument is
                real valued - allows the shape to have an unspecified dimension

    :seealso: :func:`getmatrix`, :func:`verifymatrix`, :func:`assertmatrix`
    """
    if not isinstance(m, np.ndarray):
        return False
    if m.dtype.kind == "c":
        return False
    if len(shape) != len(m.shape):
        return False
    if shape[0] is not None and shape[0] > 0 and shape[0] != m.shape[0]:
        return False
    if shape[1] is not None and shape[1] > 0 and shape[1] != m.shape[1]:
        return False
    return True


def getmatrix(
    m: ArrayLike,
    shape: Tuple[Union[int, None], Union[int, None]],
    dtype: DTypeLike = np.float64,
) -> np.ndarray:
    r"""
    Convert argument to 2D array

    :param m: input value
    :param shape: shape of returned matrix
    :type shape: 2-tuple
    :raises ValueError: if ``m`` is inconsistent with ``shape``
    :raises TypeError: if ``m`` is not required type
    :return: a 2D array
    :rtype: NumPy ndarray
    :raises TypeError: if value is not a scalar or Numpy array
    :raises ValueError: if value is not of the specified shape

    ``getmatrix(m, shape)`` is a 2D matrix with shape ``shape`` formed from
    ``m`` which can be a 2D array, 1D array-like or a scalar.

    .. runblock:: pycon

        >>> from spatialmath.base import getmatrix
        >>> import numpy as np
        >>> getmatrix(3, (1,1))
        >>> getmatrix([3,4], (1,2))
        >>> getmatrix([3,4], (2, 1))
        >>> getmatrix([3,4,5,6], (2,2))
        >>> getmatrix(np.r_[3,4,5,6], (2,2))

    .. note::

       - If ``m`` is a 2D array its shape is compared to ``shape`` - a 2-tuple
         where ``None`` stands for unspecified, ie. ``(None, 2)`` will match
         any array where the second dimension is 2.
       - If ``m`` is a 1D array its shape is checked to see if it can be
         reshaped to ``shape``.  A n-array could be reshaped as (n,1) or (1,n)
         or any other shape with the correct number of elements.  A value of
         ``None`` in the shape stands for unspecified, ie. ``(None, 2)`` will
         attempt to reshape ``m`` as an array with shape (k,2) where :math:`k \times 2 \eq n`.
       - If ``m`` is a scalar, return an array of shape (1,1)

    :seealso: :func:`ismatrix`, :func:`verifymatrix`
    :SymPy: supported
    """
    if isinstance(m, np.ndarray) and len(m.shape) == 2:
        # passed a 2D array
        mshape = m.shape

        if m.dtype == "O":
            dtype = "O"

        if (shape[0] is None or shape[0] == mshape[0]) and (
            shape[1] is None or shape[1] == mshape[1]
        ):
            return np.array(m, dtype=dtype)
        else:
            raise ValueError(f"expecting {shape} but got {mshape}")

    elif isvector(m):
        # passed a 1D array
        m = getvector(m, dtype=dtype, out="array")
        if shape[0] is not None and shape[1] is not None:
            shape = cast(Tuple[int, int], shape)
            if len(m) == np.prod(shape):
                return m.reshape(shape)
            else:
                raise ValueError("array cannot be reshaped")
        elif shape[0] is not None and shape[1] is None:
            return m.reshape((shape[0], -1))
        elif shape[0] is None and shape[1] is not None:
            return m.reshape((-1, shape[1]))
        else:
            return m.reshape((1, -1))

    else:
        raise TypeError("argument must be scalar or ndarray")


def verifymatrix(
    m: np.ndarray, shape: Tuple[Union[int, None], Union[int, None]]
) -> None:
    """
    Assert that argument is array of specified size

    :param m: value to be tested
    :param shape: desired shape of value
    :type shape: 2-tuple
    :raises TypeError: argument is not a NumPy array
    :raises ValueError: argument has incorrect shape

    Raises an exception if the argument ``m`` is not a NumPy array of the
    specified shape.

    .. note:: Unlike ``assertmatrix`` the specified shape cannot have wildcard
              dimensions.

    :seealso: :func:`assertmatrix`,:func:`getmatrix`, :func:`ismatrix`
    """
    if not isinstance(m, np.ndarray):
        raise TypeError("input must be a numPy ndarray")

    if not m.shape == shape:
        raise ValueError("incorrect matrix dimensions, expecting {0}".format(shape))


# and not np.iscomplex(m) checks every element, would need to be not np.any(np.iscomplex(m)) which seems expensive


@overload
def getvector(
    v: ArrayLike,
    dim: Optional[Union[int, None]] = None,
    out: str = "array",
    dtype: DTypeLike = np.float64,
) -> NDArray:
    ...


@overload
def getvector(
    v: ArrayLike,
    dim: Optional[Union[int, None]] = None,
    out: str = "list",
    dtype: DTypeLike = np.float64,
) -> List[float]:
    ...


@overload
def getvector(
    v: Tuple[float, ...],
    dim: Optional[Union[int, None]] = None,
    out: str = "sequence",
    dtype: DTypeLike = np.float64,
) -> Tuple[float, ...]:
    ...


@overload
def getvector(
    v: List[float],
    dim: Optional[Union[int, None]] = None,
    out: str = "sequence",
    dtype: DTypeLike = np.float64,
) -> List[float]:
    ...


def getvector(
    v: ArrayLike,
    dim: Optional[Union[int, None]] = None,
    out: str = "array",
    dtype: DTypeLike = np.float64,
) -> Union[NDArray, List[float], Tuple[float, ...]]:
    """
    Return a vector value

    :param v: passed vector
    :param dim: required dimension, or None if any length is ok
    :type dim: int or None
    :param out: output format, default is 'array'
    :type out: str
    :param dtype: datatype for numPy array return (default np.float64)
    :type dtype: numPy type
    :return: vector value in specified format
    :raises TypeError: value is not a list or NumPy array
    :raises ValueError: incorrect number of elements

    - ``getvector(vec)`` is ``vec`` converted to the output format ``out``
      where ``vec`` is any of:

        - a Python native int or float, a 1-vector
        - Python native list or tuple
        - numPy real 1D array, ie. shape=(N,)
        - numPy real 2D array with a singleton dimension, ie. shape=(1,N)
          or (N,1)

    - ``getvector(vec, N)`` as above but must be an ``N``-element vector.

    The returned vector will be in the format specified by ``out``:

    ==========  ===============================================
    format      return type
    ==========  ===============================================
    'sequence'  Python list, or tuple if a tuple was passed in
    'list'      Python list
    'array'     1D numPy array, shape=(N,)  [default]
    'row'       row vector, a 2D numPy array, shape=(1,N)
    'col'       column vector, 2D numPy array, shape=(N,1)
    ==========  ===============================================

    .. runblock:: pycon

        >>> from spatialmath.base import getvector
        >>> import numpy as np
        >>> getvector([1,2])  # list
        >>> getvector([1,2], out='row')  # list
        >>> getvector([1,2], out='col')  # list
        >>> getvector((1,2))  # tuple
        >>> getvector(np.r_[1,2,3], out='sequence')  # numpy array
        >>> getvector(1)  # scalar
        >>> getvector([1])
        >>> getvector([[1]])
        >>> getvector([1,2], 2)
        >>> # getvector([1,2], 3)  --> ValueError

    .. note::
        - For 'array', 'row' or 'col' output the NumPy dtype defaults to the
          ``dtype`` of ``v`` if it is a NumPy array, otherwise it is
          set to the value specified by the ``dtype`` keyword which defaults
          to ``np.float64``.
        - If ``v`` is symbolic the ``dtype`` is retained as ``'O'``

    :seealso: :func:`isvector`
    """
    dt = dtype

    if isinstance(v, _scalartypes):  # handle scalar case
        v = [v]  # type: ignore
    if isinstance(v, (list, tuple)):
        # list or tuple was passed in

        if issymbol(v):
            dt = None

        if dim is not None and v and len(v) != dim:
            raise ValueError(
                "incorrect vector length: expected {}, got {}".format(dim, len(v))
            )
        if out == "sequence":
            return v
        elif out == "list":
            return list(v)
        elif out == "array":
            return np.array(v, dtype=dt)
        elif out == "row":
            return np.array(v, dtype=dt).reshape(1, -1)
        elif out == "col":
            return np.array(v, dtype=dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")

    elif isinstance(v, np.ndarray):
        s = v.shape
        if dim is not None:
            if not (s == (dim,) or s == (1, dim) or s == (dim, 1)):
                raise ValueError(
                    "incorrect vector length: expected {}, got {}".format(dim, s)
                )

        v = v.flatten()

        if v.dtype.kind == "O":
            dt = "O"

        if out in ("sequence", "list"):
            return list(v.flatten())
        elif out == "array":
            return v.astype(dt)
        elif out == "row":
            return v.astype(dt).reshape(1, -1)
        elif out == "col":
            return v.astype(dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    else:
        raise TypeError("invalid input type")


def assertvector(
    v: Any, dim: Optional[Union[int, None]] = None, msg: Optional[str] = None
) -> None:
    """
    Assert that argument is a real vector

    :param v: passed vector
    :param dim: required dimension
    :type dim: int or None
    :raises ValueError: if not a vector of specified length

    - ``assertvector(vec)`` raise an exception if ``vec`` is not a vector, ie.
      it is not any of:

        - a Python native int or float, a 1-vector
        - Python native list or tuple
        - numPy real 1D array, ie. shape=(N,)
        - numPy real 2D array with a singleton dimension, ie. shape=(1,N)
          or (N,1)

    - ``assertvector(vec, N)`` as above but must also check the length is ``N``.

    :seealso: :func:`getvector`, :func:`isvector`
    """
    if not isvector(v, dim):
        raise ValueError(msg)


def isvector(v: Any, dim: Optional[int] = None) -> bool:
    """
    Test if argument is a real vector

    :param v: value to test
    :param dim: required dimension
    :type dim: int or None
    :return: whether value is a valid vector
    :rtype: bool

    - ``isvector(vec)`` is ``True`` if ``vec`` is a vector, ie. any of:

        - a Python native int or float, a 1-vector
        - Python native list or tuple
        - numPy real 1D array, ie. shape=(N,)
        - numPy real 2D array with a singleton dimension, ie. shape=(1,N)
          or (N,1)

    - ``isvector(vec, N)`` as above but must also be an ``N``-element vector.

    .. runblock:: pycon

        >>> from spatialmath.base import isvector
        >>> import numpy as np
        >>> isvector([1,2])  # list
        >>> isvector((1,2))  # tuple
        >>> isvector(np.r_[1,2,3])  # numpy array
        >>> isvector(1)  # scalar
        >>> isvector([1,2], 3)  # list

    :seealso: :func:`getvector`, :func:`assertvector`
    """
    if (
        isinstance(v, (list, tuple))
        and (dim is None or len(v) == dim)
        and all(map(lambda x: isinstance(x, _scalartypes), v))
    ):
        return True  # list or tuple

    if isinstance(v, np.ndarray):
        s = v.shape
        if dim is None:
            return (
                (len(s) == 1 and s[0] > 0)
                or (s[0] == 1 and s[1] > 0)
                or (s[0] > 0 and s[1] == 1)
            )
        else:
            return s == (dim,) or s == (1, dim) or s == (dim, 1)

    if (dim is None or dim == 1) and isinstance(v, _scalartypes):
        return True

    return False


def getunit(
    v: ArrayLike, unit: str = "rad", dim: Optional[int] = None, vector: bool = True
) -> Union[float, NDArray]:
    """
    Convert values according to angular units

    :param v: the value in radians or degrees
    :type v: array_like(m)
    :param unit: the angular unit, "rad" or "deg"
    :type unit: str
    :param dim: expected dimension of input, defaults to don't check (None)
    :type dim: int, optional
    :param vector: return a scalar as a 1d vector, defaults to True
    :type vector: bool, optional
    :return: the converted value in radians
    :rtype: ndarray(m) or float
    :raises ValueError: argument is not a valid angular unit

    The input value is assumed to be in units of ``unit`` and is converted to radians.

    .. runblock:: pycon

        >>> from spatialmath.base import getunit
        >>> import numpy as np
        >>> getunit(1.5, 'rad')
        >>> getunit(90, 'deg')
        >>> getunit(90, 'deg', vector=False) # force a scalar output
        >>> getunit(1.5, 'rad', dim=0) # check argument is scalar
        >>> getunit(1.5, 'rad', dim=3) # check argument is a 3-vector
        >>> getunit([1.5], 'rad', dim=1) # check argument is a 1-vector
        >>> getunit([1.5], 'rad', dim=3) # check argument is a 3-vector
        >>> getunit([90, 180], 'deg')
        >>> getunit(np.r_[90, 180], 'deg')
        >>> getunit(np.r_[90, 180], 'deg', dim=2) # check argument is a 2-vector
        >>> getunit([90, 180], 'deg', dim=3) # check argument is a 3-vector

    :note:
        - the input value is processed by :func:`getvector` and the argument ``dim`` can
          be used to check that ``v`` is the desired length.  Note that 0 means a scalar,
          whereas 1 means a 1-element array.
        - the output is always an ndarray except if the input is a scalar and ``vector=False``.

    :seealso: :func:`getvector`
    """
    if not isinstance(v, Iterable):
        # scalar input
        if dim is not None and dim != 0:
            raise ValueError("for dim==0 input must be a scalar")
        if vector:
            # scalar in, vector out
            if unit == "deg":
                v = np.deg2rad(v)
            elif unit != "rad":
                raise ValueError("invalid angular units")
            return np.array([v])
        else:
            # scalar in, scalar out
            if unit == "rad":
                return v
            elif unit == "deg":
                return np.deg2rad(v)
            else:
                raise ValueError("invalid angular units")

    else:
        # scalar or iterable in, ndarray out
        # iterable passed in
        if dim == 0:
            raise ValueError("for dim==0 input must be a scalar")
        v = getvector(v, dim=dim)
        if unit == "rad":
            return v
        elif unit == "deg":
            return np.deg2rad(v)
        else:
            raise ValueError("invalid angular units")


def isnumberlist(x: Any) -> bool:
    """
    Test if argument is a list of scalars

    :param x: the value to test
    :return: True if the argument is a list of real scalars
    :rtype: bool

    ``isscalarlist(x)`` is ``True`` if ``x```` is a list of scalars.

    .. runblock:: pycon

        >>> from spatialmath.base import isnumberlist
        >>> import numpy as np
        >>> isnumberlist((1,2,3))
        >>> isnumberlist([1.1, 2.2, 3.3])
        >>> isnumberlist(1)
        >>> isnumberlist(np.r_[1,2])
    """

    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(map(lambda x: isinstance(x, _scalartypes), x))
    )


def isvectorlist(x: Any, n: int) -> bool:
    """
    Test if argument is a list of vectors

    :param x: the value to test
    :return: True if the argument is a list of n-vectors
    :rtype: bool

    ``isvectorlist(x, n)`` is ``True`` if ``x`` is a list or tuple of
    1D numPy arrays of shape=(n,).


    .. runblock:: pycon

        >>> from spatialmath.base import isvectorlist
        >>> import numpy as np
        >>> isvectorlist([np.r_[1,2], np.r_[3,4], np.r_[5,6]], 2)
        >>> isvectorlist([(1,2), (3,4), (5,6)], 2)
        >>> isvectorlist([np.r_[1,2], np.r_[3,4], np.r_[5,6,7]], 2)
    """
    return islistof(x, lambda x: isinstance(x, np.ndarray) and x.shape == (n,))


def islistof(value: Any, what: Union[Type, Callable], n: Optional[int] = None):
    """
    Test if argument is a list of specified type

    :param value: the value to test
    :type value: list or tuple
    :param what: type, tuple of types or function
    :type what: type or callable
    :param n: length of list, defaults to None
    :type n: int, optional
    :return: whether ``value`` is a specified list
    :rtype: bool

    Tests that every element of ``value`` is of the desired type.  The type
    is specified by ``what`` and can be:

    * a single type, eg. ``int``
    * a tuple of types, eg. ``(int, float)``
    * a reference to a function which is passed each elemnent of the list and
      returns True if it is a valid member of the list.

    The length of the list can also be tested by specifying the argument ``n``.

    .. runblock:: pycon

        >>> from spatialmath.base import islistof
        >>> a = [3, 4, 5]
        >>> islistof(a, int)
        >>> islistof(a, int, 2)
        >>> a = [3, 4.5, 5.6]
        >>> islistof(a, int)
        >>> islistof(a, (int, float))
        >>> a = [[1,2], [3, 4], [5,6]]
        >>> islistof(a, lambda x: islistof(x, int, 2))
    """
    if not isinstance(value, (list, tuple)):
        return False
    if n is not None and len(value) != n:
        return False

    if isinstance(what, type) or isinstance(what, tuple):
        # it's a type or tuple of types
        return all([isinstance(x, what) for x in value])
    elif callable(what):
        return all([what(x) for x in value])
    else:
        raise ValueError("bad value of what")


if __name__ == "__main__":
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_argcheck.py"
        ).read()
    )  # pylint: disable=exec-used

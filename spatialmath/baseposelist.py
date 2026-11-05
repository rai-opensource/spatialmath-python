"""
Provide list super powers for spatial math objects.
"""

# pylint: disable=invalid-name
from __future__ import annotations
from collections import UserList
from abc import ABC, abstractproperty, abstractstaticmethod
import copy
import numpy as np
from spatialmath.base.argcheck import isnumberlist, isscalar
from spatialmath.base.types import *

_numtypes = (int, np.int64, float, np.float64)


class BasePoseList(UserList, ABC):
    """
    List properties for spatial math classes

    Each of the spatial math classes behaves like a regular Python object and
    an instance contains a value of a particular type, for example an SE(3)
    matrix, a unit quaternion, a twist etc.

    This class adds list-like capabilities to each of spatial math classes.  This
    means that an instance is not limited to holding just a single value (a
    singleton instance), it can hold a list of values.  That list can contain
    zero or more items.  This is helpful for:

    - storing sequences (trajectories) where it is important to know that all
      elements in the sequence are of the same time and have valid values
    - arrays of the same type to enable C++ like programming patterns

    This class inherits from ``collections.UserList`` and wraps those list-like
    methods in spatial math specific ways.  The list operations supported are:

    ==================  ============================================================
    syntax              meaning
    ==================  ============================================================
    ``C()``             create a singleton instance of ``C`` with the identity value
    ``C.Empty()``       create an instance of ``C`` with zero items
    ``C.Alloc(n)``      create an instance of ``C`` with ``n`` identity items
    ``len(x)``          return the number of items in ``x``
    ``x[i]``            return the ``i``'th item of ``x``, ``i`` is an index
                        or a slice.
    ``x[i] = y``        set the ``i``'th item of ``x`` to the singleton instance
                        ``y`` and ``i`` is an index
    ``x.append(y)``     append the value of singleton instance ``y`` to ``x``
    ``x.extend(y)``     append the items of ``y`` to ``x``
    ``x.pop()``         pop the first item of ``x``
    ``x.insert(i, y)``  insert the value of singleton instsance ``y`` into ``x``
                        at position ``i``.
    ``del x[i]``        delete the ``i``'th element of ``x``
    ``x.reverse()``     reverse the elements of ``x`` in place
    ``x.clear()``       remove all items from ``x``
    ==================  ============================================================

    where ``C`` is the class, and ``x`` and ``y`` are instances of ``C``.

    Notes:

    - The subclass must invoke ``super().__init__()``
    - ``UserList`` keeps the list in the ``.data`` attribute
    - Some list method do not make sense for spatial math, these are:
      ``count``, ``remove`` and ``sort``.
    """

    @abstractproperty
    def shape(self):
        pass

    @staticmethod
    @abstractstaticmethod
    def isvalid(x, check=True):
        pass

    @abstractstaticmethod
    def _identity():
        pass

    def _import(self, x, check=True):
        if not check or self.isvalid(x, check=check):
            return x
        else:
            return None

    @classmethod
    def Empty(cls) -> Self:
        """
        Construct an empty instance (BasePoseList superclass method)

        :return: pose instance with zero values

        Example::

            >>> x = X.Empty()
            >>> len(x)
            0

        where ``X`` is any of the SMTB classes.
        """
        x = cls()
        x.data = []
        return x

    @classmethod
    def Alloc(cls, n: Optional[int] = 1) -> Self:
        """
        Construct an instance with N default values (BasePoseList superclass method)

        :param n: Number of values, defaults to 1
        :type n: int, optional
        :return: pose instance with ``n`` default values

        ``X.Alloc(N)`` creates an instance of the pose class ``X`` with ``N``
        default values, ie. ``len(X)`` will be ``N``.

        ``X`` can be considered a vector of pose objects, and those elements
        can be referenced ``X[i]`` or assigned to ``X[i] = ...``.

        .. note:: The default value depends on the pose class and is the result
                  of the empty constructor. For ``SO2``,
                  ``SE2``, ``SO3``, ``SE3`` it is an identity matrix, for a
                  twist class ``Twist2`` or ``Twist3`` it is a zero vector,
                  for a ``UnitQuaternion`` or ``Quaternion`` it is a zero
                  vector.

        Example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10

        where ``X`` is any of the SMTB classes.
        """
        x = cls()
        x.data = [cls._identity() for i in range(n)]  # make n copies of the data
        return x

    def arghandler(
        self, arg: Any, convertfrom: Tuple = (), check: Optional[bool] = True
    ) -> bool:
        """
        Standard constructor support (BasePoseList superclass method)

        :param arg: initial value
        :param convertfrom: list of classes to accept and convert from
        :type: tuple of typles
        :param check: check value is valid, defaults to True
        :type check: bool
        :raises ValueError: bad type passed

        The value ``arg`` can be any of:

        #. None, an identity value is created
        #. a numpy.ndarray of the appropriate shape and value which is valid for the subclass
        #. a stacked numpy.ndarray of shape (N, r, c), where (r, c) is the shape of one value
           of the subclass, which is a sequence of N values
        #. a list whose elements all meet the criteria above
        #. an instance of the subclass
        #. a list whose elements are all singelton instances of the subclass

        For cases 2 and 3, a NumPy array or a list of NumPy array is passed.
        Each NumPyarray is tested for validity (if ``check`` is False a cursory
        check of shape is made, if ``check`` is True the numerical value is
        inspected) and converted to the required internal format by the
        ``_import`` method. The default ``_import`` method calls the ``isvalid``
        method for checking.  This mechanism allows equivalent forms to be
        passed, ie. 6x1 or 4x4 for an se(3).

        If ``self`` is an instance of class ``A``, and an instance of class
        ``B`` is passed and ``B`` is an element of the ``convertfrom`` argument,
        then ``B.A()`` will be invoked to perform the type conversion.

        Examples::

            SE3()
            SE3(np.identity(4))
            SE3([np.identity(4), np.identity(4)])
            SE3(SE3())
            SE3([SE3(), SE3()])
            Twist3(SE3())
        """

        if arg is None:
            # empty constructor
            self.data = [self._identity()]

        elif isinstance(arg, np.ndarray):
            # it's a numpy array

            shape = tuple(self.shape)
            if len(shape) == 2 and arg.ndim == 3:
                # a 3D array passed to a matrix pose class is a stack
                if arg.shape[1:] != shape:
                    # a stack of matrices of the wrong size is rejected,
                    # whatever the value of check
                    return False
                # a stacked (N, r, c) array is a sequence of N values
                values = [self._import(x, check=check) for x in arg]
                if any(x is None for x in values):
                    return False
                self.data = values
                return True

            x = self._import(arg, check=check)
            if x is not None:
                self.data = [x]
            else:
                return False

        elif isinstance(arg, (list, tuple)):
            # it's a list of things
            if isinstance(arg[0], np.ndarray):
                # possibly a list of numpy arrays
                self.data = [self._import(x, check=check) for x in arg]

            elif type(arg[0]) == type(self):
                # possibly a list of objects of same type
                assert all(
                    map(lambda x: type(x) == type(self), arg)
                ), "elements of list are incorrect type"
                self.data = [x.A for x in arg]

            elif (
                isnumberlist(arg) and len(self.shape) == 1 and len(arg) == self.shape[0]
            ):
                self.data = [np.array(arg)]

            else:
                # see what NumPy makes of it
                X = np.array(arg)
                if X.shape == self.shape:
                    self.data = [X]
                else:
                    # no idea what was passed
                    return False

        elif isinstance(arg, self.__class__):
            # instance of same type, clone it
            self.data = copy.copy(arg.data)

        elif arg.__class__ in convertfrom:
            # see if we can convert passed argument to this type
            #  only support class instance
            try:
                # get method to convert from arg to self types
                converter = getattr(arg.__class__, type(self).__name__)
            except AttributeError:
                raise ValueError(
                    "argument has no conversion method to this type"
                ) from None
            self.data = [converter(arg).A]

        else:
            # don't know this argument, let object __init__ deal with it
            return False

        return True

    @property
    def __array_interface__(self):
        """
        Copies the numpy array interface from the first numpy array
        so that C extenstions with this spatial math class have direct
        access to the underlying numpy array
        """
        return self.data[0].__array_interface__

    @property
    def _A(self) -> Union[List[NDArray], NDArray]:
        """
        Spatial vector as an array
        :return: Moment vector
        :rtype: numpy.ndarray, shape=(3,)
        - ``X.v`` is a 3-vector
        """
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    @property
    def A(self) -> Union[List[NDArray], NDArray]:
        """
        Array value of an instance (BasePoseList superclass method)

        :return: NumPy array value of this instance
        :rtype: ndarray

        - ``X.A`` is a NumPy array that represents the value of this instance,
          and has a shape given by ``X.shape``.

        .. note:: This assumes that ``len(X)`` == 1, ie. it is a single-valued
            instance.
        """

        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    # ------------------------------------------------------------------------ #

    def __getitem__(self, i: Union[int, slice]) -> BasePoseList:
        """
        Access value of an instance (BasePoseList superclass method)

        :param i: index of element to return
        :type i: int
        :return: the specific element of the pose
        :rtype: Quaternion or UnitQuaternion instance
        :raises IndexError: if the element is out of bounds

        Note that only a single index is supported, slices are not.

        Example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10
            >>> y = x[1]
            >>> len(y)
            1
            >>> y = x[1:5]
            >>> len(y)
            4

        where ``X`` is any of the SMTB classes.
        """

        if isinstance(i, slice):
            if i.stop is None:
                # stop not given
                end = len(self)
            elif i.stop < 0:
                # stop is negative, -
                end = i.stop + len(self) + 1
            else:
                # stop is positive, use it directly
                end = i.stop
            return self.__class__(
                [self.data[k] for k in range(i.start or 0, end, i.step or 1)]
            )
        else:
            ret = self.__class__(self.data[i], check=False)
            # ret.__array_interface__ = self.data[i].__array_interface__
            return ret
            # return self.__class__(self.data[i], check=False)

    def __setitem__(self, i: int, value: BasePoseList) -> None:
        """
        Assign a value to an instance (BasePoseList superclass method)

        :param i: index of element to assign to
        :type i: int
        :param value: the value to insert
        :type value: Quaternion or UnitQuaternion instance
        :raises ValueError: incorrect type of assigned value

        Assign the argument to an element of the object's internal list of values.
        This supports the assignement operator, for example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10
            >>> x[3] = X()   # assign to position 3 in the list

        where ``X`` is any of the SMTB classes.

        """
        if not type(self) == type(value):
            raise ValueError("can't insert different type of object")
        if len(value) > 1:
            raise ValueError(
                "can't insert a multivalued element - must have len() == 1"
            )
        self.data[i] = value.A

    # flag these binary operators as being not supported
    def __lt__(self, other: BasePoseList) -> Type[Exception]:
        return NotImplementedError

    def __le__(self, other: BasePoseList) -> Type[Exception]:
        return NotImplementedError

    def __gt__(self, other: BasePoseList) -> Type[Exception]:
        return NotImplementedError

    def __ge__(self, other: BasePoseList) -> Type[Exception]:
        return NotImplementedError

    def append(self, item: BasePoseList) -> None:
        """
        Append a value to an instance (BasePoseList superclass method)

        :param x: the value to append
        :type x: Quaternion or UnitQuaternion instance
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list of values.

        Example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10
            >>> x.append(X())   # append to the list
            >>> len(x)
            11

        where ``X`` is any of the SMTB classes.
        """
        # print('in append method')
        if not type(self) == type(item):
            raise ValueError("can't append different type of object")
        if len(item) > 1:
            raise ValueError("can't append a multivalued instance - use extend")
        super().append(item.A)

    def extend(self, iterable: BasePoseList) -> None:
        """
        Extend sequence of values in an instance (BasePoseList superclass method)

        :param x: the value to extend
        :type x: instance of same type
        :raises ValueError: incorrect type of appended object

        Appends the argument's values to the object's internal list of values.

        Example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10
            >>> x.append(X.Alloc(5))   # extend the list
            >>> len(x)
            15

        where ``X`` is any of the SMTB classes.
        """
        # print('in extend method')
        if not type(self) == type(iterable):
            raise ValueError("can't append different type of object")
        super().extend(iterable._A)

    def insert(self, i: int, item: BasePoseList) -> None:
        """
        Insert a value to an instance (BasePoseList superclass method)

        :param i: element to insert value before
        :type i: int
        :param item: the value to insert
        :type item: instance of same type
        :raises ValueError: incorrect type of inserted value

        Inserts the argument into the object's internal list of values.

        Example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10
            >>> x.insert(0, X())   # insert at start of list
            >>> len(x)
            11
            >>> x.insert(10, X())   # append to the list
            >>> len(x)
            11

        where ``X`` is any of the SMTB classes.

        .. note:: If ``i`` is beyond the end of the list, the item is appended
            to the list
        """
        if not type(self) == type(item):
            raise ValueError("can't insert different type of object")
        if len(item) > 1:
            raise ValueError(
                "can't insert a multivalued instance - must have len() == 1"
            )
        super().insert(i, item._A)

    def pop(self, i: Optional[int] = -1) -> Self:
        """
        Pop value from an instance (BasePoseList superclass method)

        :param i: item in the list to pop, default is last
        :type i: int
        :return: the popped value
        :rtype: instance of same type
        :raises IndexError: if there are no values to pop

        Removes a value from the value list and returns it.  The original
        instance is modified.

        Example::

            >>> x = X.Alloc(10)
            >>> len(x)
            10
            >>> y = x.pop()  # pop the last value x[9]
            >>> len(x)
            9
            >>> y = x.pop(0)  # pop the first value x[0]
            >>> len(x)
            8

        where ``X`` is any of the SMTB classes.
        """
        return self.__class__(super().pop(i))

    def binop(
        self,
        right: BasePoseList,
        op: Callable,
        op2: Optional[Callable] = None,
        list1: Optional[bool] = True,
    ) -> List:
        """
        Perform binary operation

        :param left: left operand
        :type left: BasePoseList subclass
        :param right: right operand
        :type right: BasePoseList subclass, scalar or array
        :param op: binary operation
        :type op: callable
        :param op2: binary operation
        :type op2: callable
        :param list1: return single array as a list, default True
        :type list1: bool
        :raises ValueError: arguments are not compatible
        :return: list of values
        :rtype: list

        The is a helper method for implementing binary operation with overloaded
        operators such as ``X * Y`` where ``X`` and ``Y`` are both subclasses
        of ``BasePoseList``.  Each operand has a list of one or more
        values and this methods computes a list of result values according to:

        =========   ==========   ====  ===================================
              Inputs                    Output
        ----------------------   -----------------------------------------
        len(left)   len(right)   len     operation
        =========   ==========   ====  ===================================
         1          1             1    ``ret = op(left, right)``
         1          M             M    ``ret[i] = op(left, right[i])``
         M          1             M    ``ret[i] = op(left[i], right)``
         M          M             M    ``ret[i] = op(left[i], right[i])``
        =========   ==========   ====  ===================================

        The arguments to ``op`` are the internal numeric values, ie. as returned
        by the ``._A`` property.

        The result is always a list, except for the first case above and
        ``list1`` is ``False``.

        If the right operand is not a ``BasePoseList`` subclass, but is a numeric
        scalar or array then then ``op2`` is invoked

        For example::

            X._binop(Y, lambda x, y: x + y)

        =========   ====  ===================================
          Input                    Output
        ---------   -----------------------------------------
        len(left)   len     operation
        =========   ====  ===================================
         1           1    ``ret = op2(left, right)``
         M           M    ``ret[i] = op2(left[i], right)``
        =========   ====  ===================================

        There is no check on the shape of ``right`` if it is an array.
        The result is always a list, except for the first case above and
        ``list1`` is ``False``.
        """
        left = self

        # class * class
        if len(left) == 1:
            # singleton *
            if isscalar(right):
                if list1:
                    return [op(left._A, right)]
                else:
                    return op(left.A, right)
            elif len(right) == 1:
                # singleton * singleton
                if list1:
                    return [op(left._A, right._A)]
                else:
                    return op(left.A, right.A)
            else:
                # singleton * non-singleton
                return [op(left.A, x) for x in right.A]
        else:
            # non-singleton *
            if isscalar(right):
                return [op(x, right) for x in left.A]
            elif len(right) == 1:
                # non-singleton * singleton
                return [op(x, right.A) for x in left.A]
            elif len(left) == len(right):
                # non-singleton * non-singleton
                return [op(x, y) for (x, y) in zip(left.A, right.A)]
            else:
                raise ValueError("length of lists to == must be same length")

        # if isinstance(right, left.__class__):
        #     # class * class
        #     if len(left) == 1:
        #         # singleton *
        #         if len(right) == 1:
        #             # singleton * singleton
        #             if list1:
        #                 return [op(left._A, right._A)]
        #             else:
        #                 return op(left.A, right.A)
        #         else:
        #             # singleton * non-singleton
        #             return [op(left.A, x) for x in right.A]
        #     else:
        #         # non-singleton *
        #         if len(right) == 1:
        #             # non-singleton * singleton
        #             return [op(x, right.A) for x in left.A]
        #         elif len(left) == len(right):
        #             # non-singleton * non-singleton
        #             return [op(x, y) for (x, y) in zip(left.A, right.A)]
        #         else:
        #             raise ValueError('length of lists to == must be same length')
        # elif op2 is not None and isinstance(right, _numtypes) or (isinstance(right, np.ndarray)):
        #     # class * (scalar or array)
        #     if len(left) == 1:
        #         if list1:
        #             return [op2(left.A, right)]
        #         else:
        #             return op2(left.A, right)
        #     else:
        #         return [op(x, right) for x in left.A]

    def unop(
        self, op: Callable, matrix: Optional[bool] = False
    ) -> Union[NDArray, List]:
        """
        Perform unary operation

        :param self: operand
        :type self: BasePoseList subclass
        :param op: unnary operation
        :type op: callable
        :param matrix: return array instead of list, default False
        :type matrix: bool
        :return: operation results
        :rtype: list or NumPy array

        The is a helper method for implementing unary operations where the
        operand has multiple value. This method computes the value of
        the operation for all input values and returns the result as either
        a list or as a matrix which vertically stacks the results.

        =========   ====  ===================================
          Input                     Output
        ---------   -----------------------------------------
        len(self)   len     operation
        =========   ====  ===================================
         1           1    ``ret = op(self)``
         M           M    ``ret[i] = op(self[i])``
         M           M    ``ret[i,;] = op(self[i])``
        =========   ====  ===================================

        The result is:

        - a list of values if ``matrix==False``, or
        - a 2D NumPy stack of values if ``matrix==True``, it is assumed
          that the value is a 1D array.

        """
        if matrix:
            return np.vstack([op(x) for x in self.data])
        else:
            return [op(x) for x in self.data]


if __name__ == "__main__":
    from spatialmath import SO3, SO2

    R = SO3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(R.eulervec())

    R = SO2([0.3, 0.4, 0.5])
    pass

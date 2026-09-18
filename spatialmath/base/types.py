# Type aliases used throughout spatialmath and spatialmath.base.
#
# This used to dispatch across three per-Python-version files
# (_types_35.py, _types_39.py, _types_311.py) via sys.version_info, from
# when Python 3.5-3.10 support coexisted. pyproject.toml has required
# Python >=3.10 since #186, so the only genuinely version-dependent piece
# left is where `Self` lives (typing_extensions before 3.11, typing from
# 3.11 on).

import sys

from typing import (
    overload,
    cast,
    Union,
    List,
    Tuple,
    Type,
    TextIO,
    Any,
    Callable,
    Optional,
    Iterator,
)
from typing import Literal as L

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from numpy import ndarray, dtype, floating
from numpy.typing import NDArray, DTypeLike

# array like

# these are input to many functions in spatialmath.base, and can be a list, tuple or
# ndarray.  The elements are generally float, but some functions accept symbolic
# arguments as well, which leads to a NumPy array with dtype=object. For now
# symbolics will throw a lint error.  Possibly create variants ArrayLikeSym that
# admits symbols and can be used for those functions that accept symbols.
#
# The variants like ArrayLike2 indicate that a list, tuple or ndarray of
# length 2 is expected.  Static checking of tuple length is possible, but not for lists.
# This might be possible in future versions of Python, but for now it is a hint to the
# coder about what is expected

# cannot be a scalar
ArrayLikePure = Union[List[float], Tuple[float, ...], ndarray[Any, dtype[floating]]]

ArrayLike = Union[float, List[float], Tuple[float, ...], ndarray[Any, dtype[floating]]]
ArrayLike2 = Union[
    List[float],
    Tuple[float, float],
    ndarray[
        Tuple[L[2]],
        dtype[floating],
    ],
]
ArrayLike3 = Union[
    List[float],
    Tuple[float, float, float],
    ndarray[
        Tuple[L[3]],
        dtype[floating],
    ],
]
ArrayLike4 = Union[
    List[float],
    Tuple[float, float, float, float],
    ndarray[
        Tuple[L[4]],
        dtype[floating],
    ],
]
ArrayLike6 = Union[
    List[float],
    Tuple[float, float, float, float, float, float],
    ndarray[
        Tuple[L[6]],
        dtype[floating],
    ],
]

# real vectors
R1 = ndarray[
    Tuple[L[1]],
    dtype[floating],
]  # R^1
R2 = ndarray[
    Tuple[L[2]],
    dtype[floating],
]  # R^2
R3 = ndarray[
    Tuple[L[3]],
    dtype[floating],
]  # R^3
R4 = ndarray[
    Tuple[L[4]],
    dtype[floating],
]  # R^4
R6 = ndarray[
    Tuple[L[6]],
    dtype[floating],
]  # R^6

R8 = ndarray[
    Tuple[L[8,]],
    dtype[floating],
]  # R^8

# real matrices
R1x1 = ndarray[Tuple[L[1], L[1]], dtype[floating]]  # R^{1x1} matrix
R2x2 = ndarray[Tuple[L[2], L[2]], dtype[floating]]  # R^{2x2} matrix
R3x3 = ndarray[Tuple[L[3], L[3]], dtype[floating]]  # R^{3x3} matrix
R4x4 = ndarray[Tuple[L[4], L[4]], dtype[floating]]  # R^{4x4} matrix
R6x6 = ndarray[Tuple[L[6], L[6]], dtype[floating]]  # R^{6x6} matrix
R8x8 = ndarray[Tuple[L[8], L[8]], dtype[floating]]  # R^{8x8} matrix
R1x3 = ndarray[Tuple[L[1], L[3]], dtype[floating]]  # R^{1x3} row vector
R3x1 = ndarray[Tuple[L[3], L[1]], dtype[floating]]  # R^{3x1} column vector
R1x2 = ndarray[Tuple[L[1], L[2]], dtype[floating]]  # R^{1x2} row vector
R2x1 = ndarray[Tuple[L[2], L[1]], dtype[floating]]  # R^{2x1} column vector

# Points2 = ndarray[Tuple[L[2, Any]], dtype[floating]]  # R^{2xN} matrix
# Points3 = ndarray[Tuple[L[3, Any]], dtype[floating]]  # R^{2xN} matrix
Points2 = NDArray  # R^{2xN} matrix
Points3 = NDArray  # R^{2xN} matrix

# RNx3 = ndarray[(Any, 3), dtype[floating]]  # R^{Nx3} matrix
RNx3 = NDArray

# Lie group elements
SO2Array = ndarray[Tuple[L[2], L[2]], dtype[floating]]  # SO(2) rotation matrix
SE2Array = ndarray[Tuple[L[3], L[3]], dtype[floating]]  # SE(2) rigid-body transform
SO3Array = ndarray[Tuple[L[3], L[3]], dtype[floating]]  # SO(3) rotation matrix
SE3Array = ndarray[Tuple[L[4], L[4]], dtype[floating]]  # SE(3) rigid-body transform

# Lie algebra elements
so2Array = ndarray[
    Tuple[L[2, 2]], dtype[floating]
]  # so(2) Lie algebra of SO(2), skew-symmetrix matrix
se2Array = ndarray[
    Tuple[L[3, 3]], dtype[floating]
]  # se(2) Lie algebra of SE(2), augmented skew-symmetrix matrix
so3Array = ndarray[
    Tuple[L[3, 3]], dtype[floating]
]  # so(3) Lie algebra of SO(3), skew-symmetrix matrix
se3Array = ndarray[
    Tuple[L[4, 4]], dtype[floating]
]  # se(3) Lie algebra of SE(3), augmented skew-symmetrix matrix

# quaternion arrays
QuaternionArray = ndarray[
    Tuple[L[4,]],
    dtype[floating],
]
UnitQuaternionArray = ndarray[
    Tuple[L[4,]],
    dtype[floating],
]

Rn = Union[R2, R3]

SOnArray = Union[SO2Array, SO3Array]
SEnArray = Union[SE2Array, SE3Array]

sonArray = Union[so2Array, so3Array]
senArray = Union[se2Array, se3Array]

Color = Union[str, ArrayLike3]

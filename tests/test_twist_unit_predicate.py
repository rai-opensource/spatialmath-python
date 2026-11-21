import numpy as np
import pytest
from spatialmath import Twist2, Twist3
from spatialmath.base import isunittwist, isunittwist2


@pytest.mark.parametrize(
    "twist, expected",
    [
        (Twist2.UnitRevolute([2, 3]), True),
        (Twist2.UnitRevolute([-4, 1]), True),
        (Twist2([3, -2, -1]), True),
        (Twist3.UnitRevolute([0, 0, 1], [2, 3, 0]), True),
        (Twist3.UnitRevolute([1, 0, 0], [2, 3, 4], pitch=2), True),
        (Twist3([3, -2, 5, 0, 0, -1]), True),
        (Twist2([0.6, 0, 0.8]), False),
        (Twist3([0.6, 0, 0, 0.8, 0, 0]), False),
        (Twist2.UnitPrismatic([3, 4]), True),
        (Twist3.UnitPrismatic([0, 3, 4]), True),
        (Twist2([0, 0, 1]), True),
        (Twist3([0, 0, 0, 0, 1, 0]), True),
        (Twist2([0, 0, 2]), False),
        (Twist3([0, 0, 0, 0, 2, 0]), False),
        (Twist2([2, 0, 0]), False),
        (Twist3([2, 0, 0, 0, 0, 0]), False),
        (Twist2(), False),
        (Twist3(), False),
    ],
)
def test_unit_screw_criteria(twist, expected):
    original = twist.S.copy()
    assert bool(twist.isunit) is expected
    predicate = isunittwist if twist.N == 3 else isunittwist2
    assert twist.isunit == predicate(twist.S)
    np.testing.assert_array_equal(twist.S, original)


@pytest.mark.parametrize("cls", [Twist2, Twist3])
def test_sequence_returns_elementwise_list(cls):
    if cls is Twist2:
        entries = [
            cls.UnitRevolute([2, 3]),
            cls([0.6, 0, 0.8]),
            cls.UnitPrismatic([3, 4]),
            cls(),
        ]
    else:
        entries = [
            cls.UnitRevolute([0, 0, 1], [2, 3, 0]),
            cls([0.6, 0, 0, 0.8, 0, 0]),
            cls.UnitPrismatic([0, 3, 4]),
            cls(),
        ]
    sequence = cls(entries)
    assert isinstance(sequence.isunit, list)
    assert sequence.isunit == [True, False, True, False]
    assert cls.Empty().isunit == []


@pytest.mark.parametrize("cls", [Twist2, Twist3])
@pytest.mark.parametrize("scale", [0.25, -2.0, 3.0])
def test_normalizing_nonzero_twist_yields_unit_screw(cls, scale):
    vector = (
        np.array([3.0, -2, 1]) if cls is Twist2 else np.array([3.0, -2, 5, 0, 0, 1])
    )
    normalized = cls(vector * scale).unit()
    assert normalized.isunit

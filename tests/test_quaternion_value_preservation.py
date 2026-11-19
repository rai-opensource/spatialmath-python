import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from spatialmath import Quaternion, UnitQuaternion


@pytest.mark.parametrize("cls", [Quaternion, UnitQuaternion])
@pytest.mark.parametrize(
    "rows",
    [
        [[0.5, 0.5, 0.5, 0.5]],
        [[-0.5, 0.5, 0.5, 0.5]],
        [[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5]],
        [[-0.5, 0.5, 0.5, 0.5]] * 6,
    ],
)
def test_negation_preserves_values_and_type(cls, rows):
    original = np.array(rows, dtype=float)
    if cls is Quaternion:
        original *= 3
    q = cls.Empty()
    q.data = [row.copy() for row in original]
    negative = -q
    assert type(negative) is cls
    assert len(negative) == len(q)
    assert_array_equal(negative.data, -original)
    assert_array_equal((-negative).data, original)
    assert_allclose(negative.norm(), q.norm())
    assert_array_equal(q.data, original)
    negative.data[0][0] += 0.1
    assert_array_equal(q.data, original)


@pytest.mark.parametrize("cls", [Quaternion, UnitQuaternion])
def test_empty_negation(cls):
    result = -cls.Empty()
    assert type(result) is cls
    assert len(result) == 0


@pytest.mark.parametrize("name", ["s", "v", "vec", "vec_xyzs"])
@pytest.mark.parametrize("cls", [Quaternion, UnitQuaternion])
@pytest.mark.parametrize("count", [1, 2, 6])
def test_exports_match_stored_coefficients(name, cls, count):
    rows = np.tile(np.array([[-0.5, 0.5, -0.5, 0.5]]), (count, 1))
    if cls is Quaternion:
        rows *= 3
    q = cls.Empty()
    q.data = [row.copy() for row in rows]
    expected = {
        "s": rows[:, 0],
        "v": rows[:, 1:],
        "vec": rows,
        "vec_xyzs": np.roll(rows, -1, axis=1),
    }[name]
    if count == 1:
        expected = expected[0]
    assert_array_equal(getattr(q, name), expected)
    assert_array_equal(q.data, rows)


def test_general_zero_negation():
    assert_array_equal((-Quaternion()).vec, np.zeros(4))


def test_signed_unit_export_from_supported_constructor():
    rows = np.array([[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5]])
    q = UnitQuaternion(rows, norm=False)
    assert_array_equal(q.vec, rows)
    assert_array_equal((-q).vec, -rows)

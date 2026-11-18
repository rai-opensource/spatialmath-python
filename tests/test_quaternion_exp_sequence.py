import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath import Quaternion, UnitQuaternion


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 0, 0, 0], [0, 0.1, -0.2, 0.3]],
        [[1, 0, 0, 0], [2, 0.2, 0.4, -0.1]],
        [[0, 0.1, 0.2, 0.3], [0.7, -0.2, 0, 0.1]],
        [[0, 4, 0, 0], [0, 0, 4, 0]],
        [[0, 0, 0, 0]] * 6,
    ],
)
@pytest.mark.parametrize("tol", [1, 20, 100])
def test_sequence_matches_independent_singletons(rows, tol):
    sequence = Quaternion([np.array(row, dtype=float) for row in rows])
    expected = [Quaternion(row).exp(tol=tol) for row in rows]
    actual = sequence.exp(tol=tol)
    expected_type = (
        UnitQuaternion
        if all(isinstance(q, UnitQuaternion) for q in expected)
        else Quaternion
    )
    assert type(actual) is expected_type
    assert len(actual) == len(rows)
    for value, scalar in zip(actual.data, expected):
        assert value.shape == (4,)
        assert_allclose(value, scalar.data[0], atol=1e-15)
    assert_allclose(sequence.data, rows)


@pytest.mark.parametrize("count", [2, 3, 6])
def test_log_exp_round_trip(count):
    rows = [np.array([1 + k, 0.2, -0.3, 0.4]) for k in range(count)]
    q = Quaternion(rows)
    assert_allclose(q.log().exp().data, rows, atol=1e-14)


def test_empty_exponential():
    assert type(Quaternion.Empty().exp()) is Quaternion
    assert len(Quaternion.Empty().exp()) == 0


@pytest.mark.parametrize(
    "row", [[0, 0, 0, 0], [0, 0.1, -0.2, 0.3], [1, 0, 0, 0], [0.7, 0.2, -0.1, 0.4]]
)
def test_singleton_analytic_formula(row):
    row = np.array(row)
    n = np.linalg.norm(row[1:])
    expected = np.exp(row[0]) * np.r_[np.cos(n), row[1:] * np.sinc(n / np.pi)]
    assert_allclose(Quaternion(row).exp().data[0], expected, atol=1e-15)


@pytest.mark.parametrize("tol", [1, 20, 100])
def test_sequence_forwards_scalar_tolerance(tol):
    eps = np.finfo(float).eps
    rows = [np.array([10 * eps, 0.2, -0.1, 0]), np.array([50 * eps, 0, 0, 0.3])]
    expected = [Quaternion(row).exp(tol=tol) for row in rows]
    actual = Quaternion(rows).exp(tol=tol)
    expected_type = (
        UnitQuaternion
        if all(isinstance(q, UnitQuaternion) for q in expected)
        else Quaternion
    )
    assert type(actual) is expected_type
    assert_allclose(actual.data, [q.data[0] for q in expected])


def test_unit_quaternion_sequence_uses_stored_values():
    rows = np.array([[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, 0.5]])
    q = UnitQuaternion(rows, norm=False)
    expected = [Quaternion(row).exp().data[0] for row in rows]
    assert type(q.exp()) is Quaternion
    assert_allclose(q.exp().data, expected)

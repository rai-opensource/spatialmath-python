import numpy as np
import pytest

from spatialmath import Twist2, Twist3


def _sequence(cls):
    """A prismatic, a revolute and a general twist, in that order."""
    if cls is Twist2:
        vectors = [[1, 0, 0], [0, 0, 1], [1, 2, 3]]
    else:
        vectors = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 2, 3, 0.1, 0.2, 0.3]]
    return cls([np.array(v, dtype=float) for v in vectors])


@pytest.mark.parametrize("cls", [Twist2, Twist3])
def test_isprismatic_sequence(cls):
    """isprismatic gives one result per twist, as for a single twist."""
    twists = _sequence(cls)
    assert list(twists.isprismatic) == [True, False, False]
    assert [t.isprismatic for t in twists] == list(twists.isprismatic)


@pytest.mark.parametrize("cls", [Twist2, Twist3])
def test_isrevolute_sequence(cls):
    """isrevolute gives one result per twist, as for a single twist."""
    twists = _sequence(cls)
    assert list(twists.isrevolute) == [False, True, False]
    assert [t.isrevolute for t in twists] == list(twists.isrevolute)


def test_twist2_se2_degree_warning_covers_sequence(capsys):
    """Degree mode warns when any twist in the sequence is prismatic."""
    _sequence(Twist2).SE2(30, unit="deg")
    assert "Twist2.SE2" in capsys.readouterr().out

    Twist2([np.array([1.0, 2.0, 3.0])] * 2).SE2(30, unit="deg")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("cls", [Twist2, Twist3])
def test_scalar_times_twist(cls):
    """``s * X`` scales each twist, the same as ``X * s``."""
    twists = _sequence(cls)
    for left in (2 * twists, 2.0 * twists):
        right = twists * 2
        assert len(left) == len(twists)
        for a, b, t in zip(left, right, twists):
            np.testing.assert_allclose(a.S, 2 * t.S)
            np.testing.assert_allclose(a.S, b.S)


def test_twist3_rand_honours_zrange():
    """The z lower bound comes from zrange, not yrange."""
    np.random.seed(0)
    poses = Twist3.Rand(N=200, xrange=(0, 1), yrange=(-3, -2), zrange=(5, 6)).exp()
    z = np.array([p.t[2] for p in poses])
    assert z.min() >= 5 and z.max() <= 6

import numpy as np
import numpy.testing as nt
import pytest
from scipy.linalg import expm

from spatialmath import SE2, Twist2
from spatialmath.base import skewa, vexa


def _bracket(a, b):
    """Lie bracket of two se(2) vectors, via their matrix commutator."""
    A, B = skewa(a), skewa(b)
    return vexa(A @ B - B @ A)


@pytest.fixture
def twists():
    rng = np.random.default_rng(7)
    return rng.normal(size=3), rng.normal(size=3)


def test_ad_is_lie_bracket(twists):
    """ad(S) @ T is the Lie bracket [S, T]."""
    a, b = twists
    ad = Twist2(a).ad()
    assert ad.shape == (3, 3)
    nt.assert_allclose(ad @ b, _bracket(a, b), atol=1e-14)


def test_ad_antisymmetry(twists):
    """[S, S] = 0, and ad(S) T = -ad(T) S."""
    a, b = twists
    nt.assert_allclose(Twist2(a).ad() @ a, 0, atol=1e-14)
    nt.assert_allclose(Twist2(a).ad() @ b, -Twist2(b).ad() @ a, atol=1e-14)


def test_exp_of_ad_is_Ad(twists):
    """The adjoint of exp(S) is the exponential of ad(S)."""
    a, _ = twists
    S = Twist2(a)
    nt.assert_allclose(S.Ad(), expm(S.ad()), atol=1e-12)
    nt.assert_allclose(S.Ad(), S.SE2().Ad(), atol=1e-14)


def test_SE2_Ad_maps_twists(twists):
    """Ad(T) S is vee(T [S] T^-1)."""
    a, _ = twists
    T = SE2(1.5, -0.7, 0.9)
    expected = vexa(T.A @ skewa(a) @ T.inv().A)
    nt.assert_allclose(T.Ad() @ a, expected, atol=1e-14)


def test_SE2_Ad_identity_and_composition():
    """Ad is a homomorphism: Ad(T1 T2) = Ad(T1) Ad(T2)."""
    nt.assert_allclose(SE2().Ad(), np.eye(3), atol=1e-14)
    T1, T2 = SE2(1, 2, 0.3), SE2(-0.5, 0.25, -1.1)
    nt.assert_allclose((T1 * T2).Ad(), T1.Ad() @ T2.Ad(), atol=1e-14)


def test_ad_sequence():
    """A sequence of twists gives a list of matrices, as for skewa()."""
    seq = Twist2([np.array([1.0, 2.0, 0.3]), np.array([0.0, 1.0, -0.4])])
    ads = seq.ad()
    assert isinstance(ads, list) and len(ads) == 2
    for m, t in zip(ads, seq):
        nt.assert_allclose(m, t.ad())

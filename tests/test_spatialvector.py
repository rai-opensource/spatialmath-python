import unittest
import numpy.testing as nt
import numpy as np

from spatialmath.spatialvector import *


class TestSpatialVector(unittest.TestCase):
    def test_list_powers(self):
        x = SpatialVelocity.Empty()
        self.assertEqual(len(x), 0)
        x.append(SpatialVelocity([1, 2, 3, 4, 5, 6]))
        self.assertEqual(len(x), 1)

        x.append(SpatialVelocity([7, 8, 9, 10, 11, 12]))
        self.assertEqual(len(x), 2)

        y = x[0]
        self.assertIsInstance(y, SpatialVelocity)
        self.assertEqual(len(y), 1)
        self.assertTrue(all(y.A == np.r_[1, 2, 3, 4, 5, 6]))

        y = x[1]
        self.assertIsInstance(y, SpatialVelocity)
        self.assertEqual(len(y), 1)
        self.assertTrue(all(y.A == np.r_[7, 8, 9, 10, 11, 12]))

        x.insert(0, SpatialVelocity([20, 21, 22, 23, 24, 25]))

        y = x[0]
        self.assertIsInstance(y, SpatialVelocity)
        self.assertEqual(len(y), 1)
        self.assertTrue(all(y.A == np.r_[20, 21, 22, 23, 24, 25]))

        y = x[1]
        self.assertIsInstance(y, SpatialVelocity)
        self.assertEqual(len(y), 1)
        self.assertTrue(all(y.A == np.r_[1, 2, 3, 4, 5, 6]))

    def test_velocity(self):
        a = SpatialVelocity([1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialVelocity)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialM6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        a = SpatialVelocity(np.r_[1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialVelocity)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialM6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        s = str(a)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 0)
        self.assertTrue(s.startswith("SpatialVelocity"))

        r = np.random.rand(6, 10)
        a = SpatialVelocity(r)
        self.assertIsInstance(a, SpatialVelocity)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialM6)
        self.assertEqual(len(a), 10)

        b = a[3]
        self.assertIsInstance(b, SpatialVelocity)
        self.assertIsInstance(b, SpatialVector)
        self.assertIsInstance(b, SpatialM6)
        self.assertEqual(len(b), 1)
        self.assertTrue(all(b.A == r[:, 3]))

        s = str(a)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 9)

    def test_acceleration(self):
        a = SpatialAcceleration([1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialAcceleration)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialM6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        a = SpatialAcceleration(np.r_[1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialAcceleration)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialM6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        s = str(a)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 0)
        self.assertTrue(s.startswith("SpatialAcceleration"))

        r = np.random.rand(6, 10)
        a = SpatialAcceleration(r)
        self.assertIsInstance(a, SpatialAcceleration)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialM6)
        self.assertEqual(len(a), 10)

        b = a[3]
        self.assertIsInstance(b, SpatialAcceleration)
        self.assertIsInstance(b, SpatialVector)
        self.assertIsInstance(b, SpatialM6)
        self.assertEqual(len(b), 1)
        self.assertTrue(all(b.A == r[:, 3]))

        s = str(a)
        self.assertIsInstance(s, str)

    def test_force(self):
        a = SpatialForce([1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialForce)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialF6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        a = SpatialForce(np.r_[1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialForce)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialF6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        s = str(a)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 0)
        self.assertTrue(s.startswith("SpatialForce"))

        r = np.random.rand(6, 10)
        a = SpatialForce(r)
        self.assertIsInstance(a, SpatialForce)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialF6)
        self.assertEqual(len(a), 10)

        b = a[3]
        self.assertIsInstance(b, SpatialForce)
        self.assertIsInstance(b, SpatialVector)
        self.assertIsInstance(b, SpatialF6)
        self.assertEqual(len(b), 1)
        self.assertTrue(all(b.A == r[:, 3]))

        s = str(a)
        self.assertIsInstance(s, str)

    def test_momentum(self):
        a = SpatialMomentum([1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialMomentum)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialF6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        a = SpatialMomentum(np.r_[1, 2, 3, 4, 5, 6])
        self.assertIsInstance(a, SpatialMomentum)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialF6)
        self.assertEqual(len(a), 1)
        self.assertTrue(all(a.A == np.r_[1, 2, 3, 4, 5, 6]))

        s = str(a)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 0)
        self.assertTrue(s.startswith("SpatialMomentum"))

        r = np.random.rand(6, 10)
        a = SpatialMomentum(r)
        self.assertIsInstance(a, SpatialMomentum)
        self.assertIsInstance(a, SpatialVector)
        self.assertIsInstance(a, SpatialF6)
        self.assertEqual(len(a), 10)

        b = a[3]
        self.assertIsInstance(b, SpatialMomentum)
        self.assertIsInstance(b, SpatialVector)
        self.assertIsInstance(b, SpatialF6)
        self.assertEqual(len(b), 1)
        self.assertTrue(all(b.A == r[:, 3]))

        s = str(a)
        self.assertIsInstance(s, str)

    def test_arith(self):
        # just test SpatialVelocity since all types derive from same superclass

        r1 = np.r_[1, 2, 3, 4, 5, 6]
        r2 = np.r_[7, 8, 9, 10, 11, 12]
        a1 = SpatialVelocity(r1)
        a2 = SpatialVelocity(r2)

        self.assertTrue(all((a1 + a2).A == r1 + r2))
        self.assertTrue(all((a1 - a2).A == r1 - r2))
        self.assertTrue(all((-a1).A == -r1))

    def test_inertia(self):
        # constructor
        i0 = SpatialInertia()
        nt.assert_equal(i0.A, np.zeros((6, 6)))

        i1 = SpatialInertia(np.eye(6, 6))
        nt.assert_equal(i1.A, np.eye(6, 6))

        i2 = SpatialInertia(m=1, r=(1, 2, 3))
        nt.assert_almost_equal(i2.A, i2.A.T)

        i3 = SpatialInertia(m=1, r=(1, 2, 3), I=np.ones((3, 3)))
        nt.assert_almost_equal(i3.A, i3.A.T)

        # addition
        m_a, m_b = 1.1, 2.2
        r = (1, 2, 3)
        i4a, i4b = SpatialInertia(m=m_a, r=r), SpatialInertia(m=m_b, r=r)
        nt.assert_almost_equal((i4a + i4b).A, SpatialInertia(m=m_a + m_b, r=r).A)

        # isvalid - note this method is very barebone, to be improved
        self.assertTrue(SpatialInertia().isvalid(np.ones((6, 6)), check=False))

    def test_products(self):
        # v x v = a  *, v x F6 = a
        # a x I, I x a
        # v x I, I x v
        # twist x v, twist x a, twist x F
        pass

    def test_spatial_transform_duality(self):
        # A pose transform acts on motion (velocity/acceleration) by the
        # adjoint Ad, and on force/momentum by the coadjoint Ad^-T, because
        # force space is dual to motion space (Featherstone, "Rigid Body
        # Dynamics Algorithms"). With a nonzero translation the SE(3) adjoint
        # is not orthogonal, so Ad^-T differs from the adjoint transpose Ad^T.
        T = SE3(1, 0, 0)
        # ordering is [v(0:3); omega(3:6)] (see SE3.Ad docstring)
        v = SpatialVelocity([0, 0, 0, 0, 1, 0])
        f = SpatialForce([0, 0, 1, 0, 0, 0])
        v2, f2 = T * v, T * f

        # Instantaneous power f . v is frame-invariant; this is the self-oracle
        # and is independent of the [v; omega] vs [omega; v] convention.
        nt.assert_almost_equal(f2.A @ v2.A, f.A @ v.A)

        # Explicit transformed wrench: the moment component flips sign.
        nt.assert_almost_equal(f2.A, np.r_[0, 0, 1, 0, -1, 0])

        # SpatialMomentum exercises the general SpatialVector.__rmul__ path.
        m2 = T * SpatialMomentum([0, 0, 1, 0, 0, 0])
        nt.assert_almost_equal(m2.A, np.r_[0, 0, 1, 0, -1, 0])


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()

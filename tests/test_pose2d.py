import numpy.testing as nt
import matplotlib.pyplot as plt
import unittest
import sys
import pytest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.pose2d import *

# from spatialmath import super_pose as sp
from spatialmath.base import *
import spatialmath.base.argcheck as argcheck
import spatialmath as sm
from spatialmath.baseposematrix import BasePoseMatrix
from spatialmath.twist import BaseTwist


def array_compare(x, y):
    if isinstance(x, BasePoseMatrix):
        x = x.A
    if isinstance(y, BasePoseMatrix):
        y = y.A
    if isinstance(x, BaseTwist):
        x = x.S
    if isinstance(y, BaseTwist):
        y = y.S
    nt.assert_array_almost_equal(x, y)


class TestSO2(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        # null case
        x = SO2()
        self.assertIsInstance(x, SO2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.eye(2, 2))

        ## from angle

        array_compare(SO2(0).A, np.eye(2))
        array_compare(SO2(pi / 2).A, rot2(pi / 2))
        array_compare(SO2(90, unit="deg").A, rot2(pi / 2))

        ## from R

        array_compare(SO2(np.eye(2, 2)).A, np.eye(2, 2))

        array_compare(SO2(rot2(pi / 2)).A, rot2(pi / 2))
        array_compare(SO2(rot2(pi)).A, rot2(pi))

        ## R,T
        array_compare(SO2(np.eye(2)).R, np.eye(2))

        array_compare(SO2(rot2(pi / 2)).R, rot2(pi / 2))

        ## vectorised forms of R
        R = SO2.Empty()
        for theta in [-pi / 2, 0, pi / 2, pi]:
            R.append(SO2(theta))
        self.assertEqual(len(R), 4)
        array_compare(R[0], rot2(-pi / 2))
        array_compare(R[3], rot2(pi))

        # TODO self.assertEqual(SO2(R).R, R)

        ## copy constructor
        r = SO2(0.3)
        c = SO2(r)
        array_compare(r, c)
        r = SO2(0.4)
        array_compare(c, SO2(0.3))

    def test_concat(self):
        x = SO2()
        xx = SO2([x, x, x, x])

        self.assertIsInstance(xx, SO2)
        self.assertEqual(len(xx), 4)

    def test_primitive_convert(self):
        # char

        s = str(SO2())
        self.assertIsInstance(s, str)

    def test_shape(self):
        a = SO2()
        self.assertEqual(a._A.shape, a.shape)

    def test_constructor_Exp(self):
        array_compare(SO2.Exp(skew(0.3)).R, rot2(0.3))
        array_compare(SO2.Exp(0.3).R, rot2(0.3))

        x = SO2.Exp([0, 0.3, 1])
        self.assertEqual(len(x), 3)
        array_compare(x[0], rot2(0))
        array_compare(x[1], rot2(0.3))
        array_compare(x[2], rot2(1))

        x = SO2.Exp([skew(x) for x in [0, 0.3, 1]])
        self.assertEqual(len(x), 3)
        array_compare(x[0], rot2(0))
        array_compare(x[1], rot2(0.3))
        array_compare(x[2], rot2(1))

    def test_isa(self):
        self.assertTrue(SO2.isvalid(rot2(0)))

        self.assertFalse(SO2.isvalid(1))

    def test_resulttype(self):
        r = SO2()
        self.assertIsInstance(r, SO2)

        self.assertIsInstance(r * r, SO2)

        self.assertIsInstance(r / r, SO2)

        self.assertIsInstance(r.inv(), SO2)

    def test_multiply(self):
        vx = np.r_[1, 0]
        vy = np.r_[0, 1]

        r0 = SO2(0)
        r1 = SO2(pi / 2)
        r2 = SO2(pi)
        u = SO2()

        ## SO2-SO2, product
        # scalar x scalar

        array_compare(r0 * u, r0)
        array_compare(u * r0, r0)

        # vector x vector
        array_compare(
            SO2([r0, r1, r2]) * SO2([r2, r0, r1]), SO2([r0 * r2, r1 * r0, r2 * r1])
        )

        # scalar x vector
        array_compare(r1 * SO2([r0, r1, r2]), SO2([r1 * r0, r1 * r1, r1 * r2]))

        # vector x scalar
        array_compare(SO2([r0, r1, r2]) * r2, SO2([r0 * r2, r1 * r2, r2 * r2]))

        ## SO2-vector product
        # scalar x scalar

        array_compare(r1 * vx, np.c_[vy])

        # vector x vector
        # array_compare(SO2([r0, r1, r0]) * np.c_[vy, vx, vx], np.c_[vy, vy, vx])

        # scalar x vector
        array_compare(r1 * np.c_[vx, vy, -vx], np.c_[vy, -vx, -vy])

        # vector x scalar
        array_compare(SO2([r0, r1, r2]) * vy, np.c_[vy, -vx, -vy])

    def test_divide(self):
        r0 = SO2(0)
        r1 = SO2(pi / 2)
        r2 = SO2(pi)
        u = SO2()

        # scalar / scalar
        # implicity tests inv

        array_compare(r1 / u, r1)
        array_compare(r1 / r1, u)

        # vector / vector
        array_compare(
            SO2([r0, r1, r2]) / SO2([r2, r1, r0]), SO2([r0 / r2, r1 / r1, r2 / r0])
        )

        # vector / scalar
        array_compare(SO2([r0, r1, r2]) / r1, SO2([r0 / r1, r1 / r1, r2 / r1]))

    def test_conversions(self):
        T = SO2(pi / 2).SE2()

        self.assertIsInstance(T, SE2)

        ## Lie stuff
        th = 0.3
        RR = SO2(th)
        array_compare(RR.log(), skew(th))

    def test_miscellany(self):
        r = SO2(
            0.3,
        )
        self.assertAlmostEqual(np.linalg.det(r.A), 1)

        self.assertEqual(r.N, 2)

        self.assertFalse(r.isSE)

    def test_printline(self):
        R = SO2(0.3)

        R.printline()
        # s = R.printline(file=None)
        # self.assertIsInstance(s, str)

        R = SO2([0.3, 0.4, 0.5])
        s = R.printline(file=None)
        # self.assertIsInstance(s, str)
        # self.assertEqual(s.count('\n'), 2)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        plt.close("all")

        R = SO2(0.3)
        R.plot(block=False)

        R2 = SO2(0.6)
        # R.animate()
        # R.animate(start=R2)


# ============================== SE2 =====================================#


class TestSE2(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        self.assertIsInstance(SE2(), SE2)

        ## null
        array_compare(SE2().A, np.eye(3, 3))

        # from x,y
        x = SE2(2, 3)
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]]))

        x = SE2([2, 3])
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]]))

        # from x,y,theta
        x = SE2(2, 3, pi / 2)
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        x = SE2([2, 3, pi / 2])
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        x = SE2(2, 3, 90, unit="deg")
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        x = SE2([2, 3, 90], unit="deg")
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        ## T
        T = transl2(1, 2) @ trot2(0.3)
        x = SE2(T)
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, T)

        ## copy constructor
        TT = SE2(x)
        array_compare(SE2(TT).A, T)
        x = SE2()
        array_compare(SE2(TT).A, T)

        ## vectorised versions

        T1 = transl2(1, 2) @ trot2(0.3)
        T2 = transl2(1, -2) @ trot2(-0.4)

        x = SE2([T1, T2, T1, T2])
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 4)
        array_compare(x[0], T1)
        array_compare(x[1], T2)

    def test_shape(self):
        a = SE2()
        self.assertEqual(a._A.shape, a.shape)

    def test_concat(self):
        x = SE2()
        xx = SE2([x, x, x, x])

        self.assertIsInstance(xx, SE2)
        self.assertEqual(len(xx), 4)

    def test_constructor_Exp(self):
        array_compare(SE2.Exp(skewa([1, 2, 0])), transl2(1, 2))
        array_compare(SE2.Exp(np.r_[1, 2, 0]), transl2(1, 2))

        x = SE2.Exp([(1, 2, 0), (3, 4, 0), (5, 6, 0)])
        self.assertEqual(len(x), 3)
        array_compare(x[0], transl2(1, 2))
        array_compare(x[1], transl2(3, 4))
        array_compare(x[2], transl2(5, 6))

        x = SE2.Exp([skewa(x) for x in [(1, 2, 0), (3, 4, 0), (5, 6, 0)]])
        self.assertEqual(len(x), 3)
        array_compare(x[0], transl2(1, 2))
        array_compare(x[1], transl2(3, 4))
        array_compare(x[2], transl2(5, 6))

    def test_isa(self):
        self.assertTrue(SE2.isvalid(trot2(0)))
        self.assertFalse(SE2.isvalid(1))

    def test_resulttype(self):
        t = SE2()
        self.assertIsInstance(t, SE2)
        self.assertIsInstance(t * t, SE2)
        self.assertIsInstance(t / t, SE2)
        self.assertIsInstance(t.inv(), SE2)
        self.assertIsInstance(t + t, np.ndarray)
        self.assertIsInstance(t + 1, np.ndarray)
        self.assertIsInstance(t - 1, np.ndarray)
        self.assertIsInstance(1 + t, np.ndarray)
        self.assertIsInstance(1 - t, np.ndarray)
        self.assertIsInstance(2 * t, np.ndarray)
        self.assertIsInstance(t * 2, np.ndarray)

    def test_inverse(self):
        T1 = transl2(1, 2) @ trot2(0.3)
        TT1 = SE2(T1)

        # test inverse
        array_compare(TT1.inv().A, np.linalg.inv(T1))

        array_compare(TT1 * TT1.inv(), np.eye(3))
        array_compare(TT1.inv() * TT1, np.eye(3))

        # vector case
        TT2 = SE2([TT1, TT1])
        u = [np.eye(3), np.eye(3)]
        array_compare(TT2.inv() * TT1, u)

    def test_Rt(self):
        TT1 = SE2.Rand()
        T1 = TT1.A
        R1 = t2r(T1)
        t1 = transl2(T1)

        array_compare(TT1.A, T1)
        array_compare(TT1.R, R1)
        array_compare(TT1.t, t1)
        self.assertEqual(TT1.x, t1[0])
        self.assertEqual(TT1.y, t1[1])

        TT = SE2([TT1, TT1, TT1])
        array_compare(TT.t, [t1, t1, t1])

    def test_arith(self):
        TT1 = SE2.Rand()
        T1 = TT1.A
        TT2 = SE2.Rand()
        T2 = TT2.A

        I = SE2()

        ## SE2, * SE2, product
        # scalar x scalar

        array_compare(TT1 * TT2, T1 @ T2)
        array_compare(TT2 * TT1, T2 @ T1)
        array_compare(TT1 * I, T1)
        array_compare(TT2 * I, TT2)

        # vector x vector
        array_compare(
            SE2([TT1, TT1, TT2]) * SE2([TT2, TT1, TT1]),
            SE2([TT1 * TT2, TT1 * TT1, TT2 * TT1]),
        )

        # scalar x vector
        array_compare(TT1 * SE2([TT2, TT1]), SE2([TT1 * TT2, TT1 * TT1]))

        # vector x scalar
        array_compare(SE2([TT1, TT2]) * TT2, SE2([TT1 * TT2, TT2 * TT2]))

        ## SE2, * vector product
        vx = np.r_[1, 0]
        vy = np.r_[0, 1]

        # scalar x scalar

        array_compare(TT1 * vy, h2e(T1 @ e2h(vy)))

        # # vector x vector
        # array_compare(SE2([TT1, TT2]) * np.c_[vx, vy], np.c_[h2e(T1 @ e2h(vx)), h2e(T2 @ e2h(vy))])

        # scalar x vector
        array_compare(TT1 * np.c_[vx, vy], h2e(T1 @ e2h(np.c_[vx, vy])))

        # vector x scalar
        array_compare(
            SE2([TT1, TT2, TT1]) * vy,
            np.c_[h2e(T1 @ e2h(vy)), h2e(T2 @ e2h(vy)), h2e(T1 @ e2h(vy))],
        )

    def test_defs(self):
        # log
        # x = SE2.Exp([2, 3, 0.5])
        # array_compare(x.log(), np.array([[0, -0.5, 2], [0.5, 0, 3], [0, 0, 0]]))
        pass

    def test_conversions(self):
        ##  SE2,                     convert to SE2, class

        TT = SE2(1, 2, 0.3)

        array_compare(TT, transl2(1, 2) @ trot2(0.3))

        ## xyt
        array_compare(TT.xyt(), np.r_[1, 2, 0.3])

        ## Lie stuff
        x = TT.log()
        self.assertTrue(isskewa(x))

    def test_interp(self):
        TT = SE2(2, -4, 0.6)
        I = SE2()

        z = I.interp(TT, s=0)
        self.assertIsInstance(z, SE2)

        array_compare(I.interp(TT, s=0), I)
        array_compare(I.interp(TT, s=1), TT)
        array_compare(I.interp(TT, s=0.5), SE2(1, -2, 0.3))

        R1 = SO2(math.pi - 0.1)
        R2 = SO2(-math.pi + 0.2)
        array_compare(R1.interp(R2, s=0.5, shortest=False), SO2(0.05))
        array_compare(R1.interp(R2, s=0.5, shortest=True), SO2(-math.pi + 0.05))

        T1 = SE2(0, 0, math.pi - 0.1)
        T2 = SE2(0, 0, -math.pi + 0.2)
        array_compare(T1.interp(T2, s=0.5, shortest=False), SE2(0, 0, 0.05))
        array_compare(T1.interp(T2, s=0.5, shortest=True), SE2(0, 0, -math.pi + 0.05))

    def test_interp1(self):
        # interpolate from the identity pose
        TT = SE2(2, -4, 0.6)
        array_compare(TT.interp1(0), SE2())
        array_compare(TT.interp1(1), TT)
        array_compare(TT.interp1(0.5), SE2(1, -2, 0.3))

        z = TT.interp1([0, 0.5, 1])
        self.assertEqual(len(z), 3)
        array_compare(z[2], TT)

        R = SO2(0.6)
        array_compare(R.interp1(0), SO2())
        array_compare(R.interp1(1), R)
        array_compare(R.interp1(0.5), SO2(0.3))
        self.assertEqual(len(SE2([TT, TT]).interp1(0.5)), 2)

    def test_miscellany(self):
        TT = SE2(1, 2, 0.3)

        self.assertEqual(TT.A.shape, (3, 3))

        self.assertTrue(TT.isSE)

        self.assertIsInstance(TT, SE2)

    def test_display(self):
        T1 = SE2.Rand()

        T1.printline()

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_graphics(self):
        plt.close("all")
        T1 = SE2.Rand()
        T2 = SE2.Rand()

        T1.plot(block=False, dims=[-2, 2])

        T1.animate(repeat=False, dims=[-2, 2], nframes=10)
        T1.animate(T0=T2, repeat=False, dims=[-2, 2], nframes=10)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main(buffer=True)

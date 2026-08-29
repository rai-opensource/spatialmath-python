import numpy.testing as nt
import matplotlib.pyplot as plt
import unittest
import sys
import pytest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath import SE3, SO3, SE2, UnitQuaternion
import numpy as np
from spatialmath.base import *
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


class TestSO3(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        # null constructor
        R = SO3()
        nt.assert_equal(len(R), 1)
        array_compare(R, np.eye(3))
        self.assertIsInstance(R, SO3)

        # empty constructor
        R = SO3.Empty()
        nt.assert_equal(len(R), 0)
        self.assertIsInstance(R, SO3)

        # construct from matrix
        R = SO3(rotx(0.2))
        nt.assert_equal(len(R), 1)
        array_compare(R, rotx(0.2))
        self.assertIsInstance(R, SO3)

        # construct from canonic rotation
        R = SO3.Rx(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, rotx(0.2))
        self.assertIsInstance(R, SO3)

        R = SO3.Ry(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, roty(0.2))
        self.assertIsInstance(R, SO3)

        R = SO3.Rz(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, rotz(0.2))
        self.assertIsInstance(R, SO3)

        # OA
        R = SO3.OA([0, 1, 0], [0, 0, 1])
        nt.assert_equal(len(R), 1)
        array_compare(R, np.eye(3))
        self.assertIsInstance(R, SO3)

        np.random.seed(32)
        # random
        R = SO3.Rand()
        nt.assert_equal(len(R), 1)
        self.assertIsInstance(R, SO3)

        # random constrained
        R = SO3.Rand(theta_range=(0.1, 0.7))
        self.assertIsInstance(R, SO3)
        self.assertEqual(R.A.shape, (3, 3))
        self.assertLessEqual(R.angvec()[0], 0.7)
        self.assertGreaterEqual(R.angvec()[0], 0.1)

        # copy constructor
        R = SO3.Rx(pi / 2)
        R2 = SO3(R)
        R = SO3.Ry(pi / 2)
        array_compare(R2, rotx(pi / 2))

    def test_constructor_Eul(self):
        R = SO3.Eul([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SO3)

        R = SO3.Eul(0.1, 0.2, 0.3)
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SO3)

        R = SO3.Eul(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SO3)

        R = SO3.Eul([10, 20, 30], unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([10, 20, 30], unit="deg"))
        self.assertIsInstance(R, SO3)

        R = SO3.Eul(10, 20, 30, unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([10, 20, 30], unit="deg"))
        self.assertIsInstance(R, SO3)

        # matrix input

        angles = np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        )
        R = SO3.Eul(angles)
        self.assertIsInstance(R, SO3)
        nt.assert_equal(len(R), 4)
        for i in range(4):
            array_compare(R[i], eul2r(angles[i, :]))

        angles *= 10
        R = SO3.Eul(angles, unit="deg")
        self.assertIsInstance(R, SO3)
        nt.assert_equal(len(R), 4)
        for i in range(4):
            array_compare(R[i], eul2r(angles[i, :], unit="deg"))

    def test_constructor_RPY(self):
        R = SO3.RPY(0.1, 0.2, 0.3, order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        self.assertIsInstance(R, SO3)

        R = SO3.RPY(10, 20, 30, unit="deg", order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([10, 20, 30], order="zyx", unit="deg"))
        self.assertIsInstance(R, SO3)

        R = SO3.RPY([0.1, 0.2, 0.3], order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        self.assertIsInstance(R, SO3)

        R = SO3.RPY(np.r_[0.1, 0.2, 0.3], order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        self.assertIsInstance(R, SO3)

        # check default
        R = SO3.RPY([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        self.assertIsInstance(R, SO3)

        # XYZ order

        R = SO3.RPY(0.1, 0.2, 0.3, order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="xyz"))
        self.assertIsInstance(R, SO3)

        R = SO3.RPY(10, 20, 30, unit="deg", order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([10, 20, 30], order="xyz", unit="deg"))
        self.assertIsInstance(R, SO3)

        R = SO3.RPY([0.1, 0.2, 0.3], order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="xyz"))
        self.assertIsInstance(R, SO3)

        R = SO3.RPY(np.r_[0.1, 0.2, 0.3], order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="xyz"))
        self.assertIsInstance(R, SO3)

        # matrix input

        angles = np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        )
        R = SO3.RPY(angles, order="zyx")
        self.assertIsInstance(R, SO3)
        nt.assert_equal(len(R), 4)
        for i in range(4):
            array_compare(R[i], rpy2r(angles[i, :], order="zyx"))

        angles *= 10
        R = SO3.RPY(angles, unit="deg", order="zyx")
        self.assertIsInstance(R, SO3)
        nt.assert_equal(len(R), 4)
        for i in range(4):
            array_compare(R[i], rpy2r(angles[i, :], unit="deg", order="zyx"))

    def test_constructor_AngVec(self):
        # angvec
        R = SO3.AngVec(0.2, [1, 0, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, rotx(0.2))
        self.assertIsInstance(R, SO3)

        R = SO3.AngVec(0.3, [0, 1, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, roty(0.3))
        self.assertIsInstance(R, SO3)

    def test_constructor_TwoVec(self):
        # Randomly selected vectors
        v1 = [1, 73, -42]
        v2 = [0, 0.02, 57]
        v3 = [-2, 3, 9]

        # x and y given
        R = SO3.TwoVectors(x=v1, y=v2)
        self.assertIsInstance(R, SO3)
        nt.assert_almost_equal(R.det(), 1, 5)
        # x axis should equal normalized x vector
        nt.assert_almost_equal(R.R[:, 0], v1 / np.linalg.norm(v1), 5)

        # y and z given
        R = SO3.TwoVectors(y=v2, z=v3)
        self.assertIsInstance(R, SO3)
        nt.assert_almost_equal(R.det(), 1, 5)
        # y axis should equal normalized y vector
        nt.assert_almost_equal(R.R[:, 1], v2 / np.linalg.norm(v2), 5)

        # x and z given
        R = SO3.TwoVectors(x=v3, z=v1)
        self.assertIsInstance(R, SO3)
        nt.assert_almost_equal(R.det(), 1, 5)
        # x axis should equal normalized x vector
        nt.assert_almost_equal(R.R[:, 0], v3 / np.linalg.norm(v3), 5)

    def test_conversion(self):
        R = SO3.AngleAxis(0.7, [1, 2, 3])
        q = UnitQuaternion([11, 7, 3, -6])

        R_from_q = SO3(q.R)
        q_from_R = UnitQuaternion(R)

        nt.assert_array_almost_equal(R.UnitQuaternion(), q_from_R)
        nt.assert_array_almost_equal(R.UnitQuaternion().SO3(), R)

        nt.assert_array_almost_equal(q.SO3(), R_from_q)
        nt.assert_array_almost_equal(q.SO3().UnitQuaternion(), q)

    def test_shape(self):
        a = SO3()
        self.assertEqual(a._A.shape, a.shape)

    def test_about(self):
        R = SO3()
        R.about

    def test_str(self):
        R = SO3()

        s = str(R)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 3)

        s = repr(R)
        self.assertIsInstance(s, str)
        self.assertEqual(s.count("\n"), 2)

    def test_printline(self):
        R = SO3.Rx(0.3)

        R.printline()
        # s = R.printline(file=None)
        # self.assertIsInstance(s, str)

        R = SO3.Rx([0.3, 0.4, 0.5])
        s = R.printline(file=None)
        # self.assertIsInstance(s, str)
        # self.assertEqual(s.count('\n'), 2)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        plt.close("all")

        R = SO3.Rx(0.3)
        R.plot(block=False)

        R2 = SO3.Rx(0.6)
        # R.animate()
        # R.animate(start=R.inv())

    def test_listpowers(self):
        R = SO3()
        R1 = SO3.Rx(0.2)
        R2 = SO3.Ry(0.3)

        R.append(R1)
        R.append(R2)
        nt.assert_equal(len(R), 3)
        self.assertIsInstance(R, SO3)

        array_compare(R[0], np.eye(3))
        array_compare(R[1], R1)
        array_compare(R[2], R2)

        R = SO3([rotx(0.1), rotx(0.2), rotx(0.3)])
        nt.assert_equal(len(R), 3)
        self.assertIsInstance(R, SO3)
        array_compare(R[0], rotx(0.1))
        array_compare(R[1], rotx(0.2))
        array_compare(R[2], rotx(0.3))

        R = SO3([SO3.Rx(0.1), SO3.Rx(0.2), SO3.Rx(0.3)])
        nt.assert_equal(len(R), 3)
        self.assertIsInstance(R, SO3)
        array_compare(R[0], rotx(0.1))
        array_compare(R[1], rotx(0.2))
        array_compare(R[2], rotx(0.3))

    def test_tests(self):
        R = SO3()

        self.assertEqual(R.isrot(), True)
        self.assertEqual(R.isrot2(), False)
        self.assertEqual(R.ishom(), False)
        self.assertEqual(R.ishom2(), False)

    def test_properties(self):
        R = SO3()

        self.assertEqual(R.isSO, True)
        self.assertEqual(R.isSE, False)

        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])

        nt.assert_equal(R.N, 3)
        nt.assert_equal(R.shape, (3, 3))

        R = SO3.Rx(0.3)
        array_compare(R.inv() * R, np.eye(3, 3))

    def test_arith(self):
        R = SO3()

        # sum
        a = R + R
        self.assertNotIsInstance(a, SO3)
        array_compare(a, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))

        a = R + 1
        self.assertNotIsInstance(a, SO3)
        array_compare(a, np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))

        # a = 1 + R
        # self.assertNotIsInstance(a, SO3)
        # array_compare(a, np.array([ [2,1,1], [1,2,1], [1,1,2]]))

        a = R + np.eye(3)
        self.assertNotIsInstance(a, SO3)
        array_compare(a, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))

        # a =  np.eye(3) + R
        # self.assertNotIsInstance(a, SO3)
        # array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        #  this invokes the __add__ method for numpy

        # difference
        R = SO3()

        a = R - R
        self.assertNotIsInstance(a, SO3)
        array_compare(a, np.zeros((3, 3)))

        a = R - 1
        self.assertNotIsInstance(a, SO3)
        array_compare(a, np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]))

        # a = 1 - R
        # self.assertNotIsInstance(a, SO3)
        # array_compare(a, -np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))

        a = R - np.eye(3)
        self.assertNotIsInstance(a, SO3)
        array_compare(a, np.zeros((3, 3)))

        # a =  np.eye(3) - R
        # self.assertNotIsInstance(a, SO3)
        # array_compare(a, np.zeros((3,3)))

        # multiply
        R = SO3()

        a = R * R
        self.assertIsInstance(a, SO3)
        array_compare(a, R)

        a = R * 2
        self.assertNotIsInstance(a, SO3)
        array_compare(a, 2 * np.eye(3))

        a = 2 * R
        self.assertNotIsInstance(a, SO3)
        array_compare(a, 2 * np.eye(3))

        R = SO3()
        R *= SO3.Rx(pi / 2)
        self.assertIsInstance(R, SO3)
        array_compare(R, rotx(pi / 2))

        R = SO3()
        R *= 2
        self.assertNotIsInstance(R, SO3)
        array_compare(R, 2 * np.eye(3))

        array_compare(SO3.Rx(pi / 2) * SO3.Ry(pi / 2) * SO3.Rx(-pi / 2), SO3.Rz(pi / 2))

        array_compare(SO3.Ry(pi / 2) * [1, 0, 0], np.c_[0, 0, -1].T)

        # SO3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]

        def cv(v):
            return np.c_[v]

        nt.assert_equal(isinstance(SO3.Rx(pi / 2) * vx, np.ndarray), True)
        print(vx)
        print(SO3.Rx(pi / 2) * vx)
        print(cv(vx))
        array_compare(SO3.Rx(pi / 2) * vx, cv(vx))
        array_compare(SO3.Rx(pi / 2) * vy, cv(vz))
        array_compare(SO3.Rx(pi / 2) * vz, cv(-vy))

        array_compare(SO3.Ry(pi / 2) * vx, cv(-vz))
        array_compare(SO3.Ry(pi / 2) * vy, cv(vy))
        array_compare(SO3.Ry(pi / 2) * vz, cv(vx))

        array_compare(SO3.Rz(pi / 2) * vx, cv(vy))
        array_compare(SO3.Rz(pi / 2) * vy, cv(-vx))
        array_compare(SO3.Rz(pi / 2) * vz, cv(vz))

        # divide
        R = SO3.Ry(0.3)
        a = R / R
        self.assertIsInstance(a, SO3)
        array_compare(a, np.eye(3))

        a = R / 2
        self.assertNotIsInstance(a, SO3)
        array_compare(a, roty(0.3) / 2)

        # power

        R = SO3.Rx(pi / 2)
        R = R**2
        array_compare(R, SO3.Rx(pi))

        R = SO3.Rx(pi / 2)
        R **= 2
        array_compare(R, SO3.Rx(pi))

        R = SO3.Rx(pi / 4)
        R = R ** (-2)
        array_compare(R, SO3.Rx(-pi / 2))

        R = SO3.Rx(pi / 4)
        R **= -2
        array_compare(R, SO3.Rx(-pi / 2))

    def test_arith_vect(self):
        rx = SO3.Rx(pi / 2)
        ry = SO3.Ry(pi / 2)
        rz = SO3.Rz(pi / 2)
        u = SO3()

        # multiply
        R = SO3([rx, ry, rz])
        a = R * rx
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * rx)
        array_compare(a[2], rz * rx)

        a = rx * R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], rx * ry)
        array_compare(a[2], rx * rz)

        a = R * R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * ry)
        array_compare(a[2], rz * rz)

        a = R * 2
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * 2)
        array_compare(a[1], ry * 2)
        array_compare(a[2], rz * 2)

        a = 2 * R
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * 2)
        array_compare(a[1], ry * 2)
        array_compare(a[2], rz * 2)

        a = R
        a *= rx
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * rx)
        array_compare(a[2], rz * rx)

        a = rx
        a *= R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], rx * ry)
        array_compare(a[2], rx * rz)

        a = R
        a *= R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * ry)
        array_compare(a[2], rz * rz)

        a = R
        a *= 2
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * 2)
        array_compare(a[1], ry * 2)
        array_compare(a[2], rz * 2)

        # SO3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]

        a = R * vx
        array_compare(a[:, 0], (rx * vx).flatten())
        array_compare(a[:, 1], (ry * vx).flatten())
        array_compare(a[:, 2], (rz * vx).flatten())

        a = rx * np.vstack((vx, vy, vz)).T
        array_compare(a[:, 0], (rx * vx).flatten())
        array_compare(a[:, 1], (rx * vy).flatten())
        array_compare(a[:, 2], (rx * vz).flatten())

        # divide
        R = SO3([rx, ry, rz])
        a = R / rx
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], ry / rx)
        array_compare(a[2], rz / rx)

        a = rx / R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], rx / ry)
        array_compare(a[2], rx / rz)

        a = R / R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], np.eye(3))
        array_compare(a[1], np.eye(3))
        array_compare(a[2], np.eye(3))

        a = R / 2
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / 2)
        array_compare(a[1], ry / 2)
        array_compare(a[2], rz / 2)

        a = R
        a /= rx
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], ry / rx)
        array_compare(a[2], rz / rx)

        a = rx
        a /= R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], rx / ry)
        array_compare(a[2], rx / rz)

        a = R
        a /= R
        self.assertIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], np.eye(3))
        array_compare(a[1], np.eye(3))
        array_compare(a[2], np.eye(3))

        a = R
        a /= 2
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / 2)
        array_compare(a[1], ry / 2)
        array_compare(a[2], rz / 2)

        # add
        R = SO3([rx, ry, rz])
        a = R + rx
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + rx)
        array_compare(a[1], ry + rx)
        array_compare(a[2], rz + rx)

        a = rx + R
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + rx)
        array_compare(a[1], rx + ry)
        array_compare(a[2], rx + rz)

        a = R + R
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + rx)
        array_compare(a[1], ry + ry)
        array_compare(a[2], rz + rz)

        a = R + 1
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + 1)
        array_compare(a[1], ry + 1)
        array_compare(a[2], rz + 1)

        # subtract
        R = SO3([rx, ry, rz])
        a = R - rx
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - rx)
        array_compare(a[1], ry - rx)
        array_compare(a[2], rz - rx)

        a = rx - R
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - rx)
        array_compare(a[1], rx - ry)
        array_compare(a[2], rx - rz)

        a = R - R
        self.assertNotIsInstance(a, SO3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - rx)
        array_compare(a[1], ry - ry)
        array_compare(a[2], rz - rz)

    def test_functions(self):
        # inv
        # .T

        # conversion to SE2
        poseSE3 = SE3.Tx(3.3) * SE3.Rz(1.5)
        poseSE2 = poseSE3.yaw_SE2()
        nt.assert_almost_equal(poseSE3.R[0:2, 0:2], poseSE2.R[0:2, 0:2])
        nt.assert_equal(poseSE3.x, poseSE2.x)
        nt.assert_equal(poseSE3.y, poseSE2.y)

        posesSE3 = SE3([poseSE3, poseSE3])
        posesSE2 = posesSE3.yaw_SE2()
        nt.assert_equal(len(posesSE2), 2)

    def test_functions_vect(self):
        # inv
        # .T
        pass

    def test_functions_lie(self):
        R = SO3.EulerVec([0.42, 0.73, -1.17])

        # Check log and exponential map
        nt.assert_equal(R, SO3.Exp(R.log()))
        np.testing.assert_equal((R.inv() * R).log(), np.zeros([3, 3]))

        # Check euler vector map
        nt.assert_equal(R, SO3.EulerVec(R.eulervec()))
        np.testing.assert_equal((R.inv() * R).eulervec(), np.zeros(3))

    def test_rotatedvector(self):
        v1 = [1, 2, 3]
        R = SO3.Eul(0.3, 0.4, 0.5)
        v2 = R * v1
        Re = SO3.RotatedVector(v1, v2)
        np.testing.assert_almost_equal(v2, Re * v1)

        Re = SO3.RotatedVector(v1, v1)
        np.testing.assert_almost_equal(Re, np.eye(3))

        # Antipodal case: v1 and v2 point in exactly opposite directions.
        # The cross product used to find the rotation axis is zero here
        # too, same as the parallel case above, but the correct answer is
        # a 180 degree flip, not identity -- regression test for a bug
        # where this silently returned identity instead.
        for v1 in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [-3, 1, 2]):
            v1 = unitvec(v1)
            v2 = [-x for x in v1]
            Re = SO3.RotatedVector(v1, v2)
            np.testing.assert_almost_equal(np.asarray(Re * v1).flatten(), v2)
            # must actually be a 180 degree rotation, not identity
            assert not np.allclose(Re.A, np.eye(3))

        R = SO3()  # identity matrix case

        # Check log and exponential map
        nt.assert_equal(R, SO3.Exp(R.log()))
        np.testing.assert_equal((R.inv() * R).log(), np.zeros([3, 3]))

        # Check euler vector map
        nt.assert_equal(R, SO3.EulerVec(R.eulervec()))
        np.testing.assert_equal((R.inv() * R).eulervec(), np.zeros(3))

    def test_mean(self):
        rpy = np.ones((100, 1)) @ np.c_[0.1, 0.2, 0.3]
        R = SO3.RPY(rpy)
        self.assertEqual(len(R), 100)
        m = R.mean()
        self.assertIsInstance(m, SO3)
        array_compare(m, R[0])

        # range of angles, mean should be the middle one, index=25
        R = SO3.Rz(np.linspace(start=0.3, stop=0.7, num=51))
        m = R.mean()
        self.assertIsInstance(m, SO3)
        array_compare(m, R[25])

        # now add noise
        rng = np.random.default_rng(0)  # reproducible random numbers
        rpy += rng.normal(scale=0.00001, size=(100, 3))
        R = SO3.RPY(rpy)
        m = R.mean()
        array_compare(m, SO3.RPY(0.1, 0.2, 0.3))


# ============================== SE3 =====================================#


class TestSE3(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        # null constructor
        R = SE3()
        nt.assert_equal(len(R), 1)
        array_compare(R, np.eye(4))
        self.assertIsInstance(R, SE3)

        # construct from matrix
        R = SE3(trotx(0.2))
        nt.assert_equal(len(R), 1)
        array_compare(R, trotx(0.2))
        self.assertIsInstance(R, SE3)

        # construct from canonic rotation
        R = SE3.Rx(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, trotx(0.2))
        self.assertIsInstance(R, SE3)

        R = SE3.Ry(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, troty(0.2))
        self.assertIsInstance(R, SE3)

        R = SE3.Rz(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, trotz(0.2))
        self.assertIsInstance(R, SE3)

        # construct from canonic translation
        R = SE3.Tx(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, transl(0.2, 0, 0))
        self.assertIsInstance(R, SE3)

        R = SE3.Ty(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, transl(0, 0.2, 0))
        self.assertIsInstance(R, SE3)

        R = SE3.Tz(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, transl(0, 0, 0.2))
        self.assertIsInstance(R, SE3)

        # triple angle
        R = SE3.Eul([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2tr([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SE3)

        R = SE3.Eul(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2tr([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SE3)

        R = SE3.Eul([10, 20, 30], unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2tr([10, 20, 30], unit="deg"))
        self.assertIsInstance(R, SE3)

        R = SE3.RPY([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SE3)

        R = SE3.RPY(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([0.1, 0.2, 0.3]))
        self.assertIsInstance(R, SE3)

        R = SE3.RPY([10, 20, 30], unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([10, 20, 30], unit="deg"))
        self.assertIsInstance(R, SE3)

        R = SE3.RPY([0.1, 0.2, 0.3], order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([0.1, 0.2, 0.3], order="xyz"))
        self.assertIsInstance(R, SE3)

        # angvec
        R = SE3.AngVec(0.2, [1, 0, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, trotx(0.2))
        self.assertIsInstance(R, SE3)

        R = SE3.AngVec(0.3, [0, 1, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, troty(0.3))
        self.assertIsInstance(R, SE3)

        # OA
        R = SE3.OA([0, 1, 0], [0, 0, 1])
        nt.assert_equal(len(R), 1)
        array_compare(R, np.eye(4))
        self.assertIsInstance(R, SE3)

        np.random.seed(65)
        # random
        R = SE3.Rand()
        nt.assert_equal(len(R), 1)
        self.assertIsInstance(R, SE3)

        # random
        T = SE3.Rand()
        R = T.R
        t = T.t
        T = SE3.Rt(R, t)
        self.assertIsInstance(T, SE3)
        self.assertEqual(T.A.shape, (4, 4))

        nt.assert_equal(T.R, R)
        nt.assert_equal(T.t, t)

        nt.assert_equal(T.x, t[0])
        nt.assert_equal(T.y, t[1])
        nt.assert_equal(T.z, t[2])

        TT = SE3([T, T, T])
        desired_shape = (3,)
        nt.assert_equal(TT.x.shape, desired_shape)
        nt.assert_equal(TT.y.shape, desired_shape)
        nt.assert_equal(TT.z.shape, desired_shape)

        ones = np.ones(desired_shape)
        nt.assert_equal(TT.x, ones * t[0])
        nt.assert_equal(TT.y, ones * t[1])
        nt.assert_equal(TT.z, ones * t[2])

        # random constrained
        T = SE3.Rand(theta_range=(0.1, 0.7))
        self.assertIsInstance(T, SE3)
        self.assertEqual(T.A.shape, (4, 4))
        self.assertLessEqual(T.angvec()[0], 0.7)
        self.assertGreaterEqual(T.angvec()[0], 0.1)

        # copy constructor
        R = SE3.Rx(pi / 2)
        R2 = SE3(R)
        R = SE3.Ry(pi / 2)
        array_compare(R2, trotx(pi / 2))

        # SO3
        T = SE3(SO3())
        nt.assert_equal(len(T), 1)
        self.assertIsInstance(T, SE3)
        nt.assert_equal(T.A, np.eye(4))

        # SE2
        T = SE3(SE2(1, 2, 0.4))
        nt.assert_equal(len(T), 1)
        self.assertIsInstance(T, SE3)
        self.assertEqual(T.A.shape, (4, 4))
        nt.assert_equal(T.t, [1, 2, 0])

        # Bad number of arguments
        with self.assertRaises(ValueError):
            T = SE3(1.0, 0.0)
        with self.assertRaises(TypeError):
            T = SE3(1.0, 0.0, 0.0, 0.0)

    def test_shape(self):
        a = SE3()
        self.assertEqual(a._A.shape, a.shape)

    def test_listpowers(self):
        R = SE3()
        R1 = SE3.Rx(0.2)
        R2 = SE3.Ry(0.3)

        R.append(R1)
        R.append(R2)
        nt.assert_equal(len(R), 3)
        self.assertIsInstance(R, SE3)

        array_compare(R[0], np.eye(4))
        array_compare(R[1], R1)
        array_compare(R[2], R2)

        R = SE3([trotx(0.1), trotx(0.2), trotx(0.3)])
        nt.assert_equal(len(R), 3)
        self.assertIsInstance(R, SE3)
        array_compare(R[0], trotx(0.1))
        array_compare(R[1], trotx(0.2))
        array_compare(R[2], trotx(0.3))

        R = SE3([SE3.Rx(0.1), SE3.Rx(0.2), SE3.Rx(0.3)])
        nt.assert_equal(len(R), 3)
        self.assertIsInstance(R, SE3)
        array_compare(R[0], trotx(0.1))
        array_compare(R[1], trotx(0.2))
        array_compare(R[2], trotx(0.3))

    def test_tests(self):
        R = SE3()

        self.assertEqual(R.isrot(), False)
        self.assertEqual(R.isrot2(), False)
        self.assertEqual(R.ishom(), True)
        self.assertEqual(R.ishom2(), False)

    def test_properties(self):
        R = SE3()

        self.assertEqual(R.isSO, False)
        self.assertEqual(R.isSE, True)

        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])

        nt.assert_equal(R.N, 3)
        nt.assert_equal(R.shape, (4, 4))

        # Testing the CopyFrom function
        mutable_array = np.eye(4)
        pass_by_ref = SE3(mutable_array)
        pass_by_val = SE3.CopyFrom(mutable_array)
        mutable_array[0, 3] = 5.0
        nt.assert_allclose(pass_by_val.data[0], np.eye(4))
        nt.assert_allclose(pass_by_ref.data[0], mutable_array)
        nt.assert_raises(
            AssertionError, nt.assert_allclose, pass_by_val.data[0], pass_by_ref.data[0]
        )

    def test_arith(self):
        T = SE3(1, 2, 3)

        # sum
        a = T + T
        self.assertNotIsInstance(a, SE3)
        array_compare(
            a, np.array([[2, 0, 0, 2], [0, 2, 0, 4], [0, 0, 2, 6], [0, 0, 0, 2]])
        )

        a = T + 1
        self.assertNotIsInstance(a, SE3)
        array_compare(
            a, np.array([[2, 1, 1, 2], [1, 2, 1, 3], [1, 1, 2, 4], [1, 1, 1, 2]])
        )

        # a = 1 + T
        # self.assertNotIsInstance(a, SE3)
        # array_compare(a, np.array([ [2,1,1], [1,2,1], [1,1,2]]))

        a = T + np.eye(4)
        self.assertNotIsInstance(a, SE3)
        array_compare(
            a, np.array([[2, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3], [0, 0, 0, 2]])
        )

        # a =  np.eye(3) + T
        # self.assertNotIsInstance(a, SE3)
        # array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        #  this invokes the __add__ method for numpy

        # difference
        T = SE3(1, 2, 3)

        a = T - T
        self.assertNotIsInstance(a, SE3)
        array_compare(a, np.zeros((4, 4)))

        a = T - 1
        self.assertNotIsInstance(a, SE3)
        array_compare(
            a,
            np.array([[0, -1, -1, 0], [-1, 0, -1, 1], [-1, -1, 0, 2], [-1, -1, -1, 0]]),
        )

        # a = 1 - T
        # self.assertNotIsInstance(a, SE3)
        # array_compare(a, -np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))

        a = T - np.eye(4)
        self.assertNotIsInstance(a, SE3)
        array_compare(
            a, np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]])
        )

        # a =  np.eye(3) - T
        # self.assertNotIsInstance(a, SE3)
        # array_compare(a, np.zeros((3,3)))

        a = T
        a -= T
        self.assertNotIsInstance(a, SE3)
        array_compare(a, np.zeros((4, 4)))

        # multiply
        T = SE3(1, 2, 3)

        a = T * T
        self.assertIsInstance(a, SE3)
        array_compare(a, transl(2, 4, 6))

        a = T * 2
        self.assertNotIsInstance(a, SE3)
        array_compare(a, 2 * transl(1, 2, 3))

        a = 2 * T
        self.assertNotIsInstance(a, SE3)
        array_compare(a, 2 * transl(1, 2, 3))

        T = SE3(1, 2, 3)
        T *= SE3.Ry(pi / 2)
        self.assertIsInstance(T, SE3)
        array_compare(
            T, np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]])
        )

        T = SE3()
        T *= 2
        self.assertNotIsInstance(T, SE3)
        array_compare(T, 2 * np.eye(4))

        array_compare(SE3.Rx(pi / 2) * SE3.Ry(pi / 2) * SE3.Rx(-pi / 2), SE3.Rz(pi / 2))

        array_compare(SE3.Ry(pi / 2) * [1, 0, 0], np.c_[0, 0, -1].T)

        # SE3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]

        def cv(v):
            return np.c_[v]

        nt.assert_equal(isinstance(SE3.Tx(pi / 2) * vx, np.ndarray), True)
        array_compare(SE3.Rx(pi / 2) * vx, cv(vx))
        array_compare(SE3.Rx(pi / 2) * vy, cv(vz))
        array_compare(SE3.Rx(pi / 2) * vz, cv(-vy))

        array_compare(SE3.Ry(pi / 2) * vx, cv(-vz))
        array_compare(SE3.Ry(pi / 2) * vy, cv(vy))
        array_compare(SE3.Ry(pi / 2) * vz, cv(vx))

        array_compare(SE3.Rz(pi / 2) * vx, cv(vy))
        array_compare(SE3.Rz(pi / 2) * vy, cv(-vx))
        array_compare(SE3.Rz(pi / 2) * vz, cv(vz))

        # divide
        T = SE3.Ry(0.3)
        a = T / T
        self.assertIsInstance(a, SE3)
        array_compare(a, np.eye(4))

        a = T / 2
        self.assertNotIsInstance(a, SE3)
        array_compare(a, troty(0.3) / 2)

    def test_arith_vect(self):
        rx = SE3.Rx(pi / 2)
        ry = SE3.Ry(pi / 2)
        rz = SE3.Rz(pi / 2)
        u = SE3()

        # multiply
        T = SE3([rx, ry, rz])
        a = T * rx
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * rx)
        array_compare(a[2], rz * rx)

        a = rx * T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], rx * ry)
        array_compare(a[2], rx * rz)

        a = T * T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * ry)
        array_compare(a[2], rz * rz)

        a = T * 2
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * 2)
        array_compare(a[1], ry * 2)
        array_compare(a[2], rz * 2)

        a = 2 * T
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * 2)
        array_compare(a[1], ry * 2)
        array_compare(a[2], rz * 2)

        a = T
        a *= rx
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * rx)
        array_compare(a[2], rz * rx)

        a = rx
        a *= T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], rx * ry)
        array_compare(a[2], rx * rz)

        a = T
        a *= T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * rx)
        array_compare(a[1], ry * ry)
        array_compare(a[2], rz * rz)

        a = T
        a *= 2
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx * 2)
        array_compare(a[1], ry * 2)
        array_compare(a[2], rz * 2)

        # SE3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]

        a = T * vx
        array_compare(a[:, 0], (rx * vx).flatten())
        array_compare(a[:, 1], (ry * vx).flatten())
        array_compare(a[:, 2], (rz * vx).flatten())

        a = rx * np.vstack((vx, vy, vz)).T
        array_compare(a[:, 0], (rx * vx).flatten())
        array_compare(a[:, 1], (rx * vy).flatten())
        array_compare(a[:, 2], (rx * vz).flatten())

        # divide
        T = SE3([rx, ry, rz])
        a = T / rx
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], ry / rx)
        array_compare(a[2], rz / rx)

        a = rx / T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], rx / ry)
        array_compare(a[2], rx / rz)

        a = T / T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], np.eye(4))
        array_compare(a[1], np.eye(4))
        array_compare(a[2], np.eye(4))

        a = T / 2
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / 2)
        array_compare(a[1], ry / 2)
        array_compare(a[2], rz / 2)

        a = T
        a /= rx
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], ry / rx)
        array_compare(a[2], rz / rx)

        a = rx
        a /= T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / rx)
        array_compare(a[1], rx / ry)
        array_compare(a[2], rx / rz)

        a = T
        a /= T
        self.assertIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], np.eye(4))
        array_compare(a[1], np.eye(4))
        array_compare(a[2], np.eye(4))

        a = T
        a /= 2
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx / 2)
        array_compare(a[1], ry / 2)
        array_compare(a[2], rz / 2)

        # add
        T = SE3([rx, ry, rz])
        a = T + rx
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + rx)
        array_compare(a[1], ry + rx)
        array_compare(a[2], rz + rx)

        a = rx + T
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + rx)
        array_compare(a[1], rx + ry)
        array_compare(a[2], rx + rz)

        a = T + T
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + rx)
        array_compare(a[1], ry + ry)
        array_compare(a[2], rz + rz)

        a = T + 1
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx + 1)
        array_compare(a[1], ry + 1)
        array_compare(a[2], rz + 1)

        # subtract
        T = SE3([rx, ry, rz])
        a = T - rx
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - rx)
        array_compare(a[1], ry - rx)
        array_compare(a[2], rz - rx)

        a = rx - T
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - rx)
        array_compare(a[1], rx - ry)
        array_compare(a[2], rx - rz)

        a = T - T
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - rx)
        array_compare(a[1], ry - ry)
        array_compare(a[2], rz - rz)

        a = T - 1
        self.assertNotIsInstance(a, SE3)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx - 1)
        array_compare(a[1], ry - 1)
        array_compare(a[2], rz - 1)

    def test_angle(self):
        # angle between SO3's
        r1 = SO3.Rx(0.1)
        r2 = SO3.Rx(0.2)
        for metric in range(6):
            self.assertAlmostEqual(r1.angdist(other=r1, metric=metric), 0.0)
            self.assertGreater(r1.angdist(other=r2, metric=metric), 0.0)
            self.assertAlmostEqual(
                r1.angdist(other=r2, metric=metric), r2.angdist(other=r1, metric=metric)
            )
        # angle between SE3's
        p1a, p1b = SE3.Rx(0.1), SE3.Rx(0.1, t=(1, 2, 3))
        p2a, p2b = SE3.Rx(0.2), SE3.Rx(0.2, t=(3, 2, 1))
        for metric in range(6):
            self.assertAlmostEqual(p1a.angdist(other=p1a, metric=metric), 0.0)
            self.assertGreater(p1a.angdist(other=p2a, metric=metric), 0.0)
            self.assertAlmostEqual(p1a.angdist(other=p1b, metric=metric), 0.0)
            self.assertAlmostEqual(
                p1a.angdist(other=p2a, metric=metric),
                p2a.angdist(other=p1a, metric=metric),
            )
            self.assertAlmostEqual(
                p1a.angdist(other=p2a, metric=metric),
                p1a.angdist(other=p2b, metric=metric),
            )
        # angdist is not implemented for mismatched types
        with self.assertRaises(ValueError):
            _ = r1.angdist(p1a)

        with self.assertRaises(ValueError):
            _ = r1._op2(right=p1a, op=r1.angdist)

        with self.assertRaises(ValueError):
            _ = p1a._op2(right=r1, op=p1a.angdist)

        # in general, the _op2 interface enforces an isinstance check.
        with self.assertRaises(TypeError):
            _ = r1._op2(right=(1, 0, 0), op=r1.angdist)

    def test_functions(self):
        # inv
        # .T
        pass

    def test_functions_vect(self):
        # inv
        # .T
        pass

    def test_rtvec(self):
        # OpenCV compatibility functions
        T = SE3.RTvec([0, 1, 0], [2, 3, 4])
        nt.assert_equal(T.t, [2, 3, 4])
        nt.assert_equal(T.R, SO3.Ry(1))

        rvec, tvec = T.rtvec()
        nt.assert_equal(rvec, [0, 1, 0])
        nt.assert_equal(tvec, [2, 3, 4])

    def test_interp(self):
        # This data is taken from https://github.com/rai-opensource/spatialmath-python/issues/165
        se3_1 = SE3()
        se3_1.t = np.array(
            [0.5705748101710814, 0.29623210833184527, 0.10764106509086407]
        )
        se3_1.R = np.array(
            [
                [0.2852875203191073, 0.9581330588259315, -0.024332536551692617],
                [0.9582072394229962, -0.28568756930438033, -0.014882844564011068],
                [-0.021211248608609852, -0.019069722856395098, -0.9995931315303468],
            ]
        )
        assert SE3.isvalid(se3_1.A)

        se3_2 = SE3()
        se3_2.t = np.array(
            [0.5150284150005691, 0.25796537207802533, 0.1558725490743694]
        )
        se3_2.R = np.array(
            [
                [0.42058255728234184, 0.9064420651629983, -0.038380919906699236],
                [0.9070822373513454, -0.4209501599465646, -0.0016665901233428627],
                [-0.01766712176680449, -0.0341137119645545, -0.9992617912561634],
            ]
        )
        assert SE3.isvalid(se3_2.A)

        path_se3 = se3_1.interp(end=se3_2, s=15, shortest=False)

        angle = None
        for i in range(len(path_se3) - 1):
            assert SE3.isvalid(path_se3[i].A)

            if angle is None:
                angle = path_se3[i].angdist(path_se3[i + 1])
            else:
                test_angle = path_se3[i].angdist(path_se3[i + 1])
                assert abs(test_angle - angle) < 1e-6

    def test_mean(self):
        rpy = np.ones((100, 1)) @ np.c_[0.1, 0.2, 0.3]
        T = SE3.RPY(rpy)
        self.assertEqual(len(T), 100)
        m = T.mean()
        self.assertIsInstance(m, SE3)
        array_compare(m, T[0])

        # range of angles, mean should be the middle one, index=25
        T = SE3.Rz(np.linspace(start=0.3, stop=0.7, num=51))
        m = T.mean()
        self.assertIsInstance(m, SE3)
        array_compare(m, T[25])

        # now add noise
        rng = np.random.default_rng(0)  # reproducible random numbers
        rpy += rng.normal(scale=0.00001, size=(100, 3))
        T = SE3.RPY(rpy)
        m = T.mean()
        array_compare(m, SE3.RPY(0.1, 0.2, 0.3))

        T = SE3.Tz(np.linspace(start=-2, stop=1, num=51))
        m = T.mean()
        self.assertIsInstance(m, SE3)
        array_compare(m, T[25])


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()

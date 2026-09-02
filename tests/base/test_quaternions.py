# This file is part of the SpatialMath toolbox for Python
# https://github.com/petercorke/spatialmath-python
#
# MIT License
#
# Copyright (c) 1993-2020 Peter Corke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Contributors:
#
#     1. Luis Fernando Lara Tobar and Peter Corke, 2008
#     2. Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan, 2017 (robopy)
#     3. Peter Corke, 2020

import numpy.testing as nt
import unittest

from spatialmath.base.vectors import *
import spatialmath.base as tr
from spatialmath.base.quaternions import *
import spatialmath as sm
import io


class TestQuaternion(unittest.TestCase):
    def test_ops(self):
        nt.assert_array_almost_equal(qeye(), np.r_[1, 0, 0, 0])

        nt.assert_array_almost_equal(qpure(np.r_[1, 2, 3]), np.r_[0, 1, 2, 3])
        nt.assert_array_almost_equal(qpure([1, 2, 3]), np.r_[0, 1, 2, 3])
        nt.assert_array_almost_equal(qpure((1, 2, 3)), np.r_[0, 1, 2, 3])

        nt.assert_equal(qnorm(np.r_[1, 2, 3, 4]), math.sqrt(30))
        nt.assert_equal(qnorm([1, 2, 3, 4]), math.sqrt(30))
        nt.assert_equal(qnorm((1, 2, 3, 4)), math.sqrt(30))

        nt.assert_array_almost_equal(
            qunit(np.r_[1, 2, 3, 4]), np.r_[1, 2, 3, 4] / math.sqrt(30)
        )
        nt.assert_array_almost_equal(
            qunit([1, 2, 3, 4]), np.r_[1, 2, 3, 4] / math.sqrt(30)
        )

        nt.assert_array_almost_equal(
            qqmul(np.r_[1, 2, 3, 4], np.r_[5, 6, 7, 8]), np.r_[-60, 12, 30, 24]
        )
        nt.assert_array_almost_equal(
            qqmul([1, 2, 3, 4], [5, 6, 7, 8]), np.r_[-60, 12, 30, 24]
        )
        nt.assert_array_almost_equal(
            qqmul(np.r_[1, 2, 3, 4], np.r_[1, 2, 3, 4]), np.r_[-28, 4, 6, 8]
        )

        nt.assert_array_almost_equal(
            qmatrix(np.r_[1, 2, 3, 4]) @ np.r_[5, 6, 7, 8], np.r_[-60, 12, 30, 24]
        )
        nt.assert_array_almost_equal(
            qmatrix([1, 2, 3, 4]) @ np.r_[5, 6, 7, 8], np.r_[-60, 12, 30, 24]
        )
        nt.assert_array_almost_equal(
            qmatrix(np.r_[1, 2, 3, 4]) @ np.r_[1, 2, 3, 4], np.r_[-28, 4, 6, 8]
        )

        nt.assert_array_almost_equal(qpow(np.r_[1, 2, 3, 4], 0), np.r_[1, 0, 0, 0])
        nt.assert_array_almost_equal(qpow(np.r_[1, 2, 3, 4], 1), np.r_[1, 2, 3, 4])
        nt.assert_array_almost_equal(qpow([1, 2, 3, 4], 1), np.r_[1, 2, 3, 4])
        nt.assert_array_almost_equal(qpow(np.r_[1, 2, 3, 4], 2), np.r_[-28, 4, 6, 8])
        nt.assert_array_almost_equal(qpow(np.r_[1, 2, 3, 4], -1), np.r_[1, -2, -3, -4])
        nt.assert_array_almost_equal(
            qpow(np.r_[1, 2, 3, 4], -2), np.r_[-28, -4, -6, -8]
        )

        nt.assert_equal(qisequal(np.r_[1, 2, 3, 4], np.r_[1, 2, 3, 4]), True)
        nt.assert_equal(qisequal(np.r_[1, 2, 3, 4], np.r_[5, 6, 7, 8]), False)
        nt.assert_equal(
            qisequal(
                np.r_[1, 1, 0, 0] / math.sqrt(2),
                np.r_[-1, -1, 0, 0] / math.sqrt(2),
                unitq=True,
            ),
            True,
        )
        nt.assert_equal(isunitvec(qrand()), True)

    def test_display(self):
        s = q2str(np.r_[1, 2, 3, 4])
        nt.assert_equal(isinstance(s, str), True)
        nt.assert_equal(s, " 1.0000 <  2.0000,  3.0000,  4.0000 >")

        s = q2str([1, 2, 3, 4])
        nt.assert_equal(s, " 1.0000 <  2.0000,  3.0000,  4.0000 >")

        s = q2str([1, 2, 3, 4], delim=("<<", ">>"))
        nt.assert_equal(s, " 1.0000 <<  2.0000,  3.0000,  4.0000 >>")

        s = q2str([1, 2, 3, 4], fmt="{:20.6f}")
        nt.assert_equal(
            s,
            "            1.000000 <             2.000000,             3.000000,             4.000000 >",
        )

        # would be nicer to do this with redirect_stdout() from contextlib but that
        # fails because file=sys.stdout is maybe assigned at compile time, so when
        # contextlib changes sys.stdout, qprint() doesn't see it

        f = io.StringIO()
        qprint(np.r_[1, 2, 3, 4], file=f)
        nt.assert_equal(f.getvalue().rstrip(), " 1.0000 <  2.0000,  3.0000,  4.0000 >")

    def test_rotation(self):
        # rotation matrix to quaternion
        nt.assert_array_almost_equal(r2q(tr.rotx(180, "deg")), np.r_[0, 1, 0, 0])
        nt.assert_array_almost_equal(r2q(tr.roty(180, "deg")), np.r_[0, 0, 1, 0])
        nt.assert_array_almost_equal(r2q(tr.rotz(180, "deg")), np.r_[0, 0, 0, 1])

        # quaternion to rotation matrix
        nt.assert_array_almost_equal(q2r(np.r_[0, 1, 0, 0]), tr.rotx(180, "deg"))
        nt.assert_array_almost_equal(q2r(np.r_[0, 0, 1, 0]), tr.roty(180, "deg"))
        nt.assert_array_almost_equal(q2r(np.r_[0, 0, 0, 1]), tr.rotz(180, "deg"))

        nt.assert_array_almost_equal(q2r([0, 1, 0, 0]), tr.rotx(180, "deg"))
        nt.assert_array_almost_equal(q2r([0, 0, 1, 0]), tr.roty(180, "deg"))
        nt.assert_array_almost_equal(q2r([0, 0, 0, 1]), tr.rotz(180, "deg"))

        # quaternion - vector product
        nt.assert_array_almost_equal(
            qvmul(np.r_[0, 1, 0, 0], np.r_[0, 0, 1]), np.r_[0, 0, -1]
        )
        nt.assert_array_almost_equal(qvmul([0, 1, 0, 0], [0, 0, 1]), np.r_[0, 0, -1])

        large_rotation = math.pi + 0.01
        q1 = r2q(tr.rotx(large_rotation), shortest=False)
        q2 = r2q(tr.rotx(large_rotation), shortest=True)
        self.assertLess(q1[0], 0)
        self.assertGreater(q2[0], 0)
        self.assertTrue(qisequal(q1=q1, q2=q2, unitq=True))

    def test_slerp(self):
        q1 = np.r_[0, 1, 0, 0]
        q2 = np.r_[0, 0, 1, 0]

        nt.assert_array_almost_equal(qslerp(q1, q2, 0), q1)
        nt.assert_array_almost_equal(qslerp(q1, q2, 1), q2)
        nt.assert_array_almost_equal(
            qslerp(q1, q2, 0.5), np.r_[0, 1, 1, 0] / math.sqrt(2)
        )

        q1 = [0, 1, 0, 0]
        q2 = [0, 0, 1, 0]

        nt.assert_array_almost_equal(qslerp(q1, q2, 0), q1)
        nt.assert_array_almost_equal(qslerp(q1, q2, 1), q2)
        nt.assert_array_almost_equal(
            qslerp(q1, q2, 0.5), np.r_[0, 1, 1, 0] / math.sqrt(2)
        )

        nt.assert_array_almost_equal(
            qslerp(r2q(tr.rotx(-0.3)), r2q(tr.rotx(0.3)), 0.5), np.r_[1, 0, 0, 0]
        )
        nt.assert_array_almost_equal(
            qslerp(r2q(tr.roty(0.3)), r2q(tr.roty(0.5)), 0.5), r2q(tr.roty(0.4))
        )

    def test_slerp_same_rotation(self):
        # coincident (dot = +1) and antipodal (dot = -1) endpoints are the same
        # rotation, so every interpolate is that rotation
        q = r2q(tr.rotx(0.3))
        for s in (0, 0.25, 0.5, 1):
            for shortest in (False, True):
                nt.assert_array_almost_equal(qslerp(q, q, s, shortest=shortest), q)
                qi = qslerp(q, -q, s, shortest=shortest)
                self.assertAlmostEqual(np.linalg.norm(qi), 1)
                nt.assert_array_almost_equal(q2r(qi), tr.rotx(0.3))

    def test_slerp_near_pi(self):
        # the slerp weights are sin(...)/sin(theta), singular at theta = 0 and pi.
        # Check against the closed form cos(s.theta) q0 + sin(s.theta) v, where v is
        # the unit quaternion orthogonal to q0 in the plane of the great circle.
        q0 = r2q(tr.rpy2r(0.2, 0.3, 0.4))
        v = np.r_[0, 0, 1, 0] - np.dot(np.r_[0, 0, 1, 0], q0) * q0
        v = v / np.linalg.norm(v)

        for theta in (1e-6, 1e-3, 0.5, 1.5, math.pi - 1e-3, math.pi - 1e-6):
            q1 = math.cos(theta) * q0 + math.sin(theta) * v
            for s in (0.25, 0.5, 0.75):
                qi = qslerp(q0, q1, s)
                nt.assert_array_almost_equal(
                    qi, math.cos(s * theta) * q0 + math.sin(s * theta) * v
                )
                self.assertAlmostEqual(np.linalg.norm(qi), 1)

    def test_rotx(self):
        pass

    def test_r2q(self):
        # null rotation case
        R = np.eye(3)
        nt.assert_array_almost_equal(r2q(R), [1, 0, 0, 0])

        R = tr.rotx(np.pi / 2)
        nt.assert_array_almost_equal(r2q(R), np.r_[1, 1, 0, 0] / np.sqrt(2))

        R = tr.rotx(-np.pi / 2)
        nt.assert_array_almost_equal(r2q(R), np.r_[1, -1, 0, 0] / np.sqrt(2))

        R = tr.rotx(np.pi)
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 1, 0, 0])

        R = tr.rotx(-np.pi)
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 1, 0, 0])

        # ry
        R = tr.roty(np.pi / 2)
        nt.assert_array_almost_equal(r2q(R), np.r_[1, 0, 1, 0] / np.sqrt(2))

        R = tr.roty(-np.pi / 2)
        nt.assert_array_almost_equal(r2q(R), np.r_[1, 0, -1, 0] / np.sqrt(2))

        R = tr.roty(np.pi)
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 0, 1, 0])

        R = tr.roty(-np.pi)
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 0, 1, 0])

        # rz
        R = tr.rotz(np.pi / 2)
        nt.assert_array_almost_equal(r2q(R), np.r_[1, 0, 0, 1] / np.sqrt(2))

        R = tr.rotz(-np.pi / 2)
        nt.assert_array_almost_equal(r2q(R), np.r_[1, 0, 0, -1] / np.sqrt(2))

        R = tr.rotz(np.pi)
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 0, 0, 1])

        R = tr.rotz(-np.pi)
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 0, 0, 1])

        # github issue case
        R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(r2q(R), np.r_[0, 1, -1, 0] / np.sqrt(2))

        r1 = sm.SE3.Rx(0.1)
        q1a = np.array([9.987503e-01, 4.997917e-02, 0.000000e00, 2.775558e-17])
        q1b = np.array([4.997917e-02, 0.000000e00, 2.775558e-17, 9.987503e-01])

        nt.assert_array_almost_equal(q1a, r2q(r1.R))
        nt.assert_array_almost_equal(q1a, r2q(r1.R, order="sxyz"))
        nt.assert_array_almost_equal(q1b, r2q(r1.R, order="xyzs"))

        with self.assertRaises(ValueError):
            nt.assert_array_almost_equal(q1a, r2q(r1.R, order="aaa"))

    def test_qangle(self):
        # Test function that calculates angle between quaternions
        q1 = [1.0, 0, 0, 0]
        q2 = [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]  # 90deg rotation about y-axis
        nt.assert_almost_equal(qangle(q1, q2), np.pi / 2)

        q1 = [1.0, 0, 0, 0]
        q2 = [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]  # 90deg rotation about x-axis
        nt.assert_almost_equal(qangle(q1, q2), np.pi / 2)


if __name__ == "__main__":
    unittest.main()

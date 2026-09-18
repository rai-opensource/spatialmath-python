#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Micro-benchmarks for spatialmath: base functions and classes for SO(3),
SE(3), quaternions and twists, plus a NumPy baseline.  Run from a checkout::

    python benchmarks/benchmark_smtb.py

The output starts with a summary of the machine and package versions, so a
pasted table is self-describing.

Created on Fri Apr 10 14:22:36 2020

@author: Peter Corke
"""

import os
import platform
import subprocess
import timeit as _timeit

import numpy as np
from ansitable import ANSITable, Column

import spatialmath

N = 10_000
REPEATS = 5

table = None


def cpu_info() -> str:
    """Best-effort, portable one-line CPU description

    :return: CPU name and core count, plus clock speed if available
    :rtype: str

    No new hard dependency: psutil is used for clock speed only if already
    installed.  Some platforms (e.g. Apple Silicon) don't expose a single
    meaningful clock speed, so a missing or nonsensical reading is omitted.
    """
    system = platform.system()
    name = None

    if system == "Darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif system == "Windows":
        name = platform.processor() or None

    if not name:
        name = platform.processor() or platform.machine() or "unknown CPU"

    info = f"{name} ({os.cpu_count() or '?'} cores)"

    try:
        import psutil

        freq = psutil.cpu_freq()
        # real clock speeds are hundreds to thousands of MHz; some platforms
        # report bogus single-digit values instead of raising
        if freq and freq.max and freq.max > 100:
            info += f", {freq.max:.0f} MHz"
    except Exception:
        pass

    return info


def print_machine_summary() -> None:
    print(f"CPU:          {cpu_info()}")
    print(f"OS:           {platform.platform()}")
    print(f"Python:       {platform.python_version()}")
    print(f"numpy:        {np.__version__}")
    print(f"spatialmath:  {spatialmath.__version__}")
    print(f"Timing:       min of {REPEATS} repeats x {N} calls")


def new_table():
    return ANSITable(
        Column("Operation", headalign="^"),
        Column("Time (μs)", headalign="^", fmt="{:.2f}"),
        border="thick",
    )


def timeit(stmt, label, setup):
    # min-of-repeats suppresses system jitter (GC pauses, OS scheduling)
    # much better than a single long run of the same total work
    t = min(_timeit.repeat(stmt=stmt, setup=setup, number=N, repeat=REPEATS))
    table.row(label, t / N * 1e6)


def section(title):
    # print the previous table (if any), then start a fresh one under a
    # plain header naming the category of test that follows
    global table
    if table is not None:
        table.print()
    print(f"\n{title}\n")
    table = new_table()


print_machine_summary()

# ------------------------------------------------------------------------- #
transforms_setup = '''
from spatialmath import SE3
from spatialmath import base

import numpy as np
from collections import namedtuple
Rt = namedtuple('Rt', 'R t')
X1 = SE3.Rand()
X2 = SE3.Rand()
T1 = X1.A
T2 = X2.A
R1 = base.t2r(T1)
R2 = base.t2r(T2)
t1 = base.transl(T1)
t2 = base.transl(T2)
Rt1 = Rt(R1, t1)
Rt2 = Rt(R2, t2)
v = np.r_[1,2,3]
v2 = np.r_[1,2,3, 1]
'''

section("SO(3) / SE(3) base functions")

timeit('base.getvector(0.2)', "base.getvector(x)", transforms_setup)
timeit('base.rotx(0.2, unit="rad")', "base.rotx(x)", transforms_setup)
timeit('base.trotx(0.2, unit="rad")', "base.trotx(x)", transforms_setup)
timeit('base.t2r(T1)', "base.t2r(T1)", transforms_setup)
timeit('base.r2t(R1)', "base.r2t(R1)", transforms_setup)
timeit('T1 @ T2', "T1 @ T2 (4x4)", transforms_setup)
timeit(
    'T1[:3,:3] @ T2[:3,:3] + T1[:3,:3] @ T2[:3,3]',
    "T1 @ T2 decomposed (slice)",
    transforms_setup,
)
timeit(
    '(Rt1.R @ Rt2.R, Rt1.R @ Rt2.t)',
    "T1 @ T2 decomposed (namedtuple)",
    transforms_setup,
)
timeit('base.trinv(T1)', "base.trinv(T1)", transforms_setup)
timeit(
    '(Rt1.R.T, -Rt1.R.T @ Rt1.t)', "T1 inverse decomposed (R,t)", transforms_setup
)
timeit('np.linalg.inv(T1)', "np.linalg.inv(T1)", transforms_setup)
timeit('T1 @ v2', "T1 @ v2 (4,4)*(4,)", transforms_setup)

# ------------------------------------------------------------------------- #
section("SE3 class")

timeit('SE3()', "SE3()", transforms_setup)
timeit('SE3.Rx(0.2)', "SE3.Rx(x)", transforms_setup)
timeit('SE3(T1)', "SE3(T1)", transforms_setup)
timeit('SE3(T1, check=False)', "SE3(T1, check=False)", transforms_setup)
timeit('SE3([T1], check=False)', "SE3([T1], check=False)", transforms_setup)
timeit('T1[:3,:3]', "T1[:3,:3] (raw slice)", transforms_setup)
timeit('X1.A', "X1.A (property)", transforms_setup)
timeit('X1 * X2', "X1 * X2", transforms_setup)
timeit('X1 @ X2', "X1 @ X2 (normalized)", transforms_setup)
timeit('X1.inv()', "X1.inv()", transforms_setup)
timeit('X1 * v', "X1 * v", transforms_setup)
timeit('a = X1.log()', "X1.log()", transforms_setup)
timeit('SE3.RPY([0.1, 0.2, 0.3])', "SE3.RPY(rpy)", transforms_setup)
timeit('X1.rpy()', "X1.rpy()", transforms_setup)
timeit('SE3.Eul([0.1, 0.2, 0.3])', "SE3.Eul(eul)", transforms_setup)
timeit('X1.eul()', "X1.eul()", transforms_setup)
timeit('X1.UnitQuaternion()', "X1.UnitQuaternion()", transforms_setup)

# ------------------------------------------------------------------------- #
quat_setup = '''
from spatialmath import base
from spatialmath import UnitQuaternion, SO3
import numpy as np
q1 = base.qrand()
q2 = base.qrand()
v = np.r_[1,2,3]
R1 = SO3.Rand().R
Q1 = UnitQuaternion.Rx(0.2)
Q2 = UnitQuaternion.Ry(0.3)
'''

section("Quaternion base functions")

timeit('a = base.qqmul(q1,q2)', "base.qqmul(q1, q2)", quat_setup)
timeit('a = base.qvmul(q1,v)', "base.qvmul(q1, v)", quat_setup)

# ------------------------------------------------------------------------- #
section("UnitQuaternion class")

timeit('a = UnitQuaternion()', "UnitQuaternion()", quat_setup)
timeit('a = UnitQuaternion.Rx(0.2)', "UnitQuaternion.Rx(x)", quat_setup)
timeit('a = UnitQuaternion(R1)', "UnitQuaternion(R1)", quat_setup)
timeit('a = Q1.R', "Q1.R (property)", quat_setup)
timeit('a = Q1 * Q2', "Q1 * Q2", quat_setup)
timeit('a = Q1 * v', "Q1 * v", quat_setup)
timeit('a = Q1.SE3()', "Q1.SE3()", quat_setup)

# ------------------------------------------------------------------------- #
twist_setup = '''
from spatialmath import SE3, Twist3
from spatialmath import base
import numpy as np
from math import cos
S1 = Twist3(SE3.Rand())
S2 = Twist3(SE3.Rand())
X1 = SE3.Rand()
T1 = X1.A
A1 = X1.Ad()
se3 = S1.skewa()
s = np.r_[1,2,3,4,5,6]
v = np.r_[1,2,3]
'''

section("Twist / exponential-map base functions")

timeit('a = base.skew(v)', "base.skew(v)", twist_setup)
timeit('a = base.skewa(s)', "base.skewa(s)", twist_setup)
timeit('a = base.vexa(se3)', "base.vexa(se3)", twist_setup)
timeit('a = base.trlog(T1)', "base.trlog(T1)", twist_setup)
timeit('a = base.trlog(T1, twist=True)', "base.trlog(T1, twist=True)", twist_setup)
timeit('a = base.trexp(se3)', "base.trexp(se3)", twist_setup)
timeit('a = base.rodrigues(v)', "base.rodrigues(v)", twist_setup)
timeit('a = A1 @ s', "A1 @ s (6,6)*(6,)", twist_setup)
timeit('a = cos(0.3)', "math.cos(x)", twist_setup)
timeit('a = np.cos(0.3)', "np.cos(x)", twist_setup)

# ------------------------------------------------------------------------- #
section("Twist3 class")

# NB: Twist3 * Twist3 and Twist3.Ad() both round-trip through the SE3
# exponential/logarithm maps (trexp/trlog above), which is why they cost
# several times more than a single trexp or trlog call - see
# claude-notes/twist3-timing-investigation.md for the profiling detail.

timeit('a = Twist3()', "Twist3()", twist_setup)
timeit('a = Twist3(X1)', "Twist3(X1) (via log)", twist_setup)
timeit('a = S1.inv()', "S1.inv()", twist_setup)
timeit('a = S1 * S2', "S1 * S2 (product of exponentials)", twist_setup)
timeit('a = S1.Ad()', "S1.Ad() (via SE3 + tr2adjoint)", twist_setup)
timeit('a = S1.exp(1)', "S1.exp() -> SE3", twist_setup)

# ------------------------------------------------------------------------- #
misc_setup = """
from spatialmath import base
import numpy as np
s = np.r_[1.0,2,3,4,5,6]
s3 = np.r_[1.0,2,3]
a = np.r_[1.0, 2.0, 3.0]
b = np.r_[-5.0, 4.0, 3.0]

A = np.random.randn(6,6)
As = (A + A.T) / 2
bb = np.random.randn(6)
"""

section("NumPy linear-algebra baseline")

timeit("c = np.linalg.inv(As)", "np.linalg.inv(As)", misc_setup)
timeit("c = np.linalg.pinv(As)", "np.linalg.pinv(As)", misc_setup)
timeit("c = np.linalg.solve(As, bb)", "np.linalg.solve(As, b)", misc_setup)
timeit("c = np.cross(a,b)", "np.cross(a, b)", misc_setup)
timeit("c = base.cross(a,b)", "base.cross(a, b)", misc_setup)
timeit("a = np.inner(s,s).sum()", "np.inner(s, s).sum()", misc_setup)
timeit("a = np.linalg.norm(s) ** 2", "np.linalg.norm(s) ** 2", misc_setup)
timeit("a = base.normsq(s)", "base.normsq(s)", misc_setup)
timeit("a = (s ** 2).sum()", "(s ** 2).sum()", misc_setup)
timeit("a = np.sum(s ** 2)", "np.sum(s ** 2)", misc_setup)
timeit("a = np.linalg.norm(s)", "np.linalg.norm(s) [R6]", misc_setup)
timeit("a = base.norm(s)", "base.norm(s) [R6]", misc_setup)
timeit("a = np.linalg.norm(s3)", "np.linalg.norm(s3) [R3]", misc_setup)
timeit("a = base.norm(s3)", "base.norm(s3) [R3]", misc_setup)

table.print()

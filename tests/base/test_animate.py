#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression coverage for spatialmath.base.animate.

animate()/tranimate()/tranimate2() build a matplotlib FuncAnimation, but
under a non-interactive backend with nothing pumping a real event loop
(exactly the CI/test environment -- see conftest.py, which forces the Agg
backend for the whole session) its per-frame update() callback never
actually fires. Merely constructing the FuncAnimation, or calling
.animate() and letting the result be garbage collected, proves nothing
about whether the interpolation code (trinterp/trinterp2) runs or is
correct -- it only proves construction didn't raise.

FuncAnimation.save() is different: it synchronously steps through every
frame and invokes the real update() callback for each one, regardless of
backend -- the same thing a real playback or movie export would do. We
drive that here with PillowWriter (pure Python, ships with
matplotlib+Pillow, no external ffmpeg binary) purely to force every frame
through the real callback, and record what it actually computed, rather
than analyzing the saved output itself.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

import spatialmath.base as smb
from spatialmath import SE2, SE3


def _drive_all_frames(fa):
    tmp = tempfile.mktemp(suffix=".gif")
    try:
        fa.save(tmp, writer=PillowWriter(fps=5))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class TestAnimate2(unittest.TestCase):
    def test_drives_real_frames(self):
        plt.close("all")
        end = SE2.Rand()
        start = SE2.Rand()
        nframes = 5

        real_trinterp2 = smb.trinterp2
        calls = []

        def recording_trinterp2(*args, **kwargs):
            calls.append(kwargs["s"])
            return real_trinterp2(*args, **kwargs)

        with patch("spatialmath.base.trinterp2", side_effect=recording_trinterp2):
            fa = end.animate(start=start, dims=[-2, 2], nframes=nframes, repeat=False)
            _drive_all_frames(fa)

        self.assertEqual(len(calls), nframes)
        self.assertAlmostEqual(min(calls), 0.0)
        self.assertAlmostEqual(max(calls), 1.0)
        plt.close("all")


class TestAnimate3(unittest.TestCase):
    def test_drives_real_frames(self):
        plt.close("all")
        end = SE3.Rand()
        start = SE3.Rand()
        nframes = 5

        real_trinterp = smb.trinterp
        calls = []

        def recording_trinterp(*args, **kwargs):
            calls.append(kwargs["s"])
            return real_trinterp(*args, **kwargs)

        with patch("spatialmath.base.trinterp", side_effect=recording_trinterp):
            fa = end.animate(start=start, dims=[-2, 2], nframes=nframes, repeat=False)
            _drive_all_frames(fa)

        self.assertEqual(len(calls), nframes)
        self.assertAlmostEqual(min(calls), 0.0)
        self.assertAlmostEqual(max(calls), 1.0)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()

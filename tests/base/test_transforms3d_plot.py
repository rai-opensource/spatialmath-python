#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""


import numpy as np
import numpy.testing as nt
import unittest
from math import pi
import math
from scipy.linalg import logm, expm
import os
import pytest
import sys

from spatialmath.base.transforms3d import *
from spatialmath.base.transformsNd import isR, t2r, r2t, rt2tr

import matplotlib.pyplot as plt
from spatialmath.base.animate import Animate


class Test3D(unittest.TestCase):
    def test_animate_nonidentity_start(self):
        # the axis artists of an Animate must sit where a static trplot puts them, both straight after
        # trplot(end, start=start) and after _draw(end); losing either half of the fix breaks one of these
        def axis_geometry(artists):
            geometry = []
            for h in artists[:3]:  # the three axes of the frame
                # arrow style: compare the shaft only, the head wings differ
                if hasattr(h, "_segments3d"):
                    geometry.append(np.asarray(h._segments3d[0]).T)
                else:
                    geometry.append(np.asarray(h.get_data_3d()))
            return geometry

        def static_geometry(pose, style):
            fig = plt.figure()
            try:
                ax = fig.add_subplot(projection="3d")
                trplot(pose, ax=ax, style=style, frame="moving")
                kinds = ("Line3DCollection",) if style == "arrow" else ("Line3D",)
                return axis_geometry(
                    [a for a in ax.get_children() if type(a).__name__ in kinds]
                )
            finally:
                plt.close(fig)

        start = transl(-1, 0, 2) @ trotz(0.4)
        end = transl(1, 2, 1) @ trotx(0.3)
        for style in ("line", "arrow", "rviz"):
            fig = plt.figure()
            try:
                anim = Animate(dim=[-4, 4])
                anim.trplot(end, start=start, style=style, frame="moving")
                artists = [x.h for x in anim.displaylist]
                for actual, expected in zip(
                    axis_geometry(artists), static_geometry(start, style)
                ):
                    nt.assert_allclose(actual, expected, atol=1e-12)
                anim._draw(end)
                for actual, expected in zip(
                    axis_geometry(artists), static_geometry(end, style)
                ):
                    nt.assert_allclose(actual, expected, atol=1e-12)
            finally:
                plt.close(fig)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true"
        or (sys.platform.startswith("darwin") and sys.version_info < (3, 11)),
        reason="no display in CI / tkinter bug on mac",
    )
    def test_plot(self):
        plt.figure()
        # test options
        trplot(
            transl(1, 2, 3),
            block=False,
            frame="A",
            style="line",
            width=1,
            dims=[0, 10, 0, 10, 0, 10],
        )
        trplot(
            transl(1, 2, 3),
            block=False,
            frame="A",
            style="arrow",
            width=1,
            dims=[0, 10, 0, 10, 0, 10],
        )
        trplot(
            transl(1, 2, 3),
            block=False,
            frame="A",
            style="rgb",
            width=1,
            dims=[0, 10, 0, 10, 0, 10],
        )
        trplot(transl(3, 1, 2), block=False, color="red", width=3, frame="B")
        trplot(
            transl(4, 3, 1) @ trotx(math.pi / 3),
            block=False,
            color="green",
            frame="c",
            dims=[0, 4, 0, 4, 0, 4],
        )

        # test for iterable
        plt.clf()
        T = [transl(1, 2, 3), transl(2, 3, 4), transl(3, 4, 5)]
        trplot(T)

        plt.close("all")

    def test_plot_rejects_unknown_kwarg(self):
        # trplot() used to silently swallow an unrecognized keyword via a
        # bare **kwargs sink instead of raising -- e.g. a typo'd
        # framelabel= (the real parameter is frame=) drew the frame with
        # no label and no error at all.
        with self.assertRaises(TypeError):
            trplot(transl(1, 2, 3), framelabel="A", block=False)

    def test_plot_anaglyph_forwards_params(self):
        # the anaglyph branch recurses into trplot() for each eye; it used
        # to forward only a small fixed subset of parameters (ax, frame,
        # length, style, wtl, flo, d2), silently dropping textcolor,
        # labels, originsize, origincolor, axislabel, axissubscript, width
        # and projection for both eyes.
        plt.figure()
        ax = trplot(
            transl(1, 2, 3),
            frame="A",
            anaglyph=True,
            block=False,
            textcolor="k",
            originsize=0,
        )
        # frame label + 3 axis labels, per eye
        self.assertEqual(len(ax.texts), 8)
        self.assertTrue(all(t.get_color() == "k" for t in ax.texts))
        plt.close("all")

    def test_plot_iterable_forwards_axissubscript(self):
        # the "T is an iterable of transforms" branch used to forward
        # every named parameter except axissubscript when recursing per
        # transform, so axissubscript=False on the outer call was ignored.
        plt.figure()
        T = [transl(1, 2, 3), transl(2, 3, 4)]
        ax = trplot(T, frame="F", axissubscript=False, block=False)
        axis_labels = [t.get_text() for t in ax.texts if t.get_text() != r"$\{F\}$"]
        self.assertTrue(axis_labels)
        self.assertTrue(all("_{" not in text for text in axis_labels))
        plt.close("all")

    @pytest.mark.skipif(
        plt.get_backend().lower() == "agg"
        or os.environ.get("CI") == "true"
        or (sys.platform.startswith("darwin") and sys.version_info < (3, 11)),
        reason="animation wait=True busy-loop never terminates under the "
        "non-interactive Agg backend (no event loop to deregister the "
        "timer callback); needs a real display",
    )
    def test_animate(self):
        tranimate(transl(1, 2, 3), repeat=False, wait=True)

        tranimate(transl(1, 2, 3), repeat=False, wait=True)
        # run again, with axes already created
        tranimate(transl(1, 2, 3), repeat=False, wait=True, dims=[0, 10, 0, 10, 0, 10])

        plt.close("all")
        # test animate with line not arrow, text, test with SO(3)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()

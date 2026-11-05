import unittest
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
from spatialmath.base import *

# test graphics primitives
# TODO check they actually create artists


class TestWindowTitle(unittest.TestCase):
    def teardown_method(self, method):
        plt.close("all")

    def test_sets_the_title_of_a_managed_figure(self):
        from types import SimpleNamespace
        from unittest import mock

        manager = mock.Mock()
        fig = SimpleNamespace(canvas=SimpleNamespace(manager=manager))
        self.assertTrue(set_window_title(fig, "a title"))
        manager.set_window_title.assert_called_once_with("a title")

    def test_pyplot_figure(self):
        # the Agg manager has no window, but does accept a title
        fig = plt.figure()
        self.assertIsInstance(set_window_title(fig, "a title"), bool)

    def test_figure_without_a_manager(self):
        from matplotlib.figure import Figure

        self.assertFalse(set_window_title(Figure(), "a title"))

    def test_manager_without_the_method(self):
        from types import SimpleNamespace

        class Manager:
            pass

        fig = SimpleNamespace(canvas=SimpleNamespace(manager=Manager()))
        self.assertFalse(set_window_title(fig, "a title"))


class TestGraphics(unittest.TestCase):
    def teardown_method(self, method):
        plt.close("all")

    def test_expand_dims_unspecified(self):
        self.assertIsNone(expand_dims())
        for nd in (2, 3):
            with self.subTest(nd=nd):
                self.assertIsNone(expand_dims(nd=nd))
                self.assertIsNone(expand_dims(None, nd=nd))

    def test_expand_dims_specified(self):
        for nd in (2, 3):
            with self.subTest(nd=nd):
                # Zero is an explicit extent, not an unspecified one.
                np.testing.assert_array_equal(expand_dims(0, nd), [0, 0] * nd)
                np.testing.assert_array_equal(expand_dims(3, nd), [-3, 3] * nd)
                np.testing.assert_array_equal(expand_dims([-2, 5], nd), [-2, 5] * nd)
                limits = np.arange(2 * nd)
                np.testing.assert_array_equal(expand_dims(limits, nd), limits)
                with self.assertRaises(ValueError):
                    expand_dims([1, 2, 3], nd)

    def test_expand_dims_invalid_nd(self):
        for nd in (1, 4):
            for dim in (None, 1):
                with self.subTest(nd=nd, dim=dim):
                    with self.assertRaisesRegex(ValueError, "nd is 2 or 3"):
                        expand_dims(dim, nd)

    def test_expand_dims(self):
        np.testing.assert_array_equal(expand_dims(3), [-3, 3, -3, 3])
        np.testing.assert_array_equal(expand_dims(-3), [-3, 3, -3, 3])
        np.testing.assert_array_equal(expand_dims([1, 2]), [1, 2, 1, 2])
        np.testing.assert_array_equal(expand_dims([1, 2, 3, 4]), [1, 2, 3, 4])
        # reversed limits are passed through, matplotlib inverts the axis
        np.testing.assert_array_equal(expand_dims([2, 1]), [2, 1, 2, 1])

        np.testing.assert_array_equal(expand_dims(-3, nd=3), [-3, 3, -3, 3, -3, 3])
        np.testing.assert_array_equal(expand_dims([1, 2], nd=3), [1, 2] * 3)
        np.testing.assert_array_equal(
            expand_dims([1, 2, 3, 4, 5, 6], nd=3), [1, 2, 3, 4, 5, 6]
        )

        with self.assertRaises(ValueError):
            expand_dims([1, 2, 3])
        with self.assertRaises(ValueError):
            expand_dims([1, 2, 3, 4], nd=3)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plotvol2(self):
        plotvol2(5)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plotvol3(self):
        plotvol3(5)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot_point(self):
        plot_point((2, 3))
        plot_point(np.r_[2, 3])
        plot_point((2, 3), "x")
        plot_point((2, 3), "x", text="foo")

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot_text(self):
        plot_text((2, 3), "foo")
        plot_text(np.r_[2, 3], "foo")

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot_box(self):
        plot_box("r--", centre=(-2, -3), wh=(1, 1))
        plot_box(lt=(1, 1), rb=(2, 0), filled=True, color="b")
        plot_box(lrbt=(1, 2, 0, 1), filled=True, color="b")
        plot_box(ltrb=(1, 0, 2, 0), filled=True, color="b")
        plot_box(lt=(1, 2), wh=(2, 3))
        plot_box(lbwh=(1, 2, 3, 4))
        plot_box(centre=(1, 2), wh=(2, 3))

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot_circle(self):
        plot_circle(1, (0, 0), "r")  # red circle
        plot_circle(2, (0, 0), "b--")  # blue dashed circle
        plot_circle(0.5, (0, 0), filled=True, color="y")  # yellow filled circle

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_ellipse(self):
        plot_ellipse(np.diag((1, 2)), (0, 0), "r")  # red ellipse
        plot_ellipse(np.diag((1, 2)), (0, 0), "b--")  # blue dashed ellipse
        plot_ellipse(
            np.diag((1, 2)), centre=(1, 1), filled=True, color="y"
        )  # yellow filled ellipse

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot_homline(self):
        plot_homline((1, 2, 3))
        plot_homline((2, 1, 3))
        plot_homline((1, -2, 3), "k--")

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_cuboid(self):
        plot_cuboid((1, 2, 3), color="g")
        plot_cuboid((1, 2, 3), centre=(2, 3, 4), color="g")
        plot_cuboid((1, 2, 3), filled=True, color="y")

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_sphere(self):
        plot_sphere(0.3, color="r")
        plot_sphere(1, centre=(1, 1, 1), filled=True, color="b")

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_ellipsoid(self):
        plot_ellipsoid(np.diag((1, 2, 3)), color="r")  # red ellipsoid
        plot_ellipsoid(
            np.diag((1, 2, 3)), centre=(1, 2, 3), filled=True, color="y"
        )  # yellow filled ellipsoid

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_cylinder(self):
        plot_cylinder(radius=0.2, centre=(0.5, 0.5, 0), height=[-0.2, 0.2])
        plot_cylinder(
            radius=0.2,
            centre=(0.5, 0.5, 0),
            height=[-0.2, 0.2],
            filled=True,
            resolution=5,
            color="red",
        )

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_cone(self):
        plot_cone(radius=0.2, centre=(0.5, 0.5, 0), height=0.3)
        plot_cone(
            radius=0.2,
            centre=(0.5, 0.5, 0),
            height=0.3,
            filled=True,
            resolution=5,
            color="red",
        )


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main(buffer=True)

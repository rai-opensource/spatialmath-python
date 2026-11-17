import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

from spatialmath import Plane3


@pytest.mark.parametrize(
    "normal,point",
    [
        ([1, 0, 0], [1, 0, 0]),
        ([0, 1, 0], [0, -1, 0]),
        ([1, 2, 0], [1, 0, 0]),
        ([0, 0, 1], [0, 0, 0.5]),
        ([1, 2, 3], [0, 0, 0.5]),
        ([1, 2, 1e-20], [0, 0, 0]),
    ],
)
@pytest.mark.parametrize("scale", [1, -3, 1e-10])
def test_plane_plot_finite_surface(normal, point, scale, monkeypatch):
    bounds = np.array([-2, 3, -3, 2, -1, 4])
    plane = Plane3.PointNormal(point, np.array(normal) * scale)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    surfaces = []
    original = ax.plot_surface

    def record_surface(x, y, z, **kwargs):
        surfaces.append(np.stack((x, y, z)))
        return original(x, y, z, **kwargs)

    monkeypatch.setattr(ax, "plot_surface", record_surface)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plane.plot(bounds=bounds, ax=ax, alpha=0.4)
        assert len(surfaces) == 1
        xyz = surfaces[0].reshape(3, -1)
        xyz = xyz[:, np.all(np.isfinite(xyz), axis=0)]
        assert xyz.shape[1] > 0
        nt.assert_allclose(
            np.array(normal) @ (xyz - np.array(point)[:, None]), 0, atol=1e-12
        )
        assert np.all(xyz >= bounds[::2, None])
        assert np.all(xyz <= bounds[1::2, None])
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("normal", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_plane_plot_uses_axes_bounds(normal):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-2, 2), ylim=(-3, 3), zlim=(-4, 4))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            Plane3.PointNormal([0, 0, 0], normal).plot(ax=ax)
        assert len(ax.collections) == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)

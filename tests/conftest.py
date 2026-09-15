import os

# Force a non-interactive Matplotlib backend for the whole test session,
# before any test module or package code gets a chance to import
# matplotlib.pyplot. Several modules (geom2d, geom3d, spline, animate)
# import pyplot at module load time, so this has to happen here, in
# conftest.py, which pytest guarantees to load before collecting tests.
#
# CI already sets MPLBACKEND=Agg via the workflow env, so this mainly
# fixes local runs, which otherwise use the platform's interactive
# backend and pop up real windows / can hang on plt.pause(). setdefault
# (not a hard override) leaves an escape hatch: run with
# MPLBACKEND=MacOSX pytest ... to actually see a plot when you want to.
os.environ.setdefault("MPLBACKEND", "Agg")

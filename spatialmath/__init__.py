# print("in spatialmath/__init__")

from spatialmath.pose2d import SO2, SE2
from spatialmath.pose3d import SO3, SE3
from spatialmath.baseposematrix import BasePoseMatrix
from spatialmath.geom2d import Line2, LineSegment2, Polygon2, Ellipse
from spatialmath.geom3d import Line3, Plane3
from spatialmath.twist import Twist3, Twist2
from spatialmath.spatialvector import (
    SpatialVelocity,
    SpatialAcceleration,
    SpatialForce,
    SpatialMomentum,
    SpatialInertia,
)
from spatialmath.quaternion import Quaternion, UnitQuaternion
from spatialmath.DualQuaternion import DualQuaternion, UnitDualQuaternion
from spatialmath.spline import BSplineSE3, InterpSplineSE3, SplineFit


__all__ = [
    # pose
    "SO2",
    "SE2",
    "SO3",
    "SE3",
    "BasePoseMatrix",
    "Quaternion",
    "UnitQuaternion",
    "DualQuaternion",
    "UnitDualQuaternion",
    "Twist3",
    "Twist2",
    "SpatialVelocity",
    "SpatialAcceleration",
    "SpatialForce",
    "SpatialMomentum",
    "SpatialInertia",
    "Line3",
    "Plane3",
    "Line2",
    "LineSegment2",
    "Polygon2",
    "Ellipse",
    "BSplineSE3",
    "InterpSplineSE3",
    "SplineFit",
]

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("spatialmath-python")
except importlib.metadata.PackageNotFoundError:
    # running from a source checkout without an installed/editable
    # spatialmath-python distribution -- e.g. importing straight from
    # the repo root
    __version__ = "unknown"

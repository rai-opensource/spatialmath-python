# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

# matplotlib inline

# line.set_data()
# text.set_position()
# quiver.set_offsets(), quiver.set_UVC()
# FancyArrow.set_xy()
from __future__ import annotations
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import spatialmath.base as smb
from collections.abc import Iterable, Iterator
from spatialmath.base.types import *

# global variable holds reference to FuncAnimation object, this is essential
# for animatiion to work
_ani = None


class Animate:
    """
    Animate objects for matplotlib 3d

    An instance of this class behaves like an Axes3D and supports proxies for

    - ``plot``
    - ``quiver``
    - ``text``
    - ``scatter``

    which renders them and also places corresponding objects into a display
    list. These objects are ``Line``, ``Quiver`` and ``Text``.  Only these
    primitives will be animated.

    The objects are all drawn relative to the origin, and will be transformed
    according to the transform that is being animated.

    Example::

        anim = animate.Animate(dims=[0,2]) # set up the 3D axes
        anim.trplot(T, frame='A', color='green')  # draw the frame
        anim.run(repeat=True)  # animate it
    """

    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        dim: Optional[ArrayLike] = None,
        projection: Optional[str] = "ortho",
        labels: Optional[Tuple[str, str, str]] = ("X", "Y", "Z"),
        **kwargs,
    ):
        """
        Construct an Animate object

        :param ax: the axes to plot into, defaults to current axes
        :type ax: Axes3D reference
        :param dim: dimension of plot volume, using the same shorthand as
            ``plotvol3``: a scalar ``A`` gives ``[-A,A]`` on every axis, a
            2-vector ``[A,B]`` gives ``[A,B]`` on every axis, and a 6-vector
            gives ``[xmin, xmax, ymin, ymax, zmin, zmax]`` explicitly.
        :type dim: scalar, array_like(2) or array_like(6)
        :param projection: 3D projection: ortho [default] or persp
        :type projection: str
        :param labels: labels for the axes, defaults to X, Y and Z
        :type labels: 3-tuple of strings

        Will setup to plot into an existing or a new Axes3D instance.

        """
        self.trajectory = None
        self.displaylist = []

        if ax is None:
            #     # no axes specified

            #     fig = plt.gcf()
            #     # check any current axes
            #     for a in fig.axes:
            #         if a.name != "3d":
            #             # if they are not 3D axes, remove them, otherwise will
            #             # get plot errors
            #             a.remove()
            #     if len(fig.axes) == 0:
            #         # no axes in the figure, create a 3D axes
            #         axes = fig.add_subplot(111, projection="3d", proj_type=projection)
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
            #         ax.set_zlabel(labels[2])
            #         ax.autoscale(enable=True, axis="both")
            #     else:
            #         # reuse an existing axis
            #         axes = plt.gca()

            # if dims is not None:
            #     if len(dims) == 2:
            #         dims = dims * 3
            #     ax.set_xlim(dims[0:2])
            #     ax.set_ylim(dims[2:4])
            #     ax.set_zlim(dims[4:6])
            #     # ax.set_aspect('equal')
            ax = smb.plotvol3(ax=ax, dim=dim)
        if dim is not None:
            # same shorthand as plotvol3: scalar A -> [-A,A]*3, [A,B] -> [A,B]*3,
            # or a full [xmin,xmax,ymin,ymax,zmin,zmax]
            dim = smb.expand_dims(dim, nd=3)
            ax.set_xlim(dim[0:2])
            ax.set_ylim(dim[2:4])
            ax.set_zlim(dim[4:])

        self.ax = ax

        # TODO set flag for 2d or 3d axes, flag errors on the methods called later

    def trplot(
        self,
        end: Union[SO3Array, SE3Array],
        start: Optional[Union[SO3Array, SE3Array]] = None,
        **kwargs,
    ):
        """
        Define the transform to animate

        :param end: the final pose SE(3) or SO(3) to display as a coordinate frame
        :type end: ndarray(4,4) or ndarray(3,3)
        :param start: the initial pose SE(3) or SO(3) to display as a coordinate frame, defaults to null
        :type start: ndarray(4,4) or ndarray(3,3)
        :param start: an

        Is polymorphic with ``base.trplot`` and accepts the same parameters.
        This sets up the animation but doesn't execute it.

        :seealso: :func:`run`

        """
        self.trajectory = None
        if not isinstance(end, (np.ndarray, np.generic)) and isinstance(end, Iterable):
            try:
                if len(end) == 1:
                    end = end[0]
                elif len(end) >= 2:
                    self.trajectory = end
            except TypeError:
                # a generator has no len()
                self.trajectory = end

        # stash the final value
        if smb.isrot(end):
            self.end = smb.r2t(end)
        else:
            self.end = end

        if start is None:
            self.start = np.identity(4)
        else:
            if smb.isrot(start):
                self.start = smb.r2t(start)
            else:
                self.start = start

        # draw axes at the origin
        smb.trplot(self.start, ax=self, **kwargs)

    def set_proj_type(self, proj_type: str):
        self.ax.set_proj_type(proj_type)

    def run(
        self,
        movie: Optional[str] = None,
        axes: Optional[plt.Axes] = None,
        repeat: bool = False,
        interval: int = 50,
        nframes: int = 100,
        wait: bool = False,
        **kwargs,
    ):
        """
        Run the animation

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param repeat: animate in endless loop [default False]
        :type repeat: bool
        :param nframes: number of steps in the animation [default 100]
        :type nframes: int
        :param interval: number of milliseconds between frames [default 50]
        :type interval: int
        :param movie: name of file to write MP4 movie into, or True
        :type movie: str, bool
        :param wait: wait until animation is complete, default False
        :type wait: bool

        Animates a 3D coordinate frame moving from the world frame to a frame
        represented by the SO(3) or SE(3) matrix to the current axes.

        .. note::

            - the ``movie`` option requires the ffmpeg package to be installed:
              ``conda install -c conda-forge ffmpeg``
            - if ``movie=True`` then return an HTML5 video which can be displayed in a notebook
              using ``HTML()``
            - invokes the draw() method of every object in the display list
        """

        def update(frame, animation):
            # frame is the result of calling next() on a iterator or generator
            # seemingly the animation framework isn't checking StopException
            # so there is no way to know when this is no longer called.
            # we implement a rather hacky heartbeat style timeout

            if isinstance(frame, float):
                # passed a single transform, interpolate it
                T = smb.trinterp(start=self.start, end=self.end, s=frame)
            elif isinstance(frame, np.ndarray):
                # type is SO3Array or SE3Array when Animate.trajectory is not None
                T = frame
            else:
                # [unlikely] other types are converted to np array
                T = np.array(frame)

            if T.shape == (3, 3):
                T = smb.r2t(T)

            # update the scene
            animation._draw(T)
            self.count += 1  # say we're still running

        if movie is not None:
            repeat = False

        self.count = 1
        if self.trajectory is not None:
            if not isinstance(self.trajectory, Iterator):
                # make it iterable, eg. if a list or tuple
                self.trajectory = iter(self.trajectory)
            frames = self.trajectory
        else:
            frames = iter(np.linspace(0, 1, nframes))

        global _ani
        fig = self.ax.get_figure()
        _ani = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=frames,
            fargs=(self,),
            # blit=False,  # blit leaves a trail and first frame, set to False
            interval=interval,
            repeat=repeat,
            save_count=nframes,
        )

        if movie is True:
            plt.close(fig)
            return _ani.to_html5_video()
        elif isinstance(movie, str):
            # Set up formatting for the movie files
            if os.path.exists(movie):
                print("overwriting movie", movie)
            else:
                print("creating movie", movie)
            FFwriter = animation.FFMpegWriter(
                fps=1000 / interval, extra_args=["-vcodec", "libx264"]
            )
            _ani.save(movie, writer=FFwriter)

        if wait:
            # wait for the animation to finish.  Dig into the timer for this
            # animation and wait for its callback to be deregistered.
            while True:
                plt.pause(0.25)
                if _ani.event_source is None or len(_ani.event_source.callbacks) == 0:
                    break
        return _ani

    def __repr__(self) -> str:
        """
        Human readable version of the display list

        :param self: the animation
        :type self: Animate
        :returns: readable version of the display list
        :rtype: str
        """
        return "Animate(" + ", ".join([x.type for x in self.displaylist]) + ")"

    def __str__(self) -> str:
        return f"Animate(len={len(self.displaylist)}"

    def artists(self) -> List[plt.Artist]:
        """
        List of artists that need to be updated

        :param self: the animation
        :type self: Animate
        :returns: list of artists
        :rtype: list
        """
        return [x.h for x in self.displaylist]

    def _draw(self, T):
        for x in self.displaylist:
            x.draw(T)

    # ------------------- plot()

    class _Line:
        def __init__(self, anim: Animate, h, xs, ys, zs):
            # form 4xN matrix, columns are first/last point in homogeneous form
            p = np.vstack([xs, ys, zs])
            self.p = np.vstack([p, np.ones((p.shape[1],))])
            self.h = h
            self.type = "line"
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            self.h.set_data(p[0, :], p[1, :])
            self.h.set_3d_properties(p[2, :])

    def plot(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, *args: List, **kwargs):
        """
        Plot a polyline

        :param x: list of x-coordinates
        :type x: array_like
        :param y: list of y-coordinates
        :type y: array_like
        :param z: list of z-coordinates
        :type z: array_like

        Other arguments as accepted by the matplotlib method.

        All arrays must have the same length.

        :seealso: :func:`matplotlib.pyplot.plot`
        """

        (h,) = self.ax.plot(x, y, z, *args, **kwargs)
        self.displaylist.append(Animate._Line(self, h, x, y, z))
        return h

    # ------------------- quiver()

    class _Quiver:
        def __init__(self, anim, h):
            self.type = "quiver"
            self.anim = anim
            # for matplotlib 3.1.x
            # ._segments3d is 3x2x3
            #   first index: line segment in the collection
            #   second index: 0 = start, 1 = end
            #   third index: x, y, z components
            # https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib
            #
            # for matplotlib 3.3.x
            # ._segments3d is a 3-element list, each element is 2x3

            # turn to homogeneous form, with columns per point, alternating start, end

            if isinstance(h._segments3d, np.ndarray):
                self.p = np.vstack(
                    [h._segments3d.reshape(6, 3).T, np.ones((1, 6))]
                )  # result is 4x6
            else:
                self.p = np.vstack(
                    [np.hstack([x.T for x in h._segments3d]), np.ones((1, 6))]
                )
            self.h = h
            self.type = "arrow"
            self.anim = anim

        def draw(self, T):
            # import ipdb; ipdb.set_trace()
            p = T @ self.p

            # reshape it
            p = p[0:3, :].T.reshape(3, 2, 3)
            self.h.set_segments(p)

    def quiver(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        u: ArrayLike,
        v: ArrayLike,
        w: ArrayLike,
        *args: List,
        **kwargs,
    ):
        """
        Plot a quiver

        :param x: list of base x-coordinates
        :type x: array_like
        :param y: list of base y-coordinates
        :type y: array_like
        :param z: list of base z-coordinates
        :type z: array_like
        :param u: list of vector x-coordinates
        :type u: array_like
        :param v: list of vector y-coordinates
        :type v: array_like
        :param w: list of vector z-coordinates
        :type w: array_like

        Draws a series of arrows, the bases defined by corresponding elements
        of (x,y,z) and the vector has components defined by corresponding
        elements of (u,v,w).

        Other arguments as accepted by the matplotlib method.

        :seealso: :func:`matplotlib.pyplot.quiver`
        """
        h = self.ax.quiver(x, y, z, u, v, w, *args, **kwargs)
        self.displaylist.append(Animate._Quiver(self, h))

    # ------------------- text()

    class _Text:
        def __init__(self, anim, h, x, y, z):
            self.type = "text"
            self.h = h
            self.p = np.r_[x, y, z, 1]
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            # x2, y2, _ = proj3d.proj_transform(
            #   p[0], p[1], p[2], self.anim.ax.get_proj())
            # self.h.set_position((x2, y2))
            self.h.set_position((p[0], p[1]))
            self.h.set_3d_properties(z=p[2], zdir="x")

    def text(self, x: float, y: float, z: float, *args: List, **kwargs):
        """
        Plot text

        :param x: x-coordinate
        :type x: float
        :param y: float
        :type y: float
        :param z: z-coordinate
        :type z: float
        :param kwargs: Other arguments as accepted by the matplotlib method.

        ``.text(x, y, z, s)`` display the string ``s`` at coordinate
        (``x``, ``y``, ``z``).

        :seealso: :func:`~matplotlib.pyplot.text`
        """
        h = self.ax.text3D(x, y, z, *args, **kwargs)
        self.displaylist.append(Animate._Text(self, h, x, y, z))

    # ------------------- scatter()

    def scatter(
        self, xs: ArrayLike, ys: ArrayLike, zs: ArrayLike, s: float = 0, **kwargs
    ):
        h = self.plot(xs, ys, zs, ".", markersize=0, **kwargs)
        self.displaylist.append(Animate._Line(self, h, xs, ys, zs))

    # ------------------- wrappers for Axes primitives

    def set_xlim(self, *args: List, **kwargs):
        self.ax.set_xlim(*args, **kwargs)

    def set_ylim(self, *args: List, **kwargs):
        self.ax.set_ylim(*args, **kwargs)

    def set_zlim(self, *args: List, **kwargs):
        self.ax.set_zlim(*args, **kwargs)

    def set_xlabel(self, *args: List, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args: List, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)

    def set_zlabel(self, *args: List, **kwargs):
        self.ax.set_zlabel(*args, **kwargs)


class Animate2:
    """
    Animate objects for matplotlib 2d

    An instance of this class behaves like an Axes3D and supports proxies for

    - ``plot``
    - ``quiver``
    - ``text``

    which renders them and also places corresponding objects into a display
    list. These objects are ``Line``, ``Quiver`` and ``Text``.  Only these
    primitives will be animated.

    The objects are all drawn relative to the origin, and will be transformed
    according to the transform that is being animated.

    Example::

        anim = animate.Animate(dims=[0,2]) # set up the 3D axes
        anim.trplot(T, frame='A', color='green')  # draw the frame
        anim.run(loop=True)  # animate it
    """

    def __init__(
        self,
        axes: Optional[plt.Axes] = None,
        dims: Optional[ArrayLike] = None,
        labels: Tuple[str, str] = ("X", "Y"),
        **kwargs,
    ):
        """
        Construct an Animate object

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param dims: dimension of plot volume as [xmin, xmax, ymin, ymax]. If
            dims is [min, max] those limits are applied to the x- and y-axes.
        :type dims: array_like(4) or array_like(2)
        :param projection: 3D projection: ortho [default] or persp
        :type projection: str
        :param labels: labels for the axes, defaults to X, Y and Z
        :type labels: 3-tuple of strings

        Will setup to plot into an existing or a new Axes3D instance.

        """
        self.trajectory = None
        self.displaylist = []

        if axes is None:
            # create an axes
            fig = plt.gcf()
            if fig.axes is None:
                # no axes in the figure, create a 3D axes
                axes = fig.add_subplot(111)
                axes.set_xlabel(labels[0])
                axes.set_ylabel(labels[1])
                axes.autoscale(enable=True, axis="both")
            else:
                # reuse an existing axis
                axes = plt.gca()

        if dims is not None:
            if len(dims) == 2:
                dims = dims * 2
            axes.set_xlim(dims[0:2])
            axes.set_ylim(dims[2:4])
            # ax.set_aspect('equal')

        self.ax = axes

        # set flag for 2d or 3d axes, flag errors on the methods called later

    def trplot2(
        self,
        end: Union[SO2Array, SE2Array],
        start: Optional[Union[SO2Array, SE2Array]] = None,
        **kwargs,
    ):
        """
        Define the transform to animate

        :param end: the final pose SE(2) or SO(2) to display as a coordinate frame
        :type end: ndarray(3,3) or ndarray(2,2)
        :param start: the initial pose SE(2) or SO(2) to display as a coordinate frame, defaults to null
        :type start: ndarray(3,3) or ndarray(2,2)

        Is polymorphic with ``base.trplot`` and accepts the same parameters.
        This sets up the animation but doesn't execute it.

        :seealso: :func:`run`

        """
        if not isinstance(end, (np.ndarray, np.generic)) and isinstance(end, Iterable):
            if len(end) == 1:
                end = end[0]
            elif len(end) >= 2:
                self.trajectory = end

        # stash the final value
        if smb.isrot2(end):
            self.end = smb.r2t(end)
        else:
            self.end = end

        if start is None:
            self.start = np.identity(3)
        else:
            if smb.isrot2(start):
                self.start = smb.r2t(start)
            else:
                self.start = start

        # draw axes at the origin
        smb.trplot2(self.start, ax=self, block=False, **kwargs)

    def run(
        self,
        movie: Optional[str] = None,
        axes: Optional[plt.Axes] = None,
        repeat: bool = False,
        interval: int = 50,
        nframes: int = 100,
        wait: bool = False,
        **kwargs,
    ):
        """
        Run the animation

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes reference
        :param nframes: number of steps in the animation [defaault 100]
        :type nframes: int
        :param repeat: animate in endless loop [default False]
        :type repeat: bool
        :param interval: number of milliseconds between frames [default 50]
        :type interval: int
        :param movie: name of file to write MP4 movie into or True
        :type movie: str, bool
        :returns: Matplotlib animation object
        :rtype: Matplotlib animation object

        Animates a 3D coordinate frame moving from the world frame to a frame
        represented by the SO(2) or SE(2) matrix to the current axes.

        .. note::

            - the ``movie`` option requires the ffmpeg package to be installed:
              ``conda install -c conda-forge ffmpeg``
            - if ``movie=True`` then return an HTML5 video which can be displayed in a notebook
              using ``HTML()``
            - invokes the draw() method of every object in the display list
        """

        def update(frame, animation):
            # frame is the result of calling next() on a iterator or generator
            # seemingly the animation framework isn't checking StopException
            # so there is no way to know when this is no longer called.
            # we implement a rather hacky heartbeat style timeout

            if isinstance(frame, float):
                # passed a single transform, interpolate it
                T = smb.trinterp2(start=self.start, end=self.end, s=frame)
            else:
                # assume it is an SO(2) or SE(2)
                T = frame
            # ensure result is SE(2)
            if T.shape == (2, 2):
                T = smb.r2t(T)

            # update the scene
            animation._draw(T)
            self.count += 1  # say we're still running

        if movie is not None:
            repeat = False

        self.count = 1
        if self.trajectory is not None:
            if not isinstance(self.trajectory, Iterator):
                # make it iterable, eg. if a list or tuple
                self.trajectory = iter(self.trajectory)
            frames = self.trajectory
        else:
            frames = iter(np.linspace(0, 1, nframes))

        global _ani
        fig = self.ax.get_figure()
        _ani = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=frames,
            fargs=(self,),
            # blit=False,
            interval=interval,
            repeat=repeat,
            save_count=nframes,
        )

        if movie is True:
            plt.close(fig)
            return _ani.to_html5_video()
        elif isinstance(movie, str):
            # Set up formatting for the movie files
            if os.path.exists(movie):
                print("overwriting movie", movie)
            else:
                print("creating movie", movie)
            FFwriter = animation.FFMpegWriter(
                fps=1000 / interval, extra_args=["-vcodec", "libx264"]
            )
            _ani.save(movie, writer=FFwriter)

        if wait:
            # wait for the animation to finish.  Dig into the timer for this
            # animation and wait for its callback to be deregistered.
            while True:
                plt.pause(0.25)
                if _ani.event_source is None or len(_ani.event_source.callbacks) == 0:
                    break

        return _ani

    def __repr__(self):
        """
        Human readable version of the display list

        :param self: the animation
        :type self: Animate
        :returns: readable version of the display list
        :rtype: str
        """
        return "Animate2(" + ", ".join([x.type for x in self.displaylist]) + ")"

    def __str__(self):
        return f"Animate2(len={len(self.displaylist)}"

    def artists(self):
        """
        List of artists that need to be updated

        :param self: the animation
        :type self: Animate
        :returns: list of artists
        :rtype: list
        """
        return [x.h for x in self.displaylist]

    def _draw(self, T):
        for x in self.displaylist:
            x.draw(T)

    # ------------------- plot()

    def set_aspect(self, *args, **kwargs):
        self.ax.set_aspect(*args, **kwargs)

    def autoscale(self, *args, **kwargs):
        # self.ax.autoscale(*args, **kwargs)
        pass

    class _Line:
        def __init__(self, anim, h, xs, ys):
            # form 3xN matrix, columns are first/last point in homogeneous form
            p = np.vstack([xs, ys])
            self.p = np.vstack([p, np.ones((p.shape[1],))])
            self.h = h
            self.type = "line"
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            self.h.set_data(p[0, :], p[1, :])

    def plot(self, x, y, *args, **kwargs):
        """
        Plot a polyline

        :param x: list of x-coordinates
        :type x: array_like
        :param y: list of y-coordinates
        :type y: array_like

        Other arguments as accepted by the matplotlib method.

        All arrays must have the same length.

        :seealso: :func:`matplotlib.pyplot.plot`
        """

        (h,) = self.ax.plot(x, y, *args, **kwargs)
        self.displaylist.append(Animate2._Line(self, h, x, y))
        return h

    # ------------------- quiver()

    class _Quiver:
        def __init__(self, anim, h, x, y, u, v):
            self.type = "quiver"
            self.anim = anim

            self.h = h
            self.type = "arrow"
            self.anim = anim

            self.p = np.c_[u - x, v - y].T

        def draw(self, T):
            R, t = smb.tr2rt(T)
            p = R @ self.p
            # specific to a single Quiver
            self.h.set_offsets(t)  # shift the origin
            self.h.set_UVC(p[0], p[1])

    def quiver(self, x, y, u, v, *args, **kwargs):
        """
        Plot a quiver

        :param x: list of base x-coordinates
        :type x: array_like
        :param y: list of base y-coordinates
        :type y: array_like
        :param u: list of vector x-coordinates
        :type u: array_like
        :param v: list of vector y-coordinates
        :type v: array_like


        Draws a series of arrows, the bases defined by corresponding elements
        of (x,y,z) and the vector has components defined by corresponding
        elements of (u,v,w).

        Other arguments as accepted by the matplotlib method.

        :seealso: :func:`matplotlib.pyplot.quiver`
        """
        h = self.ax.quiver(x, y, u, v, *args, **kwargs)
        self.displaylist.append(Animate2._Quiver(self, h, x, y, u, v))

    # ------------------- text()

    class _Text:
        def __init__(self, anim, h, x, y):
            self.type = "text"
            self.h = h
            self.p = np.r_[x, y, 1]
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            # x2, y2, _ = proj3d.proj_transform(
            #   p[0], p[1], p[2], self.anim.ax.get_proj())
            # self.h.set_position((x2, y2))
            self.h.set_position((p[0], p[1]))

    def text(self, x, y, *args, **kwargs):
        """
        Plot text

        :param x: x-coordinate
        :type x: float
        :param y: float
        :type y: float
        :param z: z-coordinate
        :type z: float
        :param kwargs: Other arguments as accepted by the matplotlib method.

        ``.text(x, y, s)`` display the string ``s`` at coordinate
        (``x``, ``y``).

        :seealso: :func:`matplotlib.pyplot.text`
        """
        h = self.ax.text(x, y, *args, **kwargs)
        self.displaylist.append(Animate2._Text(self, h, x, y))

    # ------------------- scatter()

    def scatter(self, x, y, s=0, **kwargs):
        h = self.plot(x, y, ".", markersize=0, **kwargs)
        self.displaylist.append(Animate2._Line(self, h, x, y))

    # ------------------- wrappers for Axes primitives

    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)


if __name__ == "__main__":
    # from spatialmath import UnitQuaternion
    # from spatialmath.base import tranimate, r2t

    # J = np.array([[2, -1, 0], [-1, 4, 0], [0, 0, 3]])
    # dt = 0.05
    # def attitude():
    #     attitude = UnitQuaternion()
    #     w = 0.2 * np.r_[1, 2, 2].T
    #     for t in np.arange(0, 3, dt):
    #         wd =  -np.linalg.inv(J) @ (np.cross(w, J @ w))
    #         w += wd * dt
    #         attitude.increment(w * dt)
    #         yield attitude.R
    # plt.figure()
    # plotvol3(2)
    # tranimate(attitude())

    # T = smb.rpy2r(0.3, 0.4, 0.5)
    # # smb.tranimate(T, wait=True)
    # s = smb.tranimate(T, movie=True)
    # with open("zz.html", "w") as f:
    #     print(f"<html>{s}</html>", file=f)

    T = smb.rot2(2)
    # smb.tranimate2(T, wait=True)
    s = smb.tranimate2(T, movie=True)
    with open("zz.html", "w") as f:
        print(f"<html>{s}</html>", file=f)

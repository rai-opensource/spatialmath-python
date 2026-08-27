from spatialmath.base import plotvol2, plot_point
ax = plotvol2(5)
plot_point((0,0))
plot_point((1,1), 'r*')
plot_point((2,2), 'r*', 'foo')
ax.grid()
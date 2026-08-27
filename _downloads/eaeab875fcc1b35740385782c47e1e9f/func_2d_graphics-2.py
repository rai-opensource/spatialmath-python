from spatialmath.base import plotvol2, plot_arrow
ax = plotvol2(5)
ax.grid()
plot_arrow(
    (-2, -2), (2, 4), label=r"$\mathit{p}_3$", color="r", width=0.1
)
plt.show(block=True)
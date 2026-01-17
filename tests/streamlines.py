import numpy as np
from xali_tools.plots.streamlines import plotStreamlinesFromFlatArray

# Create a sample vector field on a 20x20 grid
n, m = 20, 20
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, m)
xx, yy = np.meshgrid(x, y)

# Simple rotational field: vx = -y, vy = x, vz = 0
vx = -yy.flatten()
vy = xx.flatten()
vz = np.zeros(n * m)

# Flatten to [vx0, vy0, vz0, vx1, vy1, vz1, ...]
vector_field = np.column_stack([vx, vy, vz]).flatten()

# Plot streamlines
plotStreamlinesFromFlatArray(
    vector_field,
    n=n, m=m,
    x_range=(-2, 2),
    y_range=(-2, 2),
    density=1.5,
    title="Rotational Flow"
)
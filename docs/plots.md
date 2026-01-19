# Plots

## Plotting Streamlines

```python
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
```

## Plotting Vector Field (Quiver)

```python
import numpy as np
from xali_tools.plots.vectorfield import plotVectorFieldFromFlatArray

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

# Plot vector field with arrows
plotVectorFieldFromFlatArray(
    vector_field,
    n=n, m=m,
    x_range=(-2, 2),
    y_range=(-2, 2),
    skip=1,                    # Plot every arrow (use 2+ for sparser grids)
    normalize_arrows=False,    # True to show direction only
    title="Rotational Vector Field"
)
```

## Plotting Iso-contours

```python
import numpy as np
from xali_tools.plots.isocontours import plotIsoContoursFromFlatArray

# Create a sample scalar field on a 50x50 grid
n, m = 50, 50
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, m)
xx, yy = np.meshgrid(x, y)

# Gaussian-like scalar field
scalar_field = np.exp(-(xx**2 + yy**2)).flatten()

# Plot iso-contours
plotIsoContoursFromFlatArray(
    scalar_field,
    n=n, m=m,
    x_range=(-2, 2),
    y_range=(-2, 2),
    num_levels=15,
    cmap="viridis",
    title="Gaussian Field"
)
```

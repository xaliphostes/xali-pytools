from xali_tools.plots.isocontours import plotIsoContoursFromFlatArray
from xali_tools.io.tsurf_filter import load_tsurf

# Load the first surface from surfaces.ts
surf = load_tsurf("surfaces.ts", index=0)

print(f"Surface: {surf.name}")
print(f"Vertices: {len(surf.positions) // 3}")
print(f"Properties: {list(surf.properties.keys())}")

# Get the unique property name
prop_name = list(surf.properties.keys())[0]
scalar_field = surf.properties[prop_name]

print(f"Plotting property: {prop_name}")
print(f"  min: {scalar_field.min():.4f}, max: {scalar_field.max():.4f}")

# The surface has 961 vertices = 31 x 31 regular grid
n, m = 31, 31

# Get x, y ranges from positions
positions = surf.positions.reshape(-1, 3)
x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

print(f"  x range: [{x_min:.4f}, {x_max:.4f}]")
print(f"  y range: [{y_min:.4f}, {y_max:.4f}]")

# Plot iso-contours
plotIsoContoursFromFlatArray(
    scalar_field,
    n=n, m=m,
    x_range=(x_min, x_max),
    y_range=(y_min, y_max),
    num_levels=20,
    cmap="viridis",
    title=f"Iso-contours of '{prop_name}'"
)

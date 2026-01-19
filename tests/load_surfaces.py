from xali_tools.io.surface_loader import (
    load_surfaces, save_surfaces
)

# Load all surfaces from a file
surfaces = load_surfaces("surfaces.ts")
print(f"Found {len(surfaces)} surfaces")

for surf in surfaces:
    n_verts = len(surf.positions) // 3
    n_tris = len(surf.indices) // 3
    print(f"  {surf.name}: {n_verts} vertices, {n_tris} triangles")
    # print(list(surf.properties.keys()))
    for name in surf.properties:
      values = surf.properties[name]
      print(f"    -> {name}: min={values.min():.4f}, max={values.max():.4f}")

# Load a specific surface by index (0-based)
# second_surface = load_tsurf("model.ts", index=1)

# Load a specific surface by name
# fault = load_tsurf("model.ts", name="fault_1")

# Save multiple surfaces to a single file
save_surfaces(surfaces, "combined.ts")
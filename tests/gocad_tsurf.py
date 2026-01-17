from xali_tools.io.tsurf_filter import (
    load_all_tsurf, load_tsurf, save_all_tsurf
)

# Load all surfaces from a file
surfaces = load_all_tsurf("surfaces.ts")
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
save_all_tsurf(surfaces, "combined.ts")
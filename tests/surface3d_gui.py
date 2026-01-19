"""
GUI application for 3D triangulated surface visualization with PyVista.

Features:
- Load mesh files (.ts, .gcd, .obj, .stl, .ply, .vtk)
- Select property or coordinate for coloring
- Choose display mode (shading or iso-contours)
- Select colormap
- Interactive 3D view that updates without closing
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from xali_tools.plots import create_pyvista_mesh


# Available colormaps
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdYlBu", "RdYlGn", "Spectral",
    "jet", "rainbow", "turbo",
    "Blues", "Greens", "Reds", "Oranges", "Purples",
    "terrain", "ocean", "gist_earth",
    "bone", "gray", "hot", "cool",
]


class Surface3DViewer:
    """
    Tkinter GUI for viewing 3D triangulated surfaces with PyVista.
    The 3D view stays open and updates interactively.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("3D Surface Viewer")
        self.root.geometry("500x650")

        # Data storage
        self.vertices = None
        self.triangles = None
        self.properties = {}  # name -> array
        self.current_file = None

        # PyVista plotter (persistent)
        self.plotter = None
        self.mesh_actor = None
        self.contour_actor = None
        self.pv_mesh = None

        self._create_widgets()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Clean up plotter on window close."""
        if self.plotter is not None:
            try:
                self.plotter.close()
            except:
                pass
        self.root.destroy()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === File Loading Section ===
        file_frame = ttk.LabelFrame(main_frame, text="File", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        load_btn = ttk.Button(file_frame, text="Load Mesh...", command=self._load_file)
        load_btn.pack(side=tk.RIGHT)

        # === Mesh Info Section ===
        info_frame = ttk.LabelFrame(main_frame, text="Mesh Info", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=4, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X)

        # === Property Selection Section ===
        prop_frame = ttk.LabelFrame(main_frame, text="Property / Coordinate", padding="5")
        prop_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(prop_frame, text="Select field to display:").pack(anchor=tk.W)

        self.property_var = tk.StringVar(value="Z coordinate")
        self.property_combo = ttk.Combobox(
            prop_frame,
            textvariable=self.property_var,
            state="readonly",
            values=["X coordinate", "Y coordinate", "Z coordinate"]
        )
        self.property_combo.pack(fill=tk.X, pady=(5, 0))
        self.property_combo.bind("<<ComboboxSelected>>", lambda e: self._update_view())

        # === Display Mode Section ===
        mode_frame = ttk.LabelFrame(main_frame, text="Display Mode", padding="5")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.mode_var = tk.StringVar(value="shading")

        ttk.Radiobutton(
            mode_frame, text="Shading (colored surface)",
            variable=self.mode_var, value="shading",
            command=self._update_view
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            mode_frame, text="Iso-contours (black lines)",
            variable=self.mode_var, value="isocontours",
            command=self._update_view
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            mode_frame, text="Colored iso-contours",
            variable=self.mode_var, value="colored_contours",
            command=self._update_view
        ).pack(anchor=tk.W)

        # Iso-contour options
        iso_options = ttk.Frame(mode_frame)
        iso_options.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(iso_options, text="Number of contours:").pack(side=tk.LEFT)
        self.num_contours_var = tk.IntVar(value=15)
        contour_spin = ttk.Spinbox(
            iso_options, from_=3, to=50,
            textvariable=self.num_contours_var, width=5,
            command=self._update_view
        )
        contour_spin.pack(side=tk.LEFT, padx=(5, 0))
        contour_spin.bind("<Return>", lambda e: self._update_view())

        # === Colormap Section ===
        cmap_frame = ttk.LabelFrame(main_frame, text="Colormap", padding="5")
        cmap_frame.pack(fill=tk.X, pady=(0, 10))

        self.cmap_var = tk.StringVar(value="viridis")
        self.cmap_combo = ttk.Combobox(
            cmap_frame,
            textvariable=self.cmap_var,
            state="readonly",
            values=COLORMAPS
        )
        self.cmap_combo.pack(fill=tk.X)
        self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._update_view())

        # === Options Section ===
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.show_edges_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame, text="Show mesh edges",
            variable=self.show_edges_var,
            command=self._update_view
        ).pack(anchor=tk.W)

        self.show_scalar_bar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Show scalar bar",
            variable=self.show_scalar_bar_var,
            command=self._update_view
        ).pack(anchor=tk.W)

        # Opacity slider
        opacity_frame = ttk.Frame(options_frame)
        opacity_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(opacity_frame, text="Surface opacity:").pack(side=tk.LEFT)
        self.opacity_var = tk.DoubleVar(value=1.0)
        opacity_scale = ttk.Scale(
            opacity_frame, from_=0.1, to=1.0,
            variable=self.opacity_var, orient=tk.HORIZONTAL,
            command=lambda v: self._update_view()
        )
        opacity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Contour line width
        lw_frame = ttk.Frame(options_frame)
        lw_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(lw_frame, text="Contour line width:").pack(side=tk.LEFT)
        self.line_width_var = tk.DoubleVar(value=2.0)
        lw_scale = ttk.Scale(
            lw_frame, from_=0.5, to=6.0,
            variable=self.line_width_var, orient=tk.HORIZONTAL,
            command=lambda v: self._update_view()
        )
        lw_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # === Buttons Section ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        show_btn = ttk.Button(
            btn_frame, text="Show 3D View",
            command=self._show_plotter
        )
        show_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        reset_btn = ttk.Button(
            btn_frame, text="Reset Camera",
            command=self._reset_camera
        )
        reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # === Status Bar ===
        self.status_var = tk.StringVar(value="Ready - Load a mesh or click 'Load Sample'")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _init_plotter(self):
        """Initialize or get the PyVista plotter."""
        if self.plotter is None:
            self.plotter = pv.Plotter(title="3D Surface Viewer")
            self.plotter.set_background("white")
            self.plotter.add_axes()

    def _show_plotter(self):
        """Show the plotter window."""
        if self.vertices is None:
            messagebox.showwarning("Warning", "No mesh loaded. Please load a file first.")
            return

        self._init_plotter()
        self._rebuild_scene()

        # Show non-blocking
        self.plotter.show(interactive_update=True, auto_close=False)

    def _reset_camera(self):
        """Reset camera to fit the mesh."""
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.update()

    def _update_view(self):
        """Update the 3D view with current settings."""
        if self.plotter is None or self.vertices is None:
            return

        try:
            self._rebuild_scene()
            self.plotter.update()
        except Exception as e:
            self.status_var.set(f"Update error: {str(e)}")

    def _rebuild_scene(self):
        """Rebuild the entire scene with current settings."""
        if self.plotter is None or self.vertices is None:
            return

        # Clear existing actors
        self.plotter.clear_actors()

        # Get current settings
        scalar_field, scalar_name = self._get_scalar_field()
        mode = self.mode_var.get()
        cmap = self.cmap_var.get()
        show_edges = self.show_edges_var.get()
        show_scalar_bar = self.show_scalar_bar_var.get()
        opacity = self.opacity_var.get()
        num_contours = self.num_contours_var.get()
        line_width = self.line_width_var.get()

        # Create PyVista mesh
        self.pv_mesh = create_pyvista_mesh(self.vertices, self.triangles)
        self.pv_mesh["scalars"] = scalar_field

        vmin, vmax = scalar_field.min(), scalar_field.max()
        levels = np.linspace(vmin, vmax, num_contours)

        # Add surface
        if mode == "shading":
            self.plotter.add_mesh(
                self.pv_mesh,
                scalars="scalars",
                cmap=cmap,
                show_edges=show_edges,
                edge_color="gray",
                opacity=opacity,
                clim=(vmin, vmax),
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={"title": scalar_name}
            )

        elif mode == "isocontours":
            # Add semi-transparent surface
            self.plotter.add_mesh(
                self.pv_mesh,
                scalars="scalars",
                cmap=cmap,
                show_edges=show_edges,
                edge_color="gray",
                opacity=opacity,
                clim=(vmin, vmax),
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={"title": scalar_name}
            )
            # Add black contours
            contours = self.pv_mesh.contour(isosurfaces=levels, scalars="scalars")
            if contours.n_points > 0:
                self.plotter.add_mesh(
                    contours,
                    color="black",
                    line_width=line_width
                )

        elif mode == "colored_contours":
            # Add very transparent surface
            self.plotter.add_mesh(
                self.pv_mesh,
                scalars="scalars",
                cmap=cmap,
                show_edges=show_edges,
                edge_color="gray",
                opacity=opacity * 0.3,
                clim=(vmin, vmax),
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={"title": scalar_name}
            )
            # Add colored contours
            contours = self.pv_mesh.contour(isosurfaces=levels, scalars="scalars")
            if contours.n_points > 0:
                self.plotter.add_mesh(
                    contours,
                    scalars="scalars",
                    cmap=cmap,
                    line_width=line_width,
                    clim=(vmin, vmax),
                    show_scalar_bar=False
                )

        # Re-add axes
        self.plotter.add_axes()

        # Update title
        title = f"{scalar_name}"
        if self.current_file:
            title = f"{os.path.basename(self.current_file)} - {scalar_name}"
        self.plotter.add_title(title, font_size=10)

        self.status_var.set(f"Displaying: {scalar_name} ({mode})")

    def _load_file(self):
        """Load a mesh file."""
        filetypes = [
            ("All supported", "*.ts *.gcd *.obj *.stl *.ply *.vtk *.vtp"),
            ("TS files", "*.ts *.gcd"),
            ("OBJ files", "*.obj"),
            ("STL files", "*.stl"),
            ("PLY files", "*.ply"),
            ("VTK files", "*.vtk *.vtp"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Open Mesh File",
            filetypes=filetypes
        )

        if not filename:
            return

        try:
            self.status_var.set(f"Loading {os.path.basename(filename)}...")
            self.root.update()

            ext = os.path.splitext(filename)[1].lower()

            if ext == ".ts" or ext == ".gcd":
                self._load_ts_file(filename)
            else:
                self._load_pyvista_file(filename)

            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))
            self._update_info()
            self._update_property_list()
            self.status_var.set(f"Loaded: {os.path.basename(filename)}")

            # Auto-show if plotter exists
            if self.plotter is not None:
                self._update_view()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_var.set("Error loading file")

    def _load_ts_file(self, filename):
        """Load a .ts or .gcd (GOCAD TSurf) file."""
        vertices = []
        triangles = []
        properties = {}
        prop_names = []

        with open(filename, 'r') as f:
            vertex_map = {}  # GOCAD vertex ID -> array index
            idx = 0

            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == "PVRTX" or parts[0] == "VRTX":
                    # VRTX id x y z [prop1 prop2 ...]
                    vid = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    vertices.append([x, y, z])
                    vertex_map[vid] = idx

                    # Load property values if present
                    if len(parts) > 5:
                        for i, val in enumerate(parts[5:]):
                            prop_name = f"property_{i}" if i >= len(prop_names) else prop_names[i]
                            if prop_name not in properties:
                                properties[prop_name] = []
                            try:
                                properties[prop_name].append(float(val))
                            except ValueError:
                                properties[prop_name].append(0.0)

                    idx += 1

                elif parts[0] == "TRGL":
                    # TRGL v1 v2 v3
                    v1 = vertex_map.get(int(parts[1]), int(parts[1]) - 1)
                    v2 = vertex_map.get(int(parts[2]), int(parts[2]) - 1)
                    v3 = vertex_map.get(int(parts[3]), int(parts[3]) - 1)
                    triangles.append([v1, v2, v3])

                elif parts[0] == "PROPERTIES":
                    # PROPERTIES prop1 prop2 ...
                    prop_names = parts[1:]

        self.vertices = np.array(vertices)
        self.triangles = np.array(triangles)
        self.properties = {k: np.array(v) for k, v in properties.items()}

    def _load_pyvista_file(self, filename):
        """Load a mesh using PyVista."""
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required to load this file type")

        mesh = pv.read(filename)

        # Extract vertices and faces
        self.vertices = np.array(mesh.points)

        # Extract triangles from faces
        if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0:
            faces = mesh.faces
            triangles = []
            i = 0
            while i < len(faces):
                n = faces[i]
                if n == 3:
                    triangles.append(faces[i+1:i+4])
                i += n + 1
            self.triangles = np.array(triangles) if triangles else np.array([]).reshape(0, 3)
        else:
            # Try to extract from cells for other mesh types
            self.triangles = np.array([]).reshape(0, 3)

        # Extract properties
        self.properties = {}
        for name in mesh.array_names:
            arr = mesh[name]
            if arr is not None and len(arr) == len(self.vertices):  # Point data
                self.properties[name] = np.array(arr).flatten()

    def _update_info(self):
        """Update mesh info display."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        if self.vertices is not None:
            info = f"Vertices: {len(self.vertices)}\n"
            info += f"Triangles: {len(self.triangles)}\n"
            info += f"Properties: {len(self.properties)}"
            if self.properties:
                info += f" ({', '.join(self.properties.keys())})"
            info += "\n"

            bounds = (
                f"Bounds: X[{self.vertices[:, 0].min():.1f}, {self.vertices[:, 0].max():.1f}] "
                f"Y[{self.vertices[:, 1].min():.1f}, {self.vertices[:, 1].max():.1f}] "
                f"Z[{self.vertices[:, 2].min():.1f}, {self.vertices[:, 2].max():.1f}]"
            )
            info += bounds
            self.info_text.insert(tk.END, info)

        self.info_text.config(state=tk.DISABLED)

    def _update_property_list(self):
        """Update the property selection combobox."""
        values = ["X coordinate", "Y coordinate", "Z coordinate"]
        values.extend(sorted(self.properties.keys()))
        self.property_combo['values'] = values

        if self.property_var.get() not in values:
            self.property_var.set("Z coordinate")

    def _get_scalar_field(self):
        """Get the scalar field based on current selection."""
        prop = self.property_var.get()

        if prop == "X coordinate":
            return self.vertices[:, 0].copy(), "X"
        elif prop == "Y coordinate":
            return self.vertices[:, 1].copy(), "Y"
        elif prop == "Z coordinate":
            return self.vertices[:, 2].copy(), "Z"
        elif prop in self.properties:
            return self.properties[prop].copy(), prop
        else:
            return self.vertices[:, 2].copy(), "Z"


def create_sample_mesh():
    """Create a sample mesh for testing without loading a file."""
    n = 40
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X * 2) * np.cos(Y * 2) + 0.5 * np.sin(X + Y)

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)

    # Create sample properties
    distance = np.sqrt(X.ravel()**2 + Y.ravel()**2)
    curvature = np.abs(np.sin(X.ravel() * 3) * np.cos(Y.ravel() * 3))
    wave = np.sin(X.ravel() * 2 + Y.ravel())

    properties = {
        "distance": distance,
        "curvature": curvature,
        "wave": wave,
    }

    return vertices, triangles, properties


class Surface3DViewerWithSample(Surface3DViewer):
    """Extended viewer with option to load sample data."""

    def _create_widgets(self):
        super()._create_widgets()

        # Add sample data button after load button
        for widget in self.root.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.LabelFrame) and child.cget("text") == "File":
                    sample_btn = ttk.Button(
                        child, text="Load Sample",
                        command=self._load_sample
                    )
                    sample_btn.pack(side=tk.RIGHT, padx=(5, 0))
                    break

    def _load_sample(self):
        """Load sample mesh data."""
        try:
            self.status_var.set("Creating sample mesh...")
            self.root.update()

            self.vertices, self.triangles, self.properties = create_sample_mesh()
            self.current_file = "sample_mesh"
            self.file_label.config(text="Sample Mesh (wavy surface)")

            self._update_info()
            self._update_property_list()
            self.status_var.set("Sample mesh loaded - Click 'Show 3D View'")

            # Auto-update if plotter exists
            if self.plotter is not None:
                self._update_view()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample:\n{str(e)}")
            self.status_var.set("Error")


def main():
    """Main entry point."""
    if not PYVISTA_AVAILABLE:
        print("Error: PyVista is required. Install with: pip install pyvista")
        return

    root = tk.Tk()
    app = Surface3DViewerWithSample(root)

    # Keep reference to prevent garbage collection
    root.app = app

    root.mainloop()


if __name__ == "__main__":
    main()

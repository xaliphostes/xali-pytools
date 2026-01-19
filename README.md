# Xali Python Tools

A collection of utility tools for geophysics, visualization, and file I/O.

## Installation

```sh
pip install xali-tools
```

Or for development:
```sh
pip install -e .
```

## Documentation

- [Plots](docs/plots.md) - Streamlines, vector fields, iso-contours
- [File I/O](docs/io.md) - STL, OBJ, PLY, OFF, Gocad TSurf loading/saving
- [Math](docs/math.md) - Displacement and stress tensor operations
- [Geophysics](docs/geophysics.md) - Stress inversion from fractures, stylolites, dykes

## Generating the Wheel

Install the necessary packages:
```sh
pip install build
```

Create the wheel:
```sh
python -m build
```

This creates `dist/xali_tools-0.1.0-py3-none-any.whl`.

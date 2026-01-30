#!/usr/bin/env python
"""Convert VTP files to Gocad TSurf format."""

import argparse
from pathlib import Path
from xali_tools.io import load_surface, save_surface


def main():
    parser = argparse.ArgumentParser(
        description="Convert VTP files to Gocad TSurf format"
    )
    parser.add_argument(
        "input",
        help="Input VTP file path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output TSurf file path (default: same name with .ts extension)"
    )
    parser.add_argument(
        "-n", "--name",
        help="Surface name in the TSurf file (default: input filename)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".ts")

    # Load and convert
    print(f"Loading: {input_path}")
    surface = load_surface(str(input_path))

    if args.name:
        surface.name = args.name

    print(f"Saving:  {output_path}")
    save_surface(surface, str(output_path))

    print(f"Done. Converted {surface.n_vertices} vertices, {surface.n_triangles} triangles")
    if surface.properties:
        print(f"Properties: {list(surface.properties.keys())}")

    return 0


if __name__ == "__main__":
    exit(main())
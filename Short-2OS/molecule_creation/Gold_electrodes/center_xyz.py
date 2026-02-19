#!/usr/bin/env python3
"""
Center an XYZ structure by PURE TRANSLATION (no rotation).

Centers can be:
- com      : center of mass
- centroid : simple average of positions
- bbox     : center of bounding box ( (min+max)/2 )

Usage:
  python center_xyz_translate_only.py input.xyz output.xyz --mode com
"""

import numpy as np

# Extend as needed
MASS = {
    "H": 1.00784, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Br": 79.904, "I": 126.90447,
    "Au": 196.96657,
}

def read_xyz(path: str):
    with open(path, "r") as f:
        first = f.readline()
        if not first:
            raise ValueError("Empty file.")
        n = int(first.strip())
        comment = f.readline().rstrip("\n")

        elems = []
        xyz = np.zeros((n, 3), dtype=float)
        extras = []  # keep trailing columns (if any)

        for i in range(n):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading atom {i+1}/{n}")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ line {i+3}: {line!r}")

            elems.append(parts[0])
            xyz[i, 0] = float(parts[1])
            xyz[i, 1] = float(parts[2])
            xyz[i, 2] = float(parts[3])
            extras.append(parts[4:])  # may be empty

    return n, comment, elems, xyz, extras

def write_xyz(path: str, n: int, comment: str, elems, xyz: np.ndarray, extras):
    with open(path, "w") as f:
        f.write(f"{n}\n")
        f.write(comment + "\n")
        for e, (x, y, z), extra in zip(elems, xyz, extras):
            tail = (" " + " ".join(extra)) if extra else ""
            f.write(f"{e:2s} {x: .10f} {y: .10f} {z: .10f}{tail}\n")

def compute_center(elems, xyz: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "centroid":
        return xyz.mean(axis=0)

    if mode == "bbox":
        return 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))

    if mode == "com":
        masses = np.array([MASS.get(el, None) for el in elems], dtype=object)
        if any(m is None for m in masses):
            missing = sorted({el for el, m in zip(elems, masses) if m is None})
            raise ValueError(
                f"Missing masses for elements: {missing}. "
                "Add them to the MASS dict or use --mode centroid/bbox."
            )
        masses = masses.astype(float)
        return (xyz * masses[:, None]).sum(axis=0) / masses.sum()

    raise ValueError("mode must be one of: com, centroid, bbox")

def run(input_xyz: str, output_xyz: str, mode: str = "com", keep_comment: bool = False):
    n, comment, elems, xyz, extras = read_xyz(input_xyz)
    center = compute_center(elems, xyz, mode)

    # PURE TRANSLATION ONLY (no rotation)
    xyz_centered = xyz - center

    if keep_comment:
        new_comment = comment
    else:
        new_comment = f"{comment} | centered by translation ({mode}), shift = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})"

    write_xyz(output_xyz, n, new_comment, elems, xyz_centered, extras)

if __name__ == "__main__":
    # Set input/output paths here
    input_xyz = "Au_electrodes/left.xyz"
    output_xyz = "Au_electrodes_centered/left.xyz"
    mode = "com"  # com, centroid, bbox
    keep_comment = False

    run(input_xyz, output_xyz, mode=mode, keep_comment=keep_comment)

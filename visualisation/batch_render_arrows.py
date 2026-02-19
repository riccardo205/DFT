#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
import pymol2
from pymol import cgo
import numpy as np

# ─── 1) USER CONFIG ────────────────────────────────────────────────────────────
initial_label = "dimers_run2/slow_lowT_lowest_forces"  # subfolder in Short-2OS/runs/ and visualisation/
label_out     = "dimers_run2/slow_lowT_lowest_forces_arrows_rot_x_0_y_0"
ROOT_DIR      = Path(f"Short-2OS/runs/{initial_label}")
OUTPUT_DIR    = Path(f"visualisation/pymol_rendered_cmap/{label_out}")
CSV_PATH      = Path(f"visualisation/{initial_label}/all_spins.csv")
ROTATIONS     = [("x", 0),("y", 0)]  # e.g. [("y", 90)] to rotate camera before saving
PNG_WIDTH     = 800
PNG_DPI       = 500
CONF_ID       = None  # Set to a specific conf_id (e.g., "conf_0") to render only that one, or None for all
# ────────────────────────────────────────────────────────────────────────────────

def load_spin_map(csv_path):
    """Return a dict: molecule_name -> spin_final_all string"""
    spin_map = {}
    if csv_path is not None:
        with open(csv_path, newline='') as csvfile:
            for row in csv.DictReader(csvfile):
                spin_map[row["Folder"].strip()] = row["Spin_final_all"].strip()
    return spin_map

def color_by_spin_colormap(cmd, model_name, spin_str):
    """
    spin_str like "6:-0.210; 22:0.181; …"
    Maps each atom index to a color on a white→red (pos) / white→blue (neg) gradient.
    """
    # parse into list of (idx, intensity)
    items = []
    for token in spin_str.split(';'):
        token = token.strip()
        if not token:
            continue
        idx_s, val_s = token.split(':')
        items.append((int(idx_s), float(val_s)))

    if not items:
        return

    # find maximum absolute intensity for scaling
    max_abs = max(abs(val) for _, val in items)
    if max_abs == 0:
        max_abs = 1.0
        
    carbonGrey = (0.6, 0.6, 0.6)
    for atom_idx, intensity in items:
        t = abs(intensity) / max_abs
        if intensity >= 0:
            # grey → red
            r = carbonGrey[0] + t * (1.0 - carbonGrey[0])
            g = carbonGrey[1] * (1.0 - t)
            b = carbonGrey[2] * (1.0 - t)
        else:
            # grey → blue
            r = carbonGrey[0] * (1.0 - t)
            g = carbonGrey[1] * (1.0 - t)
            b = carbonGrey[2] + t * (1.0 - carbonGrey[2])

        color_name = f"spin_{model_name}_{atom_idx}"
        cmd.set_color(color_name, [r, g, b])
        sel = f"model {model_name} and index {atom_idx}"
        cmd.color(color_name, sel)


def parse_spins(spin_str):
    """Parse spin_str into dict of atom_idx -> intensity (deduplicated, last value wins)."""
    items_dict = {}
    for token in spin_str.split(';'):
        token = token.strip()
        if not token:
            continue
        idx_s, val_s = token.split(':')
        items_dict[int(idx_s)] = float(val_s)
    return items_dict


def get_max_min_spins(items_dict):
    """Find atoms with max positive and max negative Sz values."""
    if not items_dict:
        return None, None
    
    positives = {k: v for k, v in items_dict.items() if v >= 0}
    negatives = {k: v for k, v in items_dict.items() if v < 0}
    
    max_positive = max(positives.items(), key=lambda x: x[1]) if positives else None
    max_negative = min(negatives.items(), key=lambda x: x[1]) if negatives else None
    
    return max_positive, max_negative


def get_vertical_direction(cmd):
    """
    Get the direction vector that points 'up' in the current view.
    Returns (dx, dy, dz) tuple representing the vertical direction in world coordinates.
    """
    view = cmd.get_view()
    # Use rotation matrix column 2 (screen-up direction in world coordinates)
    # Rotation matrix is row-major in view[0:9]
    dx, dy, dz = view[1], view[4], view[7]
    # Normalize
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length > 0:
        return dx/length, dy/length, dz/length
    return 0, 1, 0  # fallback to Y axis


def draw_arrow(cmd, model_name, atom_idx, direction_sign, color, intensity, max_intensity, base_length=0.5):
    """
    Draw an arrow at the specified atom using PyMOL CGO, oriented vertically in view space.
    direction_sign: 1 for up (positive), -1 for down (negative)
    color: [r, g, b] tuple
    intensity: the Sz value (used for scaling arrow length)
    max_intensity: max |Sz| value for normalization
    base_length: base arrow length (will be scaled by normalized intensity)
    """
    try:
        # Get atom coordinates (returns list [x, y, z, ...])
        coords = cmd.get_atom_coords(f"model {model_name} and index {atom_idx}")
        if not coords or len(coords) < 3:
            return
        
        x, y, z = coords[0], coords[1], coords[2]
        
        # Scale arrow length by normalized intensity magnitude
        normalized_intensity = abs(intensity) / max_intensity if max_intensity > 0 else 1.0
        arrow_length = base_length * normalized_intensity
        
        # Get the vertical direction in current view
        dx, dy, dz = get_vertical_direction(cmd)
        
        # Apply direction and length
        dx *= direction_sign * arrow_length
        dy *= direction_sign * arrow_length
        dz *= direction_sign * arrow_length
        
        # Arrow tip position
        x_end = x + dx
        y_end = y + dy
        z_end = z + dz
        
        # Arrow head starts 30% of the way from the tip
        x_head_start = x_end - 0.3 * dx
        y_head_start = y_end - 0.3 * dy
        z_head_start = z_end - 0.3 * dz
        
        # Create arrow using CGO: cylinder shaft + cone head
        arrow_obj = [
            # Cylinder shaft
            cgo.CYLINDER,
            x, y, z,                          # base of cylinder
            x_head_start, y_head_start, z_head_start,  # end of cylinder
            0.18,                             # radius of cylinder
            color[0], color[1], color[2],     # color
            color[0], color[1], color[2],     # color
            # Cone arrowhead
            cgo.CONE,
            x_head_start, y_head_start, z_head_start,  # base of cone
            x_end, y_end, z_end,              # tip of cone
            0.375, 0.0,                       # radius at base, radius at tip
            color[0], color[1], color[2],     # color at base
            color[0], color[1], color[2],     # color at tip
            1, 1                              # cap_at_start, cap_at_end
        ]
        
        cgo_name = f"arrow_{model_name}_{atom_idx}"
        cmd.load_cgo(arrow_obj, cgo_name)
    except Exception as e:
        print(f"[!] Could not draw arrow for atom {atom_idx}: {e}")


def draw_arrows_for_spins(cmd, model_name, spin_str, threshold=0.3):
    """Draw arrows for atoms where |Sz| > threshold."""
    if not spin_str:
        return
    
    items_dict = parse_spins(spin_str)
    
    # Find max intensity for scaling
    max_intensity = max(abs(v) for v in items_dict.values()) if items_dict else 1.0
    
    for atom_idx, intensity in items_dict.items():
        if abs(intensity) > threshold:
            # Positive Sz: red arrow pointing up
            # Negative Sz: blue arrow pointing down
            direction_sign = 1 if intensity >= 0 else -1
            color = [1, 0, 0] if intensity >= 0 else [0, 0, 1]  # red for +, blue for -
            print(f"   Arrow {'↑' if intensity >= 0 else '↓'} at atom {atom_idx} (Sz={intensity:.3f})")
            draw_arrow(cmd, model_name, atom_idx, direction_sign=direction_sign, color=color, 
                      intensity=intensity, max_intensity=max_intensity, base_length=1.8)


def apply_rotations(cmd, rotations):
    """Rotate the camera by a list of (axis, angle_deg) pairs."""
    for axis, angle in rotations:
        cmd.turn(axis, angle)


def save_png(cmd, obj_name, out_dir, width, dpi):
    out_png = (out_dir / obj_name).with_suffix(".png")
    cmd.png(str(out_png), width=width, dpi=dpi, ray=1)
    print(f"✅ {obj_name} → {out_png.name}")

def main(root: Path, out: Path, csv_path: Path, conf_id=None):
    out.mkdir(parents=True, exist_ok=True)
    spin_map = load_spin_map(csv_path)

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.bg_color("white")
        # base colors
        cmd.set_color('carbonGrey', [0.6, 0.6, 0.6])
        base_view = cmd.get_view()

        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            
            # Filter by conf_id if specified
            if conf_id is not None:
                # Extract numeric ID from folder name (e.g., "conf_39_ferro" -> 39)
                try:
                    # Split by underscore and get the second part (the number after "conf_")
                    parts = sub.name.split('_')
                    if len(parts) >= 2 and parts[0] == 'conf':
                        folder_conf_id = int(parts[1])
                        if folder_conf_id != conf_id:
                            continue
                    else:
                        # If format doesn't match, skip
                        continue
                except (ValueError, IndexError):
                    # If parsing fails, skip this folder
                    continue

            xyzs = list(sub.glob("*.xyz"))
            if not xyzs:
                print(f"[!] No .xyz in {sub.name}, skipping.")
                continue

            xyz_path = xyzs[0]
            name     = sub.name

            cmd.set_view(base_view)

            cmd.load(str(xyz_path), name)
            cmd.show('sticks',  name)
            cmd.show('spheres', name)
            cmd.set('sphere_scale', 0.20)
            cmd.color('carbonGrey',      f'model {name} and elem C')
            
            spin_str = spin_map.get(name, "")
            if spin_str:
                color_by_spin_colormap(cmd, name, spin_str)
            
            cmd.center(name)
            cmd.zoom(name, buffer=4)
            if ROTATIONS:
                apply_rotations(cmd, ROTATIONS)
            
            # Save view before drawing arrows so they don't affect zoom
            final_view = cmd.get_view()
            
            if spin_str:
                draw_arrows_for_spins(cmd, name, spin_str)
                cmd.set_view(final_view)  # Restore view to prevent arrow zoom
            
            cmd.color('white',      f'model {name} and elem H')

            save_png(cmd, name, out, width=PNG_WIDTH, dpi=PNG_DPI)
            
            # Clean up: delete model and all associated arrows
            cmd.delete(name)
            # Delete all arrows for this molecule
            cmd.delete(f"arrow_{name}_*")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        ROOT_DIR, OUTPUT_DIR, CSV_PATH = map(Path, sys.argv[1:])
    print(f"Rendering {ROOT_DIR} → {OUTPUT_DIR} using {CSV_PATH}")
    if CONF_ID:
        print(f"Filtering for conf_id: {CONF_ID}")
    main(ROOT_DIR, OUTPUT_DIR, CSV_PATH, conf_id=CONF_ID)

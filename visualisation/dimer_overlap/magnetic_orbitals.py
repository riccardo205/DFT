#!/usr/bin/env python3
"""
The two magnetic orbitals of the dominant intermolecular J of ONE dimer, drawn on top of each other.

1. dominant pair (a, b) = largest |J| among J_13, J_14, J_23, J_24 (same as inspect_dimer.py);
2. BS state: SPIN_CONFIG (e.g. "uddu"; a and b must have opposite spins) or, with None, the state in which
   only b is flipped (e.g. uudu for the pair AC; else the one with only a flipped). A state that was not
   computed is replaced by its global spin reverse (uddu -> duud, same Ms = 0 state with alpha <-> beta);
   the radical with spin "u" sits on the alpha corresponding orbital, the one with "d" on the beta one;
3. magnetic UCO pairs = the min(#u, #d) genuine pairs with the smallest alpha/beta overlap S_UCO (one for a
   single flip, two for a double flip); both corresponding orbitals of each candidate are written as cubes
   with orca_plot (from <state>.uco; cached in OUT/<run>_<dimer>/magnetic_orbitals/cubes) and the pair with the largest
   w_alpha(on its radical) * w_beta(on its radical) is the one of the pair (a, b);
4. everything rotated into the inspect_dimer.py frame: fluorenyl of a flat in xy, fluorenyl of b above it (+z);
5. pyvista render: structure as sticks (the two fluorenyls coloured, the rest grey) and the
   +/- isosurfaces of orbital 1 (on a) and orbital 2 (on b), transparent.
   A pi lobe and its opposite-sign partner lie on the same line along z, so looking down z the two
   colours blend; with FACING_LOBES_TOP the top view keeps only the lobes that face each other
   (orbital 1 above the plane of a, orbital 2 below the plane of b) -- the region that makes the overlap.
Printed check: fraction of |psi|^2 on the radical halves a / b for both orbitals, S_UCO and the grid
integrals int psi1 psi2 and int |psi1 psi2|. With ALIGN_PHASE the sign of orbital 2 is chosen so that
int psi1 psi2 >= 0 (orbital signs are arbitrary); same-colour-type overlap then means constructive.
6. one more top view in the frame of the anthryl of a (the unit bonded to the fluorenyl C9 of a, i.e.
   "fragment 2" of the chain fluorenyl(1)-anthryl(2)-anthryl(3)-fluorenyl(4)): origin = its centroid,
   z = its normal towards the anthryl of b, y = centroid -> anthryl carbon bonded to C9; the two anthryls
   are drawn bold (<state>_anthryl_top.png); this view has its own isovalue and opacity
   (ANTHRYL_ISOVALUE, ANTHRYL_OPACITY) so its lobes can be smaller and more transparent.
Output in OUT_DIR/<run>_<dimer>/magnetic_orbitals/:
  <state>_{top,side,3d,anthryl_top}.png, <state>_top_side.png, cubes/ and <state>_summary.csv (the printed values: one row
  per candidate UCO pair with S_UCO and the |psi|^2 fractions on A-D of its alpha and beta orbital; the chosen
  row also has J, the grid integrals and the display phase of orbital 2).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from ase.io import read
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from inspect_dimer import COL, INTER, RADICALS, fluorenyls, plane, unit  # noqa: E402
from orca_orbitals.cube import BOHR_TO_ANGSTROM, read_cube  # noqa: E402
from orca_orbitals.isosurface import extract_lobes  # noqa: E402
from orca_orbitals.orca_output import parse_output  # noqa: E402
from orca_orbitals.orca_plot import CubeRequest, find_orca_plot, generate_cubes, probe  # noqa: E402

ORB_COL = {1: ("#1f4fd1", "#6fd3ff"), 2: ("#c8141a", "#ffd83a")}   # (positive, negative) lobe of orbital 1 / 2
CONTEXT_COL = "#9a9a9a"


def flipped(lab):
    return "".join("d" if r == lab else "u" for r in RADICALS)


def reverse(cfg):
    return cfg.translate(str.maketrans("ud", "du"))


def pick_state(d, a, b, config=None):
    """(config, label on the alpha orbital, label on the beta orbital, parsed output)."""
    wanted = [config] if config else [flipped(b), flipped(a)]
    for want in wanted:
        if want[RADICALS.index(a)] == want[RADICALS.index(b)]:
            raise ValueError(f"{want}: {a} and {b} have parallel spins, no magnetic pair between them")
        for cfg in (want, reverse(want)):
            out = d / cfg / f"{cfg}.out"
            if out.exists() and (d / cfg / f"{cfg}.uco").exists():
                o = parse_output(out)
                if o.terminated_normally and o.uco_overlaps:
                    if cfg != want:
                        print(f"  {want} not computed, using its spin reverse {cfg}")
                    up = a if cfg[RADICALS.index(a)] == "u" else b
                    return cfg, up, b if up == a else a, o
    raise FileNotFoundError(f"no finished {' / '.join(wanted)} state (or spin reverse) with a .uco file in {d}")


def magnetic_candidates(o, cfg):
    """[(UCO index, S_UCO)]: the min(#u, #d) genuine pairs (i < N_beta) with the smallest overlap."""
    n_pair = round(min(o.n_alpha, o.n_beta))
    ov = {i: s for i, s in o.uco_overlaps.items() if i < n_pair}
    return sorted(ov.items(), key=lambda t: t[1])[:min(cfg.count("u"), cfg.count("d"))]


def make_cubes(uco, index, grid, cube_dir):
    """{operator: cube path} for the alpha (0) and beta (1) corresponding orbital `index`."""
    paths = {op: cube_dir / f"{uco.stem}_uco{index}_{'ab'[op]}_g{grid}.cube" for op in (0, 1)}
    todo = [CubeRequest(index, op, p) for op, p in paths.items() if not p.exists()]
    if todo:
        exe = find_orca_plot()
        info = probe(exe, uco, cube_dir / "_probe")
        print(f"  orca_plot: {len(todo)} cube(s) from {uco.name} (grid {grid}) ...")
        res, log = generate_cubes(exe, uco, todo, grid, cube_dir / "_work", info)
        bad = [m for ok, m in res.values() if not ok]
        if bad:
            (cube_dir / "orca_plot.log").write_text(log)
            raise RuntimeError(f"orca_plot failed: {bad}")
    return paths


def anthryls(numbers, halves, fl):
    """{radical: anthryl atom indices}: heavy atoms of the radical half minus its fluorenyl."""
    out = {}
    for lab, (fl_idx, _) in fl.items():
        an = sorted(k for k in halves[lab] if numbers[k] > 1 and k not in fl_idx)
        out[lab] = an
    return out


def frame(pos, unit_p, ref_p, unit_q):
    """Origin = centroid of unit_p, z = its normal towards unit_q, y = centroid -> atom ref_p projected in the plane
    (with the fluorenyl of a and its C9: the inspect_dimer.py frame)."""
    cP, nP, _ = plane(pos[unit_p])
    cQ = pos[unit_q].mean(0)
    z = nP if (cQ - cP) @ nP >= 0 else -nP
    y = unit((pos[ref_p] - cP) - ((pos[ref_p] - cP) @ z) * z)
    return cP, np.vstack([np.cross(y, z), y, z])


def half_weights(cube, halves):
    """{radical: fraction of psi^2 on the grid points closest to an atom of that radical half}."""
    pts = cube.cartesian_grid().reshape(-1, 3)
    owner = cKDTree(cube.positions).query(pts)[1]
    w = cube.volume().ravel() ** 2
    w = w / w.sum()
    return {lab: float(w[np.isin(owner, sorted(idx))].sum()) for lab, idx in halves.items()}


def structure_meshes(X, numbers, units, a, b, show_h, radius):
    """[(mesh, colour)]: sticks, bonds < 1.75 A (C-H < 1.2 A); units[a] / units[b] coloured, the rest grey."""
    target = {k: COL["P"] if lab == a else COL["Q"] for lab in (a, b) for k in units[lab]}
    keep = np.where((numbers > 1) | show_h)[0]
    d = np.linalg.norm(X[keep, None] - X[None, keep], axis=-1)
    by_col = {}
    for p, q in zip(*np.nonzero(np.triu(d < 1.75, 1))):
        u, v = keep[p], keep[q]
        if (numbers[u] == 1 or numbers[v] == 1) and d[p, q] > 1.2:
            continue
        col = target[u] if u in target and v in target and target[u] == target[v] else CONTEXT_COL
        r = radius * (1.5 if col != CONTEXT_COL else 1.0)
        by_col.setdefault((col, r), []).append(
            pv.Cylinder(center=(X[u] + X[v]) / 2, direction=X[v] - X[u], radius=r,
                        height=float(np.linalg.norm(X[v] - X[u])), resolution=16, capping=False))
    return [(pv.merge(c), col) for (col, _), c in by_col.items()]


def lobe_mesh(v, f):
    m = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
    return m.compute_normals(point_normals=True, cell_normals=False, consistent_normals=True)


def facing(mesh, z_cut, keep_above):
    """Part of a lobe mesh above (keep_above) or below z_cut."""
    m = mesh.copy()
    m["z"] = m.points[:, 2]
    return m.clip_scalar("z", value=z_cut, invert=not keep_above)


def render(png, mol, lobes, direction, up, X, opacity, width_px, margin):
    """Orthographic render looking from `direction` (unit vector, camera side) with screen-up `up`."""
    direction, up = np.asarray(direction, float), np.asarray(up, float)
    right = np.cross(-direction, up)
    P = X - X.mean(0)
    half_w = np.ptp(P @ right) / 2 + margin
    half_h = np.ptp(P @ up) / 2 + margin
    centre = X.mean(0) + ((P @ right).max() + (P @ right).min()) / 2 * right + \
        ((P @ up).max() + (P @ up).min()) / 2 * up
    win = [int(width_px), int(width_px * half_h / half_w)]
    pl = pv.Plotter(off_screen=True, window_size=win, lighting="light_kit")
    pl.set_background("white")
    for m, c in mol:
        pl.add_mesh(m, color=c, smooth_shading=True, specular=0.2, ambient=0.35, diffuse=0.7)
    for m, c in lobes:
        pl.add_mesh(m, color=c, opacity=opacity, smooth_shading=True, specular=0.3)
    pl.enable_depth_peeling(number_of_peels=12)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [tuple(centre + 100 * direction), tuple(centre), tuple(up)]
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = half_h
    pl.camera.clipping_range = (1.0, 300.0)
    pl.screenshot(str(png))
    pl.close()


def extract_raw(c1, c2, phase2, isovalue):
    """[(vertices, faces, colour, orbital)] of the +/- lobes of orbital 1 and 2, cube frame."""
    raw = []
    for k, (cube, ph) in enumerate(((c1, 1.0), (c2, phase2)), 1):
        lb, notes, _ = extract_lobes(cube, isovalue, phase=ph)
        for note in notes:
            print(f"  [orbital {k}] {note}")
        for name, col in zip(("positive", "negative"), ORB_COL[k]):
            if lb[name] is not None:
                raw.append((lb[name].vertices, lb[name].faces, col, k))
    return raw


def main(key, dimers_root, csv_root, out_dir, grid, isovalue, opacity, side_axis, show_h, align_phase,
         stick_radius, width_px, facing_top, spin_config, render_images=True,
         anthryl_isovalue=None, anthryl_opacity=None):
    """Returns the summary row of the chosen UCO pair; render_images=False skips the pictures.
    anthryl_isovalue / anthryl_opacity apply only to <state>_anthryl_top.png (None = same as the others)."""
    run, dimer = key.split("/", 1)
    d = dimers_root / f"{run}_broken_symmetry" / "relaxed" / dimer
    J = pd.read_csv(csv_root / f"{run}_broken_symmetry" / "exchange_couplings.csv")
    J = J[(J["stage"] == "relaxed") & (J["dimer"] == dimer)].iloc[0]
    pair = max(INTER, key=lambda p: abs(J[f"J_{p}_meV"]))
    a, b = RADICALS[int(pair[0]) - 1], RADICALS[int(pair[1]) - 1]

    cfg, on_alpha, on_beta, o = pick_state(d, a, b, spin_config)
    out = out_dir / key.replace("/", "_") / "magnetic_orbitals"
    halves = {k: set(v) for k, v in json.loads((d / "radical_fragments.json").read_text())["fragments"].items()}
    best, rows = None, []
    for i, s_uco in magnetic_candidates(o, cfg):
        paths = make_cubes(d / cfg / f"{cfg}.uco", i, grid, out / "cubes")
        ca, cb = read_cube(paths[0]), read_cube(paths[1])
        wa, wb = half_weights(ca, halves), half_weights(cb, halves)
        score = wa[on_alpha] * wb[on_beta]
        print(f"  {cfg} UCO {i}: S = {s_uco:.4f}, alpha on " + ", ".join(f"{k} {v:.2f}" for k, v in wa.items())
              + "; beta on " + ", ".join(f"{k} {v:.2f}" for k, v in wb.items()))
        rows.append({"dimer": key, "state": cfg, "uco": i, "S_uco": s_uco,
                     **{f"alpha_on_{k}": v for k, v in wa.items()}, **{f"beta_on_{k}": v for k, v in wb.items()}})
        if best is None or score > best[0]:
            best = (score, i, s_uco, ca, cb)
    _, i, s_uco, ca, cb = best
    cubes = {on_alpha: ca, on_beta: cb}
    c1, c2 = cubes[a], cubes[b]                                   # orbital 1 on a, orbital 2 on b

    atoms = read(d / cfg / "geometry.xyz")
    pos, numbers = c1.positions_angstrom, atoms.numbers
    fl = fluorenyls(atoms, halves)
    dv = abs(np.linalg.det(c1.axes))
    s_grid = float((c1.volume() * c2.volume()).sum() * dv)
    s_abs = float(np.abs(c1.volume() * c2.volume()).sum() * dv)
    phase2 = -1.0 if align_phase and s_grid < 0 else 1.0
    w1, w2 = half_weights(c1, halves), half_weights(c2, halves)
    print(f"  {key}: J_{pair} ({a}{b}) = {J[f'J_{pair}_meV']:+.3f} meV;  state {cfg}, UCO {i}, S_UCO = {s_uco:.4f}")
    print(f"  orbital 1 ({'alpha' if a == on_alpha else 'beta'}): |psi|^2 on {a} {w1[a]:.2f}, on {b} {w1[b]:.2f}")
    print(f"  orbital 2 ({'alpha' if b == on_alpha else 'beta'}): |psi|^2 on {b} {w2[b]:.2f}, on {a} {w2[a]:.2f}")
    print(f"  grid: int psi1 psi2 = {s_grid:+.4f}   int |psi1 psi2| = {s_abs:.4f}"
          + ("   (orbital 2 sign flipped for display)" if phase2 < 0 else ""))
    for r in rows:
        r.update({"chosen": r["uco"] == i, "pair": f"J_{pair} ({a}{b})", "J_meV": float(J[f"J_{pair}_meV"])}
                 if r["uco"] == i else {"chosen": False})
        if r["chosen"]:
            r.update({"orbital1": f"{a} ({'alpha' if a == on_alpha else 'beta'})",
                      "orbital2": f"{b} ({'alpha' if b == on_alpha else 'beta'})",
                      "int_psi1_psi2": s_grid, "int_abs_psi1_psi2": s_abs, "orbital2_sign_flipped": phase2 < 0})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{cfg}_summary.csv", index=False)
    chosen = next(r for r in rows if r["chosen"])
    if not render_images:
        return chosen

    raw = extract_raw(c1, c2, phase2, isovalue)                    # (vertices, faces, colour, orbital), cube frame

    cP, R = frame(pos, fl[a][0], fl[a][1], fl[b][0])
    X = (pos - cP) @ R.T
    mol = structure_meshes(X, numbers, {lab: fl[lab][0] for lab in (a, b)}, a, b, show_h, stick_radius)
    lobes = [(lobe_mesh((v - cP) @ R.T, f), col, k) for v, f, col, k in raw]   # (mesh, colour, orbital)
    d_perp = X[fl[b][0]].mean(0)[2]
    top_lobes = [(facing(m, 0.0, True) if k == 1 else facing(m, d_perp, False), c, k) for m, c, k in lobes] \
        if facing_top else lobes

    heavy = X[numbers > 1]
    h = np.eye(3)[0 if side_axis == "x" else 1]
    views = {"top": ([0, 0, 1], [0, 1, 0]),
             "side": (np.cross(h, [0, 0, 1]), [0, 0, 1]),             # screen-right = side_axis, up = z
             "3d": (unit(np.array([0.9, -1.3, 0.9])), unit(np.array([-0.3, 0.45, 1.0])))}
    views["3d"] = (views["3d"][0], unit(views["3d"][1] - (views["3d"][1] @ views["3d"][0]) * views["3d"][0]))
    out.mkdir(parents=True, exist_ok=True)
    for name, (dv_, up) in views.items():
        sel = top_lobes if name == "top" else lobes
        render(out / f"{cfg}_{name}.png", mol, [(m, c) for m, c, _ in sel if m.n_points], dv_, up,
               heavy, opacity, width_px, 2.5)

    head = (f"{key}   J_{pair} ({a}{b}) = {J[f'J_{pair}_meV']:+.2f} meV   {cfg}, UCO {i}: "
            f"S = {s_uco:.3f}")
    key_txt = (f"orbital 1 on {a}: + blue / − cyan      orbital 2 on {b}: + red / − yellow      "
               f"∫ψ₁ψ₂ = {s_grid * phase2:+.3f}, ∫|ψ₁ψ₂| = {s_abs:.3f}, iso ±{isovalue}")
    imgs = [plt.imread(out / f"{cfg}_{v}.png") for v in ("top", "side")]
    hs = [im.shape[0] / im.shape[1] for im in imgs]
    fig, axes = plt.subplots(2, 1, figsize=(7, 7 * sum(hs) + 0.9), gridspec_kw={"height_ratios": hs})
    for ax, im, t in zip(axes, imgs, ("top view (down z)" + (", facing lobes only" if facing_top else ""), f"side view (along {'y' if side_axis == 'x' else 'x'})")):
        ax.imshow(im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{head}\n{key_txt}", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / f"{cfg}_top_side.png", dpi=250)
    plt.close(fig)

    # top view in the frame of the anthryl bonded to the radical fluorenyl of a ("fragment 2")
    an = anthryls(numbers, halves, fl)
    attach = {lab: min(an[lab], key=lambda k: np.linalg.norm(pos[k] - pos[fl[lab][1]])) for lab in (a, b)}
    cA, RA = frame(pos, an[a], attach[a], an[b])
    XA = (pos - cA) @ RA.T
    molA = structure_meshes(XA, numbers, {lab: an[lab] for lab in (a, b)}, a, b, show_h, stick_radius)
    iso_a = isovalue if anthryl_isovalue is None else anthryl_isovalue
    raw_a = raw if iso_a == isovalue else extract_raw(c1, c2, phase2, iso_a)
    lobesA = [(lobe_mesh((v - cA) @ RA.T, f), col, k) for v, f, col, k in raw_a]
    if facing_top:
        dA = XA[an[b]].mean(0)[2]
        lobesA = [(facing(m, 0.0, True) if k == 1 else facing(m, dA, False), c, k) for m, c, k in lobesA]
    render(out / f"{cfg}_anthryl_top.png", molA, [(m, c) for m, c, _ in lobesA if m.n_points], [0, 0, 1], [0, 1, 0],
           XA[numbers > 1], opacity if anthryl_opacity is None else anthryl_opacity, width_px, 2.5)
    print(f"  wrote {out}")
    return chosen


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    KEYS = ["run2/conf_11"]                               # one or more "<run>/<dimer>"
    SPIN_CONFIG = None                                   # BS state, e.g. "uddu"; None = only the second radical flipped
    GRID = 100                                           # orca_plot grid points per axis
    ISOVALUE = 0.02                                      # +/- orbital amplitude of the isosurfaces
    OPACITY = 1                                        # isosurface opacity
    ANTHRYL_ISOVALUE = 0.035                             # <state>_anthryl_top.png only: larger = smaller lobes
    ANTHRYL_OPACITY = 0.45                               # <state>_anthryl_top.png only
    FACING_LOBES_TOP = False                              # top view: only the lobes between the two planes
    SIDE_AXIS = "x"                                      # horizontal axis of the side view: "x" or "y"
    SHOW_HYDROGENS = False
    ALIGN_PHASE = True                                   # flip orbital 2 so that int psi1 psi2 >= 0
    STICK_RADIUS = 0.09                                  # Angstrom (the two fluorenyls 1.5x)
    WIDTH_PX = 1600
    DIMERS_ROOT = HERE.parents[1] / "runs" / "orca" / "short-2OS" / "dimers_run2"
    CSV_ROOT = HERE.parents[1] / "visualisation" / "output" / "orca" / "dimers_run2"
    OUT_DIR = CSV_ROOT / "dimer_overlap"

    for k in KEYS:
        main(k, DIMERS_ROOT, CSV_ROOT, OUT_DIR, GRID, ISOVALUE, OPACITY, SIDE_AXIS, SHOW_HYDROGENS,
             ALIGN_PHASE, STICK_RADIUS, WIDTH_PX, FACING_LOBES_TOP, SPIN_CONFIG,
             anthryl_isovalue=ANTHRYL_ISOVALUE, anthryl_opacity=ANTHRYL_OPACITY)

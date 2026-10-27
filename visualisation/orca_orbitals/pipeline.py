"""
Orchestration: discovery -> output parsing -> common view/alignment -> cubes ->
isosurfaces -> images -> contact sheets -> metadata.

Every calculation is processed inside its own try/except, so one bad job never
stops a directory-wide run; the reason is written to skipped_calculations.csv.
"""

from __future__ import annotations

import datetime
import fnmatch
import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import alignment as al
from .cube import read_cube
from .discovery import discover
from .isosurface import extract_lobes, phase_factor
from .orca_output import OrcaOutputError, check_usable, parse_output, parse_uno_occupations, select_frontier
from .orca_plot import CubeRequest, OrcaPlotError, find_orca_plot, generate_cubes, probe, validate_cube_file
from .render import (Camera, RenderStyle, camera_contains, contact_sheet, frame_camera, molecule_meshes,
                     render_orbital, symbols_to_numbers)
from .tracking import assign_identity

VIEW_FILE = "view_settings.json"
SPIN_OPERATOR = {"alpha": 0, "closed": 0, "beta": 1}
SPIN_SYMBOL = {"alpha": "α", "beta": "β", "closed": ""}
GEOM_TOL = 2e-3    # Angstrom: orbital-file geometry vs final geometry of the output
COLUMNS = [
    "calculation", "calculation_dir", "calculation_name", "electronic_state", "efield_au",
    "orbital_type", "spin", "orbital_index", "orbital_label", "energy_eh", "energy_ev", "occupation",
    "uco_overlap", "cube_path", "image_path", "isovalue", "grid_size", "phase_convention", "sign_flipped",
    "positive_lobe", "negative_lobe", "min_amplitude", "max_amplitude", "status", "message",
    "identity", "identity_method", "alignment_rmsd_A", "n_alpha", "n_beta", "s2", "final_energy_eh",
]


@dataclass
class Settings:
    root: Path
    recursive: bool = True
    orbitals: list = field(default_factory=lambda: ["HOMO-1", "HOMO", "LUMO", "LUMO+1"])
    spins: list = field(default_factory=lambda: ["alpha", "beta"])
    grid_size: int = 90
    isovalue: float = 0.02
    render_mo: bool = True
    render_uno: bool = True
    render_uco: bool = True
    output_dir_name: str = "orbital_analysis"
    central_output: bool = True
    orca_plot: str | None = None
    regenerate_cubes: bool = False
    keep_cubes: bool = True
    smoke_test: bool = False
    smoke_test_calc: str | None = None
    include_patterns: list = field(default_factory=list)
    exclude_patterns: list = field(default_factory=list)
    ignore_dir_names: list = field(default_factory=list)
    align: bool = True
    align_atoms: list | None = None
    reference: str | None = None
    view_rotation_deg: tuple = (0.0, 0.0, 0.0)
    frame_margin: float = 3.0
    reuse_saved_view: bool = True
    phase_convention: str = "none"
    uno_occupation_range: tuple = (0.05, 1.95)
    uno_max_orbitals: int = 6
    uco_max_overlap: float = 0.95
    uco_max_pairs: int = 4
    style: RenderStyle = field(default_factory=RenderStyle)


class CalcSkip(Exception):
    pass


class SetupError(RuntimeError):
    pass


class Log:
    def __init__(self, path=None, prefix="", echo=True):
        self.path, self.prefix, self.echo, self.lines = path, prefix, echo, []

    def __call__(self, msg, level="INFO"):
        self.lines.append(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {level:7s} {msg}")
        if self.echo:
            print(f"{self.prefix}{'' if level == 'INFO' else level + ': '}{msg}", flush=True)

    def write(self, path=None):
        p = Path(path or self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(self.lines) + "\n")


# ---------------------------------------------------------------------------
# common view: reference geometry, alignment, display frame, camera
# ---------------------------------------------------------------------------

@dataclass
class View:
    frame: al.RigidTransform
    camera: Camera
    align: bool
    align_atoms: list
    reference: str
    ref_symbols: list
    ref_coords: np.ndarray
    view_rotation_deg: tuple
    window_size: tuple

    def calc_transform(self, symbols, coords):
        """(display transform, alignment RMSD or None) for one geometry."""
        if not self.align:
            return self.frame, None
        al.check_alignment_atoms(symbols, self.ref_symbols, self.align_atoms)
        P, Q = coords[self.align_atoms], self.ref_coords[self.align_atoms]
        T = al.kabsch(P, Q)
        return T.then(self.frame), al.rmsd(T.apply(P), Q)

    def to_dict(self):
        return {
            "created": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "reference": self.reference, "align": self.align, "align_atoms": list(self.align_atoms),
            "view_rotation_deg": list(self.view_rotation_deg), "window_size": list(self.window_size),
            "frame": self.frame.to_dict(), "camera": self.camera.to_dict(),
            "ref_symbols": list(self.ref_symbols), "ref_coords_angstrom": self.ref_coords.tolist(),
            "note": "display coords = frame(kabsch(calc -> reference)); camera looks down -z, orthographic",
        }

    @staticmethod
    def from_dict(d):
        return View(al.RigidTransform.from_dict(d["frame"]), Camera.from_dict(d["camera"]), d["align"],
                    list(d["align_atoms"]), d["reference"], list(d["ref_symbols"]),
                    np.array(d["ref_coords_angstrom"]), tuple(d["view_rotation_deg"]), tuple(d["window_size"]))


def _pick(parsed, key, root, what):
    """The single parsed calculation matching `key` (path or substring of its label)."""
    kp = Path(key).expanduser()
    kp = kp if kp.is_absolute() else root / kp
    hits = [(c, o) for c, o in parsed if kp.resolve() in (c.out_path.resolve(), c.directory.resolve())]
    if not hits:
        hits = [(c, o) for c, o in parsed if key in c.label(root)]
    if len(hits) != 1:
        names = ", ".join(c.label(root) for c, _ in hits[:5])
        raise SetupError(f"{what} {key!r} matches {len(hits)} usable calculations{': ' + names if names else ''}")
    return hits[0]


def build_view(parsed, s, root, root_out, log):
    path = root_out / VIEW_FILE
    if path.exists() and s.reuse_saved_view:
        view = View.from_dict(json.loads(path.read_text()))
        clash = []
        if view.align != s.align:
            clash.append(f"ALIGN saved={view.align} now={s.align}")
        if s.align and s.align_atoms is not None and list(s.align_atoms) != view.align_atoms:
            clash.append("ALIGN_ATOMS differ")
        if tuple(view.view_rotation_deg) != tuple(s.view_rotation_deg):
            clash.append("VIEW_ROTATION_DEG differ")
        if tuple(view.window_size) != tuple(s.style.window_size):
            clash.append("window size differs")
        if clash:
            raise SetupError(f"saved view {path} does not match the configuration ({'; '.join(clash)}). "
                             f"Delete it or set REUSE_SAVED_VIEW = False to build a new one.")
        log(f"reusing saved view/alignment: {path} (reference {view.reference})")
    else:
        ref_c, ref_o = _pick(parsed, s.reference, root, "REFERENCE") if s.reference else parsed[0]
        heavy = [i for i, x in enumerate(ref_o.symbols) if x != "H"]
        if len(heavy) < 3:
            heavy = list(range(len(ref_o.symbols)))
        atoms = list(s.align_atoms) if s.align_atoms is not None else heavy
        if s.align:
            al.check_alignment_atoms(ref_o.symbols, ref_o.symbols, atoms)
        frame = al.principal_frame(ref_o.coords[heavy], s.view_rotation_deg)
        view = View(frame, None, s.align, atoms, ref_c.label(root), list(ref_o.symbols), ref_o.coords.copy(),
                    tuple(s.view_rotation_deg), tuple(s.style.window_size))
        try:
            view.calc_transform(ref_o.symbols, ref_o.coords)
        except ValueError as e:
            raise SetupError(f"reference {view.reference} cannot be aligned: {e}") from None
        pts = []
        for c, o in parsed:
            try:
                T, _ = view.calc_transform(o.symbols, o.coords)
                pts.append(T.apply(o.coords))
            except ValueError:
                continue
        view.camera = frame_camera(np.vstack(pts), s.frame_margin, s.style.window_size)
        root_out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(view.to_dict(), indent=1))
        log(f"new view/alignment saved to {path}")
        log(f"  reference: {view.reference}; align={s.align}; {len(atoms)} alignment atoms; "
            f"framing from {len(pts)} geometries")
    return view


# ---------------------------------------------------------------------------
# per-calculation stages
# ---------------------------------------------------------------------------

def _check_orbital_file(info, out, expected_ops, kind):
    if info.n_operators != expected_ops:
        raise CalcSkip(f"{kind}: orbital file has {info.n_operators} operator(s), expected {expected_ops}")
    if out.basis_dimension is not None and info.n_orbitals != out.basis_dimension:
        raise CalcSkip(f"{kind}: orbital file has {info.n_orbitals} orbitals but the output's basis dimension is "
                       f"{out.basis_dimension}; it does not belong to this output")
    if [x.capitalize() for x in info.symbols] != [x.capitalize() for x in out.symbols]:
        raise CalcSkip(f"{kind}: atoms in the orbital file differ from the output geometry")
    d = al.rmsd(info.coords, out.coords)
    if d > GEOM_TOL:
        raise CalcSkip(f"{kind}: orbital-file geometry differs from the final output geometry (RMSD {d:.4f} A)")


class Context:
    """Everything one calculation's stages share."""

    def __init__(self, calc, out, view, s, exe, root, log):
        self.calc, self.out, self.view, self.s, self.exe, self.root, self.log = calc, out, view, s, exe, root, log
        self.od = calc.output_dir
        self.T, self.rmsd = view.calc_transform(out.symbols, out.coords)
        self.mol = molecule_meshes(out.symbols, self.T.apply(out.coords), s.style)
        self.Z = symbols_to_numbers(out.symbols)


def _note(row, msg):
    row["message"] = "; ".join(m for m in (row.get("message"), msg) if m)


def _cubes_and_images(ctx, kind, source, expected_ops, items, subdir):
    """
    items: dicts with operator, index, file_stem, caption and metadata fields.
    Returns metadata rows (one per item) with status/paths filled in.
    """
    s, log = ctx.s, ctx.log
    cubes_dir = ctx.od / "cubes" / subdir
    img_dir = ctx.od / "images" / subdir
    work = ctx.od / f"_orca_plot_work_{kind}"
    rows = []
    for it in items:
        rows.append({k: v for k, v in it.items() if k not in ("operator", "file_stem", "caption")}
                    | {"orbital_type": kind, "isovalue": s.isovalue, "grid_size": s.grid_size,
                       "phase_convention": s.phase_convention, "status": it.get("status", "pending"),
                       "message": it.get("message", "")})
    active = [k for k, r in enumerate(rows) if r["status"] == "pending"]
    if not active:
        return rows

    try:
        info = probe(ctx.exe, source, work)
        _check_orbital_file(info, ctx.out, expected_ops, kind)
    except OrcaPlotError as e:
        (ctx.od / "logs").mkdir(parents=True, exist_ok=True)
        (ctx.od / "logs" / f"orca_plot_probe_{kind}.log").write_text(e.log)
        raise CalcSkip(f"{kind}: orca_plot could not open {source.name}: {e}") from None
    log(f"{kind}: {source.name} has {info.n_orbitals} orbitals, {info.n_operators} operator(s); "
        f"geometry matches output")

    requests, req_rows = [], []
    for k in active:
        it = items[k]
        dest = cubes_dir / f"{it['file_stem']}.cube"
        rows[k]["cube_path"] = str(dest)
        if dest.exists() and not s.regenerate_cubes:
            try:
                validate_cube_file(dest, it["index"], it["operator"], s.grid_size)
                _note(rows[k], "cube reused")
                continue
            except Exception as e:
                log(f"existing {dest.name} unusable ({e}); regenerating", "WARNING")
        requests.append(CubeRequest(it["index"], it["operator"], dest))
        req_rows.append(k)
    if requests:
        try:
            results, plot_log = generate_cubes(ctx.exe, source, requests, s.grid_size, work, info)
        except OrcaPlotError as e:
            results, plot_log = {j: (False, str(e)) for j in range(len(requests))}, e.log
        failed = [j for j, (ok, _) in results.items() if not ok]
        if failed:
            (ctx.od / "logs").mkdir(parents=True, exist_ok=True)
            lp = ctx.od / "logs" / f"orca_plot_{kind}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
            lp.write_text(plot_log)
            log(f"{kind}: {len(failed)} cube(s) failed; orca_plot output saved to {lp}", "ERROR")
        for j, k in enumerate(req_rows):
            ok, msg = results[j]
            _note(rows[k], msg)
            if not ok:
                rows[k]["status"] = "cube_failed"
        log(f"{kind}: generated {len(requests) - len(failed)}/{len(requests)} cube(s) at {s.grid_size}^3")

    for k in active:
        r, it = rows[k], items[k]
        if r["status"] != "pending":
            continue
        try:
            cube = read_cube(r["cube_path"])
            if cube.orca_mo() not in (None, (it["index"], it["operator"])):
                raise ValueError(f"cube header says MO/op {cube.orca_mo()}")
            if not np.array_equal(cube.atomic_numbers, ctx.Z):
                raise ValueError("cube atoms differ from the output geometry")
            d = al.rmsd(cube.positions_angstrom, ctx.out.coords)
            if d > GEOM_TOL:
                raise ValueError(f"cube atom positions differ from the output geometry (RMSD {d:.4f} A)")
            ph = phase_factor(cube, s.phase_convention)
            lobes, notes, stats = extract_lobes(cube, s.isovalue, phase=ph)
            r.update(sign_flipped=ph < 0, positive_lobe=lobes["positive"] is not None,
                     negative_lobe=lobes["negative"] is not None, **stats)
            for n in notes:
                _note(r, n)
            if lobes["positive"] is None and lobes["negative"] is None:
                r["status"] = "no_surface"
                log(f"{it['file_stem']}: no isosurface at +/-{s.isovalue} (|psi|max "
                    f"{max(abs(stats['min_amplitude']), stats['max_amplitude']):.3g}); image not written", "WARNING")
                continue
            disp = {n: (ctx.T.apply(l.vertices), l.faces) if l is not None else None for n, l in lobes.items()}
            png = img_dir / f"{it['file_stem']}.png"
            render_orbital(png, ctx.mol, disp, ctx.view.camera, s.style)
            r.update(image_path=str(png), status="ok")
            for n in notes:
                log(f"{it['file_stem']}: {n}", "WARNING")
        except Exception as e:
            r["status"] = "render_failed"
            _note(r, str(e))
            log(f"{it['file_stem']}: {e}", "ERROR")
        finally:
            if not s.keep_cubes and Path(r.get("cube_path", "")).is_file():
                Path(r["cube_path"]).unlink()
    return rows


def mo_stage(ctx):
    s, out = ctx.s, ctx.out
    try:
        frontier = select_frontier(out, s.orbitals, s.spins)
    except OrcaOutputError as e:
        raise CalcSkip(f"frontier orbitals: {e}") from None
    items = []
    for f in frontier:
        sp, sym = f["spin"], SPIN_SYMBOL[f["spin"]]
        e_txt = f"{f['energy_ev']:.3f} eV" if "energy_ev" in f else "n/a"
        items.append({
            "spin": sp, "operator": SPIN_OPERATOR[sp], "index": f["index"], "orbital_index": f["index"],
            "orbital_label": f["label"], "energy_eh": f.get("energy_eh"), "energy_ev": f.get("energy_ev"),
            "occupation": f.get("occupation"),
            "status": "pending" if f["status"] == "ok" else f["status"], "message": f["note"],
            "file_stem": f"{sp}_{f['label']}_mo{f['index']}",
            "caption": f"{sym} {f['label'].replace('-', '−')}  (MO {f['index']})\n{e_txt}".strip(),
        })
        if f["note"]:
            ctx.log(f"{sp} {f['label']} (MO {f['index']}): {f['note']}", "NOTE")
    ctx.log("frontier: " + ", ".join(f"{f['spin']} {f['label']}=MO {f['index']}" for f in frontier))
    rows = _cubes_and_images(ctx, "MO", ctx.calc.gbw, 1 if out.restricted else 2, items, "")
    channels = ["closed"] if out.restricted else [x for x in s.spins if x in out.orbitals]
    sheet_rows = []
    for ch in channels:
        panels = []
        for label in s.orbitals:
            k = next(i for i, it in enumerate(items) if it["spin"] == ch and it["orbital_label"] == label)
            panels.append((rows[k].get("image_path"), items[k]["caption"]))
        sheet_rows.append(({"alpha": "alpha (α)", "beta": "beta (β)", "closed": "closed shell"}[ch], panels))
    _sheet(ctx, sheet_rows, "frontier MOs (energy ordering)", "frontier_orbitals.png")
    return rows


def uno_stage(ctx):
    calc, s = ctx.calc, ctx.s
    if calc.uno is None:
        ctx.log("UNO: no .uno file; stage skipped")
        return []
    occ = parse_uno_occupations(calc.property_txt, calc.out_path.read_text(errors="replace"))
    if not occ:
        ctx.log(f"UNO: {calc.uno.name} present but no UNO occupations found in .property.txt/.out; skipped", "WARNING")
        return [{"orbital_type": "UNO", "status": "skipped", "message": "UNO occupations not found"}]
    lo, hi = s.uno_occupation_range
    cand = sorted((i for i, n in occ.items() if lo <= n <= hi), key=lambda i: (abs(occ[i] - 1.0), i))
    chosen = sorted(cand[:s.uno_max_orbitals])
    ctx.log(f"UNO: {len(cand)} orbital(s) with {lo} <= n <= {hi}; plotting {chosen}")
    if not chosen:
        return [{"orbital_type": "UNO", "status": "skipped", "message": f"no UNO with occupation in [{lo}, {hi}]"}]
    items = [{"spin": "", "operator": 0, "index": i, "orbital_index": i, "orbital_label": f"UNO {i}",
              "occupation": occ[i], "file_stem": f"UNO_{i}", "caption": f"UNO {i}\nn = {occ[i]:.3f}"}
             for i in chosen]
    rows = _cubes_and_images(ctx, "UNO", calc.uno, 1, items, "UNO")
    _sheet(ctx, [("UNO", [(r.get("image_path"), it["caption"]) for r, it in zip(rows, items)])],
           "unrestricted natural orbitals (n = occupation)", "UNO.png")
    return rows


def select_uco(overlaps, n_alpha, n_beta, max_overlap, max_pairs):
    """
    (candidate pairs, plotted pairs, unpaired alpha indices) from the UCO overlap table.

    Only i < min(N_alpha, N_beta) are genuine alpha/beta pairs. ORCA weights the printed
    overlap by the product of occupations, so for Ms > 0 the entry i = N_beta pairs an occupied
    alpha orbital with an EMPTY beta orbital and reads S = 0 -- that is not a magnetic pair.
    The unmatched alpha corresponding orbitals N_beta .. N_alpha-1 hold the unpaired electrons.
    """
    n_a, n_b = round(n_alpha), round(n_beta)
    n_pair = min(n_a, n_b)
    cand = sorted((i for i, v in overlaps.items() if i < n_pair and v < max_overlap), key=lambda i: (overlaps[i], i))
    return cand, sorted(cand[:max_pairs]), (list(range(n_b, n_a)) if n_a > n_b else [])


def uco_stage(ctx):
    calc, s, out = ctx.calc, ctx.s, ctx.out
    if calc.uco is None:
        ctx.log("UCO: no .uco file; stage skipped")
        return []
    if not out.uco_overlaps:
        ctx.log(f"UCO: {calc.uco.name} present but no corresponding-orbital overlap table in the output; skipped",
                "WARNING")
        return [{"orbital_type": "UCO", "status": "skipped", "message": "UCO overlap table not found in output"}]
    ov = out.uco_overlaps
    cand, pairs, unpaired = select_uco(ov, out.n_alpha, out.n_beta, s.uco_max_overlap, s.uco_max_pairs)
    ctx.log(f"UCO: {len(cand)} paired orbital(s) with overlap < {s.uco_max_overlap}; plotting pairs {pairs}"
            + (f"; unpaired alpha corresponding orbitals {unpaired}" if unpaired else ""))
    if not pairs and not unpaired:
        return [{"orbital_type": "UCO", "status": "skipped",
                 "message": f"no corresponding pair with overlap < {s.uco_max_overlap} and no unpaired electrons"}]
    items = []
    for i in pairs:
        for sp in ("alpha", "beta"):
            items.append({"spin": sp, "operator": SPIN_OPERATOR[sp], "index": i, "orbital_index": i,
                          "orbital_label": f"UCO pair {i}", "uco_overlap": ov[i],
                          "file_stem": f"UCO_{i}_{sp}",
                          "caption": f"UCO {i} {SPIN_SYMBOL[sp]}\nS(α,β) = {ov[i]:.3f}"})
    for i in unpaired:
        items.append({"spin": "alpha", "operator": 0, "index": i, "orbital_index": i,
                      "orbital_label": f"UCO {i} unpaired", "file_stem": f"UCO_{i}_alpha_unpaired",
                      "caption": f"UCO {i} α (unpaired)\nno β partner"})
    rows = _cubes_and_images(ctx, "UCO", calc.uco, 2, items, "UCO")
    by_key = {(it["spin"], it["orbital_label"]): (r.get("image_path"), it["caption"]) for r, it in zip(rows, items)}
    columns = [f"UCO pair {i}" for i in pairs] + [f"UCO {i} unpaired" for i in unpaired]
    sheet = [("alpha (α)", [by_key[("alpha", c)] for c in columns])]
    if pairs:
        sheet.append(("beta (β)", [by_key.get(("beta", c), (None, "β: none (unpaired α)")) for c in columns]))
    _sheet(ctx, sheet, "UHF corresponding orbitals (S = α/β overlap; unpaired = singly occupied α)", "UCO.png")
    return rows


def _sheet(ctx, rows, what, filename):
    if not any(img for _, panels in rows for img, _ in panels):
        ctx.log(f"no images for {filename}; contact sheet not written", "WARNING")
        return
    title = (f"{ctx.calc.label(ctx.root)}  |  {ctx.out.electronic_state()}  |  {what}  |  "
             f"isovalue \u00b1{ctx.s.isovalue:g}, blue > 0, red < 0")
    p = ctx.od / "contact_sheets" / filename
    contact_sheet(rows, title, p)
    ctx.log(f"contact sheet: {p}")


def process_calculation(calc, out, view, s, exe, root):
    od = calc.output_dir
    od.mkdir(parents=True, exist_ok=True)
    log = Log(od / "processing_log.txt", prefix="    ")
    base = {
        "calculation": calc.label(root), "calculation_dir": str(calc.directory), "calculation_name": calc.base,
        "electronic_state": out.electronic_state(), "efield_au": " ".join(f"{v:g}" for v in out.efield) if out.efield else "",
        "n_alpha": out.n_alpha, "n_beta": out.n_beta, "s2": out.s2, "final_energy_eh": out.final_energy,
    }
    rows, skip = [], None
    try:
        log(f"output {calc.out_path}")
        log(f"orbitals {calc.gbw.name}; uno: {calc.uno.name if calc.uno else '-'}; uco: {calc.uco.name if calc.uco else '-'}")
        log(f"ORCA {out.version}; state {out.electronic_state()}; N(alpha)={out.n_alpha}, N(beta)={out.n_beta}, "
            f"<S^2>={out.s2}; EField={out.efield}")
        try:
            ctx = Context(calc, out, view, s, exe, root, log)
        except ValueError as e:
            raise CalcSkip(f"alignment: {e}") from None
        if ctx.rmsd is not None:
            log(f"aligned to {view.reference} on {len(view.align_atoms)} atoms, RMSD {ctx.rmsd:.4f} A")
        base["alignment_rmsd_A"] = ctx.rmsd
        if not camera_contains(view.camera, ctx.T.apply(out.coords), s.style.window_size):
            log("some atoms fall outside the saved camera framing", "WARNING")
        (od / "view_and_alignment.json").write_text(json.dumps({
            "display_transform": ctx.T.to_dict(), "alignment_rmsd_A": ctx.rmsd, "reference": view.reference,
            "align_atoms": view.align_atoms, "camera": view.camera.to_dict(), "style": asdict(s.style),
            "isovalue": s.isovalue, "grid_size": s.grid_size, "phase_convention": s.phase_convention,
        }, indent=1))
        if s.render_mo:
            rows += mo_stage(ctx)
            if not any(r["status"] == "ok" for r in rows):
                raise CalcSkip("no frontier-orbital image could be produced")
        if s.render_uno:
            rows += uno_stage(ctx)
        if s.render_uco:
            rows += uco_stage(ctx)
    except CalcSkip as e:
        skip = str(e)
        log(skip, "ERROR")
    except Exception as e:  # keep the directory-wide run alive
        skip = f"unexpected error: {e!r}"
        log(traceback.format_exc(), "ERROR")
    finally:
        rows = [{**base, **r} for r in rows]
        assign_identity(rows)
        if rows:
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(od / "orbital_metadata.csv", index=False)
        log.write()
    return rows, skip


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def run(s):
    root = Path(s.root).expanduser().resolve()
    if not root.is_dir():
        raise SetupError(f"ROOT {root} is not a directory")
    root_out = root / s.output_dir_name
    runlog = Log()
    for label in s.orbitals:
        from .orca_output import parse_frontier_label
        parse_frontier_label(label)                  # fail fast on a bad configuration
    bad_spins = set(s.spins) - {"alpha", "beta"}
    if bad_spins:
        raise SetupError(f"unknown spin channel(s) {bad_spins}")
    exe = find_orca_plot(s.orca_plot)
    runlog(f"orca_plot: {exe}")
    runlog(f"root: {root}")

    def selected(p):
        rel = str(p.relative_to(root))
        if s.include_patterns and not any(fnmatch.fnmatch(rel, pat) for pat in s.include_patterns):
            return False
        return not any(fnmatch.fnmatch(rel, pat) for pat in s.exclude_patterns)

    calcs, problems = discover(root, s.recursive, s.output_dir_name, s.ignore_dir_names, selected,
                               central_output=s.central_output)
    skipped = [{"calculation": str(p), "stage": "discovery", "reason": r} for p, r in problems]
    runlog(f"found {len(calcs)} calculation(s) with matching .gbw; {len(problems)} output(s) not usable at discovery")

    parsed = []
    for c in calcs:
        try:
            o = parse_output(c.out_path)
            check_usable(o)
            parsed.append((c, o))
        except (OrcaOutputError, OSError) as e:
            skipped.append({"calculation": c.label(root), "stage": "output", "reason": str(e)})
    runlog(f"{len(parsed)} calculation(s) with a usable final wavefunction")
    if not parsed:
        _write_summary(root_out, [], skipped, runlog)
        return [], skipped

    view = build_view(parsed, s, root, root_out, runlog)
    todo = parsed
    if s.smoke_test:
        if s.smoke_test_calc:
            todo = [_pick(parsed, s.smoke_test_calc, root, "SMOKE_TEST_CALC")]
        else:
            todo = [next(((c, o) for c, o in parsed if c.label(root) == view.reference), parsed[0])]
        runlog(f"SMOKE TEST: processing only {todo[0][0].label(root)}")

    all_rows = []
    for k, (c, o) in enumerate(todo, 1):
        runlog(f"[{k}/{len(todo)}] {c.label(root)}  ({o.electronic_state()})")
        rows, skip = process_calculation(c, o, view, s, exe, root)
        all_rows += rows
        if skip:
            skipped.append({"calculation": c.label(root), "stage": "processing", "reason": skip})
            runlog(f"    skipped: {skip}", "WARNING")
        else:
            n_ok = sum(r["status"] == "ok" for r in rows)
            runlog(f"    {n_ok}/{len(rows)} image(s) written to {c.output_dir}")
    _write_summary(root_out, all_rows, skipped, runlog)
    return all_rows, skipped


def _write_summary(root_out, rows, skipped, runlog):
    root_out.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(root_out / "orbital_summary.csv", index=False)
    pd.DataFrame(skipped, columns=["calculation", "stage", "reason"]).to_csv(
        root_out / "skipped_calculations.csv", index=False)
    runlog(f"summary: {root_out / 'orbital_summary.csv'}; skipped: {len(skipped)} "
           f"(see {root_out / 'skipped_calculations.csv'})")
    runlog.write(root_out / "run_log.txt")

#!/usr/bin/env python3
"""
Local ORCA spin-flip TDA (SF-TDA) validation of the broken-symmetry (BS) exchange
splittings of short-2OS on four representative geometries.

For each geometry two SF-TDA calculations are run from the Ms = 1 triplet UKS
reference (charge 0, multiplicity 3): PBE0 and BHANDHLYP, def2-TZVP.
Singlet-like and triplet-like roots are identified from <S^2> (never from the
root order), and

    dE_ST = E(triplet root) - E(singlet root)      > 0: singlet lower

which has the same qualitative sign as the existing BS convention
DeltaE = E_HS - E_BS (> 0: BS / singlet-like lower). In the repository convention H = -J S1.S2
(J > 0 FM, exchange_convention.py), dE_ST = E_T - E_S = -J.

Runs locally: ORCA is started directly with subprocess; ORCA itself launches its
parallel modules according to %pal. No batch system, no scratch handling.

Usage:  python3 run_sf_tda_validation.py
        (set RUN_CALCULATIONS = False to only (re)parse existing outputs)
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =====================================================================
# USER SETTINGS
# =====================================================================
ORCA = Path.home() / "software" / "orca-6.1.1" / "orca"

NPROCS = 16              # DelftBlue node (orca_job.sh: 16 tasks); use 4-8 for local runs on the Mac
MAXCORE = 3000          # MB per process (DelftBlue: 4 GB per task available)

RUN_CALCULATIONS = True
OVERWRITE = False       # True: delete the previous ORCA files of a run and recompute

# Local OpenMPI used by ORCA for NPROCS > 1 (conda env "DFT"). On macOS the dynamic
# loader variables are stripped before ORCA reaches mpirun, so the library path is
# passed to mpirun explicitly through ORCA's second command-line argument.
# Set to None if your MPI is already found without help.
OPENMPI_PREFIX = Path.home() / "miniforge3" / "envs" / "DFT"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
OUTPUT_ROOT = SCANS / "sf_tda_validation"

# The geometries are those of the existing PBE0 HS/BS single points, so SF-TDA and
# BS-DFT are compared at exactly the same structures.
GEOMETRIES = {
    "M1": SCANS / "visualization_orbitals" / "M1" / "high_spin" / "F0" / "geometry.xyz",
    "symmetric_90_90": SCANS / "angle_scan_exchange_analysis" / "L090_C080_R090" / "high_spin" / "geometry.xyz",
    "asymmetric_negative": SCANS / "angle_scan_exchange_analysis" / "L135_C120_R090" / "high_spin" / "geometry.xyz",
    "reemergent_AFM": SCANS / "angle_scan_C100_L090_exchange" / "L090_C100_R170" / "high_spin" / "geometry.xyz",
}

# Optional converged PBE0 high-spin wavefunctions (same geometry) used as SCF guess.
USE_EXISTING_GBW = True
USE_PBE0_GBW_FOR_BHANDHLYP = True    # explicitly enabled: PBE0 triplet orbitals as BHANDHLYP SCF guess
GBW_FILES = {
    "M1": SCANS / "visualization_orbitals" / "M1" / "high_spin" / "F0" / "high_spin.gbw",
    "symmetric_90_90": SCANS / "angle_scan_exchange_analysis" / "L090_C080_R090" / "high_spin" / "high_spin.gbw",
    "asymmetric_negative": SCANS / "angle_scan_exchange_analysis" / "L135_C120_R090" / "high_spin" / "high_spin.gbw",
    "reemergent_AFM": SCANS / "angle_scan_C100_L090_exchange" / "L090_C100_R170" / "high_spin" / "high_spin.gbw",
}

# Functionals: name -> ORCA simple-input keyword.
# BHANDHLYP was checked with ORCA 6.1.1: accepted directly as B88 exchange + LYP
# correlation with 50 % HF exchange (ScalHFX = ScalDFX = 0.5), also in the TDA kernel.
FUNCTIONALS = {"pbe0": "PBE0", "bhandhlyp": "BHANDHLYP"}

NROOTS = 8
ETOL = 1e-7
RTOL = 1e-7

# Root classification by <S^2>
SINGLET_S2_MAX = 0.3
TRIPLET_S2_MIN = 1.7
TRIPLET_S2_MAX = 2.3
NEAR_DEGENERATE_MEV = 0.5
# Pair identification (see gap()): a root belongs to the diradical (SOMO-flip) manifold if at least this
# weight sits in alpha-SOMO -> beta-SOMO excitations; the two lowest such roots form a spin-mixed S/T pair
# only if neither is dominated by a single determinant.
SOMO_MANIFOLD_MIN_WEIGHT = 0.5
UNMIXED_SINGLE_DET_WEIGHT = 0.75

# Existing PBE0 BS-DFT results at the same geometries (dE = E_HS - E_BS; > 0: BS lower).
# E_T_minus_E_S_yamaguchi_meV = -J with J = -2 * raw / (<S2>_HS - <S2>_BS) (Yamaguchi, H = -J S1.S2), the
# quantity directly comparable to the SF-TDA dE_ST. Set to {} to omit the comparison.
BS_RESULTS = {
    "M1": {"raw_meV": 1.3115, "E_T_minus_E_S_yamaguchi_meV": 2.6325,
           "source": "visualization_orbitals/M1/{high_spin,broken_symmetry}/F0"},
    "symmetric_90_90": {"raw_meV": 0.0961, "E_T_minus_E_S_yamaguchi_meV": 0.1922,
                        "source": "angle_scan_exchange_analysis/exchange_analysis.csv"},
    "asymmetric_negative": {"raw_meV": -0.3069, "E_T_minus_E_S_yamaguchi_meV": -0.6137,
                            "source": "angle_scan_exchange_analysis/exchange_analysis.csv"},
    "reemergent_AFM": {"raw_meV": 2.5967, "E_T_minus_E_S_yamaguchi_meV": 5.1786,
                       "source": "outer_scan_analysis/outer_scans_exchange_points.csv"},
}

CALC_NAME = "sf_tda"   # files: sf_tda.inp / sf_tda.out / sf_tda.err

# =====================================================================
# CONSTANTS
# =====================================================================
EH_TO_MEV = 27211.386245988
EH_TO_CM = 219474.6313632
CM_TO_MEV = EH_TO_MEV / EH_TO_CM

STATE_RE = re.compile(
    r"^STATE\s+(\d+):\s+E=\s+(-?\d+\.\d+)\s+au\s+(-?\d+\.\d+)\s+eV\s+(-?\d+\.\d+)\s+cm\*\*-1\s+"
    r"<S\*\*2>\s+=\s+(-?\d+\.\d+)(?:\s+Mult\s+(\d+))?", re.M)
CONF_RE = re.compile(r"^\s+(\d+[ab])\s+->\s+(\d+[ab])\s+:\s+(-?\d+\.\d+)\s+\(c=\s*(-?\d+\.\d+)\)")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
ESCF_RE = re.compile(r"E\(SCF\)\s+=\s+(-?\d+\.\d+)\s+Eh")
NALPHA_RE = re.compile(r"N\(Alpha\)\s+:\s+(\d+\.\d+)")
NBETA_RE = re.compile(r"N\(Beta\)\s+:\s+(\d+\.\d+)")
TOTAL_E_RE = re.compile(r"Total Energy\s+:\s+(-?\d+\.\d+)\s+Eh")


# =====================================================================
# INPUT GENERATION
# =====================================================================
def render_input(keyword, moread):
    head = f"! {keyword} def2-TZVP TightSCF UKS DEFGRID3"
    if moread:
        head += " NoAutoStart MORead"
    lines = [head, ""]
    if moread:
        lines += ['%moinp "guess.gbw"', ""]
    lines += [
        "%scf", "    MaxIter 500", "    ConvCheckMode 0", "end", "",
        "%tddft", "    SF true", f"    NRoots {NROOTS}", f"    ETol {ETOL:g}", f"    RTol {RTOL:g}", "end", "",
        "%pal", f"    nprocs {NPROCS}", "end", "",
        f"%maxcore {MAXCORE}", "",
        "* xyzfile 0 3 geometry.xyz", "",   # SF-TDA from the high-spin triplet reference
    ]
    return "\n".join(lines)


def prepare(geom, func, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GEOMETRIES[geom], run_dir / "geometry.xyz")   # source is only read
    moread = USE_EXISTING_GBW and geom in GBW_FILES and (func == "pbe0" or USE_PBE0_GBW_FOR_BHANDHLYP)
    if moread:
        shutil.copy2(GBW_FILES[geom], run_dir / "guess.gbw")
    (run_dir / f"{CALC_NAME}.inp").write_text(render_input(FUNCTIONALS[func], moread))
    return moread


def clean_previous(run_dir):
    for p in run_dir.iterdir():
        if p.name.startswith(CALC_NAME + ".") and p.suffix != ".inp" or p.name == "run_log.json":
            p.unlink()


# =====================================================================
# EXECUTION
# =====================================================================
def run_orca(run_dir):
    args = [str(ORCA), f"{CALC_NAME}.inp"]
    env = os.environ.copy()
    if NPROCS > 1 and OPENMPI_PREFIX is not None:
        env["PATH"] = f"{OPENMPI_PREFIX / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        if sys.platform == "darwin":
            args.append(f"-x DYLD_LIBRARY_PATH={OPENMPI_PREFIX / 'lib'}")   # forwarded by ORCA to mpirun
    start = datetime.now()
    t0 = time.time()
    with open(run_dir / f"{CALC_NAME}.out", "w") as out, open(run_dir / f"{CALC_NAME}.err", "w") as err:
        try:
            rc = subprocess.run(args, cwd=run_dir, stdout=out, stderr=err, env=env).returncode
        except OSError as exc:   # e.g. ORCA path wrong; never stop the batch
            err.write(f"failed to start ORCA: {exc}\n")
            rc = -1
    log = {"start": start.isoformat(timespec="seconds"), "finish": datetime.now().isoformat(timespec="seconds"),
           "return_code": rc, "runtime_s": round(time.time() - t0, 1), "command": " ".join(args),
           "nprocs": NPROCS, "maxcore": MAXCORE}
    (run_dir / "run_log.json").write_text(json.dumps(log, indent=2))
    return log


# =====================================================================
# PARSING
# =====================================================================
def parse_property_energies(prop_path):
    """Full-precision energies of SF roots relative to root 1 (cm^-1), from the first
    &ExcitationEnergies / &States blocks of the property file. {root: cm^-1}."""
    if not prop_path.exists():
        return {}
    lines = prop_path.read_text(errors="ignore").splitlines()

    def block(tag):
        for i, ln in enumerate(lines):
            if ln.strip().startswith(tag):
                n = int(re.search(r"&Dim \((\d+),", ln).group(1))
                rows = []
                for row in lines[i + 3:i + 3 + n]:
                    rows.append([float(x) for x in row.split()[1:]])
                return rows
        return None

    ex, st = block("&ExcitationEnergies"), block("&States")
    if not ex or not st or len(ex) != len(st):
        return {}
    out = {1: 0.0}
    for e_row, s_row in zip(ex, st):
        if int(s_row[0]) != 1:
            return {}
        out[int(s_row[2])] = e_row[1]   # column 1 = cm^-1
    return out


def parse_output(run_dir):
    out_path = run_dir / f"{CALC_NAME}.out"
    rec = {"status": "NOT_RUN", "roots": [], "ref_S2": None, "E_ref_Eh": None, "notes": []}
    if not out_path.exists() or out_path.stat().st_size == 0:
        return rec
    text = out_path.read_text(errors="ignore")
    normal = "ORCA TERMINATED NORMALLY" in text
    scf_ok = ("SCF CONVERGED AFTER" in text and "SCF NOT CONVERGED" not in text
              and "wavefunction IS NOT CONVERGED" not in text)
    scf_started = "SCF SETTINGS" in text   # a crash before the SCF is an ORCA_ERROR, not an SCF failure
    s2 = S2_RE.findall(text)
    rec["ref_S2"] = float(s2[0]) if s2 else None
    e = ESCF_RE.findall(text) or TOTAL_E_RE.findall(text)
    rec["E_ref_Eh"] = float(e[-1]) if e else None
    i = text.rfind("SF-TDA EXCITED STATES")
    roots = []
    if i >= 0:
        block = text[i:]
        end = block.find("TD-DFT/TDA-EXCITATION SPECTRA")
        block = block[:end] if end > 0 else block
        blines = block.splitlines()
        for k, ln in enumerate(blines):
            m = STATE_RE.match(ln)
            if not m:
                continue
            confs = []
            for nxt in blines[k + 1:]:
                c = CONF_RE.match(nxt)
                if not c:
                    break
                confs.append((c.group(1), c.group(2), float(c.group(3)), float(c.group(4))))
            roots.append({"root": int(m.group(1)), "E_exc_Eh_printed": float(m.group(2)),
                          "E_exc_eV_printed": float(m.group(3)), "E_exc_cm_printed": float(m.group(4)),
                          "S2": float(m.group(5)), "mult_printed": int(m.group(6)) if m.group(6) else None,
                          "configurations": confs})
    # full-precision energies relative to root 1 (property file), checked against the .out
    prop = parse_property_energies(run_dir / f"{CALC_NAME}.property.txt")
    if roots:
        e1 = next(r for r in roots if r["root"] == 1)["E_exc_cm_printed"] if any(r["root"] == 1 for r in roots) else 0.0
        use_prop = bool(prop) and all(r["root"] in prop for r in roots)
        if use_prop:
            worst = max(abs((r["E_exc_cm_printed"] - e1) - prop[r["root"]]) for r in roots)
            if worst > 0.2:
                use_prop = False
                rec["notes"].append(f"property-file energies disagree with output by {worst:.2f} cm-1; output used")
        for r in roots:
            rel_cm = prop[r["root"]] if use_prop else r["E_exc_cm_printed"] - e1
            r["E_rel_root1_cm"] = rel_cm
            r["energy_source"] = "property.txt (full precision)" if use_prop else "output (0.1 cm-1 rounding)"
            r["E_exc_Eh"] = (e1 + rel_cm) / EH_TO_CM if use_prop else r["E_exc_Eh_printed"]
            r["E_abs_Eh"] = rec["E_ref_Eh"] + r["E_exc_Eh"] if rec["E_ref_Eh"] is not None else None
            r["E_eV"] = r["E_exc_Eh"] * EH_TO_MEV / 1000.0
            s = r["S2"]
            r["spin_class"] = ("singlet-like" if s <= SINGLET_S2_MAX else
                               "triplet-like" if TRIPLET_S2_MIN <= s <= TRIPLET_S2_MAX else "contaminated/ambiguous")
    na, nb = NALPHA_RE.findall(text), NBETA_RE.findall(text)
    somo = set(range(round(float(nb[-1])), round(float(na[-1])))) if na and nb else set()
    rec["somo_indices"] = sorted(somo)
    for r in roots:
        w_somo = sum(w for a, b, w, _ in r["configurations"]
                     if int(a[:-1]) in somo and int(b[:-1]) in somo and a.endswith("a") and b.endswith("b"))
        r["somo_weight"] = w_somo
        r["max_config_weight"] = max((w for *_, w, _ in r["configurations"]), default=0.0)
        r["n_somo_configs_ge_0.05"] = sum(1 for a, b, w, _ in r["configurations"]
                                          if int(a[:-1]) in somo and int(b[:-1]) in somo and w >= 0.05)
    rec["roots"] = roots
    if scf_started and not scf_ok:
        rec["status"] = "SCF_FAILED"
    elif scf_ok and not roots:
        rec["status"] = "SF_FAILED"      # SCF fine, SF-TDA step missing / crashed
    elif not normal:
        rec["status"] = "ORCA_ERROR"
    else:
        rec["status"] = "OK"
        if len(roots) < NROOTS:
            rec["notes"].append(f"only {len(roots)} of {NROOTS} roots printed")
    if rec["ref_S2"] is not None and abs(rec["ref_S2"] - 2.0) > 0.3:
        rec["notes"].append(f"triplet reference <S2> = {rec['ref_S2']:.3f} (spin-contaminated reference)")
    return rec


def gap(rec):
    """Singlet-triplet gap from the two lowest SOMO-flip roots.

    In the Ms = 0 spin-flip space of a diradical the four alpha-SOMO -> beta-SOMO determinants give the triplet
    (Ms = 0 component), the open-shell singlet and two ionic singlets. The S/T pair is the two lowest roots of this
    manifold. It is only meaningful if the two roots are spin-mixed combinations of these determinants; if each is
    a single unmixed determinant (<S2> ~ 1, BS-like), the SF-TDA (UKS-reference, spin-incomplete) has not coupled
    them and their energy difference is NOT a singlet-triplet gap.
    Assigning by a fixed <S2> window is not used: near-degenerate S/T states are strongly spin-mixed in SF-TDA, and
    the window then selects the ionic singlets instead."""
    res = {"pair_type": "no SOMO-flip pair"}
    cand = sorted((r for r in rec["roots"] if r.get("somo_weight", 0) >= SOMO_MANIFOLD_MIN_WEIGHT),
                  key=lambda r: r["E_rel_root1_cm"])
    if len(cand) < 2:
        return res
    r1, r2 = cand[0], cand[1]
    split_cm = r2["E_rel_root1_cm"] - r1["E_rel_root1_cm"]
    res.update({"pair_roots": f"{r1['root']},{r2['root']}", "pair_S2": f"{r1['S2']:.3f},{r2['S2']:.3f}",
                "pair_somo_weights": f"{r1['somo_weight']:.2f},{r2['somo_weight']:.2f}",
                "pair_max_single_det": f"{r1['max_config_weight']:.2f},{r2['max_config_weight']:.2f}",
                "pair_splitting_meV": split_cm * CM_TO_MEV, "energy_source": r1["energy_source"]})
    mixed = all(r["max_config_weight"] < UNMIXED_SINGLE_DET_WEIGHT and r["n_somo_configs_ge_0.05"] >= 2
                for r in (r1, r2))
    if not mixed:
        res["pair_type"] = "unmixed determinants (no S/T pair)"
        res["label"] = "not determinable"
        return res
    s_root, t_root = sorted((r1, r2), key=lambda r: r["S2"])
    cm = t_root["E_rel_root1_cm"] - s_root["E_rel_root1_cm"]
    res.update({"pair_type": "spin-mixed S/T pair", "singlet_root": s_root["root"], "S2_singlet": s_root["S2"],
                "triplet_root": t_root["root"], "S2_triplet": t_root["S2"], "dE_ST_cm": cm,
                "dE_ST_Eh": cm / EH_TO_CM, "dE_ST_meV": cm * CM_TO_MEV,
                "triplet_root_vs_reference_meV": t_root["E_exc_Eh"] * EH_TO_MEV})
    res["label"] = label(res["dE_ST_meV"])
    return res


def label(mev):
    if mev is None or mev != mev:
        return "n/a"
    if abs(mev) < NEAR_DEGENERATE_MEV:
        return "near-degenerate"
    return "singlet lower" if mev > 0 else "triplet lower"


# =====================================================================
# REPORTING
# =====================================================================
def fmt(v, f="{:+.3f}"):
    return "–" if v is None or v != v else f.format(v)


def write_outputs(results):
    roots_rows, summary_rows = [], []
    for (geom, func), (rec, g, log, moread) in results.items():
        pair = [int(x) for x in g.get("pair_roots", "").split(",") if x]
        for r in rec["roots"]:
            roots_rows.append({
                "geometry": geom, "functional": func, "root": r["root"], "E_exc_Eh": r["E_exc_Eh"],
                "E_exc_eV": r["E_eV"], "E_rel_root1_cm": r["E_rel_root1_cm"],
                "E_rel_root1_meV": r["E_rel_root1_cm"] * CM_TO_MEV, "E_abs_Eh": r["E_abs_Eh"], "S2": r["S2"],
                "mult_printed": r["mult_printed"], "spin_class_S2_window": r["spin_class"],
                "somo_weight": r.get("somo_weight"), "max_single_config_weight": r.get("max_config_weight"),
                "in_lowest_SOMO_pair": r["root"] in pair,
                "configurations": "; ".join(f"{a}->{b} {w:.3f} (c={c:+.3f})" for a, b, w, c in r["configurations"]),
                "energy_source": r["energy_source"]})
        summary_rows.append({
            "geometry": geom, "functional": func, "status": rec["status"], "pair_type": g.get("pair_type"),
            "pair_roots": g.get("pair_roots"), "pair_S2": g.get("pair_S2"),
            "pair_max_single_det": g.get("pair_max_single_det"), "pair_splitting_meV": g.get("pair_splitting_meV"),
            "singlet_root": g.get("singlet_root"), "S2_singlet": g.get("S2_singlet"),
            "triplet_root": g.get("triplet_root"), "S2_triplet": g.get("S2_triplet"),
            "dE_ST_Eh": g.get("dE_ST_Eh"), "dE_ST_meV": g.get("dE_ST_meV"), "dE_ST_cm-1": g.get("dE_ST_cm"),
            "label": g.get("label", "n/a"), "reference_S2": rec["ref_S2"], "E_ref_Eh": rec["E_ref_Eh"],
            "somo_indices": ",".join(map(str, rec.get("somo_indices", []))),
            "triplet_root_vs_reference_meV": g.get("triplet_root_vs_reference_meV"),
            "energy_source": g.get("energy_source"), "moread_pbe0_guess": moread,
            "return_code": log.get("return_code"), "runtime_s": log.get("runtime_s"), "notes": " | ".join(rec["notes"])})
    for name, rows in (("sf_tda_roots.csv", roots_rows), ("sf_tda_summary.csv", summary_rows)):
        if rows:
            with open(OUTPUT_ROOT / name, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    write_markdown(summary_rows)


def write_markdown(rows):
    S = {(r["geometry"], r["functional"]): r for r in rows}
    geoms = list(GEOMETRIES)
    yam = lambda g: BS_RESULTS.get(g, {}).get("E_T_minus_E_S_yamaguchi_meV")
    L = ["# SF-TDA validation of the BS-DFT exchange splittings\n",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/orca/run_sf_tda_validation.py`. "
         f"Collinear SF-TDA from the UKS triplet reference (charge 0, multiplicity 3), def2-TZVP, DEFGRID3, "
         f"NRoots {NROOTS}. ΔE_ST = E_T − E_S (> 0: singlet lower), the same sign convention as the BS "
         "ΔE = E_HS − E_BS.\n",
         "## How the singlet and triplet are identified\n",
         "In the Ms = 0 spin-flip space, the four α-SOMO → β-SOMO determinants give the triplet (Ms = 0), the "
         "open-shell singlet and two ionic singlets. The singlet/triplet pair is taken as the two lowest roots of this "
         f"SOMO-flip manifold (≥ {SOMO_MANIFOLD_MIN_WEIGHT:.0%} SOMO-flip weight). The gap is reported only if these two "
         "roots are spin-mixed combinations of the SOMO determinants. If each is one unmixed determinant "
         f"(> {UNMIXED_SINGLE_DET_WEIGHT:.0%} single-configuration weight, ⟨S²⟩ ≈ 1 like a BS determinant), SF-TDA has not "
         "coupled them; their energy difference is an artifact of the spin-incomplete UKS-based SF-TDA (in exact theory "
         "the two determinants are degenerate) and **no gap is reported**. A fixed ⟨S²⟩ window is not used: "
         "near-degenerate singlet and triplet are strongly spin-mixed in SF-TDA, and the window then picks the ionic "
         "singlets ~0.85 eV higher.\n",
         "## Per calculation\n",
         "| geometry | functional | status | lowest SOMO pair (roots) | ⟨S²⟩ | max single-det weight | pair type | "
         "splitting (meV) | ΔE_ST (meV) | label | ref ⟨S²⟩ |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in geoms:
        for f in FUNCTIONALS:
            r = S.get((g, f))
            if not r:
                continue
            L.append(f"| {g} | {f} | {r['status']} | {r['pair_roots'] or '–'} | {r['pair_S2'] or '–'} | "
                     f"{r['pair_max_single_det'] or '–'} | {r['pair_type'] or '–'} | {fmt(r['pair_splitting_meV'], '{:.3f}')} | "
                     f"{fmt(r['dE_ST_meV'])} | {r['label']} | {fmt(r['reference_S2'], '{:.3f}')} |")
    L += ["", "*splitting*: energy difference of the two lowest SOMO-flip roots (always ≥ 0). It equals |ΔE_ST| only for a "
          "spin-mixed pair. *ref ⟨S²⟩*: ⟨S²⟩ of the UKS triplet reference (ideal 2.0).\n"]
    if BS_RESULTS:
        L += ["## Comparison with BS-DFT (PBE0, same geometries)\n",
              "BS: ΔE = E_HS − E_BS; J = −2ΔE/(⟨S²⟩_HS − ⟨S²⟩_BS) (Yamaguchi, H = −J S₁·S₂, J > 0 FM); "
              "E_T − E_S = −J is the BS quantity directly comparable to ΔE_ST.\n",
              "| geometry | ΔE = E_HS − E_BS (meV) | E_T − E_S = −J (meV) | PBE0 SF-TDA | BHANDHLYP SF-TDA |", "|---|---|---|---|---|"]
        for g in geoms:
            bs = BS_RESULTS.get(g, {})
            cells = []
            for f in FUNCTIONALS:
                r = S.get((g, f), {})
                if r.get("status") != "OK":
                    cells.append(r.get("status", "n/a"))
                elif r.get("dE_ST_meV") is not None:
                    cells.append(f"{r['dE_ST_meV']:+.3f} ({r['label']})")
                else:
                    cells.append(f"no S/T pair (determinants split by {r['pair_splitting_meV']:.1f} meV)")
            L.append(f"| {g} | {fmt(bs.get('raw_meV'))} | {fmt(yam(g))} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Conclusions per geometry\n"]
    names = {"M1": "M1 (free minimum)", "symmetric_90_90": "symmetric 90/90 (L090_C080_R090)",
             "asymmetric_negative": "asymmetric, small BS HS preference (L135_C120_R090)",
             "reemergent_AFM": "re-emergent AFM, one outer ring at 90° (L090_C100_R170)"}
    for g in geoms:
        parts = []
        for f in FUNCTIONALS:
            r = S.get((g, f), {})
            if r.get("status") != "OK":
                parts.append(f"{f}: {r.get('status', 'n/a')}")
            elif r.get("dE_ST_meV") is not None:
                parts.append(f"{f}: spin-mixed pair, singlet-like root (⟨S²⟩ {r['S2_singlet']:.2f}) "
                             f"{'below' if r['dE_ST_meV'] > 0 else 'above'} the triplet-like root (⟨S²⟩ "
                             f"{r['S2_triplet']:.2f}) by {abs(r['dE_ST_meV']):.2f} meV")
            else:
                parts.append(f"{f}: two unmixed determinants (⟨S²⟩ {r['pair_S2']}) {r['pair_splitting_meV']:.1f} meV "
                             "apart, no S/T pair")
        L.append(f"**{names.get(g, g)}.** BS E_T − E_S = −J (Yamaguchi) = {fmt(yam(g))} meV. SF-TDA: " + "; ".join(parts) + ".")
        L.append("")
    ok = [r for r in rows if r["status"] == "OK"]
    mixed = [r for r in ok if r.get("dE_ST_meV") is not None]
    unmixed = [r for r in ok if r.get("pair_type", "").startswith("unmixed")]
    contaminated = [r for r in ok if r["reference_S2"] and r["reference_S2"] > 2.3]
    L += ["## Overall\n"]
    if mixed:
        L.append(f"- SF-TDA gives a usable singlet/triplet pair in {len(mixed)} of {len(ok)} finished calculations ("
                 + ", ".join(f"{r['geometry']}/{r['functional']}" for r in mixed) + "). There the singlet-like root is "
                 + ("lower in all cases" if all(r["dE_ST_meV"] > 0 for r in mixed) else "not consistently lower")
                 + ", the same sign as BS-DFT. The two roots are strongly spin-mixed, so the magnitude is only "
                 "indicative.")
    if unmixed:
        L.append(f"- In {len(unmixed)} calculations (" + ", ".join(f"{r['geometry']}/{r['functional']}" for r in unmixed)
                 + ") the two lowest SOMO-flip roots are unmixed BS-like determinants split by "
                 + ", ".join(f"{r['pair_splitting_meV']:.0f}" for r in unmixed) + " meV, far more than any plausible "
                 "exchange. SF-TDA from a UKS triplet cannot resolve the coupling at these geometries (localized, "
                 "inequivalent radicals): it neither confirms nor refutes the BS result. A spin-complete method "
                 "(e.g. CASSCF(2,2)/NEVPT2, as used by Zuo et al. for the isolated molecule) would be needed there.")
    if contaminated:
        L.append("- The BHANDHLYP triplet references are strongly spin-contaminated (⟨S²⟩ = "
                 + ", ".join(f"{r['reference_S2']:.2f}" for r in contaminated) + "), which further degrades the "
                 "BHANDHLYP SF-TDA states.")
    L.append("- Differences below ~0.5 meV are not interpreted.")
    (OUTPUT_ROOT / "sf_tda_summary.md").write_text("\n".join(L) + "\n")


# =====================================================================
# MAIN
# =====================================================================
def main():
    missing = [f"{k}: {p}" for k, p in GEOMETRIES.items() if not Path(p).is_file()]
    if USE_EXISTING_GBW:
        missing += [f"{k} (gbw): {p}" for k, p in GBW_FILES.items() if k in GEOMETRIES and not Path(p).is_file()]
    if RUN_CALCULATIONS and not ORCA.is_file():
        missing.append(f"ORCA executable: {ORCA}")
    if missing:
        sys.exit("Missing files, nothing was run:\n  " + "\n  ".join(missing))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = {}
    # functional-major order: all PBE0 runs first (fast, closest guesses), then BHANDHLYP
    for func in FUNCTIONALS:
        for geom in GEOMETRIES:
            run_dir = OUTPUT_ROOT / geom / func
            out = run_dir / f"{CALC_NAME}.out"
            if run_dir.exists() and OVERWRITE:
                clean_previous(run_dir)
            if not out.exists():
                moread = prepare(geom, func, run_dir)
                if RUN_CALCULATIONS:
                    print(f"[{datetime.now():%H:%M:%S}] running {geom}/{func} ...", flush=True)
                    log = run_orca(run_dir)
                    print(f"    return code {log['return_code']}, {log['runtime_s'] / 60:.1f} min", flush=True)
            else:
                print(f"{geom}/{func}: existing output kept (OVERWRITE = False)")
            moread = "MORead" in (run_dir / f"{CALC_NAME}.inp").read_text() if (run_dir / f"{CALC_NAME}.inp").exists() else False
            log_path = run_dir / "run_log.json"
            log = json.loads(log_path.read_text()) if log_path.exists() else {}
            rec = parse_output(run_dir)
            g = gap(rec) if rec["status"] == "OK" else {}
            results[(geom, func)] = (rec, g, log, moread)
            print(f"    {geom}/{func}: {rec['status']}"
                  + (f", dE_ST = {g['dE_ST_meV']:+.3f} meV (roots S{g['singlet_root']}/T{g['triplet_root']})"
                     if g.get("dE_ST_meV") is not None else ""))
    write_outputs(results)
    print(f"Wrote {OUTPUT_ROOT / 'sf_tda_summary.md'}")


if __name__ == "__main__":
    main()

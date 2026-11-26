#!/usr/bin/env python3
"""
Lowest-energy broken-symmetry state of every dimer, rendered with spin arrows and collected on compact pages.

For every dimer of <run>_broken_symmetry/relaxed (run2, run3: <dimer>/<config>; run4: <set>/<dimer>/<config>):
  1. read FINAL SINGLE POINT ENERGY of every configuration (uuuu, duuu, ..., duud) that terminated normally;
  2. take the lowest one (the "BS ground state");
  3. render it with batch_render_orca_arrows.py (spin colours, molecule planes, arrows; that script's
     settings apply) into OUT_DIR/renders/<run>_<dimer>/<config>.png (skipped if it exists, unless RERENDER),
     plus a copy as <config>.pdf;
  4. place all renders on A4 pages, NROWS x NCOLS per page, title "<run>/<dimer>" only:
     OUT_DIR/bs_ground_states.pdf (all pages) and OUT_DIR/bs_ground_states_page<k>.png.
OUT_DIR/bs_ground_states.csv lists the chosen configuration, its energy, the gap to the next state and
the number of finished configurations per dimer. Dimers without finished outputs are listed and skipped.

Settings in the `if __name__ == "__main__":` block.
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batch_render_orca_arrows as R  # noqa: E402

CONFIGS = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
HARTREE_TO_MEV = 27211.386245988


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def find_dimers(dimers_root, runs, stage):
    """[(run, dimer_label, dimer_dir)], dimer_label e.g. conf_7 or free/docked_1."""
    out = []
    for run in runs:
        base = dimers_root / f"{run}_broken_symmetry" / stage
        if not base.exists():
            print(f"[!] {base} not found, {run} skipped")
            continue
        dirs = sorted({p.parent for p in base.rglob("uuuu") if p.is_dir()}, key=natural_key)
        out += [(run, d.relative_to(base).as_posix(), d) for d in dirs]
    return out


def bs_energies(dimer_dir):
    """{config: (E_Eh, out_file)} for every configuration that terminated normally."""
    res = {}
    for c in CONFIGS:
        f = dimer_dir / c / f"{c}.out"
        if not f.exists():
            continue
        text = f.read_text(errors="ignore")
        e = ENERGY_RE.findall(text)
        if "ORCA TERMINATED NORMALLY" in text[-5000:] and e:
            res[c] = (float(e[-1]), f)
    return res


def png_to_pdf(png, pdf, dpi=300):
    """Single-page PDF of a render (the PyMOL ray trace is a raster image: the PDF embeds it losslessly)."""
    img = plt.imread(png)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.figimage(img, resize=False)
    fig.savefig(pdf, dpi=dpi)
    plt.close(fig)


def trim_white(img, thr=0.97, pad=6):
    """Crop the white margins of a rendered PNG (keeps a small padding)."""
    rgb = img[..., :3]
    mask = (rgb < thr).any(axis=2)
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1])
    return img[r0:r1, c0:c1]


def make_pages(items, out_dir, nrows, ncols, page_size, title_size, dpi):
    """items: [(title, png)]; one page per nrows*ncols panels."""
    per_page = nrows * ncols
    pages = [items[k:k + per_page] for k in range(0, len(items), per_page)]
    pdf_path = out_dir / "bs_ground_states.pdf"
    with PdfPages(pdf_path) as pdf:
        for k, page in enumerate(pages, 1):
            fig, axes = plt.subplots(nrows, ncols, figsize=page_size, squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, (title, png) in zip(axes.ravel(), page):
                ax.imshow(trim_white(plt.imread(png)))
                ax.set_title(title, fontsize=title_size, pad=2)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, wspace=0.03, hspace=0.12)
            pdf.savefig(fig, dpi=dpi)
            fig.savefig(out_dir / f"bs_ground_states_page{k}.png", dpi=dpi)
            plt.close(fig)
    print(f"wrote {pdf_path} ({len(pages)} pages) and bs_ground_states_page1..{len(pages)}.png")


def main(dimers_root, runs, stage, out_dir, nrows, ncols, page_size, title_size, dpi, rerender):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, items, missing = [], [], []
    for run, dimer, d in find_dimers(dimers_root, runs, stage):
        E = bs_energies(d)
        if not E:
            missing.append(f"{run}/{dimer}")
            continue
        order = sorted(E, key=lambda c: E[c][0])
        gs = order[0]
        gap = (E[order[1]][0] - E[gs][0]) * HARTREE_TO_MEV if len(order) > 1 else np.nan
        rows.append({"run": run, "dimer": dimer, "ground_state_config": gs, "E_Eh": f"{E[gs][0]:.9f}",
                     "gap_to_next_meV": f"{gap:.3f}", "next_config": order[1] if len(order) > 1 else "",
                     "n_finished_configs": len(E)})
        render_dir = out_dir / "renders" / f"{run}_{dimer.replace('/', '_')}"
        png = render_dir / f"{gs}.png"
        if rerender or not png.exists():
            R.main(E[gs][1].parent, render_dir)          # renders <config>.png into render_dir
        pdf = png.with_suffix(".pdf")
        if png.exists() and (rerender or not pdf.exists() or pdf.stat().st_mtime < png.stat().st_mtime):
            png_to_pdf(png, pdf)
        items.append((f"{run}/{dimer}", png))
    with (out_dir / "bs_ground_states.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    if missing:
        print(f"[!] no finished BS outputs, skipped ({len(missing)}): {', '.join(missing)}")
    make_pages(items, out_dir, nrows, ncols, page_size, title_size, dpi)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    REPO_ROOT = HERE.parent
    DIMERS_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
    RUNS = ["run2", "run3", "run4"]
    STAGE = "relaxed"
    OUT_DIR = REPO_ROOT / "visualisation" / "pymol_rendered_cmap" / "orca" / "dimers_run2" / "bs_ground_states"
    NROWS, NCOLS = 6, 4                  # panels per page
    PAGE_SIZE = (8.27, 11.69)            # inches (A4 portrait)
    TITLE_SIZE = 8
    DPI = 300
    RERENDER = False                     # True: re-render even if the PNG exists (e.g. after changing render settings)

    main(DIMERS_ROOT, RUNS, STAGE, OUT_DIR, NROWS, NCOLS, PAGE_SIZE, TITLE_SIZE, DPI, RERENDER)

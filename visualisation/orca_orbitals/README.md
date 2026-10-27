# ORCA orbital images

Renders frontier MOs, plus UNOs and UCOs when those files exist, from ORCA calculations that have already converged. Every image uses the same isovalue, style and camera. Geometries are rigidly aligned, so images from different calculations can be compared side by side.

No new ORCA calculations are run. The cubes come from the existing `.gbw`, `.uno` and `.uco` files through `orca_plot`. The original calculation files are never modified.

## Installation

```bash
conda activate DFT                      # already has everything except pytest
pip install -r orca_orbitals/requirements.txt
```

You also need ORCA's `orca_plot`, which was tested with 6.1.1. The script finds it on `PATH`, through `$ORCA_DIR`, or in `~/software/orca*/`. To use a specific copy, set `ORCA_PLOT`.

## Running

1. Edit the `CONFIGURATION` block at the top of `visualisation/render_orca_orbitals.py`. Usually you only change `ROOT`.
2. Leave `SMOKE_TEST = True` for the first run. That processes one calculation (the reference) so you can check the images.
3. Set `SMOKE_TEST = False` and run again to process everything under `ROOT`.

```bash
cd visualisation
python render_orca_orbitals.py
```

`ROOT` can be a single calculation folder or a whole tree; no folder layout is assumed. To restrict a run to part of a tree, use `INCLUDE_PATTERNS` / `EXCLUDE_PATTERNS`, which are fnmatch patterns on `.out` paths relative to `ROOT`, for example `["*/F0/*"]`.

Cost: at the default `GRID_SIZE = 90` (90³ points), one calculation takes about 30 s and the 8 frontier cubes take about 80 MB. To delete cubes after rendering, set `KEEP_CUBES = False`. Existing cubes are reused unless `REGENERATE_CUBES = True`.

## Output

```
<calculation dir>/orbital_analysis/          (/<job name>/ appended if the dir holds several jobs)
    cubes/            alpha_HOMO_mo177.cube ...   cubes/UNO/, cubes/UCO/
    images/           alpha_HOMO_mo177.png ...    images/UNO/, images/UCO/
    contact_sheets/   frontier_orbitals.png  UNO.png  UCO.png
    orbital_metadata.csv
    processing_log.txt
    view_and_alignment.json        (this job's display transform, camera, style)
    logs/                          (orca_plot output for any failed cube)
<ROOT>/orbital_analysis/
    orbital_summary.csv            (all rows of every job)
    skipped_calculations.csv       (job, stage, reason)
    view_settings.json             (reference geometry, frame, camera)
    run_log.txt
```

The individual PNGs have no text on them. The label, the ORCA index and the energy (or occupation / overlap) are written above each panel of the contact sheet and stored in the CSV.

## How each step works

**Matching files to a calculation.** A calculation is an ORCA `.out` file, recognised by its banner. Its companion files must be in the same directory and named after the job name, which is taken from the `NAME = <job>.inp` line. Files such as `guess.gbw`, `high_spin_reference.gbw` or a `.gbw` in a parent folder are never used. Two further checks catch an orbital file that belongs to a different calculation:

- The geometry stored in the orbital file must match the final geometry in the `.out` (RMSD < 0.002 Å).
- Its orbital count must equal the basis dimension.

A mismatch skips the calculation and records the reason.

**Choosing the orbitals.** Everything comes from the *last* `ORBITAL ENERGIES` section, because BS jobs print the HS step first and optimisations print one section per step. Alpha and beta are handled separately. The pipeline requires integer, aufbau occupations and checks them against N(α)/N(β). Indices are ORCA's zero-based numbers, which is also what `orca_plot` expects. A calculation is skipped if:

- its final SCF did not converge,
- ORCA terminated abnormally, or
- its occupations are fractional or non-aufbau.

Near-degenerate pairs (closer than 10 meV) are flagged in `message`.

**Electronic state.** The state is inferred from the final N(α) − N(β) and ⟨S²⟩, not from the folder name. FlipSpin jobs declare `Mult 3` in their input but end with Ms = 0.

**Generating the cubes with `orca_plot`.** All cubes for one file are made in a single session in a private work directory, on a symlink to the original file. This is needed because `orca_plot` writes `<job>.xyz` and the cubes next to the file it opens. Menu used: `4` grid → `5`/`7` Gaussian cube → (`2` MO, `3` operator, `11` plot) per orbital → `12` exit.

- Menu `1` (plot type) is avoided: in 6.1.1 it aborts unless `.densitiesinfo` exists. MO-PLOT is the default and is checked instead.
- An operator that does not exist produces an *empty* cube. So the operator count is read first, and every cube header ("Molecular orbital N of operator k") is checked against the request.

**Reading the cubes.** The parser handles positive and negative atom counts, the orbital-ID record (which may wrap over several lines), lengths in bohr (positive voxel count) or Å (negative), non-orthogonal voxel vectors, and multi-orbital cubes. A wrong number of values raises an error; nothing is silently cut off.

**Isosurfaces.** Marching cubes runs in grid-index space. The vertices are then mapped with the full transform r = origin + ijk·V. The values are amplitudes, so the surfaces are ψ = +iso (blue) and ψ = −iso (red), with the same `ISOVALUE` for every image. If neither sign reaches the isovalue, no image is written (status `no_surface`). A surface that touches the box edge is flagged.

**Alignment and camera.** Each geometry is Kabsch-aligned onto the reference using `ALIGN_ATOMS`, with proper rotations only (no reflections). The same transform is applied to the atoms and to the orbital surfaces. The display frame is the reference's principal axes, plus `VIEW_ROTATION_DEG` if set.

The camera is orthographic, looks down −z, and is sized once from the atoms of *all* geometries plus `FRAME_MARGIN`. It never depends on the size of a lobe. It is saved in `view_settings.json` and reused on later runs, so images added later match the earlier ones. If you change `ALIGN`, `ALIGN_ATOMS`, `VIEW_ROTATION_DEG` or the window size, the run stops until you delete that file.

Default `ALIGN_ATOMS = 0..13` is the heavy atoms of fragment 2 in the short-2OS numbering. For field sweeps or HS/BS comparisons at one geometry, any choice works, since the geometries are identical.

**UNO stage.** It runs when `<job>.uno` exists. ORCA 6 writes UNO occupations only to `<job>.property.txt` (the last `$Natural_Orbitals / &OccUno` block), not to the `.out`. Orbitals with n inside `UNO_OCCUPATION_RANGE` are candidates, and those closest to n = 1 are plotted first. Labels are `UNO <index>` with the occupation, never HOMO or LUMO. The `.uno` file must contain exactly one operator.

**UCO stage.** It runs when `<job>.uco` exists. Pair overlaps come from the last "UHF CORRESPONDING ORBITALS" table in the `.out`. Pairs with S < `UCO_MAX_OVERLAP` are plotted, lowest S first, with α (operator 0) and β (operator 1) side by side. The `.uco` file must contain two operators.

To produce both files, add `! UNO UCO` to the input. On the BS run, `UCO` is sufficient for the overlap table.

## Interpreting the images

- Labels follow energy ordering. "HOMO" is an index, not an identity: across a field sweep or a torsion scan, orbitals can swap order. `tracking.py` is where overlap-based identity tracking would go, and every row records `identity_method = energy_order`.
- Orbital signs are arbitrary. `PHASE_CONVENTION = "none"` keeps ORCA's sign. `"largest_positive"` applies an explicit flip for display only, recorded in `sign_flipped`. A colour swap between two images is not a physical change.
- An isosurface picture is not a quantitative measure of localisation.

## Tests

```bash
cd visualisation
python -m unittest discover -s orca_orbitals/tests -t .      # or: pytest orca_orbitals/tests
```

The tests cover:

- frontier selection: HS and BS, restricted, unconverged, fractional and non-aufbau cases, and real outputs;
- cube parsing: hand-written Gaussian and ORCA-layout files, Å units, wrapped orbital records, non-orthogonal grids checked against an analytic function, and malformed files;
- lobe position and sign;
- Kabsch alignment (recovers rotations, never reflects);
- identical camera state and image size for small and large lobes, and repeat renders that are pixel-identical;
- an end-to-end run on a real ORCA 6.1.1 broken-symmetry fixture with `! UNO UCO` (`tests/data/bs_methyl`, two CH₃ radicals 5 Å apart).

The end-to-end run checks that:

- the original files are unchanged;
- the α and β magnetic orbitals (HOMO, and UCO pair 8 with S = 0.025) sit on *different* carbons, each centred within 0.5 Å of its carbon;
- missing `.uno`/`.uco` files, missing UNO occupations, a geometry-mismatched `.gbw`, and smoke-test mode are all handled.

Its contact sheets (`bs_methyl/orbital_analysis/contact_sheets/` in the test's temporary directory) serve as validation images. Tests that need `orca_plot` are skipped if it is not installed.

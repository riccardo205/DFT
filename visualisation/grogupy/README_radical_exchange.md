# Exchange extraction for the four delocalized radical sites

## What the groups mean

The group file contains four mutually disjoint carbon clusters, in SIESTA 1-based
atom indices:

| entity | group | physical meaning |
|---|---:|---|
| A | 1 | molecule 1, positive-spin sublattice |
| B | 2 | molecule 1, negative-spin sublattice |
| C | 3 | molecule 2, positive-spin sublattice |
| D | 4 | molecule 2, negative-spin sublattice |

The calculation should first be fitted as a four-site model

\[
H = \sum_{i<j} {\bf e}_i^\mathsf{T}{\bf J}_{ij}{\bf e}_j +
    \sum_i {\bf e}_i^\mathsf{T}{\bf K}_i{\bf e}_i .
\]

The six requested pair tensors are AB, CD, AC, AD, BC, and BD.  Do not start
from two "molecular" moments, because the nearly compensated A+B and C+D sums
are not stable local spin degrees of freedom.

## 1. Validate the supplied SIESTA output

```bash
python extract_radical_exchange.py \
  --out siesta.out \
  --groups conf_11_antiferro1_Sz_groups86_separated_atoms.txt \
  --report radical_spin_report.json
```

Expected checks for the attached calculation:
- The four groups partition the 108 carbon atoms: C 1--54 and C 87--140.
- Each cluster has a coherent net local moment: `|sum_i m_i| / sum_i |m_i|`
  should be close to one.
- A and C are parallel, B and D are parallel, and the two sectors are
  antiparallel.  Small deviations quantify the intentional 5-degree canting.
- The total moment of the supercell should be near zero.

This is a validation of the local-moment partition, *not* an exchange
calculation.

## 2. Run the torque calculation

Run in the original SIESTA results directory.  The original `.fdf` must be
present, together with the side files referenced by `SystemLabel` (normally a
`.TSHS` and density-matrix files).  The attached `.out` indicates that
`SaveHS`, `Write.DM.NetCDF`, and `Write.DMHS.NetCDF` were enabled.

```bash
python -m pip install "grogupy==0.3.1"

python extract_radical_exchange.py \
  --fdf dimers-short-2OS.fdf \
  --groups conf_11_antiferro1_Sz_groups86_separated_atoms.txt \
  --run-grogupy \
  --kmesh 1 1 1 \
  --eset 400 --esetp 10000 \
  --scf-axis 0 0 1 \
  --assume-no-soc \
  --orbital-model all \
  --magnopy radical_4site_allPAO.magnopy \
  --exchange-json radical_4site_allPAO.json
```

For the attached calculation, the final local vectors are almost ±z, so the
reference axis `[0, 0, 1]` is appropriate despite SIESTA running in
non-collinear mode.

The script converts the 1-based SIESTA atom indices to the 0-based indices
used by `sisl`/`grogupy`; do not subtract one manually.

## 3. Validation / convergence protocol

Run the following before reporting a numerical \(J\).

1. **Built-in code tests**

   ```bash
   python extract_radical_exchange.py --self-test
   ```

   This checks the group/index logic, tensor decomposition, DM-sign convention,
   and global-rotation invariance of a known isotropic model.

2. **Contour convergence**

   Repeat the torque calculation with `--eset 200`, `400`, and `800`.
   The isotropic parts of all six pair tensors should be stable within your
   target error bar.

3. **Projector sensitivity**

   Repeat with the carbon p shells only:

   ```bash
   ... --orbital-model p \
       --magnopy radical_4site_pPAO.magnopy \
       --exchange-json radical_4site_pPAO.json
   ```

   A physically meaningful four-spin mapping should not qualitatively change
   under this reasonable change of local projector.  If a pair changes sign or
   changes by tens of percent, its individual `J_ij` is not robust; report the
   uncertainty rather than a single number.

4. **Solver sensitivity**

   The script uses `generalised-grogu` by default.  As a secondary
   implementation check, repeat a converged run with
   `--spin-model generalised-fit`.  It should reproduce the robust pairwise
   isotropic couplings within numerical uncertainty; do not use the fitted
   variant as the sole result.

5. **No-SOC symmetry**

   When spin-orbit coupling is absent, the fitted exchange tensor should be
   close to \(J_\mathrm{iso} I\), with small traceless-symmetric and
   Dzyaloshinskii–Moriya components.  The `--assume-no-soc` diagnostic writes
   these ratios to the JSON output.  A large anisotropic/DM term in a no-SOC
   run normally signals insufficient contour convergence or an unsuitable
   local partition.

6. **Moment rigidity**

   Compare the `local_S`/`total_S` printed by grogupy across the three contour
   runs and across the `all` versus `p` projectors.  The charge and moment of
   every entity should remain stable.  Large variation means the electrons are
   too itinerant/overlapping for a unique local-spin model.

## Interpretation

Grogupy outputs the energy-normalized convention

\[
H = \frac12 \sum_{i\ne j}{\bf e}_i^\mathsf{T}{\bf J}_{ij}{\bf e}_j
+ \sum_i {\bf e}_i^\mathsf{T}{\bf K}_i{\bf e}_i ,
\]

where \({\bf e}_i\) are unit vectors.  Thus `J_iso_meV > 0` favors
antiparallel alignment (AFM), while `J_iso_meV < 0` favors parallel alignment
(FM).  Keep this normalized-spin convention for the first report.  Converting
to a quantum-spin Hamiltonian later requires an explicit and defensible choice
of the effective local spin length \(S_i\).

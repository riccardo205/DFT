import numpy as np
from ase.io import read, write
from ase import Atoms


def as_array(x):
    """Convert 1-based indices to 0-based for internal numpy indexing."""
    return np.array(x, dtype=int) - 1


def center_of_atoms(atoms: Atoms, idx):
    idx = as_array(idx)
    return atoms.positions[idx].mean(axis=0)


def fit_plane_normal(points: np.ndarray, tip_pos: np.ndarray):
    """
    Fit plane to points (N,3). Returns unit normal pointing towards tip.
    Uses SVD: normal = last right-singular vector of centered coordinates.
    The normal is flipped if necessary to point from plane towards tip.
    """
    pts = np.asarray(points, float)
    c = pts.mean(axis=0)
    X = pts - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n)
    
    # Flip n so it points from plane towards tip
    v = tip_pos - c
    if np.dot(n, v) < 0:
        n = -n
    
    return n


def rotation_matrix_from_vectors(a, b):
    """
    Rotate unit vector a to unit vector b.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        # 180° rotation: pick any perpendicular axis
        axis = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        return np.eye(3) + 2 * K @ K

    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    return R


def apply_rotation(atoms: Atoms, R: np.ndarray, about_point: np.ndarray):
    """
    Rotate atoms by R around about_point.
    """
    p = atoms.positions - about_point
    atoms.positions = (p @ R.T) + about_point


def _plane_normal(points: np.ndarray):
    pts = np.asarray(points, float)
    c = pts.mean(axis=0)
    X = pts - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n)
    return n


def _end_outward_normal(mol: Atoms, end_idx, mol_center: np.ndarray):
    pts = mol.positions[as_array(end_idx)]
    end_center = pts.mean(axis=0)
    fallback = end_center - mol_center
    if np.linalg.norm(fallback) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
    if len(pts) == 1:
        n = fallback
    elif len(pts) == 2:
        s = pts[1] - pts[0]
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-8:
            n = fallback
        else:
            s_unit = s / s_norm
            n = fallback - s_unit * np.dot(fallback, s_unit)
            if np.linalg.norm(n) < 1e-8:
                axis = np.array([1.0, 0.0, 0.0])
                if np.abs(np.dot(axis, s_unit)) > 0.9:
                    axis = np.array([0.0, 1.0, 0.0])
                n = np.cross(s_unit, axis)
    else:
        n = _plane_normal(pts)
        if np.dot(n, fallback) < 0:
            n = -n
    n /= np.linalg.norm(n)
    return n


def _mol_end_axis(mol: Atoms, mol_end1, mol_end2):
    m1 = center_of_atoms(mol, mol_end1)
    m2 = center_of_atoms(mol, mol_end2)
    mol_center = mol.positions.mean(axis=0)

    def end_normal(end_idx, other_center):
        pts = mol.positions[as_array(end_idx)]
        if len(pts) < 3:
            extras = [other_center, mol_center]
            need = 3 - len(pts)
            pts = np.vstack([pts, *extras[:need]])
        return fit_plane_normal(pts, other_center)

    n1 = end_normal(mol_end1, m2)
    n2 = end_normal(mol_end2, m1)
    axis = n1 + n2
    if np.linalg.norm(axis) < 1e-8:
        axis = m2 - m1
    axis /= np.linalg.norm(axis)
    if np.dot(axis, (m2 - m1)) < 0:
        axis = -axis
    return axis


def build_junction(
    mol_path="mol.xyz",
    left_path="el_L.xyz",
    right_path="el_R.xyz",
    out_path="junction.xyz",
    # indices are 1-based
    mol_end1=(0,),
    mol_end2=(1,),
    L_tip=(0,),
    R_tip=(0,),
    L_plane_atoms=(1, 2, 3),
    R_plane_atoms=(1, 2, 3),
    gap_L=2.2,
    gap_R=2.2,
    # axis along which to align the entire junction
    junction_axis=(1.0, 0.0, 0.0),
):
    mol = read(mol_path)
    elL = read(left_path)
    elR = read(right_path)

    # --- molecule ends and axis
    m1 = center_of_atoms(mol, mol_end1)
    m2 = center_of_atoms(mol, mol_end2)
    mol_axis = (m2 - m1)
    mol_axis /= np.linalg.norm(mol_axis)
    mol_center = mol.positions.mean(axis=0)
    
    # Align molecule axis with junction_axis
    if junction_axis is None or (
        isinstance(junction_axis, str) and junction_axis.lower() == "auto"
    ):
        tgt_axis = _mol_end_axis(mol, mol_end1, mol_end2)
    else:
        tgt_axis = np.asarray(junction_axis, float)
        tgt_axis /= np.linalg.norm(tgt_axis)
    R_mol = rotation_matrix_from_vectors(mol_axis, tgt_axis)
    apply_rotation(mol, R_mol, about_point=mol_center)
    
    # Recompute ends after rotation
    m1 = center_of_atoms(mol, mol_end1)
    m2 = center_of_atoms(mol, mol_end2)
    mol_axis = tgt_axis  # now aligned with junction_axis
    mol_center = mol.positions.mean(axis=0)
    n1_out = _end_outward_normal(mol, mol_end1, mol_center)
    n2_out = _end_outward_normal(mol, mol_end2, mol_center)
    # A helper to orient an electrode:
    def orient_electrode(el: Atoms, tip_idx, plane_idx, desired_tip_dir):
        tip = center_of_atoms(el, tip_idx)
        plane_pts = el.positions[as_array(plane_idx)]
        n = fit_plane_normal(plane_pts, tip)

        # Rotate plane normal -> desired_tip_dir (so planes are parallel)
        tgt = np.asarray(desired_tip_dir, float)
        tgt /= np.linalg.norm(tgt)

        R1 = rotation_matrix_from_vectors(n, tgt)
        apply_rotation(el, R1, about_point=tip)

    # --- Orient electrodes along junction_axis (parallel guess)
    # Left electrode: normal points toward +junction_axis (into molecule)
    orient_electrode(elL, L_tip, L_plane_atoms, desired_tip_dir=+tgt_axis)
    # Right electrode: normal points toward -junction_axis (into molecule)
    orient_electrode(elR, R_tip, R_plane_atoms, desired_tip_dir=-tgt_axis)

    # --- Tilt electrodes to match the outward normals of each mol end
    # This allows each electrode plane to follow its local mol_end orientation.
    L_tip_pos = center_of_atoms(elL, L_tip)
    R_tip_pos = center_of_atoms(elR, R_tip)
    R_left = rotation_matrix_from_vectors(-tgt_axis, n1_out)
    R_right = rotation_matrix_from_vectors(+tgt_axis, n2_out)
    apply_rotation(elL, R_left, about_point=L_tip_pos)
    apply_rotation(elR, R_right, about_point=R_tip_pos)

    # --- Translate electrodes to desired gaps using the end-plane outward normals
    # Place left tip at m1 + gap_L * n1_out (so tip is just outside the molecule end)
    L_tip_pos = center_of_atoms(elL, L_tip)
    target_L = m1 + gap_L * n1_out
    elL.translate(target_L - L_tip_pos)

    # Place right tip at m2 + gap_R * n2_out
    R_tip_pos = center_of_atoms(elR, R_tip)
    target_R = m2 + gap_R * n2_out
    elR.translate(target_R - R_tip_pos)

    # --- Combine
    junction = mol + elL + elR
    write(out_path, junction)
    print(f"Wrote {out_path} with {len(junction)} atoms.")


if __name__ == "__main__":
    # IMPORTANT: indices here are 1-based!
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    # --- Configuration ---
    mol_folder   = Path("/Users/riccardo/Documenti/TuDelft/phd/Analysis_Code/DFT/Short-2OS/runs/main_conf/slow_lowT_lowest_forces")        # folder with molecule xyz files
    out_folder   = Path("main_conf/junction_SM_test")     # output folder
    left_path    = "Au_electrodes_relaxed/right.xyz"
    right_path   = "Au_electrodes_relaxed/right.xyz"

    mol_end1     = (59, 56, 42)       #(59, 56,) anchoring atoms at one end (1-based)
    mol_end2     = (66, 70, 29)       #(66, 70) anchoring atoms at other end (1-based)
    L_tip        = (1,6,11) ## (1,)
    R_tip        = (1,6, 11) ## (1,)
    L_plane_atoms = (20, 16) ##(11, 20, 17)
    R_plane_atoms = (20, 16) ##(11, 20, 17)
    gap          = 2.3
    # ---------------------

    out_folder.mkdir(parents=True, exist_ok=True)

    mol_files = [
        xyz
        for antidir in mol_folder.iterdir()
        if antidir.is_dir() and antidir.name.endswith("antiferro")
        for xyz in antidir.glob("*.xyz")
    ]

    print(mol_files)
    
    # mol_files = sorted(mol_folder.glob("*.xyz"))
    
    if not mol_files:
        print(f"No .xyz files found in {mol_folder}")
        sys.exit(1)

    for i, mol_path in enumerate(mol_files, start=1):
        out_path = out_folder / f"junction_{i}.xyz"
        print(f"Building junction for {mol_path.name} -> {out_path.name}")
        build_junction(
            mol_path=str(mol_path),
            left_path=left_path,
            right_path=right_path,
            out_path=str(out_path),
            mol_end1=mol_end1,
            mol_end2=mol_end2,
            L_tip=L_tip,
            R_tip=R_tip,
            L_plane_atoms=L_plane_atoms,
            R_plane_atoms=R_plane_atoms,
            gap_L=gap,
            gap_R=gap,
            junction_axis=(1.0, 0.0, 0.0),
        )

    print("Done.")

    # --- Old version: gap sweep over a single molecule ---
    # gap_list = np.linspace(1.0, 3.0, num=21)
    # print(f"Gap list: {gap_list}")
    #
    # for i, gap in enumerate(gap_list, start=1):
    #     build_junction(
    #         mol_path="main_conf/molecule/short-2OS.xyz",
    #         left_path="Au_electrodes_relaxed/right.xyz",
    #         right_path="Au_electrodes_relaxed/right.xyz",
    #         out_path=f"main_conf/junctions_2/conf_{i}.xyz",
    #
    #         mol_end1=(59, 56),
    #         mol_end2=(66, 70),
    #
    #         L_tip=(1,),
    #         R_tip=(1,),
    #
    #         L_plane_atoms=(11, 20, 17),
    #         R_plane_atoms=(11, 20, 17),
    #
    #         gap_L=gap,
    #         gap_R=gap,
    #         junction_axis=(1.0, 0.0, 0.0),
    #     )

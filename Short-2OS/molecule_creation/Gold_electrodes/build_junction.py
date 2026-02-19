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
    tgt_axis = np.asarray(junction_axis, float)
    tgt_axis /= np.linalg.norm(tgt_axis)
    R_mol = rotation_matrix_from_vectors(mol_axis, tgt_axis)
    apply_rotation(mol, R_mol, about_point=mol_center)
    
    # Recompute ends after rotation
    m1 = center_of_atoms(mol, mol_end1)
    m2 = center_of_atoms(mol, mol_end2)
    mol_axis = tgt_axis  # now aligned with junction_axis

    # A helper to orient an electrode:
    def orient_electrode(el: Atoms, tip_idx, plane_idx, desired_tip_dir):
        tip = center_of_atoms(el, tip_idx)
        plane_pts = el.positions[as_array(plane_idx)]
        n = fit_plane_normal(plane_pts, tip)
        print("n (pointing towards tip):", n)

        # Rotate plane normal -> desired_tip_dir (so planes are parallel)
        tgt = np.asarray(desired_tip_dir, float)
        tgt /= np.linalg.norm(tgt)

        R1 = rotation_matrix_from_vectors(n, tgt)
        apply_rotation(el, R1, about_point=tip)

    # --- Orient electrodes along junction_axis
    # Left electrode: normal points toward +junction_axis (into molecule)
    orient_electrode(elL, L_tip, L_plane_atoms, desired_tip_dir=+tgt_axis)
    # Right electrode: normal points toward -junction_axis (into molecule)
    orient_electrode(elR, R_tip, R_plane_atoms, desired_tip_dir=-tgt_axis)

    # --- Translate electrodes to desired gaps
    # Place left tip at m1 - gap_L * junction_axis (so tip is just outside the molecule end)
    L_tip_pos = center_of_atoms(elL, L_tip)
    target_L = m1 - gap_L * tgt_axis
    elL.translate(target_L - L_tip_pos)

    # Place right tip at m2 + gap_R * junction_axis
    R_tip_pos = center_of_atoms(elR, R_tip)
    target_R = m2 + gap_R * tgt_axis
    elR.translate(target_R - R_tip_pos)

    # --- Combine
    junction = mol + elL + elR
    write(out_path, junction)
    print(f"Wrote {out_path} with {len(junction)} atoms.")


if __name__ == "__main__":
    # IMPORTANT: indices here are 1-based!
    build_junction(
        mol_path="main_conf/molecule/short-2OS.xyz",
        left_path="Au_electrodes_relaxed/right.xyz",
        right_path="Au_electrodes_relaxed/right.xyz",
        out_path="main_conf/junctions/junction12.xyz",

        # EXAMPLES (replace with yours)
        mol_end1=(59,56),           # e.g. anchoring atom at one end
        mol_end2=(66,70),          # anchoring atom at other end

        L_tip=(1,),
        R_tip=(1,),

        L_plane_atoms=(11, 20, 17),   # atoms on the plane away from tip
        R_plane_atoms=(11, 20, 17),

        gap_L=3.9,
        gap_R=3.9,
        junction_axis=(1.0, 0.0, 0.0),
    )
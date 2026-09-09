"""
Direct quantum circuit synthesis via recursive Cosine-Sine Decomposition (CSD),
following the method of:
  Chen & Wang, "Qcompiler: quantum compilation with CSD method", arXiv:1208.0194
  (Section 2: general recursive CSD scheme; Section 3: real-matrix simplification)

Verified on RandReal.txt (8x8 real orthogonal matrix, n=3 qubits):
    28 pure Ry gates + 1 global Pi (Z) gate on qubit w1 (q0), unconditioned
    = 29 total gates, reconstruction error ~1e-14.

Wire convention (matches user's existing pipeline):
    w1 = q0 = MSB,  w2 = q1,  w3 = q2 = LSB

Core algorithm
--------------
1. LAPACK CSD (scipy.linalg.cossin) recursively splits U -> U1 @ CS(theta) @ V1
   where U1 = blkdiag(u1,u2), V1 = blkdiag(v1h,v2h), each half the size.
2. Recursion bottoms out at 2x2 blocks: any real orthogonal 2x2 matrix is either
   a pure rotation Ry(theta) (det=+1) or a reflection Ry(theta) @ Z (det=-1).
3. Key identity used to collapse ALL discrete sign information into a single
   trailing diagonal correction D (instead of leaving reflections scattered
   throughout the circuit):

        diag(d0, d1) @ Ry(theta) = Ry(theta') @ diag(d0, d1)
        where theta' = theta if d0 == d1, else theta' = -theta

   This lets any diagonal +-1 correction be commuted through an entire gate
   sequence (angles flip conditionally; D's values never change, only its
   position in time).
4. Each level's leftover post-factor correction (D_V1) is commuted leftward
   through that level's CS gate and pre-factor gates (D_U1 branch), and the
   two branches' corrections combine multiplicatively (D_final = D_U1 * D_V1).
   Recursing this bottom-up collapses all discrete corrections into ONE final
   diagonal matrix D, decomposed (trivially, since it's diagonal +-1) into a
   layered Pi-gate structure per Eq. 24 of the paper.
5. An additional angle-normalization step (theta -> theta - pi, with a
   compensating global -1 folded into D) keeps all Ry angles in [-pi/2, pi/2],
   matching the convention used in the paper's own circuit diagrams.

This file is a from-scratch reference implementation, independent of any
existing matrix_io_fixed.py pipeline, intended for cross-checking gate counts
and angles against known circuits.
"""

import numpy as np
from scipy.linalg import cossin


# ---------------------------------------------------------------------------
# Leaf-level (2x2) decomposition
# ---------------------------------------------------------------------------

def leaf_info(B):
    """Decompose a 2x2 real orthogonal block into (theta, reflect)."""
    d = np.linalg.det(B)
    if d > 0:
        theta = np.arctan2(B[1, 0], B[0, 0])
        reflect = False
    else:
        R = B @ np.diag([1.0, -1.0])
        theta = np.arctan2(R[1, 0], R[0, 0])
        reflect = True
    return theta, reflect


def leaf_gate_and_D(B):
    """
    Returns (theta, D) with |theta| <= pi/2 such that:
        B = diag(D) @ Ry(theta)      (D applied LAST, gate applied FIRST)
    """
    theta, reflect = leaf_info(B)
    extra_sign = 1.0
    if theta > np.pi / 2:
        theta -= np.pi
        extra_sign = -1.0
    elif theta < -np.pi / 2:
        theta += np.pi
        extra_sign = -1.0

    if reflect:
        gate_theta = -theta
        D = np.array([1.0, -1.0]) * extra_sign
    else:
        gate_theta = theta
        D = np.array([1.0, 1.0]) * extra_sign
    return gate_theta, D


# ---------------------------------------------------------------------------
# Recursive CSD decomposition (real orthogonal matrices)
# ---------------------------------------------------------------------------

def decompose_real(U, qubits, controls):
    """
    Recursively decompose a real orthogonal 2^k x 2^k matrix U.

    qubits:   qubit indices still to be resolved at this node (split order,
              MSB-first, e.g. [0,1,2] for a 3-qubit problem)
    controls: dict of already-fixed qubit -> 0/1 from coarser recursion levels

    Returns (tree, D):
        tree: either a flat list of gate dicts {target, controls, theta}
              (base-case), or a nested 3-tuple (V, CS, U) where V and U are
              themselves trees of the same kind (recursively) and CS is a
              flat list -- meaning "V-part, then CS, then U-part" in time
              order. Call flatten_tree(tree) to get the final flat gate list.
              Keeping (V, CS, U) until the very end lets each recursion level
              merge its q=0/q=1 branches BY ROLE (all of branch0's V, then
              branch1's V, then branch0's CS, branch1's CS, branch0's U,
              branch1's U) instead of branch0-fully-then-branch1-fully --
              this matches the reference circuit's gate ordering.
        D:    numpy array of +-1, length = U.shape[0], such that
                  U = diag(D) @ (flatten_tree(tree) applied in listed order)
    """
    N = U.shape[0]
    if N == 2:
        theta, D = leaf_gate_and_D(U)
        gate = dict(target=qubits[0], controls=dict(controls), theta=theta)
        return [gate], D

    p = N // 2
    u, cs, vh = cossin(U, p=p, q=p, separate=True, swap_sign=True)
    u1, u2 = u
    v1h, v2h = vh
    theta = -cs.copy()  # swap_sign=True flips the CS block's off-diagonal sign convention
    q = qubits[0]
    rest = qubits[1:]
    cb0 = dict(controls); cb0[q] = 0
    cb1 = dict(controls); cb1[q] = 1

    tree_u1, Du1 = decompose_real(u1, rest, cb0)
    tree_u2, Du2 = decompose_real(u2, rest, cb1)
    tree_v1, Dv1 = decompose_real(v1h, rest, cb0)
    tree_v2, Dv2 = decompose_real(v2h, rest, cb1)

    D_U1 = np.concatenate([Du1, Du2])
    D_V1 = np.concatenate([Dv1, Dv2])

    def tag_tree(node, val):
        """Add q=val to every leaf gate's controls, preserving tree shape."""
        if isinstance(node, tuple):
            V, CS, U = node
            return (tag_tree(V, val), tag_tree(CS, val), tag_tree(U, val))
        out = []
        for g in node:
            gc = dict(g['controls']); gc[q] = val
            out.append(dict(target=g['target'], controls=gc, theta=g['theta']))
        return out

    def merge_by_role(t0, t1):
        """Combine two same-shape branch trees (q=0 and q=1) by concatenating
        WITHIN each role (V0+V1, CS0+CS1, U0+U1) rather than branch0-then-
        branch1 wholesale."""
        t0 = tag_tree(t0, 0)
        t1 = tag_tree(t1, 1)
        if isinstance(t0, tuple):
            V0, CS0, U0 = t0
            V1, CS1, U1 = t1
            return (merge_by_role(V0, V1), CS0 + CS1, merge_by_role(U0, U1))
        return t0 + t1

    merged_U_tree = merge_by_role(tree_u1, tree_u2)
    merged_V_tree = merge_by_role(tree_v1, tree_v2)

    # Flatten the multiplexed CS(theta) gate into p individual flat gates
    n_rest = len(rest)
    cs_gates = []
    for j in range(p):
        bits = [(j >> (n_rest - 1 - k)) & 1 for k in range(n_rest)]
        extra_ctrl = {rq: bits[k] for k, rq in enumerate(rest)}
        gc = dict(controls); gc.update(extra_ctrl)
        cs_gates.append(dict(target=q, controls=gc, theta=theta[j]))

    # U_local = diag(D_U1) @ merged_U @ CS(theta) @ diag(D_V1) @ merged_V
    # Commute diag(D_V1) leftward through CS(theta), then through merged_U.
    new_cs_gates = []
    for j, g in enumerate(cs_gates):
        d0, d1 = D_V1[j], D_V1[p + j]
        th = g['theta'] if d0 == d1 else -g['theta']
        new_cs_gates.append(dict(target=g['target'], controls=g['controls'], theta=th))

    def commute_diag_through_tree(D, node, n_qubits_local, qubit_list):
        """Flip each leaf gate's angle according to D (D itself never changes
        value), preserving tree shape."""
        if isinstance(node, tuple):
            V, CS, U = node
            return (commute_diag_through_tree(D, V, n_qubits_local, qubit_list),
                    commute_diag_through_tree(D, CS, n_qubits_local, qubit_list),
                    commute_diag_through_tree(D, U, n_qubits_local, qubit_list))
        out = []
        for g in node:
            t = g['target']
            full_bits = dict(g['controls']); full_bits[t] = 0
            idx0 = sum(full_bits[qubit_list[i]] << (n_qubits_local - 1 - i)
                       for i in range(n_qubits_local))
            full_bits[t] = 1
            idx1 = sum(full_bits[qubit_list[i]] << (n_qubits_local - 1 - i)
                       for i in range(n_qubits_local))
            same = (D[idx0] == D[idx1])
            new_theta = g['theta'] if same else -g['theta']
            out.append(dict(target=t, controls=dict(g['controls']), theta=new_theta))
        return out

    qubit_list_local = [q] + rest
    new_U_tree = commute_diag_through_tree(D_V1, merged_U_tree, len(qubit_list_local), qubit_list_local)

    D_final = D_U1 * D_V1
    # time order: V-part, then CS (flipped), then U-part (flipped)
    return (merged_V_tree, new_cs_gates, new_U_tree), D_final


def flatten_tree(node):
    """Expand a (V, CS, U) tree (or plain flat list) into the final flat gate list."""
    if isinstance(node, tuple):
        V, CS, U = node
        return flatten_tree(V) + CS + flatten_tree(U)
    return node


def to_paper_convention(gates, D, n_qubits):
    """
    Convert RAW gates (as returned by decompose_real/flatten_tree, which use
    Ry(theta) = [[cos, -sin], [sin, cos]] with the diagonal correction D
    applied LAST / at the end of the circuit) into the paper's convention
    (Ry(theta) = [[cos, sin], [-sin, cos]], Eq.12, with D/Pi effectively
    applied FIRST / at the start of the circuit).

    Uses the commutation identity:
        diag(d0, d1) @ Ry_std(theta) = Ry_paper(theta') @ diag(d0, d1)
        where theta' = -theta if d0 == d1, else theta' = theta
    (same identity already used internally by decompose_real to collapse
    discrete sign corrections into a single trailing D). Applying this to
    every gate, with D looked up at (this gate's own controls, target=0)
    vs (target=1), converts the WHOLE gate list at once -- this must be
    done BEFORE any Section-3.3-style gate merging, since merging groups by
    exact theta match and the two conventions are not interchangeable
    mid-reduction.

    gates: flat list of {target, controls, theta} (raw, un-reduced)
    D:     the +-1 diagonal array returned by decompose_real for this same U
    n_qubits: number of qubits (len(D) == 2**n_qubits)
    """
    def d_value(bits_dict):
        idx = sum(bits_dict.get(k, 0) << (n_qubits - 1 - k) for k in range(n_qubits))
        return D[idx]

    out = []
    for g in gates:
        t = g['target']
        b0 = dict(g['controls']); b0[t] = 0
        b1 = dict(g['controls']); b1[t] = 1
        d0, d1 = d_value(b0), d_value(b1)
        theta_out = -g['theta'] if d0 == d1 else g['theta']
        out.append(dict(target=t, controls=dict(g['controls']), theta=theta_out))
    return out




# ---------------------------------------------------------------------------
# Circuit -> matrix (for verification)
# ---------------------------------------------------------------------------

def decompose_D_to_pi_layers(D, n_qubits):
    """
    Eq. 24 of the paper: any diagonal +-1 correction D (size 2^n) can be
    written as n layers, layer for target qubit t controlled by qubits
    0..t-1 (coarser qubits). Peels off dependency on qubit (n-1) first,
    then (n-2), ..., down to qubit 0.

    Returns a list of (target, controls_dict) for every NON-TRIVIAL flip
    (i.e. every controlled-Pi gate actually needed).
    """
    N = len(D)
    f = np.array([0 if d > 0 else 1 for d in D])
    nontrivial = []
    for target in reversed(range(n_qubits)):
        ctrl_qubits = list(range(target))
        n_ctrl = len(ctrl_qubits)
        for cpat in range(2 ** n_ctrl):
            cbits = [(cpat >> (n_ctrl - 1 - i)) & 1 for i in range(n_ctrl)]

            def build_idx(tval):
                bits = [0] * n_qubits
                for i, cq in enumerate(ctrl_qubits):
                    bits[cq] = cbits[i]
                bits[target] = tval
                return sum(bits[k] << (n_qubits - 1 - k) for k in range(n_qubits))

            idx0, idx1 = build_idx(0), build_idx(1)
            if f[idx0] != f[idx1]:
                nontrivial.append((target, dict(zip(ctrl_qubits, cbits))))
    return nontrivial


def gate_matrix(N, n_qubits, qubit_order, gate):
    """Ry(theta) = [[cos, sin], [-sin, cos]]  (paper Eq.12 convention)."""
    M = np.eye(N)
    t = gate['target']
    pos_t = qubit_order.index(t)
    for idx0 in range(N):
        bits = [(idx0 >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        bmap = {qubit_order[k]: bits[k] for k in range(n_qubits)}
        if bmap[t] != 0:
            continue
        if not all(bmap[cq] == cv for cq, cv in gate['controls'].items()):
            continue
        idx1 = idx0 | (1 << (n_qubits - 1 - pos_t))
        c, s = np.cos(gate['theta']), np.sin(gate['theta'])
        M[idx0, idx0] = c; M[idx0, idx1] = s
        M[idx1, idx0] = -s; M[idx1, idx1] = c
    return M


def reconstruct(gates, D, n_qubits, qubit_order):
    N = 2 ** n_qubits
    M = np.eye(N)
    for g in gates:
        M = gate_matrix(N, n_qubits, qubit_order, g) @ M
    return np.diag(D) @ M


def pi_gate_matrix(N, n_qubits, qubit_order, target, ctrl):
    """Matrix for a controlled-Pi (Z) gate: flips sign when target=1 and all controls match."""
    M = np.eye(N)
    for idx in range(N):
        bits = [(idx >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        bmap = {qubit_order[k]: bits[k] for k in range(n_qubits)}
        if not all(bmap[cq] == cv for cq, cv in ctrl.items()):
            continue
        if bmap[target] == 1:
            M[idx, idx] = -1.0
    return M


def reconstruct_full_sequence(sig_gates, pi_layers, n_qubits, qubit_order):
    """Independent check: build the matrix by literally multiplying every gate
    in the printed sequence (Ry gates, THEN Pi gates), in that exact order."""
    N = 2 ** n_qubits
    M = np.eye(N)
    for g in sig_gates:
        M = gate_matrix(N, n_qubits, qubit_order, g) @ M
    for target, ctrl in pi_layers:
        M = pi_gate_matrix(N, n_qubits, qubit_order, target, ctrl) @ M
    return M


# ---------------------------------------------------------------------------
# Wire-notation printer (w1=q0=MSB ... wn=q(n-1)=LSB)
# ---------------------------------------------------------------------------

def print_circuit(gates, pi_layers, n_qubits, pi_first=False):
    """Returns the circuit as a plain gate list, one gate per line, no extra text.
    If pi_first, Pi gates are listed before the Ry gates (caller must already
    have flipped the appropriate Ry angles -- see main block)."""
    lines = []
    wire_of = {q: f"w{q+1}" for q in range(n_qubits)}

    def pi_lines(idx):
        out = []
        for target, ctrl in pi_layers:
            parts = []
            for q in range(n_qubits):
                if q == target:
                    parts.append(f"{wire_of[q]}=Pi")
                elif q in ctrl:
                    parts.append(f"{wire_of[q]}={ctrl[q]}")
                else:
                    parts.append(f"{wire_of[q]}=*")
            out.append(f"[{idx:2d}]  {'  '.join(parts)}")
            idx += 1
        return out, idx

    def ry_lines(idx):
        out = []
        for g in gates:
            parts = []
            for q in range(n_qubits):
                if q == g['target']:
                    parts.append(f"{wire_of[q]}=Ry({g['theta']/np.pi:+.4f}pi)")
                elif q in g['controls']:
                    parts.append(f"{wire_of[q]}={g['controls'][q]}")
                else:
                    parts.append(f"{wire_of[q]}=*")
            out.append(f"[{idx:2d}]  {'  '.join(parts)}")
            idx += 1
        return out, idx

    idx = 0
    if pi_first:
        pl, idx = pi_lines(idx)
        lines += pl
        rl, idx = ry_lines(idx)
        lines += rl
    else:
        rl, idx = ry_lines(idx)
        lines += rl
        pl, idx = pi_lines(idx)
        lines += pl

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# General matrix loading / padding to power-of-2 size (paper Eq. 10)
# ---------------------------------------------------------------------------

def load_matrix(path):
    """
    Loads a matrix from a text file.
    Supported formats:
      1) First line = N, followed by N*N values (row-major)      [RandReal.txt style]
      2) Plain whitespace/CSV-separated numeric rows, no header
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    first_tokens = lines[0].replace(',', ' ').split()
    if len(first_tokens) == 1:
        # header-style: N then N*N flat values
        n = int(first_tokens[0])
        vals = []
        for line in lines[1:]:
            vals.extend(float(x) for x in line.replace(',', ' ').split())
        if len(vals) != n * n:
            raise ValueError(f"Expected {n*n} values after header N={n}, got {len(vals)}")
        return np.array(vals).reshape(n, n)
    else:
        # plain rows
        rows = [[float(x) for x in line.replace(',', ' ').split()] for line in lines]
        return np.array(rows)


def prepare_matrix(U, atol=1e-8):
    """
    Validates U is (numerically) real orthogonal, and pads to the next
    power-of-2 size with an identity block if needed (paper Eq. 10):
        W = [[U, 0], [0, I]]
    Returns (W, n_qubits, original_size).
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {U.shape}")

    N0 = U.shape[0]
    err = np.max(np.abs(U @ U.T - np.eye(N0)))
    if err > atol:
        raise ValueError(
            f"Input matrix is not orthogonal (||U U^T - I||_max = {err:.3e} > {atol}). "
            f"CSD-based synthesis requires a real orthogonal matrix."
        )

    n_qubits = int(np.ceil(np.log2(max(N0, 2))))
    N = 2 ** n_qubits
    if N == N0:
        W = U.copy()
        # No padding available: do NOT alter the original matrix's determinant.
        # decompose_real() handles det=-1 blocks natively via leaf-level Ry@Z,
        # so a det=-1 top-level matrix is fine as-is.
    else:
        W = np.eye(N)
        W[:N0, :N0] = U
        print(f"Note: input size {N0} is not a power of 2; padded to {N} "
              f"({n_qubits} qubits) with identity block per Eq. 10 of the paper.")
        if np.linalg.det(W) < 0:
            # Safe to flip here: this column lies strictly within the padding
            # block (index >= N0), so it does not touch the original U at all.
            W[:, -1] *= -1
            print("Note: det(W) was -1 after padding; sign-flipped the last "
                  "padding column (outside the original matrix) to enforce det=+1.")

    return W, n_qubits, N0


# ---------------------------------------------------------------------------
# Demo / CLI: synthesize a circuit for any input matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '/mnt/user-data/uploads/RandReal.txt'
        print(f"No file given, defaulting to {path}\n")

    # optional 2nd argument: output txt path. Default: <input_name>_circuit.txt next to input.
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        base, _ = os.path.splitext(path)
        out_path = base + "_circuit.txt"

    U0_raw = load_matrix(path)
    U0, n_qubits, N0 = prepare_matrix(U0_raw)
    qubit_order = list(range(n_qubits))

    tree, D = decompose_real(U0, qubit_order, {})
    gates = flatten_tree(tree)

    ZERO_TOL = 1e-9
    sig_gates = [g for g in gates if abs(g['theta']) > ZERO_TOL]
    pi_layers = decompose_D_to_pi_layers(D, n_qubits)

    # Convert from internal convention (Pi last, Ry(theta)=[[c,-s],[s,c]]) to
    # paper's convention (Pi first, Ry(theta)=[[c,s],[-s,c]], Eq.12).
    # General rule: for each gate, compare D's value at (this gate's own
    # controls, target=0) vs (target=1) -- flip the angle iff they differ.
    # (A simple "same target qubit as a Pi gate" check is NOT enough when the
    # Pi gate itself has controls, since D then depends on more than just the
    # target qubit's bit.)
    def d_value(D, n_qubits, bits_dict):
        idx = sum(bits_dict.get(k, 0) << (n_qubits - 1 - k) for k in range(n_qubits))
        return D[idx]

    sig_gates_out = []
    for g in sig_gates:
        t = g['target']
        b0 = dict(g['controls']); b0[t] = 0
        b1 = dict(g['controls']); b1[t] = 1
        d0, d1 = d_value(D, n_qubits, b0), d_value(D, n_qubits, b1)
        theta_out = -g['theta'] if d0 == d1 else g['theta']
        sig_gates_out.append(dict(target=t, controls=g['controls'], theta=theta_out))

    circuit_text = print_circuit(sig_gates_out, pi_layers, n_qubits, pi_first=True)

    # Independent verification using the paper-convention gate_matrix/pi_gate_matrix
    M_check = np.eye(2 ** n_qubits)
    for target, ctrl in pi_layers:
        M_check = pi_gate_matrix(2 ** n_qubits, n_qubits, qubit_order, target, ctrl) @ M_check
    for g in sig_gates_out:
        M_check = gate_matrix(2 ** n_qubits, n_qubits, qubit_order, g) @ M_check
    err = np.max(np.abs(M_check - U0))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(circuit_text + "\n")

    print(f"\n[written to: {out_path}]  [reconstruction err: {err:.2e}]")

"""
Q-search (qubit permutation search) built on top of the verified CSD gate
synthesis in qcompiler_csd_synth.py.

Goal: find a permutation Q (acting on qubit labels, i.e. relabeling which
physical qubit is w1/w2/.../wn) minimizing the CSD gate cost of
    U' = Q^T U Q
so that the full circuit is  U = Q . U' . Q^T
with total cost = cost(Q) + cost(U') + cost(Q^T).

cost(Q): number of transpositions (SWAPs) in the permutation's decomposition
         into 2-cycles (a permutation that is k independent transpositions
         costs k). cost(Q^T) = cost(Q) since Q^T undoes the same swaps.
cost(U'): number of non-trivial Ry gates in the CSD circuit for U'
          (Pi/sign gates are NOT counted -- matches the paper's own
          "unsimplified gate count" convention, which counts continuous
          rotation gates as the cost).

Search strategy: greedy hill-climbing over single-transposition
neighbours, starting from the identity permutation, always moving to
the best-improving neighbour until no single swap improves the cost.
"""

import sys
import itertools
import numpy as np

sys.path.insert(0, '/home/claude/restart')
import qcompiler_csd_synth as synth


def permute_qubits(U, perm):
    """
    Apply a qubit permutation to an orthogonal matrix U (basis relabeling).
    perm[i] = which ORIGINAL qubit ends up at position i (0=MSB).
    Returns Q^T U Q where Q is the permutation matrix implementing this
    relabeling (a pure basis permutation of the 2^n computational states).
    """
    n = int(np.log2(U.shape[0]))
    N = U.shape[0]
    # build index mapping: new_index's bits are old_index's bits permuted
    idx_map = np.zeros(N, dtype=int)
    for new_idx in range(N):
        new_bits = [(new_idx >> (n - 1 - k)) & 1 for k in range(n)]
        old_bits = [0] * n
        for new_pos, old_pos in enumerate(perm):
            old_bits[old_pos] = new_bits[new_pos]
        old_idx = sum(old_bits[k] << (n - 1 - k) for k in range(n))
        idx_map[new_idx] = old_idx
    # U'[i,j] = U[idx_map[i], idx_map[j]]  (this is Q^T U Q for the
    # permutation matrix Q with Q[idx_map[i], i] = 1)
    return U[np.ix_(idx_map, idx_map)]


def permutation_cost(perm):
    """Number of transpositions in perm's cycle decomposition (min SWAPs)."""
    n = len(perm)
    visited = [False] * n
    cost = 0
    for i in range(n):
        if visited[i]:
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len > 1:
            cost += cycle_len - 1
    return cost


def csd_gate_cost(U, zero_tol=1e-9):
    """Total circuit gate cost for U: non-trivial Ry gates + Pi (sign-flip) gates."""
    n_qubits = int(np.log2(U.shape[0]))
    qubit_order = list(range(n_qubits))
    tree, D = synth.decompose_real(U, qubit_order, {})
    gates = synth.flatten_tree(tree)
    n_ry = sum(1 for g in gates if abs(g['theta']) > zero_tol)
    pi_layers = synth.decompose_D_to_pi_layers(D, n_qubits)
    return n_ry + len(pi_layers)


def csd_gates_paper_convention(U, zero_tol=1e-9):
    """
    Same recursive CSD as csd_gate_cost, but returns gates converted to the
    paper's Ry(theta) = [[cos, sin], [-sin, cos]] convention (Eq.12), with
    the Pi (sign) layers meant to be applied FIRST in the circuit.
    Returns (sig_gates, pi_layers, D) -- all BEFORE Section 3.3 merging.
    """
    n_qubits = int(np.log2(U.shape[0]))
    qubit_order = list(range(n_qubits))
    tree, D = synth.decompose_real(U, qubit_order, {})
    gates = synth.flatten_tree(tree)
    sig_gates = [g for g in gates if abs(g['theta']) > zero_tol]
    sig_gates = synth.to_paper_convention(sig_gates, D, n_qubits)
    pi_layers = synth.decompose_D_to_pi_layers(D, n_qubits)
    return sig_gates, pi_layers, D


def greedy_q_search(U, verbose=True, cost_fn=None):
    """
    Greedy hill-climbing search over qubit permutations, optimizing the
    TOTAL cost  Q-cost + U'-cost + Q^T-cost  (not just U'-cost alone --
    a further swap is only accepted if it reduces the total).
    cost_fn defaults to the Section-3.3-reduced cost.
    Returns (best_perm, best_U_cost, history).
    """
    if cost_fn is None:
        cost_fn = lambda M: csd_gate_cost_reduced(M)[0]

    n = int(np.log2(U.shape[0]))
    current_perm = list(range(n))
    current_u_cost = cost_fn(U)
    current_q_cost = permutation_cost(current_perm)
    current_total = 2 * current_q_cost + current_u_cost
    history = [(tuple(current_perm), current_u_cost, current_total)]

    if verbose:
        print(f"Start: perm={current_perm}  U'-cost={current_u_cost}  "
              f"Q-cost={current_q_cost}  total={current_total}")

    improved = True
    while improved:
        improved = False
        best_neighbor = None
        best_neighbor_u_cost = None
        best_neighbor_total = current_total
        for i, j in itertools.combinations(range(n), 2):
            trial_perm = current_perm.copy()
            trial_perm[i], trial_perm[j] = trial_perm[j], trial_perm[i]
            U_trial = permute_qubits(U, trial_perm)
            trial_u_cost = cost_fn(U_trial)
            trial_q_cost = permutation_cost(trial_perm)
            trial_total = 2 * trial_q_cost + trial_u_cost
            if trial_total < best_neighbor_total:
                best_neighbor_total = trial_total
                best_neighbor_u_cost = trial_u_cost
                best_neighbor = trial_perm
        if best_neighbor is not None:
            current_perm = best_neighbor
            current_u_cost = best_neighbor_u_cost
            current_total = best_neighbor_total
            current_q_cost = permutation_cost(current_perm)
            history.append((tuple(current_perm), current_u_cost, current_total))
            improved = True
            if verbose:
                print(f"  swap ->  perm={current_perm}  U'-cost={current_u_cost}  "
                      f"Q-cost={current_q_cost}  total={current_total}")

    return current_perm, current_u_cost, history


# ---------------------------------------------------------------------------
# Section 3.3 gate reduction: merge CUGs with same target + same rotation
# angle whose control patterns differ in exactly one bit (that bit becomes
# a wildcard / is dropped from controls). Iterative pairwise merge (like
# Quine-McCluskey adjacency merging) until no more merges are possible.
# ---------------------------------------------------------------------------

def _merge_control_cubes(cubes):
    """cubes: list of tuples (one entry per control-qubit slot, value in {0,1,None}).
    Repeatedly merge pairs differing in exactly one non-wildcard position."""
    cubes = list(set(cubes))
    changed = True
    while changed:
        changed = False
        n = len(cubes)
        used = [False] * n
        new_cubes = []
        for i in range(n):
            if used[i]:
                continue
            merged_i = False
            for j in range(i + 1, n):
                if used[j]:
                    continue
                diff = [k for k in range(len(cubes[i]))
                        if cubes[i][k] != cubes[j][k]]
                if len(diff) == 1:
                    k = diff[0]
                    if cubes[i][k] is not None and cubes[j][k] is not None:
                        merged = list(cubes[i])
                        merged[k] = None
                        new_cubes.append(tuple(merged))
                        used[i] = used[j] = True
                        merged_i = True
                        changed = True
                        break
            if not merged_i and not used[i]:
                new_cubes.append(cubes[i])
                used[i] = True
        cubes = list(set(new_cubes))
    return cubes


def _split_into_runs(gates, key_fn):
    """
    Partition a TIME-ORDERED gate/entry list into maximal contiguous runs
    that share the same key_fn(entry) value. Consecutive entries with the
    same key stay in one run; as soon as the key changes, a new run starts
    -- even if that same key reappears later (that later reappearance is a
    SEPARATE run, never merged with the earlier one).

    This exists because the recursive CSD circuit re-visits the same target
    qubit at multiple, non-adjacent points in the gate sequence (once per
    role/branch in the (V,CS,U) tree). Two gates that happen to share the
    same target (and, for reduce_gates, the same theta) are only the SAME
    logical multiplexed rotation -- and thus only safe to merge -- if they
    were produced within the same contiguous occurrence. Grouping by key
    alone (ignoring position) can silently fuse together gates from
    unrelated points in the circuit whenever their angles coincide, which
    changes the unitary the circuit implements (this is exactly what caused
    the S8 post-merge reconstruction failure).
    """
    runs = []
    for g in gates:
        k = key_fn(g)
        if runs and runs[-1][0] == k:
            runs[-1][1].append(g)
        else:
            runs.append([k, [g]])
    return [run for _, run in runs]


def reduce_gates(gates, control_qubits_by_target):
    """
    gates: flat list of {target, controls, theta} (already filtered to
           non-trivial angles), IN CIRCUIT TIME ORDER.
    control_qubits_by_target: dict target -> ordered list of qubits that can
           appear as controls for that target (needed to build fixed-length
           cube tuples across gates that might have different control sets).
    Returns the reduced gate list (each entry may now have FEWER controls
    than before, representing a merged CUG), IN THE SAME RELATIVE TIME ORDER
    as the input (each merged group is emitted at the position of its run).

    IMPORTANT: merging only happens WITHIN a maximal contiguous run of
    same-target gates (see _split_into_runs) -- never across two separate,
    non-adjacent occurrences of the same target elsewhere in the circuit.
    Within a run, gates are further sub-grouped by theta (as before) and
    adjacency-merged over their control patterns.
    """
    from collections import defaultdict

    target_runs = _split_into_runs(gates, key_fn=lambda g: g['target'])

    reduced = []
    for run in target_runs:
        target = run[0]['target']
        qubits = control_qubits_by_target[target]

        theta_groups = defaultdict(list)
        for g in run:
            theta_groups[round(g['theta'], 9)].append(g['controls'])

        for theta, control_dicts in theta_groups.items():
            cubes = [tuple(cd.get(q, None) for q in qubits) for cd in control_dicts]
            merged_cubes = _merge_control_cubes(cubes)
            for cube in merged_cubes:
                controls = {q: v for q, v in zip(qubits, cube) if v is not None}
                reduced.append(dict(target=target, controls=controls, theta=theta))
    return reduced


def reduce_pi_layers(pi_layers, control_qubits_by_target):
    """
    Same Section-3.3 adjacency merging as reduce_gates(), applied to the
    controlled-Pi (sign-flip) layers instead of the Ry gates. Pi gates have
    no rotation angle, so entries are grouped by target only (not
    target+theta): a controlled-Pi on target t is the same "gate type" no
    matter which controls it carries, so all control patterns for a given
    target are candidates for merging -- including collapsing all the way
    down to a single unconditional (uncontrolled) Pi gate, as happens when
    the sign pattern on that target only depends on the target qubit itself.

    Merging is restricted to one maximal contiguous run of same-target
    entries at a time (see _split_into_runs / reduce_gates), for the same
    time-order-safety reason as reduce_gates -- even though
    decompose_D_to_pi_layers currently only ever emits one contiguous run
    per target, this keeps the two reduction functions consistent and safe
    if pi_layers ever comes from a different source.
    """
    target_runs = _split_into_runs(pi_layers, key_fn=lambda entry: entry[0])

    reduced = []
    for run in target_runs:
        target = run[0][0]
        qubits = control_qubits_by_target[target]
        ctrl_dicts = [ctrl for _, ctrl in run]
        cubes = [tuple(cd.get(q, None) for q in qubits) for cd in ctrl_dicts]
        merged_cubes = _merge_control_cubes(cubes)
        for cube in merged_cubes:
            controls = {q: v for q, v in zip(qubits, cube) if v is not None}
            reduced.append((target, controls))
    return reduced


def csd_gate_cost_reduced(U, zero_tol=1e-9):
    """Gate cost AFTER applying the Section 3.3 CUG-merging reduction
    (applied to BOTH the Ry gates and the controlled-Pi sign-flip layers).
    Ry gates are converted to the paper's convention
    Ry(theta) = [[cos, sin], [-sin, cos]]  (Eq.12), with Pi layers meant to
    be applied FIRST -- BEFORE merging, since merging groups by exact theta
    match and the two conventions are not interchangeable mid-reduction."""
    n_qubits = int(np.log2(U.shape[0]))
    qubit_order = list(range(n_qubits))
    tree, D = synth.decompose_real(U, qubit_order, {})
    gates = synth.flatten_tree(tree)
    sig_gates = [g for g in gates if abs(g['theta']) > zero_tol]
    sig_gates = synth.to_paper_convention(sig_gates, D, n_qubits)

    control_qubits_by_target = {
        t: [q for q in qubit_order if q != t] for t in qubit_order
    }
    reduced = reduce_gates(sig_gates, control_qubits_by_target)

    pi_layers_raw = synth.decompose_D_to_pi_layers(D, n_qubits)
    pi_layers = reduce_pi_layers(pi_layers_raw, control_qubits_by_target)

    return len(reduced) + len(pi_layers), reduced, pi_layers


def permutation_to_swaps(perm):
    """Decompose a permutation into a sequence of transpositions (SWAP gates).
    perm[i] = which ORIGINAL qubit ends up at position i."""
    perm = perm.copy()
    n = len(perm)
    swaps = []
    pos_of = {v: i for i, v in enumerate(perm)}
    for i in range(n):
        while perm[i] != i:
            j = pos_of[i]
            swaps.append((i, j))
            perm[i], perm[j] = perm[j], perm[i]
            pos_of[perm[i]] = i
            pos_of[perm[j]] = j
    return swaps


def print_full_circuit(n_qubits, best_perm, reduced_gates, pi_layers, out_path=None):
    """Print/write the full 3-part circuit: Q (SWAPs) + U' (Pi+Ry) + Q^T (SWAPs)."""
    wire_of = {q: f"w{q+1}" for q in range(n_qubits)}
    lines = []
    idx = 0

    swaps = permutation_to_swaps(best_perm)

    lines.append("--- Q (qubit permutation, as SWAPs) ---")
    for a, b in swaps:
        lines.append(f"[{idx:2d}]  SWAP({wire_of[a]}, {wire_of[b]})")
        idx += 1

    lines.append("--- U' (reduced CSD circuit) ---")
    for target, ctrl in pi_layers:
        parts = []
        for q in range(n_qubits):
            if q == target:
                parts.append(f"{wire_of[q]}=Pi")
            elif q in ctrl:
                parts.append(f"{wire_of[q]}={ctrl[q]}")
            else:
                parts.append(f"{wire_of[q]}=*")
        lines.append(f"[{idx:2d}]  {'  '.join(parts)}")
        idx += 1

    for g in reduced_gates:
        parts = []
        for q in range(n_qubits):
            if q == g['target']:
                parts.append(f"{wire_of[q]}=Ry({g['theta']/np.pi:+.4f}pi)")
            elif q in g['controls']:
                parts.append(f"{wire_of[q]}={g['controls'][q]}")
            else:
                parts.append(f"{wire_of[q]}=*")
        lines.append(f"[{idx:2d}]  {'  '.join(parts)}")
        idx += 1

    lines.append("--- Q^T (undo permutation, same SWAPs) ---")
    for a, b in swaps:
        lines.append(f"[{idx:2d}]  SWAP({wire_of[a]}, {wire_of[b]})")
        idx += 1

    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[written to: {out_path}]")
    return text


if __name__ == "__main__":
    import os
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/RandReal.txt'
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0] + "_qsearch_circuit.txt"

    U0_raw = synth.load_matrix(path)
    U0, n_qubits, N0 = synth.prepare_matrix(U0_raw)

    baseline_cost, _, _ = csd_gate_cost_reduced(U0)
    print(f"Baseline (identity Q) U-cost (after Sec.3.3 reduction): {baseline_cost}")
    print()

    best_perm, best_cost, history = greedy_q_search(U0)

    q_cost = permutation_cost(best_perm)
    total = q_cost + best_cost + q_cost
    print()
    print(f"Best permutation found: {best_perm}")
    print(f"Q cost (SWAPs): {q_cost}")
    print(f"U' cost (reduced gate count): {best_cost}")
    print(f"Q^T cost (SWAPs): {q_cost}")
    print(f"Total cost: {q_cost} + {best_cost} + {q_cost} = {total}")
    print()

    U_best = permute_qubits(U0, best_perm)
    _, reduced_gates, pi_layers = csd_gate_cost_reduced(U_best)
    print_full_circuit(n_qubits, best_perm, reduced_gates, pi_layers, out_path)


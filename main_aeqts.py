import os
# 依重現指南要求，固定 BLAS 環境變數以確保數值一致性
os.environ["OPENBLAS_CORETYPE"] = "Haswell"

import numpy as np
from scipy.linalg import qr
import qsearch


# ==========================================
# 0. 工具函式：Permutation 與 CSD 成本計算
# ==========================================

def _extract_gate_cost(res):
    """提取 CSD Gate 成本整數值"""
    if isinstance(res, (tuple, list, np.ndarray)):
        return int(res[0])
    return int(res)

def qubit_perm_to_basis_perm_matrix(q):
    """
    將長度 n 的 qubit permutation q 展開為 2^n x 2^n 的 basis permutation 矩陣 Q
    """
    n = len(q)
    m = 1 << n
    Q = np.zeros((m, m), dtype=float)
    for i in range(m):
        target = 0
        for bit in range(n):
            if (i >> bit) & 1:
                target |= (1 << q[bit])
        Q[target, i] = 1.0
    return Q

def perm_list_to_matrix(p):
    """將 basis permutation list p 轉成 m x m 排列矩陣 P"""
    m = len(p)
    P = np.zeros((m, m), dtype=float)
    for i, target in enumerate(p):
        P[i, target] = 1.0
    return P

def compute_swap_cost(q):
    """計算 qubit permutation q 所需的 swap gate 數量 (至多 n-1)"""
    if hasattr(qsearch, 'permutation_to_swaps'):
        swaps = qsearch.permutation_to_swaps(q)
        return len(swaps) if isinstance(swaps, (list, tuple)) else int(swaps)

    n = len(q)
    visited = [False] * n
    cycles = 0
    for i in range(n):
        if not visited[i]:
            cycles += 1
            curr = i
            while not visited[curr]:
                visited[curr] = True
                curr = q[curr]
    return n - cycles

def evaluate_cnum(U, p, q):
    """
    計算總成本: c_num(U, P, Q) = CSD(P Q U Q^T P^T) + CSD(P) + CSD(P^T) + 2*s_num(Q)
    """
    m = U.shape[0]
    is_identity_p = (list(p) == list(range(m)))

    Q_mat = qubit_perm_to_basis_perm_matrix(q)
    P_mat = perm_list_to_matrix(p)

    # 依 OptQC 核心公式: U' = P @ Q @ U @ Q^T @ P^T
    U_prime = P_mat @ Q_mat @ U @ Q_mat.T @ P_mat.T

    # 計算各項 Gate 成本
    cost_U_prime = _extract_gate_cost(qsearch.csd_gate_cost_reduced(U_prime))

    if is_identity_p:
        cost_P = 0
        cost_Pt = 0
    else:
        cost_P = _extract_gate_cost(qsearch.csd_gate_cost_reduced(P_mat))
        cost_Pt = _extract_gate_cost(qsearch.csd_gate_cost_reduced(P_mat.T))

    s_q = compute_swap_cost(q)
    total = cost_U_prime + cost_P + cost_Pt + 2 * s_q
    return total, (cost_U_prime, cost_P, cost_Pt, s_q)


# ==========================================
# 1. Stage Q: 搜尋 Qubit Permutation (隨機重抽 + Greedy)
# ==========================================

def run_stage_q(U, j_max, rng):
    """
    Stage Q: 固定 P = I，每次隨機抽樣完整的 q'，成本降低才接受
    """
    n = int(np.round(np.log2(U.shape[0])))
    current_q = list(range(n))
    identity_p = list(range(U.shape[0]))

    best_cost, _ = evaluate_cnum(U, identity_p, current_q)
    best_q = list(current_q)

    for _ in range(j_max):
        candidate_q = list(rng.permutation(n))
        cost, _ = evaluate_cnum(U, identity_p, candidate_q)
        if cost < best_cost:
            best_cost = cost
            best_q = candidate_q

    return best_q, best_cost


# ==========================================
# 2. Stage P: AEQTS Currentgen 搜尋 Basis Permutation
# ==========================================

class Candidate:
    def __init__(self, measured_states, full_ranking, p, total, breakdown):
        self.measured_states = measured_states
        self.full_ranking = full_ranking
        self.p = p
        self.total = total
        self.breakdown = breakdown

def measure_basis_states(Q_state, rng):
    """對 Q_state 的每一列做 categorical 抽樣"""
    basis_size, num_states = Q_state.shape
    measured = np.zeros(basis_size, dtype=int)
    for i in range(basis_size):
        measured[i] = rng.choice(num_states, p=Q_state[i])
    return measured

def rank_basis_states(measured_states, rng):
    """
    將 measured_states 排序成合法的 permutation (full_ranking)。
    同值 (tie) 時以隨機次序打破，避免 Q_state 均勻初始化階段
    因固定 tie-break 而讓 population 多樣性被壓縮、提早卡住。
    """
    basis_size = len(measured_states)
    tie_breaker = rng.permutation(basis_size)
    full_ranking = np.lexsort((tie_breaker, measured_states))
    return full_ranking

def update_q_state_currentgen(Q_state, sorted_population, theta_max, atol=1e-9):
    """
    Currentgen 當代配對更新機率轉移矩陣:
    - 第 x 好配第 x 差，步長 delta_x = theta_max / x
    - 越界整組跳過，採 sequential in-place 更新
    """
    pop_size = len(sorted_population)
    pair_count = pop_size // 2
    basis_size = Q_state.shape[0]

    for x in range(1, pair_count + 1):
        good = sorted_population[x - 1]
        bad = sorted_population[-x]
        delta_x = theta_max / x

        for i in range(basis_size):
            good_state = int(good.measured_states[i])
            bad_state = int(bad.measured_states[i])

            if good_state == bad_state:
                continue

            if (Q_state[i, good_state] + delta_x <= 1.0 + atol and
                Q_state[i, bad_state] - delta_x >= 0.0 - atol):
                Q_state[i, good_state] += delta_x
                Q_state[i, bad_state] -= delta_x

def run_stage_p_aeqts(U, q_fixed, pop_size, max_iter, theta_max, rng):
    """Stage P: AEQTS Currentgen 搜尋最佳 basis permutation p"""
    m = U.shape[0]
    num_states = m
    Q_state = np.full((m, num_states), 1.0 / num_states, dtype=float)

    identity_p = list(range(m))
    init_cost, init_breakdown = evaluate_cnum(U, identity_p, q_fixed)

    best_p = list(identity_p)
    best_cost = init_cost
    best_breakdown = init_breakdown
    evaluated_count = 1

    cost_cache = {tuple(identity_p): (init_cost, init_breakdown)}

    for iteration in range(1, max_iter + 1):
        population = []
        for _ in range(pop_size):
            measured = measure_basis_states(Q_state, rng)
            full_ranking = rank_basis_states(measured, rng)
            p = list(full_ranking)
            p_tuple = tuple(p)

            if p_tuple in cost_cache:
                total, breakdown = cost_cache[p_tuple]
            else:
                total, breakdown = evaluate_cnum(U, p, q_fixed)
                cost_cache[p_tuple] = (total, breakdown)
                evaluated_count += 1

            population.append(Candidate(measured, full_ranking, p, total, breakdown))

        sorted_population = sorted(population, key=lambda c: c.total)
        current_best = sorted_population[0]

        if current_best.total < best_cost:
            best_cost = current_best.total
            best_p = list(current_best.p)
            best_breakdown = current_best.breakdown

        update_q_state_currentgen(Q_state, sorted_population, theta_max)

    return best_p, best_cost, best_breakdown, evaluated_count


# ==========================================
# 3. 測試進入點 (OptQC 案例 1: RandReal)
# ==========================================

if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    # OptQC Section 4.1 RandReal 矩陣（已修正第一列的轉錄錯誤：
    # 原本 0.0438 被誤放在 col2，導致 row0·row1 內積 ≈ -0.019，
    # 矩陣不是正交矩陣，QR 修復後結構會跟論文不同，
    # 使得 reference 的 p/q 解完全不適用）
    U_raw = np.array([
        [0.0438,  0.0,     0.0,     0.0,     0.9990,  0.0,     0.0,     0.0],
        [0.1297,  0.8689, -0.2956,  0.0,    -0.0057,  0.1538, -0.3423,  0.0],
        [-0.2923, 0.0,     0.6661,  0.0,     0.0128,  0.0,    -0.6861,  0.0],
        [-0.0061,-0.0412,  0.0140,  0.7058,  0.0003,  0.3008,  0.0162, -0.6397],
        [0.9147,  0.0,     0.4021,  0.0,    -0.0401,  0.0,     0.0,     0.0],
        [0.0185,  0.1242, -0.0422,  0.3961, -0.0008, -0.9073, -0.0489,  0.0],
        [0.2424, -0.4762, -0.5524,  0.0,    -0.0106,  0.0,    -0.6397,  0.0],
        [0.0051,  0.0343, -0.0117, -0.5874, -0.0002, -0.2503, -0.0135, -0.7686]
    ], dtype=float)

    # 精確正交化修復浮點截斷誤差
    q_mat, r_mat = qr(U_raw)
    d = np.diagonal(r_mat)
    ph = d / np.abs(d)
    U_randreal = q_mat * ph
    U_randreal[np.abs(U_randreal) < 1e-5] = 0.0

    print("--- 開始執行 AEQTS + OptQC 量子編譯優化 (案例 1: RandReal) ---")

    print("[1/2] 執行 Stage Q (搜尋 Qubit Permutation)...")
    best_q, cost_q = run_stage_q(U_randreal, j_max=200, rng=rng)
    print(f"      Stage Q 最佳 q: {best_q}, 成本降至: {cost_q}")

    print("[2/2] 執行 Stage P (AEQTS Currentgen 搜尋 Basis Permutation)...")
    best_p, final_cost, breakdown, evals = run_stage_p_aeqts(
        U_randreal, best_q, pop_size=40, max_iter=1000, theta_max=0.002, rng=rng
    )

    print("\n--- 優化結果 ---")
    print(f"最佳 Q (qubit 排列): {best_q}")
    print(f"最佳 P (basis 排列): {best_p}")
    print(f"最終總 Gate 數量: {final_cost}")
    print(f"成本分項 [CSD(U'), CSD(P), CSD(P^T), 2*s(Q)]: {breakdown}")
    print(f"總評估次數: {evals}")

    ref_q = [2, 1, 0]
    ref_p = [0, 1, 6, 7, 4, 5, 2, 3]
    ref_total, ref_breakdown = evaluate_cnum(U_randreal, ref_p, ref_q)
    print(f"\n[驗證] 指南 Reference 解成本: {ref_total}, 分項: {ref_breakdown}")
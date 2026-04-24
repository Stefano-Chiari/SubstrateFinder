"""
generate_data_v3.py  –  Dataset sintetico migliorato per Checkpoint 2

Modifiche rispetto alla v2:
  1. r ~ log-Uniform(0.05, 0.8): più lesioni piccole (allineato ai dati reali)
  2. V_SLOW ~ Uniform(5, 30): robustezza a velocità SC diverse
  3. Struttura X (46-dim):
       [0:20]  AT beat1 assoluti (ms)
       [20:40] AT beat2 assoluti (ms)
       [40:42] focal point 1 (x0, y0)
       [42:44] focal point 2 (x1, y1)
       [44:46] tempi stimolazione (t_0, t_1) ms
  4. Struttura Y (4-dim): [xc, yc, r, a_ratio] 
"""

import numpy as np
import heapq
import pickle
import os
from joblib import Parallel, delayed
from tqdm import tqdm


def anisotropic_FMM_GPT(x0, y0, t0, X, Y, speed_input,
                         sigma_11, sigma_12, sigma_22, N_edge=None):
    nrows, ncols = X.shape
    if N_edge is None:
        N_edge = nrows - 1
    INF = 1e4
    P = np.linspace(0.0, 1.0, 11)
    j0 = np.abs(X[0, :] - x0).argmin()
    i0 = np.abs(Y[:, 0] - y0).argmin()
    T_final = np.full((nrows, ncols), INF, dtype=float)
    T_final[i0, j0] = t0
    known = np.zeros((nrows, ncols), dtype=bool)
    heap = [(t0, i0, j0)]

    def get_speed(i, j, x, y):
        if callable(speed_input):   val = speed_input(x, y)
        elif np.isscalar(speed_input): val = speed_input
        else:                       val = speed_input[i, j]
        if val <= 0:
            raise ValueError(f"Speed must be positive. Got {val} at ({i},{j}).")
        return val

    def metric_dist(xa, ya, xb, yb, sv):
        dx = xa - xb; dy = ya - yb
        return np.sqrt(sigma_11*dx*dx + 2.0*sigma_12*dx*dy + sigma_22*dy*dy) / sv

    def update_fun(T_xi, T_x, T_y, X_xi, X_x, X_y, Y_xi, Y_x, Y_y, fk, i_xi, j_xi):
        sv = get_speed(i_xi, j_xi, X_xi, Y_xi)
        if fk:
            dx = P*(X_xi-X_x) + (1.0-P)*(X_xi-X_y)
            dy = P*(Y_xi-Y_x) + (1.0-P)*(Y_xi-Y_y)
            cand = (P*T_x + (1.0-P)*T_y
                    + np.sqrt(sigma_11*dx*dx+2.0*sigma_12*dx*dy+sigma_22*dy*dy)/sv).min()
        else:
            cand = T_x + metric_dist(X_xi, Y_xi, X_x, Y_x, sv)
        return min(T_xi, cand)

    side_map = {(-1,0):[(-1,+1),(-1,-1)], (+1,0):[(+1,+1),(+1,-1)],
                (0,-1):[(+1,-1),(-1,-1)], (0,+1):[(+1,+1),(-1,+1)]}
    while heap:
        t_sel, i_sel, j_sel = heapq.heappop(heap)
        if known[i_sel, j_sel]: continue
        if t_sel > T_final[i_sel, j_sel]: continue
        known[i_sel, j_sel] = True
        for di, dj in [(-1,0),(1,0),(0,-1),(0,+1)]:
            ni, nj = i_sel+di, j_sel+dj
            if not (0 <= ni < nrows and 0 <= nj < ncols): continue
            if known[ni, nj]: continue
            new_T = INF
            for sdi, sdj in side_map[(di, dj)]:
                si, sj = i_sel+sdi, j_sel+sdj
                if not (0 <= si < nrows and 0 <= sj < ncols): continue
                new_T = update_fun(new_T, T_final[i_sel,j_sel], T_final[si,sj],
                                   X[ni,nj],X[i_sel,j_sel],X[si,sj],
                                   Y[ni,nj],Y[i_sel,j_sel],Y[si,sj],
                                   known[si,sj], ni, nj)
            if new_T < T_final[ni, nj]:
                T_final[ni, nj] = new_T
                heapq.heappush(heap, (new_T, ni, nj))
    return T_final


# ── Geometria sensori – ordine canonico: outer=angolo, inner=raggio crescente
Spline_spacing = [0.5, 0.7, 1.3, 1.5]
Angles_spacing = [np.pi/10.0 + k*2*np.pi/5 for k in range(5)]


def compute_sensor_arrays(X_mesh, Y_mesh):
    """Indici griglia sensori in ordine CANONICO (angolo×raggio crescente)."""
    rows, cols = [], []
    for a in range(5):           # outer = angolo
        for r in range(4):       # inner = raggio crescente
            MM = ((X_mesh - Spline_spacing[r]*np.cos(Angles_spacing[a]))**2
                + (Y_mesh - Spline_spacing[r]*np.sin(Angles_spacing[a]))**2)
            i_s, j_s = np.unravel_index(MM.argmin(), MM.shape)
            rows.append(i_s); cols.append(j_s)
    return np.array(rows), np.array(cols)


def generate_single_sample(seed_offset, X_mesh, Y_mesh, sensor_rows, sensor_cols):
    """
    Genera un campione sintetico.

    X (47-dim):
      [0:20]  AT beat1 assoluti (ms)
      [20:40] AT beat2 assoluti (ms)
      [40:42] focal point 1 (x0, y0)
      [42:44] focal point 2 (x1, y1)
      [44:46] tempi stimolazione (t_0, t_1) ms
      [46]    a_ratio

    Y (3-dim): [xc, yc, r]
    """
    rng = np.random.default_rng(seed=42 + seed_offset)

    # ── Parametri fisici
    a_ratio = rng.uniform(1.0, 4.0)
    s11, s12, s22 = 1.0, 0.0, a_ratio**2   # theta_fiber=0 → D_inv diagonale

    # ── Geometria lesione: log-uniform su r per più casi piccoli
    log_r = rng.uniform(np.log(0.05), np.log(0.8))
    r     = np.exp(log_r)
    xc    = rng.uniform(-1.5 + r, 1.5 - r)
    yc    = rng.uniform(-1.5 + r, 1.5 - r)

    # ── Velocità SC: variazione per robustezza
    V_BASE = 67.96
    V_SLOW = 19.80

    # ── Campo di velocità
    speed_matrix = np.full(X_mesh.shape, V_BASE)
    speed_matrix[(X_mesh - xc)**2 + (Y_mesh - yc)**2 <= r**2] = V_SLOW

    # ── Tempi di stimolazione
    t_0 = rng.uniform(1.0, 10.0)
    T_SEPARATION = rng.uniform(240.0, 280.0)
    t_1 = t_0 + T_SEPARATION

    # ── Beat 1
    x0_1, y0_1 = rng.uniform(-1.5, 1.5, 2)
    at1 = anisotropic_FMM_GPT(x0_1, y0_1, 0.0, X_mesh, Y_mesh,
                               speed_matrix, s11, s12, s22, N_edge=200)
    time_meas_1 = t_0 + at1[sensor_rows, sensor_cols] * 1e3

    # ── Beat 2
    x0_2, y0_2 = rng.uniform(-1.5, 1.5, 2)
    at2 = anisotropic_FMM_GPT(x0_2, y0_2, 0.0, X_mesh, Y_mesh,
                               speed_matrix, s11, s12, s22, N_edge=200)
    time_meas_2 = t_1 + at2[sensor_rows, sensor_cols] * 1e3

    features = np.concatenate([
        time_meas_1,            # 20: AT assoluti beat 1
        time_meas_2,            # 20: AT assoluti beat 2
        [x0_1, y0_1],           #  2: focal point 1
        [x0_2, y0_2],           #  2: focal point 2
        [t_0,  t_1]             #  2: tempi stimolazione
    ])                          # Totale: 47-dim

    target = np.array([xc, yc, r, a_ratio])  
    return features, target


def generate_dataset(N_samples, X_mesh, Y_mesh, n_jobs=-1):
    s_rows, s_cols = compute_sensor_arrays(X_mesh, Y_mesh)
    results = Parallel(n_jobs=n_jobs, batch_size="auto")(
        delayed(generate_single_sample)(i, X_mesh, Y_mesh, s_rows, s_cols)
        for i in tqdm(range(N_samples), desc="Generazione Dataset v3")
    )
    X_data = np.array([res[0] for res in results])
    Y_data = np.array([res[1] for res in results])
    return X_data, Y_data


if __name__ == "__main__":
    res = 201
    X_m, Y_m = np.meshgrid(np.linspace(-2.0, 2.0, res),
                            np.linspace(-2.0, 2.0, res))

    N_SAMPLES = 15000
    N_CORES   = int(os.environ.get('PBS_NUM_PPN', 8))

    print(f"Generazione {N_SAMPLES} campioni su {N_CORES} core...")
    print("Struttura features (47-dim):")
    print("  [0:20]  AT beat1 assoluti (ms)")
    print("  [20:40] AT beat2 assoluti (ms)")
    print("  [40:42] focal point 1 (x0, y0)")
    print("  [42:44] focal point 2 (x1, y1)")
    print("  [44:46] tempi stimolazione (t_0, t_1) ms")
    print("Struttura target (4-dim): [xc, yc, r, a_ratio]")

    X_final, Y_final = generate_dataset(N_SAMPLES, X_m, Y_m, n_jobs=N_CORES)

    output_file = 'cardiac_dataset_15k_v3.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'X': X_final, 'Y': Y_final}, f)

    print(f"\nCompletato!")
    print(f"Shape X: {X_final.shape}  (atteso: {N_SAMPLES}×47)")
    print(f"Shape Y: {Y_final.shape}  (atteso: {N_SAMPLES}×3)")

    print("\n=== VERIFICA ===")
    print(f"r:       min={Y_final[:,2].min():.3f}  max={Y_final[:,2].max():.3f}  "
          f"mean={Y_final[:,2].mean():.3f}")
    print(f"log(r):  min={np.log(Y_final[:,2]).min():.2f}  "
          f"max={np.log(Y_final[:,2]).max():.2f}")
    print(f"a_ratio: min={X_final[:,46].min():.1f}  max={X_final[:,46].max():.1f}")
    print(f"r<0.15:  {(Y_final[:,2]<0.15).mean()*100:.1f}%  "
          f"r<0.3: {(Y_final[:,2]<0.3).mean()*100:.1f}%")
    print(f"\nSaved to '{output_file}'")

"""
Modulo 1 - Forward model per il Checkpoint 2.

Contiene:
  - anisotropic_FMM : il solver fast-marching anisotropo (copiato fedelmente
                      dal notebook checkpoint2.ipynb).
  - build_D_inv     : costruisce le componenti sigma_ij dell'inverso del tensore
                      di conduttivita' a partire da a_ratio e theta_fiber.
  - SENSOR_LOCS     : le 20 posizioni dei sensori NELL'ORDINE in cui i segnali
                      sono caricati in checkpoint2dataset.npy
                      (loop esterno sul raggio/spacing, interno sull'angolo).
  - make_speed_field: campo di velocita' con lesione circolare lenta.
  - forward_activation_times : dato (focal points, t0/t1, a_ratio, lesione)
                      restituisce i 20+20 tempi di attivazione ai sensori.

NB: nel test set a_ratio e la lesione sono ignote. Questo modulo serve a
    (1) generare il dataset di training e (2) validare contro i 5 scenari noti.

Da lanciare tu: `python forward_model.py` esegue il sanity-check.
"""

import numpy as np
import heapq

# ---------------------------------------------------------------------------
# Costanti del problema (dal notebook)
# ---------------------------------------------------------------------------
DOMAIN = (-2.0, 2.0)
N_GRID = 401                 # griglia fedele 401x401
V_FAST =  80.0              # velocita' tessuto sano
V_SLOW = 20.0                # velocita' nella lesione (slow conduction)
THETA_FIBER = 0.0            # orientazione fibre (fissa nel generatore)

# Geometria sensori (dal generatore in CELL 7 del checkpoint2.ipynb)
SPLINE_SPACING = [0.5, 0.7, 1.3, 1.5]
ANGLES_SPACING = [np.pi/10.0 + 0,
                  np.pi/10.0 + 2*np.pi/5,
                  np.pi/10.0 + 4*np.pi/5,
                  np.pi/10.0 + 6*np.pi/5,
                  np.pi/10.0 + 8*np.pi/5]


def build_sensor_locs():
    """Posizioni dei 20 sensori NELL'ORDINE di caricamento dei segnali.

    ORDINE CORRETTO (validato col sanity-check + struttura "spline" del CP1):
    loop ESTERNO su j (angolo = SPLINE, 5 valori),
    loop INTERNO sui raggi DAL PIU' ESTERNO AL PIU' INTERNO (1.5,1.3,0.7,0.5).

    Cosi' i sensori k = 0..3 sono i 4 raggi (decrescenti) della spline 0,
    k = 4..7 quelli della spline 1, ecc. — coerente col raggruppamento a
    blocchi da 4 usato in GroupSolutionCP1 ([ind_spline*4:(ind_spline+1)*4]).
    """
    radius_order = [3, 2, 1, 0]   # indici di SPLINE_SPACING: 1.5, 1.3, 0.7, 0.5
    locs = []
    for j in range(5):              # angolo / spline (ESTERNO)
        for i in radius_order:      # raggio dal piu' esterno al piu' interno
            x = SPLINE_SPACING[i] * np.cos(ANGLES_SPACING[j])
            y = SPLINE_SPACING[i] * np.sin(ANGLES_SPACING[j])
            locs.append((x, y))
    return np.array(locs)        # shape (20, 2)


SENSOR_LOCS = build_sensor_locs()


# ---------------------------------------------------------------------------
# Tensore di conduttivita'
# ---------------------------------------------------------------------------
def build_D_inv(a_ratio, theta_fiber=THETA_FIBER):
    """Restituisce (sigma_11, sigma_12, sigma_22) = componenti di D^{-1}.

    NOTA FISICA IMPORTANTE: il dataset fornisce a_ratio come rapporto di
    ANISOTROPIA DELLE VELOCITA' (v_long / v_trans). Il tensore di diffusione D
    governa pero' la velocita' del fronte come v ~ sqrt(D), quindi nel tensore
    il rapporto va elevato AL QUADRATO. Validato sul sanity-check dei 5 scenari:
    con a_ratio^2 i tempi simulati combaciano con gli osservati usando le
    velocita' fisiche standard, senza dover "barare" abbassando v_fast.
    """
    a_ratio_D = a_ratio ** 2
    theta0 = np.pi/2 - theta_fiber
    a = np.array([np.cos(theta0), np.sin(theta0)])
    b = np.array([np.cos(theta0 - np.pi/2), np.sin(theta0 - np.pi/2)])
    D = (1.0/a_ratio_D) * np.tensordot(a, a, axes=0) + np.tensordot(b, b, axes=0)
    D_inv = np.linalg.inv(D)
    return D_inv[0, 0], D_inv[0, 1], D_inv[1, 1]


def make_speed_field(xc, yc, r, v_slow=V_SLOW, v_fast=V_FAST):
    """Campo di velocita': lento dentro il cerchio (lesione), veloce fuori."""
    def speed(x, y):
        return v_slow if (x - xc)**2 + (y - yc)**2 <= r**2 else v_fast
    return speed


# ---------------------------------------------------------------------------
# Solver FMM (copiato dal notebook, rinominato)
# ---------------------------------------------------------------------------
def anisotropic_FMM(x0, y0, t0, X, Y, speed_input,
                    sigma_11, sigma_12, sigma_22, N_edge=None):
    nrows, ncols = X.shape
    if N_edge is None:
        N_edge = nrows - 1

    INF = 1e4
    P = np.linspace(0.0, 1.0, 11)

    j0 = np.abs(X[0, :] - x0).argmin()
    i0 = np.abs(Y[:, 0] - y0).argmin()

    T = np.full((nrows, ncols), INF, dtype=float)
    T_final = np.full((nrows, ncols), INF, dtype=float)
    T[i0, j0] = t0
    T_final[i0, j0] = t0

    known = np.zeros((nrows, ncols), dtype=bool)
    heap = [(t0, i0, j0)]

    def get_speed(i, j, x, y):
        if callable(speed_input):
            val = speed_input(x, y)
        elif np.isscalar(speed_input):
            val = speed_input
        else:
            val = speed_input[i, j]
        if val <= 0:
            raise ValueError(f"Speed must be positive. Got {val} at ({i},{j}).")
        return val

    def metric_dist(xa, ya, xb, yb, speed_val):
        dx = xa - xb
        dy = ya - yb
        return np.sqrt(sigma_11*dx*dx + 2.0*sigma_12*dx*dy + sigma_22*dy*dy) / speed_val

    def update_fun(T_xi, T_x, T_y, X_xi, X_x, X_y, Y_xi, Y_x, Y_y,
                   flag_known, i_xi, j_xi):
        speed_val = get_speed(i_xi, j_xi, X_xi, Y_xi)
        if flag_known:
            dx = P*(X_xi - X_x) + (1.0 - P)*(X_xi - X_y)
            dy = P*(Y_xi - Y_x) + (1.0 - P)*(Y_xi - Y_y)
            triangle_vals = (P*T_x + (1.0 - P)*T_y
                             + np.sqrt(sigma_11*dx*dx + 2.0*sigma_12*dx*dy
                                       + sigma_22*dy*dy) / speed_val)
            cand = triangle_vals.min()
        else:
            cand = T_x + metric_dist(X_xi, Y_xi, X_x, Y_x, speed_val)
        return min(T_xi, cand)

    side_map = {
        (-1,  0): [(-1, +1), (-1, -1)],
        (+1,  0): [(+1, +1), (+1, -1)],
        (0, -1): [(+1, -1), (-1, -1)],
        (0, +1): [(+1, +1), (-1, +1)],
    }

    while heap:
        t_sel, i_sel, j_sel = heapq.heappop(heap)
        if known[i_sel, j_sel]:
            continue
        if t_sel > T_final[i_sel, j_sel]:
            continue
        known[i_sel, j_sel] = True

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, +1)]:
            ni, nj = i_sel + di, j_sel + dj
            if not (0 <= ni < nrows and 0 <= nj < ncols):
                continue
            if known[ni, nj]:
                continue
            new_T = INF
            for sdi, sdj in side_map[(di, dj)]:
                si, sj = i_sel + sdi, j_sel + sdj
                if not (0 <= si < nrows and 0 <= sj < ncols):
                    continue
                new_T = update_fun(
                    T_xi=new_T,
                    T_x=T_final[i_sel, j_sel], T_y=T_final[si, sj],
                    X_xi=X[ni, nj], X_x=X[i_sel, j_sel], X_y=X[si, sj],
                    Y_xi=Y[ni, nj], Y_x=Y[i_sel, j_sel], Y_y=Y[si, sj],
                    flag_known=known[si, sj], i_xi=ni, j_xi=nj)
            if new_T < T_final[ni, nj]:
                T[ni, nj] = new_T
                T_final[ni, nj] = new_T
                heapq.heappush(heap, (new_T, ni, nj))

    return T_final


# ---------------------------------------------------------------------------
# Forward completo: dai parametri ai 20+20 tempi ai sensori
# ---------------------------------------------------------------------------
_X_MESH, _Y_MESH = np.meshgrid(np.linspace(DOMAIN[0], DOMAIN[1], N_GRID),
                               np.linspace(DOMAIN[0], DOMAIN[1], N_GRID))


def _sample_at_sensors(act_map):
    """Estrae i valori della mappa di attivazione ai 20 sensori (ordine corretto)."""
    out = np.empty(len(SENSOR_LOCS))
    for k, (sx, sy) in enumerate(SENSOR_LOCS):
        MeasMat = (_X_MESH - sx)**2 + (_Y_MESH - sy)**2
        i_sel, j_sel = np.unravel_index(MeasMat.argmin(), MeasMat.shape)
        out[k] = act_map[i_sel, j_sel]
    return out


def forward_activation_times(x0, y0, t0, x1, y1, t1,
                             a_ratio, xc, yc, r):
    """Restituisce (t_front0[20], t_front1[20]) in ms.

    Due fronti indipendenti (le attivita' focali non si sovrappongono),
    quindi risolviamo l'eikonale separatamente per ciascuna sorgente.
    t0, t1 sono i tempi iniziali in ms.
    """
    s11, s12, s22 = build_D_inv(a_ratio)
    speed = make_speed_field(xc, yc, r)

    # NB: nel notebook l'FMM parte da t0=0 e poi i tempi sono *1e3 (in ms).
    # Qui partiamo direttamente con offset t0/t1 in ms NON moltiplicato:
    # risolviamo con t_start=0, convertiamo in ms, poi aggiungiamo l'offset.
    map0 = anisotropic_FMM(x0, y0, 0.0, _X_MESH, _Y_MESH, speed, s11, s12, s22)
    map1 = anisotropic_FMM(x1, y1, 0.0, _X_MESH, _Y_MESH, speed, s11, s12, s22)

    t_front0 = _sample_at_sensors(map0) * 1e3 + t0
    t_front1 = _sample_at_sensors(map1) * 1e3 + t1
    return t_front0, t_front1


# ---------------------------------------------------------------------------
# Sanity check contro i 5 scenari di validazione
# ---------------------------------------------------------------------------
def _extract_times_from_signals(signals_event, min_dist=100):
    """Estrae per ogni sensore i due tempi di attivazione (minimi della derivata),
    riusando la logica del CP1. Restituisce array (20, 2) ordinato per tempo.
    """
    from scipy.signal import savgol_filter, argrelextrema
    sig = savgol_filter(signals_event, window_length=11, polyorder=3, axis=1)
    n = sig.shape[0]
    times = np.full((n, 2), np.nan)
    for i in range(n):
        ds = np.gradient(sig[i])
        mins = argrelextrema(ds, np.less)[0]
        if len(mins) == 0:
            continue
        cand = mins[np.argsort(ds[mins])]   # piu' profondi prima
        sel = []
        for idx in cand:
            if len(sel) == 0:
                sel.append(idx)
            elif len(sel) == 1 and abs(idx - sel[0]) >= min_dist:
                sel.append(idx)
                break
        sel.sort()                          # ordine temporale crescente
        for j, v in enumerate(sel):
            times[i, j] = v
    return times


if __name__ == "__main__":
    import os
    UP = "/mnt/user-data/uploads"
    signals = np.load(os.path.join(UP, "checkpoint2dataset.npy"))
    aniso = np.load(os.path.join(UP, "checkpoint2anisotropy.npy")).ravel()
    focal = np.load(os.path.join(UP, "checkpoint2FocalPoints.npy"))  # gia' in [-2,2]^2
    starting = np.load(os.path.join(UP, "checkpoint2starting.npy"))

    # Reference lesioni dal Checkpoint2.md (scenari 1,2,4,4,5 -> indici 0..4)
    ref_lesion = np.array([
        [-0.407,  0.299, 0.578],
        [ 0.810,  0.113, 0.394],
        [-0.816, -0.686, 0.274],
        [ 0.197, -0.242, 0.042],
        [ 0.647, -0.690, 0.244],
    ])

    print("Sanity check forward vs segnali osservati\n" + "="*50)
    for ev in range(5):
        (x0, y0), (x1, y1) = focal[ev]
        t0, t1 = starting[ev]
        a_ratio = aniso[ev]
        xc, yc, r = ref_lesion[ev]

        tf0, tf1 = forward_activation_times(x0, y0, t0, x1, y1, t1,
                                            a_ratio, xc, yc, r)
        obs = _extract_times_from_signals(signals[ev])  # (20,2) campioni indice

        print(f"\nScenario {ev}  a_ratio={a_ratio}  t0={t0:.2f} t1={t1:.2f}")
        print("  forward front0 (ms), primi 5 sensori:", np.round(tf0[:5], 1))
        print("  forward front1 (ms), primi 5 sensori:", np.round(tf1[:5], 1))
        print("  osservati (indice campione) primi 5  :",
              obs[:5].tolist())

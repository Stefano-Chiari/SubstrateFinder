"""
Modulo 2 - Stima congiunta delle velocita' (v_fast, v_slow).

Strategia (concordata):
  - Stima CONGIUNTA di (v_fast, v_slow) su TUTTI i 20 sensori, PESATI.
  - 1 campione di segnale = 1 ms; l'offset t0/t1 (in ms) e' incluso.
  - Forward FMM ESATTO ricalcolato a ogni iterazione (nessuna linearizzazione).
  - Le velocita' sono comuni a tutti i pazienti: si stimano una volta sola
    usando i 5 scenari di validazione, dove a_ratio, focal points, t0/t1 e la
    lesione di riferimento (dal Checkpoint2.md) sono noti. Poi si congelano.

Costo: ogni valutazione dell'obiettivo = 5 scenari x 2 fronti = 10 FMM 401.
       Parallelizzati su Pool(processes). Lancia tu su cluster a 28 core.

Output: salva velocities.npy = [v_fast, v_slow] e stampa il report.
"""

import os
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.signal import savgol_filter, argrelextrema

import forward_model as fm   # Modulo 1


UP = "/mnt/user-data/uploads"
N_PROC = 28                  # core del cluster

# Lesioni di riferimento dei 5 scenari (dal Checkpoint2.md). 
# >>> SOSTITUISCI con i valori esatti del .md se differiscono <<<
REF_LESION = np.array([
    [-0.407,  0.299, 0.578],
    [ 0.810,  0.113, 0.394],
    [-0.816, -0.686, 0.274],
    [ 0.197, -0.242, 0.042],
    [ 0.647, -0.690, 0.244],
])


# ---------------------------------------------------------------------------
# Estrazione tempi osservati + pesi di affidabilita' (logica CP1)
# ---------------------------------------------------------------------------
def extract_times_and_weights(signals_event, t0, t1, min_dist=100):
    """Per ogni sensore restituisce:
       - t_front0[20], t_front1[20] : tempi di attivazione in ms (= indice campione)
       - w_front0[20], w_front1[20] : pesi (affidabilita' della stima del tempo)

    L'associazione fronte0/fronte1 usa l'ordine temporale (t0 < t1 sempre).
    Il peso combina profondita' del minimo della derivata e nettezza QS.
    """
    sig = savgol_filter(signals_event, window_length=11, polyorder=3, axis=1)
    n = sig.shape[0]
    t_f0 = np.full(n, np.nan); t_f1 = np.full(n, np.nan)
    w_f0 = np.zeros(n);        w_f1 = np.zeros(n)

    for i in range(n):
        ds = np.gradient(sig[i])
        mins = argrelextrema(ds, np.less)[0]
        if len(mins) == 0:
            continue
        cand = mins[np.argsort(ds[mins])]      # piu' profondi (derivata piu' negativa) prima
        sel = []
        for idx in cand:
            if not sel:
                sel.append(idx)
            elif len(sel) == 1 and abs(idx - sel[0]) >= min_dist:
                sel.append(idx)
                break
        sel.sort()                              # ordine temporale crescente
        # peso = profondita' del minimo (modulo della derivata) normalizzata
        for j, idx in enumerate(sel):
            depth = -ds[idx]
            w = max(depth, 0.0)
            if j == 0:
                t_f0[i] = idx; w_f0[i] = w
            else:
                t_f1[i] = idx; w_f1[i] = w

    # normalizza i pesi (scala relativa, non assoluta)
    if np.nansum(w_f0) > 0: w_f0 = w_f0 / np.nanmax(w_f0)
    if np.nansum(w_f1) > 0: w_f1 = w_f1 / np.nanmax(w_f1)
    return t_f0, t_f1, w_f0, w_f1


# ---------------------------------------------------------------------------
# Forward di un singolo (scenario, fronte) - unita' per il Pool
# ---------------------------------------------------------------------------
def _forward_one(args, v_fast, v_slow):
    """Risolve l'FMM per UN fronte di UNO scenario e campiona ai 20 sensori.
    Restituisce i 20 tempi in ms (offset t_start incluso).
    """
    xs, ys, t_start, a_ratio, xc, yc, r = args
    s11, s12, s22 = fm.build_D_inv(a_ratio)
    speed = fm.make_speed_field(xc, yc, r, v_slow=v_slow, v_fast=v_fast)
    amap = fm.anisotropic_FMM(xs, ys, 0.0, fm._X_MESH, fm._Y_MESH,
                              speed, s11, s12, s22)
    return fm._sample_at_sensors(amap) * 1e3 + t_start


def build_jobs(focal, starting, aniso):
    """Costruisce la lista dei 10 job (5 scenari x 2 fronti) con i metadati."""
    jobs = []           # ognuno: (ev, front, args-per-_forward_one)
    for ev in range(5):
        (x0, y0), (x1, y1) = focal[ev]
        t0, t1 = starting[ev]
        a = aniso[ev]
        xc, yc, r = REF_LESION[ev]
        jobs.append((ev, 0, (x0, y0, t0, a, xc, yc, r)))
        jobs.append((ev, 1, (x1, y1, t1, a, xc, yc, r)))
    return jobs


# ---------------------------------------------------------------------------
# Funzione obiettivo: residuo pesato tra forward e osservati
# ---------------------------------------------------------------------------
def _forward_one_packed(packed):
    """Worker top-level Windows-friendly: tutti i parametri in un'unica tupla
    (args_geometrici, v_fast, v_slow). Niente functools.partial, cosi' e'
    serializzabile sotto 'spawn' (Windows / Jupyter)."""
    args, v_fast, v_slow = packed
    return _forward_one(args, v_fast=v_fast, v_slow=v_slow)


def evaluate_forward(params, jobs, executor=None):
    """Calcola i 10 forward per la coppia (v_fast, v_slow).
    Se executor e' fornito (ProcessPoolExecutor), li distribuisce; altrimenti
    li esegue in seriale. Restituisce la lista di predizioni allineata a jobs.
    """
    v_fast, v_slow = params
    packed = [(j[2], v_fast, v_slow) for j in jobs]
    if executor is None:
        return [_forward_one_packed(p) for p in packed]
    return list(executor.map(_forward_one_packed, packed))


def _sse_from_preds(preds, jobs, obs):
    sse = 0.0
    for (ev, front, _), pred in zip(jobs, preds):
        t_obs, w = obs[(ev, front)]
        mask = ~np.isnan(t_obs)
        d = (pred[mask] - t_obs[mask])
        sse += np.sum(w[mask] * d * d)
    return sse


def objective(params, jobs, obs, executor=None):
    v_fast, v_slow = params
    if v_fast <= 0 or v_slow <= 0 or v_slow >= v_fast:
        return 1e12
    preds = evaluate_forward(params, jobs, executor=executor)
    return _sse_from_preds(preds, jobs, obs)


def load_observations(signals, starting):
    """Estrae tempi+pesi osservati per tutti gli scenari/fronti. Riutilizzabile
    dal notebook."""
    obs = {}
    for ev in range(signals.shape[0]):
        t0, t1 = starting[ev]
        tf0, tf1, wf0, wf1 = extract_times_and_weights(signals[ev], t0, t1)
        obs[(ev, 0)] = (tf0, wf0)
        obs[(ev, 1)] = (tf1, wf1)
    return obs


def main():
    signals  = np.load(os.path.join(UP, "checkpoint2dataset.npy"))
    aniso    = np.load(os.path.join(UP, "checkpoint2anisotropy.npy")).ravel()
    focal    = np.load(os.path.join(UP, "checkpoint2FocalPoints.npy"))  # gia' in [-2,2]^2
    starting = np.load(os.path.join(UP, "checkpoint2starting.npy"))

    obs = load_observations(signals, starting)
    jobs = build_jobs(focal, starting, aniso)

    # Cluster: Pool su N_PROC core, FMM esatto nel loop.
    with Pool(processes=N_PROC) as pool:
        class _PoolExec:           # adattatore: espone .map come un executor
            def map(self, fn, it): return pool.map(fn, it)
        res = minimize(objective, x0=[80.0, 20.0],
                       args=(jobs, obs, _PoolExec()),
                       method="Nelder-Mead",
                       options={"xatol": 0.5, "fatol": 1e-2, "disp": True})

    v_fast, v_slow = res.x
    print("\n" + "="*50)
    print(f"v_fast stimata = {v_fast:.3f}")
    print(f"v_slow stimata = {v_slow:.3f}")
    print(f"rapporto v_fast/v_slow = {v_fast/v_slow:.3f}")
    print(f"SSE finale = {res.fun:.4f}")
    print("="*50)

    np.save("/mnt/user-data/outputs/velocities.npy",
            np.array([v_fast, v_slow]))
    print("Salvato velocities.npy")


if __name__ == "__main__":
    main()

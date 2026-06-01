"""
Generatore di dataset sintetico per il Checkpoint 3.

Replica fedelmente le caratteristiche dei dati del prof:
- velocity field 16x16 con macchia di slow conduction lievemente ellittica (asse maggiore verticale)
- valori del field in [~0.30, 1.0]  (stessa distribuzione del prof)
- a_ratio fisso a 3.0
- 2 battiti per paziente
- segnali unipolari (N_patients, 20, 501) con morfologia simile a quella del prof
- starting_points, FocalPoints in formato compatibile col notebook
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from fmm_solver import anisotropic_FMM_GPT


# ============================================================
# COSTANTI FISICHE / DI ACQUISIZIONE
# ============================================================
DOMAIN_HALF = 2.0          # dominio [-2, 2] x [-2, 2]
N_DENSE = 401              # come nel notebook
A_RATIO = 3.0              # FISSO da specifica checkpoint
THETA_FIBER = 0.0          # fibre orizzontali

# velocità (cm/s o equivalente; coerenti con CP2 / notebook prof)
V_BASE = 67.96              # tessuto sano
V_SLOW_MIN, V_SLOW_MAX = 0.26, 0.50   # ratio min/max della slow conduction al CENTRO della macchia
                                       # (il prof ha minimi tra 0.23 e 0.51 -> mediana ~0.37)

# segnali
N_TIMESAMPLES = 501
DT_MS = 1.0                # 1 ms per sample (501 ms totali)

# layout sensori (identico al notebook)
SPLINE_SPACING = [0.5, 0.7, 1.3, 1.5]
ANGLES_SPACING = [
    np.pi/10.0 + 0,
    np.pi/10.0 + 2*np.pi/5,
    np.pi/10.0 + 4*np.pi/5,
    np.pi/10.0 + 6*np.pi/5,
    np.pi/10.0 + 8*np.pi/5,
]


# ============================================================
# 1. GENERAZIONE VELOCITY FIELD 16x16  (stile prof)
# ============================================================
def generate_velocity_field_16(rng):
    """
    Genera un campo 16x16 con UNA macchia ellittica gaussiana, valori in [vmin, 1.0].

    Parametri statisticamente compatibili col prof:
      - centro nella regione misurabile (~ |x|,|y| < 1.2)
      - raggi tra 0.4 e 1.0
      - leggera ellissi con asse maggiore verticale (anisotropia)
      - profilo gaussiano (smooth, NON binario)
    """
    # griglia 16x16 sul dominio fisico [-2, 2]
    g = np.linspace(-DOMAIN_HALF, DOMAIN_HALF, 16)
    XX, YY = np.meshgrid(g, g, indexing='ij')

    # parametri della macchia di slow conduction
    # nel prof: macchie larghe (raggio ~ 0.7-1.2), centro mai sui bordi, leggermente più alta in y
    cx = rng.uniform(-0.9, 0.9)
    cy = rng.uniform(-0.9, 0.9)
    rx = rng.uniform(0.70, 1.10)
    ry = rng.uniform(0.85, 1.30)            # asse maggiore in y (effetto anisotropia)
    rotation = rng.uniform(-np.pi/14, np.pi/14)   # quasi assiale
    depth_min = rng.uniform(V_SLOW_MIN, V_SLOW_MAX)   # quanto è "scura" la macchia

    # ruoto le coordinate
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    Xr = cos_r*(XX - cx) + sin_r*(YY - cy)
    Yr = -sin_r*(XX - cx) + cos_r*(YY - cy)

    # gaussiana ellittica normalizzata in [0, 1]
    gauss = np.exp(-(Xr**2 / (rx**2) + Yr**2 / (ry**2)))

    # scalo in [depth_min, 1.0]: 1.0 fuori, depth_min al centro
    field = 1.0 - (1.0 - depth_min) * gauss

    # smoothing per transizioni dolci come nel prof
    field = gaussian_filter(field, sigma=0.7)
    field = np.clip(field, depth_min, 1.0)

    return field, dict(cx=cx, cy=cy, rx=rx, ry=ry, rotation=rotation, depth_min=depth_min)


def field16_to_dense_field(field16, n_dense=N_DENSE):
    """Interpola il field 16x16 alla griglia densa per il forward model."""
    from scipy.interpolate import RectBivariateSpline
    g16 = np.linspace(-DOMAIN_HALF, DOMAIN_HALF, 16)
    g_dense = np.linspace(-DOMAIN_HALF, DOMAIN_HALF, n_dense)
    interp = RectBivariateSpline(g16, g16, field16, kx=3, ky=3)
    dense = interp(g_dense, g_dense)
    return np.clip(dense, 0.05, 1.5)


# ============================================================
# 2. SENSORI / GRIGLIA DI MISURA
# ============================================================
def get_sensor_positions():
    """Calcola le posizioni dei 20 sensori (stesso ordine del notebook prof)."""
    positions = []
    for i in range(4):
        for j in range(5):
            x_s = SPLINE_SPACING[i] * np.cos(ANGLES_SPACING[j])
            y_s = SPLINE_SPACING[i] * np.sin(ANGLES_SPACING[j])
            positions.append((x_s, y_s))
    return np.array(positions)   # shape (20, 2)


def get_sensor_indices_on_dense(X_mesh, Y_mesh):
    """Indici (i,j) su griglia densa più vicini ai sensori."""
    sensors = get_sensor_positions()
    idx = []
    for x_s, y_s in sensors:
        d2 = (X_mesh - x_s)**2 + (Y_mesh - y_s)**2
        idx.append(np.unravel_index(d2.argmin(), d2.shape))
    return np.array(idx)


# ============================================================
# 3. SIMULAZIONE FISICA (FMM) PER OTTENERE TEMPI DI ATTIVAZIONE
# ============================================================
def compute_diffusion_inverse():
    """Tensore D^{-1} con anisotropia fissa e fibre orizzontali."""
    theta0 = np.pi/2 - THETA_FIBER
    a = np.array([np.cos(theta0), np.sin(theta0)])              # cross-fiber
    b = np.array([np.cos(theta0 - np.pi/2), np.sin(theta0 - np.pi/2)])  # fiber
    D = (1.0/A_RATIO) * np.outer(a, a) + np.outer(b, b)
    D_inv = np.linalg.inv(D)
    return D_inv[0, 0], D_inv[0, 1], D_inv[1, 1]


def simulate_activation_times(
    field16, focal_xy, t0_ms,
    X_mesh, Y_mesh, sensor_idx
):
    """
    Esegue il FMM e ritorna i 20 tempi di attivazione (in ms) ai sensori.
    field16   : (16, 16)  ratio in [0,1]  -> velocità = field16 * V_BASE
    focal_xy  : (x0, y0)
    t0_ms     : tempo iniziale del battito in ms
    """
    speed_dense = field16_to_dense_field(field16) * V_BASE
    s11, s12, s22 = compute_diffusion_inverse()

    T = anisotropic_FMM_GPT(
        focal_xy[0], focal_xy[1], t0_ms / 1000.0,
        X_mesh, Y_mesh, speed_dense,
        s11, s12, s22, N_edge=200
    )
    T_ms = T * 1000.0
    times_at_sensors = T_ms[sensor_idx[:, 0], sensor_idx[:, 1]]
    return times_at_sensors


# ============================================================
# 4. SIMULAZIONE SEGNALI UNIPOLARI ATTORNO AI TEMPI DI ATTIVAZIONE
# ============================================================
def synthesize_unipolar_signal(activation_times_ms, n_samples=N_TIMESAMPLES, rng=None):
    """
    Dati i 2 tempi di attivazione di un sensore, costruisce un segnale unipolare
    con la classica morfologia: piccola onda positiva seguita da deflessione
    negativa veloce centrata nel tempo di attivazione, poi rilassamento.

    Sample rate = 1 ms.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n_samples).astype(float)
    sig = np.zeros(n_samples)

    for t_act in activation_times_ms:
        if not np.isfinite(t_act):
            continue
        if t_act < 5 or t_act > n_samples - 5:
            continue

        # parametri morfologici (con piccola variabilità)
        amp_pos = rng.uniform(0.7, 1.1)
        amp_neg = rng.uniform(0.9, 1.3)
        sigma_pos = rng.uniform(6.0, 9.0)
        sigma_neg = rng.uniform(3.0, 4.5)
        delay_neg = rng.uniform(2.0, 4.0)   # la deflessione negativa è poco dopo il picco positivo

        # il punto di attivazione è dove il gradiente è minimo
        # -> mettiamo la transizione brusca centrata su t_act
        center_pos = t_act - delay_neg
        center_neg = t_act + 2.0

        sig += amp_pos * np.exp(-((t - center_pos)**2) / (2 * sigma_pos**2))
        sig -= amp_neg * np.exp(-((t - center_neg)**2) / (2 * sigma_neg**2))

    # piccolo rumore gaussiano
    sig += rng.normal(0, 0.012, n_samples)

    # baseline drift molto leggero
    drift = 0.005 * np.sin(2*np.pi*t / n_samples * rng.uniform(0.5, 1.5))
    sig += drift

    return sig


# ============================================================
# 5. PIPELINE PER UN SINGOLO PAZIENTE
# ============================================================
def generate_single_patient(seed):
    """Genera un paziente completo: velocity field + segnali + metadati."""
    rng = np.random.default_rng(seed)

    # ---- griglia densa e sensori ----
    g_dense = np.linspace(-DOMAIN_HALF, DOMAIN_HALF, N_DENSE)
    X_mesh, Y_mesh = np.meshgrid(g_dense, g_dense)
    sensor_idx = get_sensor_indices_on_dense(X_mesh, Y_mesh)

    # ---- velocity field ----
    field16, _params = generate_velocity_field_16(rng)

    # ---- focal points: 2 battiti, posizioni casuali interne ----
    focal_points = np.zeros((2, 2))
    for k in range(2):
        # bordo del dominio per simulare ingresso del fronte
        side = rng.integers(0, 4)
        if side == 0:    # ingresso da destra
            focal_points[k] = [rng.uniform(1.3, 1.6), rng.uniform(-1.5, 1.5)]
        elif side == 1:  # da sinistra
            focal_points[k] = [rng.uniform(-1.6, -1.3), rng.uniform(-1.5, 1.5)]
        elif side == 2:  # dall'alto
            focal_points[k] = [rng.uniform(-1.5, 1.5), rng.uniform(1.3, 1.6)]
        else:            # dal basso
            focal_points[k] = [rng.uniform(-1.5, 1.5), rng.uniform(-1.6, -1.3)]

    # ---- starting times: primo battito a t≈0, secondo battito a t≈260-270 ms ----
    t0_ms = rng.uniform(0.0, 8.0)
    t1_ms = rng.uniform(260.0, 270.0)
    starting_times = np.array([t0_ms, t1_ms])

    # ---- simulo i tempi di attivazione (via FMM) per ogni battito ----
    times_b1 = simulate_activation_times(field16, focal_points[0], t0_ms, X_mesh, Y_mesh, sensor_idx)
    times_b2 = simulate_activation_times(field16, focal_points[1], t1_ms, X_mesh, Y_mesh, sensor_idx)

    # ---- costruisco i 20 segnali unipolari ----
    signals = np.zeros((20, N_TIMESAMPLES))
    for s in range(20):
        signals[s] = synthesize_unipolar_signal(
            [times_b1[s], times_b2[s]],
            n_samples=N_TIMESAMPLES,
            rng=rng,
        )

    return {
        'velocity_field_16': field16,
        'signals': signals,
        'focal_points': focal_points,
        'starting_times': starting_times,
        'anisotropy': np.array([A_RATIO]),
    }


# ============================================================
# 6. GENERAZIONE PARALLELA DEL DATASET COMPLETO
# ============================================================
def generate_synthetic_dataset_parallel(n_samples, base_seed=42, n_jobs=-1, verbose=True):
    """
    Genera n_samples pazienti in parallelo.
    Ritorna un dict con array stack-ati pronti per essere salvati.
    """
    seeds = [base_seed + i for i in range(n_samples)]

    if verbose:
        print(f"Generazione di {n_samples} campioni sintetici (n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(generate_single_patient)(s) for s in seeds
    )

    out = {
        'velocity_field_16': np.stack([r['velocity_field_16'] for r in results], axis=0),
        'signals':            np.stack([r['signals']            for r in results], axis=0),
        'focal_points':       np.stack([r['focal_points']       for r in results], axis=0),
        'starting_times':     np.stack([r['starting_times']     for r in results], axis=0),
        'anisotropy':         np.stack([r['anisotropy']         for r in results], axis=0),
    }
    return out


def save_synthetic_dataset(out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'synthetic_velocityfield16.npy'), out['velocity_field_16'])
    np.save(os.path.join(out_dir, 'synthetic_dataset.npy'),         out['signals'])
    np.save(os.path.join(out_dir, 'synthetic_FocalPoints.npy'),     out['focal_points'])
    np.save(os.path.join(out_dir, 'synthetic_starting.npy'),        out['starting_times'])
    np.save(os.path.join(out_dir, 'synthetic_anisotropy.npy'),      out['anisotropy'])
    print(f"Salvato in {out_dir}:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")


# ============================================================
# 7. MAIN: SMOKE TEST SU 3 CAMPIONI
# ============================================================
if __name__ == '__main__':
    out = generate_synthetic_dataset_parallel(n_samples=3, base_seed=42, n_jobs=1, verbose=True)
    save_synthetic_dataset(out, '/home/claude/synth_smoketest')

    print("\n--- VERIFICA SHAPE (devono combaciare col prof) ---")
    print("velocity_field_16:", out['velocity_field_16'].shape, "  expected (N, 16, 16)")
    print("signals          :", out['signals'].shape,           "  expected (N, 20, 501)")
    print("focal_points     :", out['focal_points'].shape,      "  expected (N, 2, 2)")
    print("starting_times   :", out['starting_times'].shape,    "  expected (N, 2)")
    print("anisotropy       :", out['anisotropy'].shape,        "  expected (N, 1)")

    print("\n--- STATS VELOCITY FIELD ---")
    vf = out['velocity_field_16']
    for i in range(vf.shape[0]):
        f = vf[i]
        print(f"  caso {i}: min={f.min():.3f} max={f.max():.3f} mean={f.mean():.3f} std={f.std():.3f}")

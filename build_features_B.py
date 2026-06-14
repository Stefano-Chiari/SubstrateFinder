"""
Feature engineering per la Rete B: back-projection heatmap (triangolazione).

Per ogni campione, dai residui dei due fronti costruisce una mappa di
back-projection: ogni sensore "rallentato" (residuo > 0) vota lungo il
segmento focal->sensore. Il PRODOTTO delle due viste (fronte 0 x fronte 1)
e' alto solo dove ENTRAMBE votano -> triangola la lesione.

La heatmap GRID x GRID appiattita e' la feature principale per la Rete B,
piu' i focal e t0/t1 come contesto.

Tutto in formula chiusa (homogeneous_times) + algebra, nessun FMM.
Gira in locale in pochi minuti. Output: X_train_B.npy

NB: la STESSA costruzione va usata in GroupSolutionCP2 al test
(funzione build_backproj_features qui sotto, importabile).
"""
import os
import numpy as np
import forward_model as fm

DATA_DIR = r"C:\Users\simon\SCT4AMM\SubstrateFinder"
V_FAST   = 65.541
GRID     = 20                      # risoluzione heatmap (20x20, piu' fine)
DOMAIN   = (-2.0, 2.0)
SIGMA_BP = 0.12                    # voti piu' concentrati (picco piu' stretto)
N_SEG    = 16                      # piu' punti lungo il segmento (griglia fine)
RES_THRESH = 1.0                   # soglia minima residuo per votare (ms)

# griglia della heatmap (fissa, condivisa train/test)
_gx, _gy = np.meshgrid(np.linspace(DOMAIN[0], DOMAIN[1], GRID),
                       np.linspace(DOMAIN[0], DOMAIN[1], GRID))
_SEG_S = np.linspace(0.25, 1.0, N_SEG)   # frazioni lungo il segmento


def _backproject_one(focal_x, focal_y, res20):
    """Heatmap GRID x GRID di una singola vista (un fronte)."""
    heat = np.zeros((GRID, GRID))
    for k in range(20):
        w = res20[k]
        if w < RES_THRESH:
            continue
        sx, sy = fm.SENSOR_LOCS[k]
        px = focal_x + _SEG_S * (sx - focal_x)
        py = focal_y + _SEG_S * (sy - focal_y)
        for pxi, pyi in zip(px, py):
            heat += w * np.exp(-((_gx - pxi)**2 + (_gy - pyi)**2)
                               / (2 * SIGMA_BP**2))
    return heat


def build_backproj_features(tf0, tf1, x0, y0, t0, x1, y1, t1, a_ratio,
                            v_fast=V_FAST):
    """Costruisce il vettore di feature per Rete B da tempi + parametri.

    Ritorna: [heatmap_flat (GRID*GRID), x0,y0,x1,y1, t0,t1]
    """
    hom0 = fm.homogeneous_times(x0, y0, t0, a_ratio, v_fast=v_fast)
    hom1 = fm.homogeneous_times(x1, y1, t1, a_ratio, v_fast=v_fast)
    res0 = tf0 - hom0
    res1 = tf1 - hom1

    h0 = _backproject_one(x0, y0, res0)
    h1 = _backproject_one(x1, y1, res1)
    heat = h0 * h1                       # triangolazione (prodotto delle viste)

    # normalizza la heatmap (scala relativa: conta DOVE e' il picco, non quanto)
    m = heat.max()
    if m > 0:
        heat = heat / m

    return np.concatenate([heat.ravel(), [x0, y0, x1, y1, t0, t1]])


def main():
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    N = X.shape[0]
    print(f"Dataset: X={X.shape}  ->  feature B con heatmap {GRID}x{GRID}")

    # a_ratio perturbato (imita errore Rete A al test)
    rng = np.random.default_rng(2024)
    feat_dim = GRID*GRID + 6
    X_B = np.empty((N, feat_dim))

    for i in range(N):
        a_used = np.clip(y[i, 0] + rng.normal(0, 0.10), 1.0, 9.0)
        tf0 = X[i, 0:20]; tf1 = X[i, 20:40]
        x0, y0, x1, y1 = X[i, 40:44]
        t0, t1 = X[i, 44], X[i, 45]
        X_B[i] = build_backproj_features(tf0, tf1, x0, y0, t0, x1, y1, t1,
                                         a_used, v_fast=V_FAST)
        if (i+1) % 5000 == 0:
            print(f"  {i+1}/{N}")

    # diagnostica: il picco della heatmap e' vicino alla lesione vera?
    errs = []
    for i in range(0, N, 50):     # campione ridotto per velocita'
        heat = X_B[i, :GRID*GRID].reshape(GRID, GRID)
        if heat.max() <= 0:
            continue
        iy, ix = np.unravel_index(heat.argmax(), heat.shape)
        est_x, est_y = _gx[iy, ix], _gy[iy, ix]
        errs.append(np.hypot(est_x - y[i,1], est_y - y[i,2]))
    errs = np.array(errs)
    print(f"\nPicco heatmap vs lesione vera: err medio={errs.mean():.2f} "
          f"mediano={np.median(errs):.2f} (su campione ridotto)")

    np.save(os.path.join(DATA_DIR, "X_train_B.npy"), X_B)
    print(f"Salvato X_train_B.npy  shape={X_B.shape}")


if __name__ == "__main__":
    main()

"""
Post-processing del dataset: calcola le feature RESIDUO per la Rete B.

Per ogni campione:
  residuo[k] = tempo_osservato[k] - tempo_omogeneo[k]
dove tempo_omogeneo e' calcolato con la formula analitica (forma chiusa,
nessun FMM) usando a_ratio e i focal del campione.

Il residuo isola l'effetto della lesione, eliminando la componente dominante
dovuta a geometria + anisotropia -> rende la lesione molto piu' identificabile.

a_ratio usato per il residuo in TRAINING: a_ratio vero + rumore gaussiano che
imita l'errore di Rete A (che al test stima a_ratio con un piccolo errore).
Cosi' la rete e' robusta al fatto che, al test, l'a_ratio per il residuo viene
da Rete A e non e' esatto.

Gira in LOCALE in pochi secondi (nessun FMM). Output: X_train_resid.npy
con le 40 colonne dei tempi sostituite dal residuo (focal e t0/t1 invariati).
"""
import os
import numpy as np
import forward_model as fm

# >>> path locali <<<
DATA_DIR = r"C:\Users\simon\SCT4AMM\SubstrateFinder"
V_FAST   = 65.541

# Errore tipico di Rete A su a_ratio (dalla cella 4 del notebook: ~0.05-0.1).
# Usato per perturbare a_ratio nel calcolo del residuo in training.
SIGMA_A_RATIO = 0.10
SEED = 2024


def compute_residual_features(X, y, sigma_a=SIGMA_A_RATIO, seed=SEED):
    """Sostituisce le 40 colonne dei tempi (0:40) con il residuo osservato-omogeneo.

    X: (N,46) feature originali [t_f0(20), t_f1(20), x0,y0,x1,y1, t0,t1]
    y: (N,4)  target [a_ratio, xc, yc, r]
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    X_res = X.copy()

    for i in range(N):
        a_true = y[i, 0]
        # a_ratio perturbato (imita la stima di Rete A al test)
        a_used = np.clip(a_true + rng.normal(0, sigma_a), 1.0, 9.0)

        x0, y0, x1, y1 = X[i, 40], X[i, 41], X[i, 42], X[i, 43]
        t0, t1         = X[i, 44], X[i, 45]

        # tempi omogenei (formula analitica) per i due fronti
        hom0 = fm.homogeneous_times(x0, y0, t0, a_used, v_fast=V_FAST)
        hom1 = fm.homogeneous_times(x1, y1, t1, a_used, v_fast=V_FAST)

        # residuo = osservato - omogeneo  (isola l'effetto della lesione)
        X_res[i, 0:20]  = X[i, 0:20]  - hom0
        X_res[i, 20:40] = X[i, 20:40] - hom1

    return X_res


def main():
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    print(f"Dataset: X={X.shape}, y={y.shape}")

    X_res = compute_residual_features(X, y)

    # diagnostica: il residuo dovrebbe essere ~0 dove non c'e' lesione,
    # e avere code dove la lesione ritarda il fronte
    res_times = X_res[:, 0:40]
    print(f"Residuo tempi: media={res_times.mean():.2f}ms  std={res_times.std():.2f}ms")
    print(f"  |residuo| medio: {np.abs(res_times).mean():.2f}ms  max: {np.abs(res_times).max():.1f}ms")
    # correlazione tra ampiezza del residuo e raggio della lesione
    res_energy = np.abs(res_times).sum(axis=1)
    r_lesion = y[:, 3]
    corr = np.corrcoef(res_energy, r_lesion)[0, 1]
    print(f"  corr(|residuo totale|, raggio lesione) = {corr:.3f}  "
          f"(atteso positivo: lesioni grandi -> residuo grande)")

    np.save(os.path.join(DATA_DIR, "X_train_resid.npy"), X_res)
    print(f"\nSalvato X_train_resid.npy  shape={X_res.shape}")


if __name__ == "__main__":
    main()

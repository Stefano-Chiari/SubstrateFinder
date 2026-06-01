"""
Verifica delle velocita' stimate.

Tre diagnostiche:
  1) verify_residuals : SSE per-scenario per set di velocita' dati
                        (confronto 65.5/22.9 vs 80/20 o altri).
  2) refit_isotropic  : ri-stima (v_fast, v_slow) usando SOLO gli scenari
                        isotropi (a_ratio == 1), dove v_fast e' disaccoppiata
                        dall'anisotropia -> test pulito del valore vero.
  3) report           : stampa tutto in modo leggibile.

Da importare nel notebook. Usa le funzioni gia' presenti in
estimate_velocities (load_observations, build_jobs, evaluate_forward, ...).
"""
import numpy as np
import forward_model as fm
import estimate_velocities as ev
from scipy.optimize import minimize


def _sse_per_scenario(params, jobs, obs, executor=None):
    """Restituisce dict {scenario: sse} per il set di velocita' params."""
    preds = ev.evaluate_forward(params, jobs, executor=executor)
    per = {}
    for (s, front, _), pred in zip(jobs, preds):
        t_obs, w = obs[(s, front)]
        mask = ~np.isnan(t_obs)
        d = pred[mask] - t_obs[mask]
        per[s] = per.get(s, 0.0) + float(np.sum(w[mask] * d * d))
    return per


def verify_residuals(signals, focal, starting, aniso,
                     velocity_sets, executor=None):
    """Confronta piu' set di velocita' calcolando SSE per scenario e totale."""
    obs = ev.load_observations(signals, starting)
    jobs = ev.build_jobs(focal, starting, aniso)
    rows = {}
    for name, (vf, vs) in velocity_sets.items():
        per = _sse_per_scenario((vf, vs), jobs, obs, executor=executor)
        per["TOT"] = sum(per.values())
        rows[name] = per
    return rows


def refit_isotropic(signals, focal, starting, aniso, executor=None):
    """Ri-stima (v_fast, v_slow) sui soli scenari con a_ratio == 1.

    NB: con a_ratio=1 il tensore e' isotropo, quindi NON c'e' lesione che
    dipenda dall'anisotropia; v_slow resta vincolata solo dai sensori il cui
    raggio attraversa la lesione. Se pochi sensori la attraversano, v_slow
    sara' poco identificabile: in tal caso fidarsi soprattutto di v_fast.
    """
    iso_idx = [i for i in range(len(aniso)) if abs(aniso[i] - 1.0) < 1e-9]
    if not iso_idx:
        return None, iso_idx

    obs = ev.load_observations(signals, starting)
    # costruisco jobs solo per gli scenari isotropi
    jobs = []
    for s in iso_idx:
        (x0, y0), (x1, y1) = focal[s]
        t0, t1 = starting[s]
        a = aniso[s]
        xc, yc, r = ev.REF_LESION[s]
        jobs.append((s, 0, (x0, y0, t0, a, xc, yc, r)))
        jobs.append((s, 1, (x1, y1, t1, a, xc, yc, r)))

    def objective(params):
        vf, vs = params
        if vf <= 0 or vs <= 0 or vs >= vf:
            return 1e12
        preds = ev.evaluate_forward(params, jobs, executor=executor)
        sse = 0.0
        for (s, front, _), pred in zip(jobs, preds):
            t_obs, w = obs[(s, front)]
            mask = ~np.isnan(t_obs)
            d = pred[mask] - t_obs[mask]
            sse += np.sum(w[mask] * d * d)
        return sse

    res = minimize(objective, x0=[80.0, 20.0], method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 1e-2})
    return res, iso_idx


def report(rows, refit_res=None, iso_idx=None):
    print("="*60)
    print("RESIDUI (SSE) PER SCENARIO E SET DI VELOCITA'")
    print("="*60)
    scen_keys = sorted({k for r in rows.values() for k in r if k != "TOT"})
    header = f'{"set":>14} | ' + " ".join(f"s{s:>6}" for s in scen_keys) + f' | {"TOT":>9}'
    print(header)
    print("-"*len(header))
    for name, per in rows.items():
        line = f'{name:>14} | ' + " ".join(f"{per.get(s,0):7.1f}" for s in scen_keys)
        line += f' | {per["TOT"]:9.1f}'
        print(line)
    if refit_res is not None:
        print("\n" + "="*60)
        print(f"RI-STIMA SUI SOLI SCENARI ISOTROPI {iso_idx} (a_ratio=1)")
        print("="*60)
        vf, vs = refit_res.x
        print(f"  v_fast = {vf:.3f}")
        print(f"  v_slow = {vs:.3f}   (poco vincolata se la lesione e' piccola)")
        print(f"  SSE    = {refit_res.fun:.3f}")
        print("\n  --> Se v_fast qui e' ~80, il valore globale piu' basso e' un")
        print("      compromesso sugli scenari anisotropi. Se e' ~66, e' reale.")

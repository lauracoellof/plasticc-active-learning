"""Budgeted oracle-query policies for classification and anomaly discovery.

Known labels can update the classifier; 99x labels are retained as discoveries
and never added to its known-class training set. Output column names and mode
values preserve the saved experiment schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .anomalies import rank01 as _rank01, robust_zscore as _zscore_robusto
from .anomalies import gp_residual_score as gp_residual_score


def entropy(proba, eps=1e-12):
    """Return Shannon entropy per probability row using natural logarithms."""
    p = np.asarray(proba, dtype=float)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def margin(proba):
    """Return one minus the difference between the largest two class probabilities."""
    p = np.sort(np.asarray(proba, dtype=float), axis=1)
    return 1.0 - (p[:, -1] - p[:, -2])


def least_confidence(proba):
    """Return one minus the maximum class probability for each object."""
    return 1.0 - np.max(np.asarray(proba, dtype=float), axis=1)


_ESTRATEGIAS_UNC = {"entropy": entropy, "margin": margin, "least_confidence": least_confidence}


def uncertainty_score(proba, method_name="entropy"):
    """Dispatch a named entropy, margin or least-confidence acquisition score."""
    if method_name not in _ESTRATEGIAS_UNC:
        raise ValueError(f"unknown method: {method_name}. " f"Options: {list(_ESTRATEGIAS_UNC)}")
    return _ESTRATEGIAS_UNC[method_name](proba)


def fuse_novelty(*scores, score_weights=None, method_name="rank"):
    """Combine aligned novelty arrays after rank or robust-scale normalization."""
    if not scores:
        raise ValueError("At least one novelty score is required.")
    norm = _rank01 if method_name == "rank" else _zscore_robusto
    M = np.column_stack([norm(s) for s in scores])
    if score_weights is None:
        return M.mean(axis=1)
    w = np.asarray(score_weights, dtype=float)
    return (M * w).sum(axis=1) / w.sum()


def novelty_gate(s_nov, threshold_value=None, quantile_level=0.95):
    """Return a high-novelty mask and its supplied or empirical-quantile threshold."""
    s = np.asarray(s_nov, dtype=float)
    u = float(np.quantile(s, quantile_level)) if threshold_value is None else float(threshold_value)
    return s >= u, u


def stratified_seed(y, por_clase=3, random_state=42):
    """Draw up to por_clase initial labelled indices per known class without replacement."""
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    sel = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = min(por_clase, len(idx))
        sel.extend(rng.choice(idx, size=k, replace=False))
    return np.array(sorted(sel))


def select_by_quota(pool_idx, s_unc, mask_candidata, s_nov, b_cls, b_nov):
    """Select separate novelty and classification quotas from disjoint candidate regions."""
    pool_idx = np.asarray(pool_idx)
    escogidos = {"clasificacion": np.array([], dtype=int), "anomalia": np.array([], dtype=int)}

    if b_nov > 0 and mask_candidata.any():
        loc = np.where(mask_candidata)[0]
        top = loc[np.argsort(s_nov[loc])[::-1][:b_nov]]
        escogidos["anomalia"] = pool_idx[top]

    if b_cls > 0:
        loc = np.where(~mask_candidata)[0]
        if len(loc):
            top = loc[np.argsort(s_unc[loc])[::-1][:b_cls]]
            escogidos["clasificacion"] = pool_idx[top]
    return escogidos


RARE_CLASSES = (64, 53, 6, 67, 52)


def _metrics(y_true, y_pred, ia_class=90, rare_classes=RARE_CLASSES):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    raras = [c for c in rare_classes if c in np.unique(y_true)]
    return {
        "WF1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "MF1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "pureza_Ia": round(
            precision_score(y_true, y_pred, labels=[ia_class], average="macro", zero_division=0), 4
        ),
        "completitud_Ia": round(
            recall_score(y_true, y_pred, labels=[ia_class], average="macro", zero_division=0), 4
        ),
        "recall_raras": (
            round(recall_score(y_true, y_pred, labels=raras, average="macro", zero_division=0), 4)
            if raras
            else 0.0
        ),
    }


def active_learning_loop(
    X,
    y,
    idx_lab,
    idx_pool,
    X_test,
    y_test,
    make_model,
    *,
    query_strategy="entropy",
    n_iter=15,
    b_cls=20,
    b_nov=0,
    use_gate=False,
    gate_quantile=0.95,
    s_novedad_pool=None,
    ia_class=90,
    random_state=42,
    verbose=True,
):
    """Run a fixed-evaluation-set acquisition experiment from explicit labelled/pool indices.

    Fit the model after each acquisition, record classification metrics and count
    queried anomaly labels separately from the known-class training set."""
    import pandas as pd

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    idx_lab = list(np.asarray(idx_lab, dtype=int))
    idx_pool = list(np.asarray(idx_pool, dtype=int))
    idx_anom = []
    rng = np.random.default_rng(random_state)
    s_nov_map = None
    if s_novedad_pool is not None:
        s_nov_map = {int(i): float(s) for i, s in zip(idx_pool, s_novedad_pool)}

    records_list = []
    for it in range(n_iter + 1):
        estimator_model = make_model(y[idx_lab]).fit(X[idx_lab], y[idx_lab])
        y_pred = estimator_model.predict(X_test)
        fila = {
            "iter": it,
            "n_lab": len(idx_lab),
            "estrategia": query_strategy,
            "n_anomalias": len(idx_anom),
        }
        fila.update(_metrics(y_test, y_pred, ia_class))
        records_list.append(fila)
        if verbose:
            print(
                f"iter {it:2d} | n_lab={len(idx_lab):4d} | "
                f"MF1={fila['MF1']:.4f} WF1={fila['WF1']:.4f} "
                f"pureza_Ia={fila['pureza_Ia']:.4f}"
            )
        if it == n_iter or not idx_pool:
            break

        pool = np.array(idx_pool)
        if query_strategy == "random":
            k = min(b_cls + b_nov, len(pool))
            new_known = rng.choice(pool, size=k, replace=False)
            new_anomalies = np.array([], dtype=int)
        else:
            proba = estimator_model.predict_proba(X[pool])
            s_unc = uncertainty_score(proba, query_strategy)
            if use_gate and b_nov > 0:
                s_nov = (
                    np.array([s_nov_map[int(i)] for i in pool])
                    if s_nov_map is not None
                    else np.zeros(len(pool))
                )
                mask_cand, _ = novelty_gate(s_nov, quantile_level=gate_quantile)
            else:
                s_nov = np.zeros(len(pool))
                mask_cand = np.zeros(len(pool), dtype=bool)
            escog = select_by_quota(pool, s_unc, mask_cand, s_nov, b_cls, b_nov)
            new_known, new_anomalies = escog["clasificacion"], escog["anomalia"]

        for i in new_anomalies:
            idx_anom.append(int(i))
            idx_pool.remove(int(i))
        for i in new_known:
            idx_lab.append(int(i))
            idx_pool.remove(int(i))

    hist = pd.DataFrame(records_list)
    loop_state = {
        "idx_lab": np.array(idx_lab),
        "idx_pool": np.array(idx_pool),
        "idx_anomalias": np.array(idx_anom),
    }
    return hist, loop_state


def classification_al_experiment(
    X,
    y,
    make_model,
    *,
    strategies_list=("entropy", "random"),
    random_seeds=(0, 1, 2, 3, 4),
    test_size=0.25,
    por_clase=3,
    n_iter=12,
    b_cls=20,
    b_nov=0,
    use_gate=False,
    s_novedad=None,
    ia_class=90,
    verbose=False,
):
    """Compare classification query strategies across matched stratified train/test splits."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    idx = np.arange(len(y))
    all_records = []
    for s in random_seeds:
        idx_trab, idx_test = train_test_split(idx, test_size=test_size, stratify=y, random_state=s)
        sem_local = stratified_seed(y[idx_trab], por_clase=por_clase, random_state=s)
        idx_lab = idx_trab[sem_local]
        idx_pool = np.setdiff1d(idx_trab, idx_lab)
        s_nov_pool = None
        if s_novedad is not None:
            s_nov_pool = np.asarray(s_novedad)[idx_pool]
        for est_ in strategies_list:
            hist, _ = active_learning_loop(
                X,
                y,
                idx_lab,
                idx_pool,
                X[idx_test],
                y[idx_test],
                make_model,
                query_strategy=est_,
                n_iter=n_iter,
                b_cls=b_cls,
                b_nov=b_nov,
                use_gate=use_gate,
                s_novedad_pool=s_nov_pool,
                ia_class=ia_class,
                random_state=s,
                verbose=verbose,
            )
            hist["semilla"] = s
            all_records.append(hist)
            if verbose:
                print(f"  [seed {s} · {est_}] MF1 final={hist['MF1'].iloc[-1]:.4f}")
    return pd.concat(all_records, ignore_index=True)


def aggregate_curves(hist, metricas=("MF1", "WF1", "pureza_Ia")):
    """Summarize the selected learning-curve metrics across seeds at each iteration."""
    aggs = {m: ["mean", "std"] for m in metricas}
    g = hist.groupby(["estrategia", "n_lab"]).agg(aggs)
    g.columns = [f"{m}_{stat}" for m, stat in g.columns]
    return g.reset_index()


def _iter_test_splits(y_te, random_seeds, *, test_size_eval, etq_anom_min):
    """Yield matched evaluation/pool splits stratified by the known/anomaly label convention."""
    y_te = np.asarray(y_te)
    known_objects = np.where(y_te < etq_anom_min)[0]
    anomalies = np.where(y_te >= etq_anom_min)[0]
    for seed_value in random_seeds:
        conocidas_pool, conocidas_eval = train_test_split(
            known_objects,
            test_size=test_size_eval,
            random_state=seed_value,
            stratify=y_te[known_objects],
        )
        pool = np.sort(np.concatenate([conocidas_pool, anomalies]))
        yield seed_value, pool, conocidas_eval


def _reset_multiseed_output(output_path):
    """Remove an existing experiment output before a requested full multiseed rerun."""
    if output_path is not None:
        Path(output_path).unlink(missing_ok=True)


def _record_multiseed_result(all_records, hist, output_path=None, **identificadores):
    """Attach experiment identifiers and append one completed seed trajectory."""
    for item_name, valor in identificadores.items():
        hist[
            {"seed_value": "semilla", "run_mode": "modo", "query_strategy": "estrategia"}.get(
                item_name, item_name
            )
        ] = valor
    all_records.append(hist)
    if output_path is not None:
        file_path = Path(output_path)
        hist.to_csv(file_path, mode="a", index=False, header=not file_path.exists())


def aad_discovery(
    S_pool,
    y_pool,
    prior,
    *,
    run_mode="aad",
    b=50,
    n_iter=20,
    etq_anom_min=990,
    X_feat=None,
    warm_start_pos=0,
    reg_C=0.1,
    random_state=42,
    verbose=False,
):
    """Run feedback-guided anomaly retrieval on aligned descriptor scores under an oracle budget."""
    import pandas as pd

    S = np.asarray(S_pool, float)
    Sn = np.column_stack([_rank01(S[:, j]) for j in range(S.shape[1])])
    y_pool = np.asarray(y_pool)
    prior = _rank01(np.asarray(prior, float))
    n = len(y_pool)

    if X_feat is not None:
        Xf = np.asarray(X_feat, float)
        mu = Xf.mean(0)
        sd = Xf.std(0)
        sd[sd == 0] = 1.0
        Zfit = (Xf - mu) / sd
    else:
        Zfit = Sn
    vivo = np.ones(n, dtype=bool)
    total = int((y_pool >= etq_anom_min).sum())
    rng = np.random.default_rng(random_state)
    lab_idx, lab_y = [], []
    reranker = None

    def _fit_reranker():
        ly = np.asarray(lab_y)
        if ly.sum() >= 1 and (ly == 0).sum() >= 1:
            return LogisticRegression(max_iter=2000, C=reg_C, class_weight="balanced").fit(
                Zfit[lab_idx], ly
            )
        return reranker

    if run_mode == "aad" and warm_start_pos > 0:
        a_idx = np.where(y_pool >= etq_anom_min)[0]
        s_pos = rng.choice(a_idx, size=min(warm_start_pos, len(a_idx)), replace=False)
        no_idx = np.where(y_pool < etq_anom_min)[0]
        s_neg = rng.choice(no_idx, size=min(5 * warm_start_pos, len(no_idx)), replace=False)
        for i in np.concatenate([s_pos, s_neg]):
            lab_idx.append(int(i))
            lab_y.append(int(y_pool[i] >= etq_anom_min))
            vivo[i] = False
        reranker = _fit_reranker()
    discoveries = query_count = 0
    records_list = []
    for it in range(n_iter + 1):
        records_list.append(
            {
                "iter": it,
                "n_consultas": query_count,
                "n_descubiertas": discoveries,
                "recall_anom": round(discoveries / total, 4) if total else 0.0,
            }
        )
        if verbose:
            print(f"iter {it:2d} | queries={query_count:4d} | discovered={discoveries:3d}/{total}")
        if it == n_iter or not vivo.any():
            break

        if run_mode == "random":
            score = rng.random(n)
        elif run_mode == "fijo" or reranker is None:
            score = prior.copy()
        else:
            score = reranker.predict_proba(Zfit)[:, 1]
        score = score.copy()
        score[~vivo] = -np.inf
        pick = np.argsort(score)[::-1][: min(b, int(vivo.sum()))]

        for i in pick:
            vivo[i] = False
            query_count += 1
            yb = int(y_pool[i] >= etq_anom_min)
            lab_idx.append(int(i))
            lab_y.append(yb)
            if yb:
                discoveries += 1

        if run_mode == "aad":
            reranker = _fit_reranker()
    return pd.DataFrame(records_list)


def aad_discovery_multiseed(
    S,
    y,
    prior,
    *,
    modes_list=("aad", "fijo", "random"),
    random_seeds=(0, 1, 2),
    n_known_pool=None,
    b=50,
    n_iter=20,
    etq_anom_min=990,
    X_feat=None,
    warm_start_pos=0,
    reg_C=0.1,
    output_path=None,
    verbose=True,
):
    """Compare anomaly-discovery policies across matched seeds and evaluation splits."""
    import pandas as pd

    S = np.asarray(S, float)
    y = np.asarray(y)
    prior = np.asarray(prior, float)
    X_feat = None if X_feat is None else np.asarray(X_feat, float)
    con = np.where(y < etq_anom_min)[0]
    ano = np.where(y >= etq_anom_min)[0]
    _reset_multiseed_output(output_path)
    all_records = []
    for s in random_seeds:
        rng = np.random.default_rng(s)
        if n_known_pool is not None and n_known_pool < len(con):
            con_s = rng.choice(con, size=n_known_pool, replace=False)
        else:
            con_s = con
        idx = np.sort(np.concatenate([con_s, ano]))
        Xf_idx = None if X_feat is None else X_feat[idx]
        for run_mode in modes_list:
            h = aad_discovery(
                S[idx],
                y[idx],
                prior[idx],
                run_mode=run_mode,
                b=b,
                n_iter=n_iter,
                etq_anom_min=etq_anom_min,
                X_feat=Xf_idx,
                warm_start_pos=warm_start_pos,
                reg_C=reg_C,
                random_state=s,
                verbose=False,
            )
            _record_multiseed_result(all_records, h, output_path, seed_value=s, run_mode=run_mode)
            if verbose:
                print(
                    f"[seed {s} · {run_mode:10s}] discovered={int(h['n_descubiertas'].iloc[-1]):3d}"
                )
    return pd.concat(all_records, ignore_index=True)


def confidence_triage(
    X_tr,
    y_tr,
    X_pool,
    y_pool,
    X_eval,
    y_eval,
    make_model,
    *,
    run_mode="triage",
    conf_anom=0.30,
    conf_dudoso=0.60,
    use_margin=False,
    margin_conf=0.15,
    orden_dudoso="least_conf",
    budget=1000,
    b_iter=100,
    n_iter=30,
    etq_anom_min=990,
    ia_class=90,
    random_state=42,
    verbose=False,
):
    "Query low-confidence objects before the intermediate-confidence region.\n\nRemove high-confidence objects without adding pseudo-labels. Route oracle\nlabels from known classes into classifier training and retain 99x labels only\nas anomaly discoveries. Enforce the total query budget."
    import pandas as pd

    X_pool = np.asarray(X_pool, float)
    y_pool = np.asarray(y_pool)
    X_lab = np.asarray(X_tr, float)
    y_lab = np.asarray(y_tr)
    vivo = np.ones(len(X_pool), dtype=bool)
    total_anom = int((y_pool >= etq_anom_min).sum())
    rng = np.random.default_rng(random_state)
    discoveries = query_count = q_anom = q_dudoso = auto = 0
    records_list = []
    for it in range(n_iter + 1):
        estimator_model = make_model(y_lab).fit(X_lab, y_lab)
        fila = {
            "iter": it,
            "n_consultas": query_count,
            "n_descubiertas": discoveries,
            "recall_anom": round(discoveries / total_anom, 4) if total_anom else 0.0,
            "q_anom": q_anom,
            "q_dudoso": q_dudoso,
            "n_auto": auto,
        }
        fila.update(_metrics(y_eval, estimator_model.predict(X_eval), ia_class))
        records_list.append(fila)
        if verbose:
            print(
                f"iter {it:2d} | queries={query_count:4d} | discovered={discoveries:3d}/{total_anom} "
                f"| auto={auto} | MF1={fila['MF1']:.3f}"
            )
        if it == n_iter or not vivo.any() or query_count >= budget:
            break

        idx = np.where(vivo)[0]
        if run_mode == "random":
            k = min(b_iter, budget - query_count, len(idx))
            pick = rng.choice(idx, size=k, replace=False)
            anomaly_fraction = np.zeros(len(pick), dtype=bool)
        else:
            proba = estimator_model.predict_proba(X_pool[idx])
            conf = proba.max(axis=1)
            novel = conf < conf_anom
            if use_margin:

                p2 = np.sort(proba, axis=1)
                margin = p2[:, -1] - p2[:, -2]
                confident_objects = (~novel) & (margin >= margin_conf)

                urg = np.where(novel, 2.0 - conf, 1.0 - margin)
            else:

                confident_objects = conf >= conf_dudoso
                if orden_dudoso == "entropy":

                    p = np.clip(proba, 1e-12, 1.0)
                    ent = -np.sum(p * np.log(p), axis=1)
                    urg = np.where(novel, 1e3 + (1.0 - conf), ent)
                else:
                    urg = 1.0 - conf
            auto += int(confident_objects.sum())
            vivo[idx[confident_objects]] = False
            cand = np.where(~confident_objects)[0]
            if len(cand) == 0:
                continue
            sort_order = cand[np.argsort(urg[cand])[::-1]]
            k = min(b_iter, budget - query_count, len(sort_order))
            pick_local = sort_order[:k]
            pick = idx[pick_local]
            anomaly_fraction = novel[pick_local]

        new_features, new_labels = [], []
        for r, i in enumerate(pick):
            vivo[i] = False
            query_count += 1
            if run_mode != "random":
                if anomaly_fraction[r]:
                    q_anom += 1
                else:
                    q_dudoso += 1
            if y_pool[i] >= etq_anom_min:
                discoveries += 1
            else:
                new_features.append(X_pool[i])
                new_labels.append(y_pool[i])
        if new_features:
            X_lab = np.vstack([X_lab, np.asarray(new_features)])
            y_lab = np.concatenate([y_lab, np.asarray(new_labels)])

    return pd.DataFrame(records_list)


def confidence_triage_multiseed(
    X_tr,
    y_tr,
    X_te,
    y_te,
    make_model,
    *,
    modes_list=("triage", "random"),
    random_seeds=(0, 1, 2),
    test_size_eval=0.2,
    conf_anom=0.30,
    conf_dudoso=0.60,
    use_margin=False,
    margin_conf=0.15,
    orden_dudoso="least_conf",
    budget=1000,
    b_iter=100,
    n_iter=30,
    etq_anom_min=990,
    ia_class=90,
    output_path=None,
    verbose=True,
):
    """Run confidence triage and its baselines on matched evaluation splits across seeds."""
    import pandas as pd

    X_te = np.asarray(X_te, float)
    y_te = np.asarray(y_te)
    _reset_multiseed_output(output_path)
    all_records = []
    for s, pool, con_eval in _iter_test_splits(
        y_te, random_seeds, test_size_eval=test_size_eval, etq_anom_min=etq_anom_min
    ):
        Xp, yp = X_te[pool], y_te[pool]
        Xe, ye = X_te[con_eval], y_te[con_eval]
        for run_mode in modes_list:
            h = confidence_triage(
                X_tr,
                y_tr,
                Xp,
                yp,
                Xe,
                ye,
                make_model,
                run_mode=run_mode,
                conf_anom=conf_anom,
                conf_dudoso=conf_dudoso,
                use_margin=(use_margin and run_mode != "random"),
                margin_conf=margin_conf,
                orden_dudoso=orden_dudoso,
                budget=budget,
                b_iter=b_iter,
                n_iter=n_iter,
                etq_anom_min=etq_anom_min,
                ia_class=ia_class,
                random_state=s,
                verbose=False,
            )
            _record_multiseed_result(all_records, h, output_path, seed_value=s, run_mode=run_mode)
            if verbose:
                print(
                    f"[seed {s} · {run_mode:9s}] queries={int(h['n_consultas'].iloc[-1]):4d} "
                    f"descub={int(h['n_descubiertas'].iloc[-1]):3d} "
                    f"auto={int(h['n_auto'].iloc[-1])} MF1={h['MF1'].iloc[-1]:.3f}"
                )
    return pd.concat(all_records, ignore_index=True)


def compare_query_strategies(
    X_tr,
    y_tr,
    X_pool,
    y_pool,
    X_eval,
    y_eval,
    make_model,
    *,
    query_strategy="least_confidence",
    b=100,
    n_iter=10,
    etq_anom_min=990,
    ia_class=90,
    random_state=42,
    verbose=False,
):
    """Evaluate one uncertainty strategy while excluding queried 99x labels from training."""
    import pandas as pd

    X_lab = np.asarray(X_tr, float)
    y_lab = np.asarray(y_tr)
    X_pool = np.asarray(X_pool, float)
    y_pool = np.asarray(y_pool)
    vivo = np.ones(len(X_pool), dtype=bool)
    total_anom = int((y_pool >= etq_anom_min).sum())
    rng = np.random.default_rng(random_state)
    Xa, ya = [X_lab], [y_lab]
    discoveries = query_count = 0
    records_list = []
    for it in range(n_iter + 1):
        Xt = np.vstack(Xa)
        yt = np.concatenate(ya)
        estimator_model = make_model(yt).fit(Xt, yt)
        fila = {
            "iter": it,
            "estrategia": query_strategy,
            "n_consultas": query_count,
            "n_descubiertas": discoveries,
            "recall_anom": round(discoveries / total_anom, 4) if total_anom else 0.0,
        }
        fila.update(_metrics(y_eval, estimator_model.predict(X_eval), ia_class))
        records_list.append(fila)
        if verbose:
            print(
                f"iter {it:2d} | queries={query_count:4d} | discovered={discoveries:3d}/{total_anom} "
                f"| MF1={fila['MF1']:.4f} | recall_anom={fila['recall_anom']:.3f}"
            )
        if it == n_iter or not vivo.any():
            break
        idx = np.where(vivo)[0]
        if query_strategy == "random":
            pick = rng.choice(idx, size=min(b, len(idx)), replace=False)
        else:
            s_unc = uncertainty_score(estimator_model.predict_proba(X_pool[idx]), query_strategy)
            pick = idx[np.argsort(s_unc)[::-1][: min(b, len(idx))]]
        new_features, new_labels = [], []
        for i in pick:
            vivo[i] = False
            query_count += 1
            if y_pool[i] >= etq_anom_min:
                discoveries += 1
            else:
                new_features.append(X_pool[i])
                new_labels.append(y_pool[i])
        if new_features:
            Xa.append(np.asarray(new_features))
            ya.append(np.asarray(new_labels))
    return pd.DataFrame(records_list)


def compare_query_strategies_multiseed(
    X_tr,
    y_tr,
    X_te,
    y_te,
    make_model,
    *,
    strategies_list=("least_confidence", "margin", "entropy", "random"),
    random_seeds=(0, 1, 2),
    test_size_eval=0.2,
    b=100,
    n_iter=10,
    etq_anom_min=990,
    ia_class=90,
    output_path=None,
    verbose=True,
):
    """Run each classification query strategy on the same per-seed evaluation split."""
    import pandas as pd

    X_te = np.asarray(X_te, float)
    y_te = np.asarray(y_te)
    _reset_multiseed_output(output_path)
    all_records = []
    for s, pool, con_eval in _iter_test_splits(
        y_te, random_seeds, test_size_eval=test_size_eval, etq_anom_min=etq_anom_min
    ):
        Xp, yp = X_te[pool], y_te[pool]
        Xe, ye = X_te[con_eval], y_te[con_eval]
        for est_ in strategies_list:
            h = compare_query_strategies(
                X_tr,
                y_tr,
                Xp,
                yp,
                Xe,
                ye,
                make_model,
                query_strategy=est_,
                b=b,
                n_iter=n_iter,
                etq_anom_min=etq_anom_min,
                ia_class=ia_class,
                random_state=s,
                verbose=False,
            )
            _record_multiseed_result(all_records, h, output_path, seed_value=s)
            if verbose:
                print(
                    f"[seed {s} · {est_:16s}] discovered={int(h['n_descubiertas'].iloc[-1]):3d} "
                    f"recall_anom={h['recall_anom'].iloc[-1]:.3f} MF1={h['MF1'].iloc[-1]:.3f}"
                )
    return pd.concat(all_records, ignore_index=True)


def aad_triage(
    X_tr,
    y_tr,
    X_pool,
    y_pool,
    desc_pool,
    X_eval,
    y_eval,
    make_model,
    *,
    run_mode="triaje_aad",
    conf_anom=0.30,
    conf_dudoso=0.60,
    orden_dudoso="entropy",
    n_rescate=0,
    reg_C=0.1,
    budget=1000,
    b_iter=100,
    n_iter=30,
    etq_anom_min=990,
    ia_class=90,
    random_state=42,
    verbose=False,
):
    """Query the low-confidence region using a feedback-trained discovery reranker.

    Intermediate-region rescue queries are optional and share the same total budget;
    only oracle labels belonging to known classes update classifier training."""
    import pandas as pd

    X_pool = np.asarray(X_pool, float)
    y_pool = np.asarray(y_pool)
    desc_pool = np.asarray(desc_pool, float)
    X_lab = np.asarray(X_tr, float)
    y_lab = np.asarray(y_tr)
    n = len(X_pool)
    vivo = np.ones(n, dtype=bool)
    total_anom = int((y_pool >= etq_anom_min).sum())
    rng = np.random.default_rng(random_state)
    discoveries = query_count = q_anom = q_dudoso = auto = 0
    lab_idx, lab_y = [], []
    reranker = None
    records_list = []
    for it in range(n_iter + 1):
        estimator_model = make_model(y_lab).fit(X_lab, y_lab)
        fila = {
            "iter": it,
            "n_consultas": query_count,
            "n_descubiertas": discoveries,
            "recall_anom": round(discoveries / total_anom, 4) if total_anom else 0.0,
            "q_anom": q_anom,
            "q_dudoso": q_dudoso,
            "n_auto": auto,
        }
        fila.update(_metrics(y_eval, estimator_model.predict(X_eval), ia_class))
        records_list.append(fila)
        if verbose:
            print(
                f"iter {it:2d} | queries={query_count:4d} | discovered={discoveries:3d}/{total_anom} "
                f"| auto={auto} | MF1={fila['MF1']:.3f}"
            )
        if it == n_iter or not vivo.any() or query_count >= budget:
            break

        proba = estimator_model.predict_proba(X_pool)
        conf = proba.max(axis=1)
        s_nov = 1.0 - conf

        S = np.column_stack([s_nov, desc_pool])
        Sn = np.column_stack([_rank01(S[:, j]) for j in range(S.shape[1])])
        ly = np.asarray(lab_y)
        if run_mode != "random" and reranker is not None and ly.sum() >= 1 and (ly == 0).sum() >= 1:
            s_aad = reranker.predict_proba(Sn)[:, 1]
        else:
            s_aad = _rank01(s_nov)

        idx = np.where(vivo)[0]
        if run_mode == "random":
            k = min(b_iter, budget - query_count, len(idx))
            pick = rng.choice(idx, size=k, replace=False)
            anomaly_fraction = np.zeros(len(pick), dtype=bool)
        else:
            novel = conf < conf_anom
            confident_objects = conf >= conf_dudoso

            if n_rescate > 0 and reranker is not None:
                elegibles = np.where(vivo & (~novel))[0]
                if len(elegibles):
                    top = elegibles[np.argsort(s_aad[elegibles])[::-1][:n_rescate]]
                    rescate = np.zeros(n, dtype=bool)
                    rescate[top] = True
                    anom_cand = novel | rescate
                    confident_objects = confident_objects & ~rescate
                else:
                    anom_cand = novel
            else:
                anom_cand = novel
            p = np.clip(proba, 1e-12, 1.0)
            ent = -np.sum(p * np.log(p), axis=1)

            if orden_dudoso == "entropy":
                urg = np.where(anom_cand, 1e3 + s_aad, ent)
            else:
                urg = np.where(anom_cand, 1e3 + s_aad, 1.0 - conf)
            nuevo_conf = confident_objects & vivo
            auto += int(nuevo_conf.sum())
            vivo[nuevo_conf] = False
            cand = np.where((~confident_objects) & vivo)[0]
            if len(cand) == 0:
                continue
            sort_order = cand[np.argsort(urg[cand])[::-1]]
            k = min(b_iter, budget - query_count, len(sort_order))
            pick = sort_order[:k]
            anomaly_fraction = anom_cand[pick]

        new_features, new_labels = [], []
        for r, i in enumerate(pick):
            vivo[i] = False
            query_count += 1
            if run_mode != "random":
                if anomaly_fraction[r]:
                    q_anom += 1
                else:
                    q_dudoso += 1
            yb = int(y_pool[i] >= etq_anom_min)
            lab_idx.append(int(i))
            lab_y.append(yb)
            if yb:
                discoveries += 1
            else:
                new_features.append(X_pool[i])
                new_labels.append(y_pool[i])
        if new_features:
            X_lab = np.vstack([X_lab, np.asarray(new_features)])
            y_lab = np.concatenate([y_lab, np.asarray(new_labels)])

        if run_mode != "random":
            ly = np.asarray(lab_y)
            if ly.sum() >= 1 and (ly == 0).sum() >= 1:
                reranker = LogisticRegression(max_iter=2000, C=reg_C, class_weight="balanced").fit(
                    Sn[lab_idx], ly
                )

    return pd.DataFrame(records_list)


def aad_triage_multiseed(
    X_tr,
    y_tr,
    X_te,
    y_te,
    desc_te,
    make_model,
    *,
    modes_list=("triaje_aad", "triage", "random"),
    random_seeds=(0, 1, 2),
    test_size_eval=0.2,
    conf_anom=0.30,
    conf_dudoso=0.60,
    orden_dudoso="entropy",
    n_rescate=30,
    reg_C=0.1,
    budget=1000,
    b_iter=100,
    n_iter=30,
    etq_anom_min=990,
    ia_class=90,
    output_path=None,
    verbose=True,
):
    """Compare feedback-assisted triage modes on matched per-seed evaluation splits."""
    import pandas as pd

    X_te = np.asarray(X_te, float)
    y_te = np.asarray(y_te)
    desc_te = np.asarray(desc_te, float)
    _reset_multiseed_output(output_path)
    all_records = []
    for s, pool, con_eval in _iter_test_splits(
        y_te, random_seeds, test_size_eval=test_size_eval, etq_anom_min=etq_anom_min
    ):
        Xp, yp = X_te[pool], y_te[pool]
        dp = desc_te[pool]
        Xe, ye = X_te[con_eval], y_te[con_eval]
        for run_mode in modes_list:
            if run_mode in ("triaje_aad", "triaje_aad_hib"):
                h = aad_triage(
                    X_tr,
                    y_tr,
                    Xp,
                    yp,
                    dp,
                    Xe,
                    ye,
                    make_model,
                    run_mode="triaje_aad",
                    conf_anom=conf_anom,
                    conf_dudoso=conf_dudoso,
                    orden_dudoso=orden_dudoso,
                    n_rescate=(n_rescate if run_mode == "triaje_aad_hib" else 0),
                    reg_C=reg_C,
                    budget=budget,
                    b_iter=b_iter,
                    n_iter=n_iter,
                    etq_anom_min=etq_anom_min,
                    ia_class=ia_class,
                    random_state=s,
                    verbose=False,
                )
            else:
                h = confidence_triage(
                    X_tr,
                    y_tr,
                    Xp,
                    yp,
                    Xe,
                    ye,
                    make_model,
                    run_mode=run_mode,
                    conf_anom=conf_anom,
                    conf_dudoso=conf_dudoso,
                    orden_dudoso=orden_dudoso,
                    budget=budget,
                    b_iter=b_iter,
                    n_iter=n_iter,
                    etq_anom_min=etq_anom_min,
                    ia_class=ia_class,
                    random_state=s,
                    verbose=False,
                )
            _record_multiseed_result(all_records, h, output_path, seed_value=s, run_mode=run_mode)
            if verbose:
                print(
                    f"[seed {s} · {run_mode:11s}] queries={int(h['n_consultas'].iloc[-1]):4d} "
                    f"descub={int(h['n_descubiertas'].iloc[-1]):3d} "
                    f"auto={int(h['n_auto'].iloc[-1])} MF1={h['MF1'].iloc[-1]:.3f}"
                )
    return pd.concat(all_records, ignore_index=True)


def two_channel_pipeline(
    X_tr,
    y_tr,
    X_pool,
    y_pool,
    desc_pool,
    X_eval,
    y_eval,
    make_model,
    *,
    frac_desc=0.4,
    conf_dudoso=0.60,
    reg_C=0.1,
    budget=1000,
    b_iter=100,
    n_iter=30,
    etq_anom_min=990,
    ia_class=90,
    random_state=42,
    verbose=False,
):
    "Allocate a shared query budget to classification and anomaly discovery.\n\nAfter automatic removal of high-confidence objects, discovery ranks the\nremaining pool using feedback-trained score fusion. Classification queries\nuncertain objects by entropy. Known labels update both components; 99x labels\nupdate only the discovery component. No high-confidence pseudo-labels are used."
    import pandas as pd

    X_pool = np.asarray(X_pool, float)
    y_pool = np.asarray(y_pool)
    desc_pool = np.asarray(desc_pool, float)
    X_lab = np.asarray(X_tr, float)
    y_lab = np.asarray(y_tr)
    n = len(X_pool)
    vivo = np.ones(n, dtype=bool)
    total_anom = int((y_pool >= etq_anom_min).sum())
    budget_desc = int(round(budget * frac_desc))
    budget_clas = budget - budget_desc
    b_desc = int(round(b_iter * frac_desc))
    b_clas = b_iter - b_desc
    discoveries = descub_d = descub_c = 0
    cons = cons_d = cons_c = auto = 0
    lab_idx, lab_y = [], []
    reranker = None
    records_list = []
    for it in range(n_iter + 1):
        estimator_model = make_model(y_lab).fit(X_lab, y_lab)
        fila = {
            "iter": it,
            "n_consultas": cons,
            "cons_desc": cons_d,
            "cons_clas": cons_c,
            "n_auto": auto,
            "n_descubiertas": discoveries,
            "descub_desc": descub_d,
            "descub_clas": descub_c,
            "recall_anom": round(discoveries / total_anom, 4) if total_anom else 0.0,
        }
        fila.update(_metrics(y_eval, estimator_model.predict(X_eval), ia_class))
        records_list.append(fila)
        if verbose:
            print(
                f"iter {it:2d} | cons={cons:4d} (D{cons_d}/C{cons_c}) | discovered={discoveries:3d} "
                f"(D{descub_d}/C{descub_c}) | auto={auto} | MF1={fila['MF1']:.3f}"
            )
        agotado = (cons_d >= budget_desc) and (cons_c >= budget_clas)
        if it == n_iter or not vivo.any() or agotado:
            break

        proba = estimator_model.predict_proba(X_pool)
        conf = proba.max(axis=1)
        s_nov = 1.0 - conf
        S = np.column_stack([s_nov, desc_pool])
        Sn = np.column_stack([_rank01(S[:, j]) for j in range(S.shape[1])])
        ly = np.asarray(lab_y)
        if reranker is not None and ly.sum() >= 1 and (ly == 0).sum() >= 1:
            s_aad = reranker.predict_proba(Sn)[:, 1]
        else:
            s_aad = _rank01(s_nov)
        p = np.clip(proba, 1e-12, 1.0)
        ent = -np.sum(p * np.log(p), axis=1)

        confident_objects = (conf >= conf_dudoso) & vivo
        auto += int(confident_objects.sum())
        vivo[confident_objects] = False

        elegido = np.zeros(n, dtype=bool)

        kd = min(b_desc, budget_desc - cons_d, int(vivo.sum()))
        if kd > 0:
            score = s_aad.copy()
            score[~vivo] = -np.inf
            elegido[np.argsort(score)[::-1][:kd]] = True

        libre = vivo & (~elegido)
        kc = min(b_clas, budget_clas - cons_c, int(libre.sum()))
        pick_c = np.array([], dtype=int)
        if kc > 0:
            score = ent.copy()
            score[~libre] = -np.inf
            pick_c = np.argsort(score)[::-1][:kc]

        new_features, new_labels = [], []
        for i in np.where(elegido)[0]:
            vivo[i] = False
            cons += 1
            cons_d += 1
            yb = int(y_pool[i] >= etq_anom_min)
            lab_idx.append(int(i))
            lab_y.append(yb)
            if yb:
                discoveries += 1
                descub_d += 1
            else:
                new_features.append(X_pool[i])
                new_labels.append(y_pool[i])
        for i in pick_c:
            vivo[i] = False
            cons += 1
            cons_c += 1
            yb = int(y_pool[i] >= etq_anom_min)
            lab_idx.append(int(i))
            lab_y.append(yb)
            if yb:
                discoveries += 1
                descub_c += 1
            else:
                new_features.append(X_pool[i])
                new_labels.append(y_pool[i])
        if new_features:
            X_lab = np.vstack([X_lab, np.asarray(new_features)])
            y_lab = np.concatenate([y_lab, np.asarray(new_labels)])
        ly = np.asarray(lab_y)
        if ly.sum() >= 1 and (ly == 0).sum() >= 1:
            reranker = LogisticRegression(max_iter=2000, C=reg_C, class_weight="balanced").fit(
                Sn[lab_idx], ly
            )

    return pd.DataFrame(records_list)


def two_channel_multiseed(
    X_tr,
    y_tr,
    X_te,
    y_te,
    desc_te,
    make_model,
    *,
    fracs=(0.0, 0.3, 0.5, 0.7, 1.0),
    random_seeds=(0, 1, 2),
    test_size_eval=0.2,
    conf_dudoso=0.60,
    reg_C=0.1,
    budget=1000,
    b_iter=100,
    n_iter=30,
    etq_anom_min=990,
    ia_class=90,
    output_path=None,
    verbose=True,
):
    """Compare discovery-budget allocations on matched evaluation splits and random seeds."""
    import pandas as pd

    X_te = np.asarray(X_te, float)
    y_te = np.asarray(y_te)
    desc_te = np.asarray(desc_te, float)
    _reset_multiseed_output(output_path)
    all_records = []
    for s, pool, con_eval in _iter_test_splits(
        y_te, random_seeds, test_size_eval=test_size_eval, etq_anom_min=etq_anom_min
    ):
        Xp, yp = X_te[pool], y_te[pool]
        dp = desc_te[pool]
        Xe, ye = X_te[con_eval], y_te[con_eval]
        for fr in fracs:
            h = two_channel_pipeline(
                X_tr,
                y_tr,
                Xp,
                yp,
                dp,
                Xe,
                ye,
                make_model,
                frac_desc=fr,
                conf_dudoso=conf_dudoso,
                reg_C=reg_C,
                budget=budget,
                b_iter=b_iter,
                n_iter=n_iter,
                etq_anom_min=etq_anom_min,
                ia_class=ia_class,
                random_state=s,
                verbose=False,
            )
            _record_multiseed_result(all_records, h, output_path, seed_value=s, frac_desc=fr)
            if verbose:
                u = h.iloc[-1]
                print(
                    f"[seed {s} · frac_desc={fr:.2f}] discovered={int(u['n_descubiertas'])} "
                    f"(D{int(u['descub_desc'])}/C{int(u['descub_clas'])}) "
                    f"auto={int(u['n_auto'])} MF1={u['MF1']:.3f}"
                )
    return pd.concat(all_records, ignore_index=True)


ejecutar_multiseed = classification_al_experiment

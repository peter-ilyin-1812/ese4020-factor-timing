# ese 4020 - experiments after the baseline checkpoint
# ridge (course baseline), lasso, elastic net on lagged anomaly long-short returns vs historical mean
# same split, folds, gap and bootstrap as baseline.py. 2010-2024 is cut off before anything is built.
#
# usage:
#   python experiments.py --data data/ --out results_5/                                   # the 5 baseline signals
#   python experiments.py --data PredictorLSretWide.parquet --out results_all/            # every signal observed 1972-2009
#   python experiments.py --data PredictorLSretWide.parquet --doc SignalDoc.csv --pit --models hm,ridge --out results_pit/
#       --pit: at each scored year Y, only signals published before Y are inputs or targets (publication year from SignalDoc)
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import enet_path

seed = 4020
nlags, gap, K = 12, 12, 5
minStocks = 100
valStart, valEnd, testStart = "1990-01", "2009-12", "2010-01"
nboot, blk = 2000, 12
ridgeGrid = np.logspace(0, 8, 33)          # baseline used 1e0..1e6; widened because inputs go from 60 to ~1,800
enetGrid = np.logspace(1, -2, 13)          # descending, in % return units; top of grid zeroes every coefficient
l1s = {"lasso": 1.0, "enet": 0.5}
costs = [0.10, 0.25, 0.50]                 # one-way cost, % of traded notional per leg


def load(path, start):
    if os.path.isdir(path):                # baseline csv format (has leg counts -> thin-month filter)
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(path + "/*.csv"))])
        df["ym"] = pd.to_datetime(df.date).dt.to_period("M")
        R = df.pivot(index="ym", columns="signalname", values="ret").sort_index()
        bad = df[(df.Nlong < minStocks) | (df.Nshort < minStocks)].ym
        R = R[~R.index.isin(bad)].dropna()
    else:
        R = pd.read_parquet(path)
        R.index = pd.to_datetime(R.pop("date")).dt.to_period("M")
        R = R.sort_index().loc[start:]
    R = R[R.index < pd.Period(testStart)]  # test period never enters memory
    if not os.path.isdir(path):
        R = R.loc[:, R.notna().all()]      # signals observed every month 1972-2009 (chosen on dev data only)
    return R


def scale(a, b):
    mu, sd = a.mean(0), a.std(0, ddof=0)
    sd[sd == 0] = 1
    return (a - mu) / sd, (b - mu) / sd


def ridge_preds(xtr, ytr, xva, grid):
    # all penalties at once from one svd; same solution as sklearn Ridge with intercept
    ym = ytr.mean(0)
    U, s, Vt = np.linalg.svd(xtr, full_matrices=False)
    Uy = U.T @ (ytr - ym)
    xv = xva @ Vt.T
    return [ym + xv @ ((s / (s**2 + a))[:, None] * Uy) for a in grid]


def enet_one(xtr, G, y, xvas, l1, grid):
    ym = y.mean()
    _, coefs, _ = enet_path(xtr, y - ym, l1_ratio=l1, alphas=grid, precompute=G, Xy=xtr.T @ (y - ym),
                            max_iter=3000, tol=1e-3)
    return [ym + xv @ coefs for xv in xvas], (coefs != 0).sum(0)


def enet_preds(xtr, ytr, xva, l1, grid, jobs):
    G = xtr.T @ xtr
    out = Parallel(n_jobs=jobs)(delayed(enet_one)(xtr, G, ytr[:, j], [xva], l1, grid) for j in range(ytr.shape[1]))
    P = np.stack([o[0][0] for o in out], axis=2)          # rows x alphas x targets
    nz = np.stack([o[1] for o in out], axis=1)            # alphas x targets
    return [P[:, i, :] for i in range(len(grid))], nz


def fit_predict(model, xtr, ytr, xva, grid, jobs):
    if model == "ridge":
        return ridge_preds(xtr, ytr, xva, grid), None
    return enet_preds(xtr, ytr, xva, l1s[model], grid, jobs)


def pick(model, X, Y, grid, jobs):
    # blocked k-fold: contiguous folds, gap months purged on both sides, scaling fit on training folds
    n = len(X)
    err = np.zeros(len(grid))
    e = np.linspace(0, n, K + 1).astype(int)
    for k in range(K):
        va = np.arange(e[k], e[k + 1])
        tr = np.r_[np.arange(0, max(e[k] - gap, 0)), np.arange(min(e[k + 1] + gap, n), n)]
        xtr, xva = scale(X[tr], X[va])
        P, _ = fit_predict(model, xtr, Y[tr], xva, grid, jobs)
        err += [((Y[va] - p)**2).sum() for p in P]
    return int(err.argmin())


# ---------- metrics (nan = portfolio not in the universe that month) ----------
def r2(d, m):
    ok = ~np.isnan(d["r"])
    return 1 - ((d["r"] - d[m])[ok]**2).sum() / ((d["r"] - d["hm"])[ok]**2).sum()


def strat(d, m):
    return np.nanmean(np.sign(d[m]) * d["r"], axis=1)


def sr(d, m, c=0.0):
    s = strat(d, m)
    if c:
        s = s - 2 * c * turnover(d, m)
    return np.sqrt(12) * s.mean() / s.std(ddof=1)


def turnover(d, m):
    # equal weight 1/N_t, long or short each portfolio by forecast sign; sum |w_t - w_{t-1}| per month
    w = np.sign(d[m]) / np.sum(~np.isnan(d["r"]), axis=1, keepdims=True)
    w = np.nan_to_num(w)
    return np.r_[0, np.abs(np.diff(w, axis=0)).sum(1)]


def boot(d, f, rng):
    T = d["r"].shape[0]
    out = np.empty(nboot)
    for b in range(nboot):
        st = rng.integers(0, T, size=int(np.ceil(T / blk)))
        idx = np.concatenate([(s + np.arange(blk)) % T for s in st])[:T]
        out[b] = f({k: v[idx] for k, v in d.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="results_5/")
    ap.add_argument("--start", default="1972-01")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--pit", action="store_true")
    ap.add_argument("--models", default="hm,ridge,lasso,enet")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--budget", type=float, default=1e9, help="stop after this many seconds; rerun to resume")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(seed)
    models = args.models.split(",")
    fitted = [m for m in models if m != "hm"]

    R = load(args.data, args.start)
    pub = None
    if args.pit:
        doc = pd.read_csv(args.doc).set_index("Acronym")
        pub = doc.Year.reindex(R.columns)
        R = R.loc[:, pub.notna()]
        pub = pub[R.columns]
    ports = list(R.columns)

    X = pd.concat({l: R.shift(l) for l in range(1, nlags + 1)}, axis=1)
    X.columns = [f"{s}_{l}" for l, s in X.columns]
    keep = X.notna().all(axis=1)
    X, Y = X[keep], R[keep]
    vm = Y.index[(Y.index >= pd.Period(valStart)) & (Y.index <= pd.Period(valEnd))]
    assert vm.max() < pd.Period(testStart) and R.index.max() < pd.Period(testStart)

    import pickle
    ck = args.out + "/checkpoints/"
    os.makedirs(ck, exist_ok=True)
    rows, chosen = [], []
    t0 = time.time()
    for yr in sorted(set(vm.year)):
        if os.path.exists(ck + f"{yr}.pkl"):        # resume: refits are deterministic, so reuse finished years
            r_, c_ = pickle.load(open(ck + f"{yr}.pkl", "rb"))
            rows += r_; chosen.append(c_)
            continue
        if time.time() - t0 > args.budget:
            print("budget reached, rerun to continue"); raise SystemExit(3)
        nrow = len(rows)
        mths = vm[vm.year == yr]
        trainEnd = mths.min() - gap - 1
        tr = Y.index <= trainEnd
        cols = ports if pub is None else [p for p in ports if pub[p] < yr]
        xcols = [c for c in X.columns if c.rsplit("_", 1)[0] in cols]
        Xtr, Ytr = X.loc[tr, xcols].values, Y.loc[tr, cols].values
        xs, xt = scale(Xtr, X.loc[mths, xcols].values)
        P = {"hm": np.tile(Ytr.mean(0), (len(mths), 1))}
        rec = {"year": yr, "n_train_months": int(tr.sum()), "n_ports": len(cols), "n_inputs": len(xcols)}
        for m in fitted:
            grid = ridgeGrid if m == "ridge" else enetGrid
            i = pick(m, Xtr, Ytr, grid, args.jobs)
            Pm, nz = fit_predict(m, xs, Ytr, xt, grid, args.jobs)
            P[m] = Pm[i]
            rec[m + "_alpha"] = float(grid[i])
            rec[m + "_grid_edge"] = bool(i in (0, len(grid) - 1))
            if nz is not None:
                rec[m + "_mean_nonzero"] = float(nz[i].mean())
        chosen.append(rec)
        for i, t in enumerate(mths):
            for j, p in enumerate(cols):
                rows.append((str(t), p, Y.loc[t, p], *[P[m][i, j] for m in models]))
        pickle.dump((rows[nrow:], rec), open(ck + f"{yr}.pkl", "wb"))
        print(yr, rec, f"{time.time() - t0:.0f}s", flush=True)

    preds = pd.DataFrame(rows, columns=["month", "port", "r", *models])
    chosen = pd.DataFrame(chosen)
    preds.to_csv(args.out + "/val_predictions.csv", index=False)
    chosen.to_csv(args.out + "/chosen.csv", index=False)

    D = {c: preds.pivot(index="month", columns="port", values=c).values for c in ["r", *models]}
    res = {"panel": [str(R.index.min()), str(R.index.max()), len(R)], "n_signals": len(ports),
           "n_inputs_max": X.shape[1], "pit": args.pit, "val_months": preds.month.nunique(),
           "val_forecasts": len(preds), "test_period_loaded": False}
    for m in fitted:
        bs = boot(D, lambda d: r2(d, m), rng)
        res["r2os_" + m] = [r2(D, m), *np.percentile(bs, [2.5, 97.5])]
    for m in models:
        bs = boot(D, lambda d: sr(d, m), rng)
        res["sharpe_" + m] = [sr(D, m), *np.percentile(bs, [2.5, 97.5])]
        res["turnover_" + m] = float(turnover(D, m).mean())
        res["sharpe_net_" + m] = {str(c): sr(D, m, c) for c in costs}
    for m in fitted:
        bs = boot(D, lambda d: sr(d, m) - sr(d, "hm"), rng)
        res[f"sharpe_{m}_minus_hm"] = [sr(D, m) - sr(D, "hm"), *np.percentile(bs, [2.5, 97.5])]
        cs = np.linspace(0, 2, 2001)       # break-even one-way cost (%) where net sharpe ties hist. mean
        gapc = np.array([sr(D, m, c) - sr(D, "hm", c) for c in cs])
        res[f"breakeven_cost_{m}"] = float(cs[np.argmax(gapc <= 0)]) if (gapc <= 0).any() else None
        res[f"forecast_sd_{m}"] = float(preds.groupby("port")[m].std().mean())
    res["forecast_sd_hm"] = float(preds.groupby("port")["hm"].std().mean())
    json.dump(res, open(args.out + "/results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

# ese 4020 baseline - ridge on lagged factor returns vs historical mean
# usage: python baseline.py --data data/ --out results/
import argparse, glob, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression

seed = 4020
nlags = 12
gap = 12
minStocks = 100
K = 5
alphas = np.logspace(0, 6, 25)
valStart, valEnd, testStart = "1990-01", "2009-12", "2010-01"
nboot, blk = 2000, 12


def load_data(path):
    if os.path.isdir(path):
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(path + "/*.csv"))])
        df["ym"] = pd.to_datetime(df.date).dt.to_period("M")
        bad = df[(df.Nlong < minStocks) | (df.Nshort < minStocks)]
        R = df.pivot(index="ym", columns="signalname", values="ret")
        first = R.dropna().index.min()
        dropped = sorted(set(bad.ym[bad.ym >= first].astype(str)))
    else:
        R = pd.read_parquet(path) if path.endswith("parquet") else pd.read_csv(path)
        R.index = pd.to_datetime(R.pop("date")).dt.to_period("M")
        dropped = []
    R = R.sort_index()
    R = R[~R.index.astype(str).isin(dropped)].dropna()
    return R, dropped


def scale(a, b):
    mu = a.mean(0)
    sd = a.std(0, ddof=0)
    return (a - mu) / sd, (b - mu) / sd


def pick_alpha(X, Y):
    # blocked kfold, contiguous chunks, drop gap months on both sides
    X = X.values; Y = Y.values
    n = len(X)
    err = np.zeros(len(alphas))
    e = np.linspace(0, n, K + 1).astype(int)
    for k in range(K):
        va = np.arange(e[k], e[k+1])
        tr = np.r_[np.arange(0, max(e[k] - gap, 0)), np.arange(min(e[k+1] + gap, n), n)]
        xtr, xva = scale(X[tr], X[va])
        for i, a in enumerate(alphas):
            m = Ridge(alpha=a).fit(xtr, Y[tr])
            err[i] += ((Y[va] - m.predict(xva))**2).sum()
    return alphas[err.argmin()]


def r2(d, m):
    return 1 - ((d["r"] - d[m])**2).sum() / ((d["r"] - d["hm"])**2).sum()


def sr(d, m):
    s = (np.sign(d[m]) * d["r"]).mean(1)
    return np.sqrt(12) * s.mean() / s.std(ddof=1)


def boot(d, f, rng):
    # circular block bootstrap over months
    T = d["r"].shape[0]
    out = np.empty(nboot)
    for b in range(nboot):
        st = rng.integers(0, T, size=int(np.ceil(T / blk)))
        idx = np.concatenate([(s + np.arange(blk)) % T for s in st])[:T]
        out[b] = f({k: v[idx] for k, v in d.items()})
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/")
ap.add_argument("--out", default="results/")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(seed)

R, dropped = load_data(args.data)
ports = list(R.columns)

# inputs at month t = returns at t-1 ... t-12 for every portfolio, target = return at t
X = pd.concat({l: R.shift(l) for l in range(1, nlags + 1)}, axis=1)
X.columns = [f"{s}_{l}" for l, s in X.columns]
keep = X.notna().all(axis=1)
X, Y = X[keep], R[keep]
lag1 = R.shift(1).loc[Y.index]
mom12 = R.shift(1).rolling(12).mean().loc[Y.index]

vm = Y.index[(Y.index >= pd.Period(valStart)) & (Y.index <= pd.Period(valEnd))]
assert vm.max() < pd.Period(testStart)   # never touch 2010+

rows = []
chosen = []
for yr in sorted(set(vm.year)):
    mths = vm[vm.year == yr]
    trainEnd = mths.min() - gap - 1
    tr = Y.index <= trainEnd
    Xtr, Ytr = X[tr], Y[tr]
    a = pick_alpha(Xtr, Ytr)
    chosen.append({"year": yr, "alpha": a, "n_train_months": int(tr.sum())})

    xs, xt = scale(Xtr.values, X.loc[mths].values)
    pRidge = Ridge(alpha=a).fit(xs, Ytr.values).predict(xt)
    pHM = np.tile(Ytr.mean().values, (len(mths), 1))
    pOLS = np.zeros_like(pHM)
    for j, p in enumerate(ports):
        z = np.c_[lag1.loc[Xtr.index, p], mom12.loc[Xtr.index, p]]
        zt = np.c_[lag1.loc[mths, p], mom12.loc[mths, p]]
        pOLS[:, j] = LinearRegression().fit(z, Ytr[p]).predict(zt)
    for i, t in enumerate(mths):
        for j, p in enumerate(ports):
            rows.append((str(t), p, Y.loc[t, p], pHM[i, j], pOLS[i, j], pRidge[i, j]))

preds = pd.DataFrame(rows, columns=["month", "port", "r", "hm", "ols", "ridge"])
chosen = pd.DataFrame(chosen)
preds.to_csv(args.out + "/val_predictions.csv", index=False)
chosen.to_csv(args.out + "/chosen_alpha.csv", index=False)

D = {c: preds.pivot(index="month", columns="port", values=c).values for c in ["r", "hm", "ols", "ridge"]}
res = {"panel_months": [str(R.index.min()), str(R.index.max()), len(R)],
       "dropped_thin_months": dropped, "portfolios": ports, "n_inputs": X.shape[1],
       "target_months_total": len(Y),
       "dev_target_months": int((Y.index < pd.Period(testStart)).sum()),
       "test_target_months_untouched": int((Y.index >= pd.Period(testStart)).sum()),
       "val_months": preds.month.nunique(), "val_forecasts": len(preds)}

for m in ["ols", "ridge"]:
    bs = boot(D, lambda d: r2(d, m), rng)
    res["r2os_" + m] = [r2(D, m), *np.percentile(bs, [2.5, 97.5]), bs.std(ddof=1)]
for m in ["hm", "ols", "ridge"]:
    bs = boot(D, lambda d: sr(d, m), rng)
    res["sharpe_" + m] = [sr(D, m), *np.percentile(bs, [2.5, 97.5])]
bs = boot(D, lambda d: sr(d, "ridge") - sr(d, "hm"), rng)
res["sharpe_ridge_minus_hm"] = [sr(D, "ridge") - sr(D, "hm"), *np.percentile(bs, [2.5, 97.5])]

res["per_portfolio_r2os"] = {}
for j, p in enumerate(ports):
    Dj = {k: v[:, [j]] for k, v in D.items()}
    res["per_portfolio_r2os"][p] = {m: [r2(Dj, m), *np.percentile(boot(Dj, lambda x: r2(x, m), rng), [2.5, 97.5])]
                                    for m in ["ols", "ridge"]}

res["forecast_sd_pct"] = {m: float(preds.groupby("port")[m].std().mean()) for m in ["hm", "ols", "ridge"]}
res["alpha_counts"] = chosen.alpha.round(2).value_counts().sort_index().to_dict()
res["alpha_at_grid_edge_years"] = int(chosen.alpha.isin([alphas.min(), alphas.max()]).sum())
json.dump(res, open(args.out + "/results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))

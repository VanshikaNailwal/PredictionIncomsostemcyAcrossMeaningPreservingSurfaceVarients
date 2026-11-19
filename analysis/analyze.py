"""Flip-rate analysis of the 9 Colab runs (3 models x 3 seeds). Writes CSVs, PNGs and README.md into analysis/.

Inputs (read only): results_colab/flips_long.csv, results_colab/preds/*.csv, results_colab/run_log.csv,
                    results/tokenizer_stats.csv, output_FROZEN/variants_long.csv (n_changed_words, A text)
Run from the project root: python analysis/analyze.py
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts_train"))
from train_eval import MODELS, SEEDS, macro_f1, preds_path, short  # noqa: E402  same metric as model selection

COLAB, OUT = ROOT / "results_colab", ROOT / "analysis"
SEED, N_BOOT = 2026, 2000
VARIANTS = ["T1", "T2", "T3", "LC", "NE", "UC", "D", "NEG"]
TYPOS = ["T1", "T2", "T3"]
COLORS = dict(zip(MODELS, ["#4C72B0", "#DD8452", "#55A868"]))
COVARIATES = ["delta_fertility", "char_edit_dist_ci", "n_changed_words", "n_words_A", "unk_rate", "margin_A"]
REF_MODEL = MODELS[0]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})


def read(path):
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False, na_values=[""])
    print(f"{path.relative_to(ROOT)}: {len(df)} rows, columns {list(df.columns)}")
    return df


def cluster_boot(df, stat, n_boot=N_BOOT):
    """Estimate and 95% percentile CI of stat(column sums), resampling sentence ids with all their rows.

    df: column "id" plus numeric columns; stat: dict of column sums -> value (works on arrays too)."""
    g = df.groupby("id").sum(numeric_only=True)
    idx = np.random.default_rng(SEED).integers(0, len(g), (n_boot, len(g)))
    boot = {c: g[c].to_numpy(float)[idx].sum(1) for c in g.columns}
    with np.errstate(invalid="ignore", divide="ignore"):
        est = stat({c: float(g[c].sum()) for c in g.columns})
        lo, hi = np.nanpercentile(stat(boot), [2.5, 97.5])
    return est, lo, hi


def rate_ci(df, num, den="n"):
    return cluster_boot(df[["id", num, den]], lambda s: s[num] / s[den])


def fmt(est, lo, hi, pct=True):
    return f"{est:.1%} [{lo:.1%}, {hi:.1%}]" if pct else f"{est:.2f} [{lo:.2f}, {hi:.2f}]"


def md_table(df, floatfmt="{:.3f}"):
    cell = lambda x: floatfmt.format(x) if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def self_test():
    ones = pd.DataFrame({"id": list("abcd"), "x": 1, "n": 1})
    assert rate_ci(ones, "x") == (1.0, 1.0, 1.0)
    half = pd.DataFrame({"id": list("aabb"), "x": [1, 0, 1, 0], "n": 1})
    assert rate_ci(half, "x") == (0.5, 0.5, 0.5)  # every id has rate 0.5, so every resample does too
    # McNemar: 10 discordant pairs all one way -> exact p = 2 * 0.5**10
    assert abs(mcnemar([[5, 10], [0, 5]], exact=True).pvalue - 2 * 0.5 ** 10) < 1e-12


def load():
    flips = read(COLAB / "flips_long.csv")
    preds = pd.concat([read(preds_path(COLAB, m, s)) for m in MODELS for s in SEEDS], ignore_index=True)
    runs = read(COLAB / "run_log.csv")
    ts = read(ROOT / "results" / "tokenizer_stats.csv")
    vl = read(ROOT / "output_FROZEN" / "variants_long.csv")

    # consistency checks between the inputs
    assert flips.groupby(["model", "seed"]).size().eq(8352).all() and len(flips) == 9 * 8352
    assert (flips.flip == (flips.pred != flips.pred_A)).all()
    assert (flips.strict_flip == (flips.flip & (flips.pred_A == flips.label))).all()
    chk = flips.merge(preds[["model", "seed", "id", "variant", "pred"]], on=["model", "seed", "id", "variant"],
                      suffixes=("", "_p"), validate="one_to_one")
    assert len(chk) == len(flips) and (chk.pred == chk.pred_p).all(), "flips_long disagrees with preds files"

    f = flips.copy()
    f["flip"] = f.flip.astype(int)
    f["strict_flip"] = f.strict_flip.astype(int)
    f["a_ok"] = (f.pred_A == f.label).astype(int)
    f["n"] = 1
    a_text = vl[vl.variant == "A"].set_index("id").text
    # complete_tweet as stored counts a typed "..." as truncation; the alternative only counts the final "…"
    f["complete_fixed"] = f.id.map(lambda i: int(not a_text[i].rstrip().endswith("…")))
    return f, preds, runs, ts, vl


def flip_rates(f):
    rows = []
    for (m, v), d in f[f.variant != "A"].groupby(["model", "variant"]):
        fl, st = rate_ci(d, "flip"), rate_ci(d, "strict_flip", "a_ok")
        rows.append({"model": m, "variant": v, "n_rows": len(d), "n_ids": d.id.nunique(),
                     "flip_rate": fl[0], "flip_lo": fl[1], "flip_hi": fl[2],
                     "n_A_correct": int(d.a_ok.sum()), "strict_rate": st[0], "strict_lo": st[1], "strict_hi": st[2]})
    return order(pd.DataFrame(rows))


def order(df):
    df = df.assign(_m=df.model.map(MODELS.index), _v=df.variant.map((["A"] + VARIANTS).index))
    return df.sort_values(["_m", "_v"]).drop(columns=["_m", "_v"]).reset_index(drop=True)


def seed_noise(f):
    w = f.pivot_table(index=["model", "variant", "id"], columns="seed", values="pred").reset_index()
    pairs = list(combinations(SEEDS, 2))
    for s1, s2 in pairs:
        w[f"{s1}_vs_{s2}"] = (w[s1] != w[s2]).astype(int)
    w["dis"] = w[[f"{a}_vs_{b}" for a, b in pairs]].mean(1)
    w["n"] = 1
    rows = []
    for (m, v), d in w.groupby(["model", "variant"]):
        est = rate_ci(d, "dis")
        rows.append({"model": m, "variant": v, "seed_disagreement": est[0], "lo": est[1], "hi": est[2],
                     **{f"{a}_vs_{b}": d[f"{a}_vs_{b}"].mean() for a, b in pairs}})
    return order(pd.DataFrame(rows))


def a_performance(f, runs):
    rows = []
    for (m, s), d in f[f.variant == "A"].groupby(["model", "seed"]):
        r = runs[(runs.model == m) & (runs.seed == s)].iloc[0]
        rows.append({"model": m, "seed": s, "A_accuracy": (d.pred == d.label).mean(),
                     "A_macro_f1": macro_f1(d.label, d.pred), "val_macro_f1": r.val_macro_f1,
                     "best_epoch": int(r.best_epoch), "train_seconds": int(r.seconds)})
    df = pd.DataFrame(rows)
    return df.assign(_m=df.model.map(MODELS.index)).sort_values(["_m", "seed"]).drop(columns="_m")


def neg_vs_typos(f):
    neg_ids = set(f[f.variant == "NEG"].id)
    m = f[f.id.isin(neg_ids)]
    rows = []
    for model in MODELS:
        neg = m[(m.model == model) & (m.variant == "NEG")].set_index(["seed", "id"])
        for t in TYPOS:
            typ = m[(m.model == model) & (m.variant == t)].set_index(["seed", "id"])
            p = neg[["flip", "a_ok"]].join(typ[["flip"]], rsuffix="_t", how="inner").reset_index()
            assert len(p) == 3 * len(neg_ids)
            for outcome in ["flip", "strict_flip"]:
                q = p if outcome == "flip" else p[p.a_ok == 1]  # strict: pairs where the A prediction was right
                x, y = q.flip.astype(bool), q.flip_t.astype(bool)
                table = [[int((x & y).sum()), int((x & ~y).sum())], [int((~x & y).sum()), int((~x & ~y).sum())]]
                diff = cluster_boot(q.assign(n=1)[["id", "flip", "flip_t", "n"]],
                                    lambda s: (s["flip"] - s["flip_t"]) / s["n"])
                rows.append({"model": model, "typo": t, "outcome": outcome, "pairs": len(q),
                             "rate_NEG": x.mean(), "rate_typo": y.mean(), "diff_NEG_minus_typo": diff[0],
                             "diff_lo": diff[1], "diff_hi": diff[2], "NEG_only": table[0][1],
                             "typo_only": table[1][0], "mcnemar_p": mcnemar(table, exact=True).pvalue})
    df = pd.DataFrame(rows)
    df["p_holm"] = multipletests(df.mcnemar_p, method="holm")[1]
    df["significant_holm_0.05"] = df.p_holm < 0.05
    return df


def neg_by_label(f):
    neg = f[f.variant == "NEG"]
    labels = ["negative", "neutral", "positive"]
    one = neg[(neg.model == MODELS[0]) & (neg.seed == SEEDS[0])]
    share = {labels[k]: (one.label == k).mean() for k in range(3)}
    rows = []
    for m in MODELS:
        for k, lab in enumerate(labels):
            d = neg[(neg.model == m) & (neg.label == k)]
            fl = rate_ci(d, "flip")
            rows.append({"model": m, "true_label": lab, "n_ids": d.id.nunique(), "share_of_NEG_ids": share[lab],
                         "flip_rate": fl[0], "lo": fl[1], "hi": fl[2]})
    return pd.DataFrame(rows)


def regression_frame(f, ts, vl):
    tok = ts[["model", "id", "variant", "delta_fertility", "char_edit_dist_ci", "n_words_A", "unk_rate"]]
    # n_changed_words from variants_long: tokenizer_stats has NaN for the 174 NE ids whose word count changed
    nc = vl[["id", "variant", "n_changed_words"]]
    d = f.merge(tok, on=["model", "id", "variant"], how="left", validate="many_to_one")
    d = d.merge(nc, on=["id", "variant"], how="left", validate="many_to_one")
    return d


def fit_gee(d, name, covariates=COVARIATES):
    d = d.sort_values("id").reset_index(drop=True).copy()
    missing = d[covariates + ["complete_tweet"]].isna().sum()
    assert not missing.any(), f"{name}: missing covariates {missing[missing > 0].to_dict()}"
    sd = {c: d[c].std() for c in covariates}
    for c in covariates:  # z-scores, so odds ratios are per 1 SD within this subset
        d[c] = (d[c] - d[c].mean()) / sd[c] if sd[c] > 0 else 0.0
    formula = f"flip ~ {' + '.join(covariates)} + complete_tweet + C(model, Treatment('{REF_MODEL}'))"
    res = smf.gee(formula, groups="id", data=d, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = res.conf_int()
    ors = pd.DataFrame({"fit": name, "term": res.params.index, "odds_ratio": np.exp(res.params.values),
                        "lo": np.exp(ci[0].values), "hi": np.exp(ci[1].values), "p": res.pvalues.values,
                        "sd_of_raw_covariate": [sd.get(t, np.nan) for t in res.params.index]})
    ors = ors[ors.term != "Intercept"]
    X = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
    vif = pd.DataFrame({"fit": name, "term": X.columns[1:],
                        "VIF": [variance_inflation_factor(X.values, k) for k in range(1, X.shape[1])]})
    r = np.corrcoef(d.delta_fertility, d.char_edit_dist_ci)[0, 1] if "char_edit_dist_ci" in covariates else np.nan
    info = {"fit": name, "rows": len(d), "clusters": d.id.nunique(), "flip_rate": d.flip.mean(),
            "corr_delta_fertility_char_edit_dist_ci": r, "exchangeable_corr": res.model.cov_struct.dep_params}
    print(f"{name}: {info}")
    return ors, vif, info


def regressions(d):
    fits = {"a_typos": TYPOS, "b_typos_case_emoji": TYPOS + ["LC", "UC", "NE"], "c_D_only": ["D"]}
    ors, vifs, infos = [], [], []
    for name, vs in fits.items():
        o, v, i = fit_gee(d[d.variant.isin(vs)], name)
        ors.append(o), vifs.append(v), infos.append(i)
    # sensitivity: drop the covariate of the collinear pair (VIF > 10 in the main fit) that the other one covers
    for name, vs, drop in [("a_typos_without_char_edit_dist_ci", TYPOS, "char_edit_dist_ci"),
                           ("c_D_only_without_n_changed_words", ["D"], "n_changed_words")]:
        o, v, i = fit_gee(d[d.variant.isin(vs)], name, [c for c in COVARIATES if c != drop])
        ors.append(o), vifs.append(v), infos.append(i)
    return pd.concat(ors), pd.concat(vifs), pd.DataFrame(infos)


def subgroups(f):
    rows = []
    for col in ["complete_tweet", "complete_fixed"]:
        for (m, v, c), d in f[f.variant != "A"].groupby(["model", "variant", col]):
            fl = rate_ci(d, "flip")
            rows.append({"definition": col, "model": m, "variant": v, "complete": "yes" if c else "no",
                         "n_ids": d.id.nunique(), "flip_rate": fl[0], "lo": fl[1], "hi": fl[2]})
    df = pd.DataFrame(rows)
    df = df.assign(_m=df.model.map(MODELS.index), _v=df.variant.map(VARIANTS.index))
    return df.sort_values(["definition", "_m", "_v", "complete"]).drop(columns=["_m", "_v"]).reset_index(drop=True)


def bar_panel(ax, df, est, lo, hi, groups, title, floor=None):
    width = 0.8 / len(MODELS)
    x = np.arange(len(groups))
    for k, m in enumerate(MODELS):
        d = df[df.model == m].set_index(df.columns[1]).reindex(groups)
        ax.bar(x + (k - 1) * width, d[est], width, yerr=[d[est] - d[lo], d[hi] - d[est]], capsize=2,
               color=COLORS[m], label=short(m))
        if floor is not None:
            ax.hlines(floor[m], x[0] - 0.5, x[-1] + 0.5, colors=COLORS[m], linestyles=":", linewidth=1.5)
    ax.set_xticks(x, groups)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)


def figures(fr, noise, nvt, sub):
    floor = noise[noise.variant == "A"].set_index("model").seed_disagreement
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    bar_panel(axes[0], fr, "flip_rate", "flip_lo", "flip_hi", VARIANTS,
              "Flip rate vs A (dotted: seed disagreement on A, a different quantity)", floor)
    bar_panel(axes[1], fr, "strict_rate", "strict_lo", "strict_hi", VARIANTS, "Strict flip rate (A prediction correct)")
    axes[0].legend(frameon=False)
    fig.text(0.01, 0.01, "95% CIs: cluster bootstrap over sentence ids, 3 seeds pooled, 2,000 resamples", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "flip_rates.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, outcome in zip(axes, ["flip", "strict_flip"]):
        d = nvt[nvt.outcome == outcome].rename(columns={"typo": "group"})[["model", "group", "diff_NEG_minus_typo",
                                                                            "diff_lo", "diff_hi"]]
        bar_panel(ax, d, "diff_NEG_minus_typo", "diff_lo", "diff_hi", TYPOS,
                  f"NEG minus typo, {outcome.replace('_', ' ')} (352 ids)")
        ax.axhline(0, color="black", linewidth=0.8)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "neg_vs_typos.png")
    plt.close(fig)

    s = sub[sub.definition == "complete_tweet"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, m in zip(axes, MODELS):
        x = np.arange(len(VARIANTS))
        for k, (c, col) in enumerate([("yes", "#333333"), ("no", "#AAAAAA")]):
            d = s[(s.model == m) & (s.complete == c)].set_index("variant").reindex(VARIANTS)
            ax.bar(x + (k - 0.5) * 0.4, d.flip_rate, 0.4, yerr=[d.flip_rate - d.lo, d.hi - d.flip_rate],
                   capsize=2, color=col, label=f"complete tweet: {c}")
        ax.set_xticks(x, VARIANTS)
        ax.set_title(short(m))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "subgroups_complete_tweet.png")
    plt.close(fig)


def forest(ors):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (name, d) in zip(axes, ors.groupby("fit", sort=False)):
        y = np.arange(len(d))[::-1]
        ax.errorbar(d.odds_ratio, y, xerr=[d.odds_ratio - d.lo, d.hi - d.odds_ratio], fmt="o", color="#333333")
        ax.axvline(1, color="grey", linestyle="--")
        ax.set_xscale("log")
        ax.set_yticks(y, [t.replace(f"C(model, Treatment('{REF_MODEL}'))[T.", "model: ").rstrip("]") for t in d.term])
        ax.set_title(name)
        ax.set_xlabel("odds ratio (per SD; 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT / "regression_odds_ratios.png")
    plt.close(fig)


def readme(f, perf, fr, noise, nvt, nbl, ors, vif, info, sub):
    pct = lambda x: f"{x:.1%}"
    comb = fr.merge(noise[["model", "variant", "seed_disagreement", "lo", "hi"]], on=["model", "variant"])
    floor = noise[noise.variant == "A"].set_index("model").seed_disagreement
    t1 = pd.DataFrame({"model": comb.model.map(short), "variant": comb.variant,
                       "flip (same seed, variant vs A)": [fmt(*r) for r in zip(comb.flip_rate, comb.flip_lo, comb.flip_hi)],
                       "strict flip": [fmt(*r) for r in zip(comb.strict_rate, comb.strict_lo, comb.strict_hi)],
                       "seed disagreement on this variant": [fmt(*r) for r in zip(comb.seed_disagreement, comb.lo, comb.hi)],
                       "seed disagreement on A": comb.model.map(floor).map(pct)})
    t_perf = perf.assign(model=perf.model.map(short))
    t_nvt = pd.DataFrame({"model": nvt.model.map(short), "typo": nvt.typo, "outcome": nvt.outcome, "pairs": nvt.pairs,
                          "NEG": nvt.rate_NEG.map(pct), "typo rate": nvt.rate_typo.map(pct),
                          "NEG minus typo [95% CI]": [fmt(*r) for r in zip(nvt.diff_NEG_minus_typo, nvt.diff_lo, nvt.diff_hi)],
                          "McNemar p": nvt.mcnemar_p.map(lambda p: f"{p:.2g}"),
                          "Holm p": nvt.p_holm.map(lambda p: f"{p:.2g}")})
    t_nbl = pd.DataFrame({"model": nbl.model.map(short), "true label": nbl.true_label, "ids": nbl.n_ids,
                          "share of NEG ids": nbl.share_of_NEG_ids.map(pct),
                          "NEG flip rate": [fmt(*r) for r in zip(nbl.flip_rate, nbl.lo, nbl.hi)]})
    t_or = pd.DataFrame({"fit": ors.fit, "term": ors.term.str.replace(f"C(model, Treatment('{REF_MODEL}'))[T.", "model: ",
                                                                         regex=False).str.rstrip("]"),
                         "OR [95% CI]": [fmt(*r, pct=False) for r in zip(ors.odds_ratio, ors.lo, ors.hi)],
                         "p": ors.p.map(lambda p: f"{p:.2g}"), "SD of raw covariate": ors.sd_of_raw_covariate})
    t_vif = vif.pivot(index="term", columns="fit", values="VIF").reset_index()
    t_info = info.assign(exchangeable_corr=info.exchangeable_corr.map(lambda x: float(np.ravel(x)[0])))
    s = sub[sub.definition == "complete_tweet"]
    t_sub = s.pivot_table(index=["model", "variant"], columns="complete", values="flip_rate").reset_index()
    t_sub = t_sub.assign(model=t_sub.model.map(short))
    t_sub = t_sub.assign(_m=t_sub.model.map([short(m) for m in MODELS].index),
                         _v=t_sub.variant.map(VARIANTS.index)).sort_values(["_m", "_v"]).drop(columns=["_m", "_v"])
    s_ci = s.assign(ci=[fmt(*r) for r in zip(s.flip_rate, s.lo, s.hi)])
    t_sub_ci = s_ci.pivot_table(index=["model", "variant"], columns="complete", values="ci", aggfunc="first").reset_index()
    t_sub_ci = t_sub_ci.assign(model=t_sub_ci.model.map(short))
    t_sub_ci = t_sub_ci.assign(_m=t_sub_ci.model.map([short(m) for m in MODELS].index),
                               _v=t_sub_ci.variant.map(VARIANTS.index)).sort_values(["_m", "_v"]).drop(columns=["_m", "_v"])
    n_complete = f.drop_duplicates("id").complete_tweet.sum()
    n_fixed = f.drop_duplicates("id").complete_fixed.sum()

    text = f"""# Analysis of the 9 fine-tuning runs

Generated by `analysis/analyze.py`; every number below is recomputed on each run. Do not edit by hand.

## How to rerun

```
cd C:\\hinglish_thesis
python -m pip install "statsmodels==0.14.5" scipy matplotlib pandas numpy
python analysis\\analyze.py
```

## Data

- 3 models x 3 seeds (13, 42, 2026), each predicting 8 variants of 1,000 sentences + NEG for 352 sentences
  (8,352 rows per run, {len(f):,} rows in total).
- Flip: the prediction on the variant differs from the prediction on A, same model and seed.
- Strict flip rate: flips among rows whose A prediction was correct (denominator = rows with A correct).
- CIs: 95% percentile cluster bootstrap over sentence ids (all seeds and rows of an id resampled together),
  2,000 resamples, seed {SEED}.

## 1. Performance on A (per run)

{md_table(t_perf)}

## 2. Flip rates next to the seed noise floor

These are **different quantities**. *Flip* compares the variant with A **under the same model and seed**.
*Seed disagreement* compares **two seeds on the same text** (mean over the pairs 13/42, 13/2026, 42/2026).
Seed disagreement on A shows how much predictions move from retraining alone, with no change to the text.
It is a reference level, not a baseline that can be subtracted from the flip rate.

{md_table(t1)}

Figure: `flip_rates.png` (dotted lines = seed disagreement on A).

## 3. NEG vs typos on the same {f[f.variant == 'NEG'].id.nunique()} ids

Pairs = (seed, id), so each model has 3 x 352 = 1,056 pairs. For strict flip, only pairs whose A prediction
was correct are used. The p-values come from an exact McNemar test with Holm correction across all
{len(nvt)} comparisons. The CI of the difference comes from a paired cluster bootstrap over ids.
NEG changes the meaning, so a model should flip on NEG; a typo keeps the meaning, so it should not.

{md_table(t_nvt)}

True labels of the NEG sentences, and the NEG flip rate within each label:

{md_table(t_nbl)}

Figure: `neg_vs_typos.png`.

## 4. What predicts a flip (GEE logistic, exchangeable, cluster = sentence id)

`flip ~ delta_fertility + char_edit_dist_ci + n_changed_words + n_words_A + unk_rate + margin_A + complete_tweet + model`
(reference model: {short(REF_MODEL)}). The continuous covariates are z-scored within each fit, so odds
ratios are per 1 SD. That SD is listed, to convert back to raw units. D is fitted on its own.

{md_table(t_info, "{:.3f}")}

{md_table(t_or)}

VIF (variance inflation factor; above 5 means a covariate is largely explained by the others):

{md_table(t_vif, "{:.2f}")}

delta_fertility and char_edit_dist_ci are **not** too correlated to include together: their correlation is
at most {info.corr_delta_fertility_char_edit_dist_ci.abs().max():.2f}, and the VIF of delta_fertility stays
below 2. The collinear pairs are different ones. In T1-T3, char_edit_dist_ci and n_changed_words both have a
VIF above 18, because each typo changes about one character per word. In D, n_changed_words (17) and
n_words_A (13) are collinear, because almost every word changes. Their separate odds ratios are not
interpretable in those fits. The `*_without_*` fits drop one member of each pair; read those odds ratios
instead. The figure shows the three main fits only.

Figure: `regression_odds_ratios.png`.

## 5. Subgroups: complete vs truncated tweets

Uses `complete_tweet` as stored ({n_complete} of 1,000 ids complete). The stored flag treats any "..." as a
truncation. With only the final Twitter "…" counted, {n_fixed} ids would be complete. Both versions are in
`subgroups_complete_tweet.csv` (column `definition`).

{md_table(t_sub_ci)}

Figure: `subgroups_complete_tweet.png`.

## Files

`flip_rates.csv`, `seed_noise.csv`, `a_performance.csv`, `neg_vs_typos.csv`, `neg_by_label.csv`,
`regression_odds_ratios.csv`, `regression_vif.csv`, `regression_info.csv`, `subgroups_complete_tweet.csv`,
and the four PNGs.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    self_test()
    OUT.mkdir(exist_ok=True)
    f, preds, runs, ts, vl = load()

    perf = a_performance(f, runs)
    fr = flip_rates(f)
    noise = seed_noise(f)
    nvt = neg_vs_typos(f)
    nbl = neg_by_label(f)
    ors, vif, info = regressions(regression_frame(f, ts, vl))
    sub = subgroups(f)

    for name, df in [("a_performance", perf), ("flip_rates", fr), ("seed_noise", noise), ("neg_vs_typos", nvt),
                     ("neg_by_label", nbl), ("regression_odds_ratios", ors), ("regression_vif", vif),
                     ("regression_info", info), ("subgroups_complete_tweet", sub)]:
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    figures(fr, noise, nvt, sub)
    forest(ors)
    readme(f, perf, fr, noise, nvt, nbl, ors, vif, info, sub)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Build OUT/flips_long.csv from OUT/preds/*.csv: every variant's prediction compared with A's.

One row per (model, seed, id, variant), A rows included (flip is always False there):
  pred_A       prediction on the A text of the same model, seed and id
  flip         pred != pred_A
  strict_flip  flip and pred_A == label (the model was right on A)
  margin_A     top1 - top2 probability on A
  conf_shift   probability of pred_A's class under the variant minus under A (negative = less sure)
Run: python scripts_train/make_flips.py [--out_dir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_eval import MODELS, PROB_COLS, ROOT, SEEDS, preds_path

FLIP_COLS = ["model", "seed", "id", "variant", "label", "pred", "pred_A", "flip", "strict_flip", "margin_A",
             "conf_shift", "complete_tweet"]


def flips(preds):
    a = preds[preds.variant == "A"].set_index(["model", "seed", "id"])
    if a.index.duplicated().any():
        raise ValueError("more than one A row for a (model, seed, id)")
    key = pd.MultiIndex.from_frame(preds[["model", "seed", "id"]])
    a = a.reindex(key)
    if a.pred.isna().any():
        raise ValueError(f"{a.pred.isna().sum()} rows have no A row for their (model, seed, id)")
    df = preds.copy()
    df["pred_A"] = a.pred.astype(int).to_numpy()
    df["flip"] = df.pred != df.pred_A
    df["strict_flip"] = df.flip & (df.pred_A == df.label)
    df["margin_A"] = a.margin.to_numpy()
    rows = np.arange(len(df))
    df["conf_shift"] = df[PROB_COLS].to_numpy()[rows, df.pred_A] - a[PROB_COLS].to_numpy()[rows, df.pred_A]
    return df[FLIP_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    out = Path(ap.parse_args().out_dir)
    paths = [preds_path(out, m, s) for m in MODELS for s in SEEDS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} of {len(paths)} preds files missing: {missing}")
    preds = pd.concat([pd.read_csv(p, dtype={"id": str}) for p in paths if p.exists()], ignore_index=True)
    df = flips(preds)
    df.to_csv(out / "flips_long.csv", index=False, encoding="utf-8")
    print(f"wrote {out / 'flips_long.csv'}: {len(df)} rows")
    print(df[df.variant != "A"].groupby(["model", "variant"])[["flip", "strict_flip"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

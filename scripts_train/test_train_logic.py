"""Checks for the data and metric logic, without torch. Run: python scripts_train/test_train_logic.py"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from make_flips import flips
from train_eval import LABEL2ID, SRC, check_leakage, load_data, macro_f1, pred_frame


def test_label_map():
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}


def test_leakage_fires():
    df = lambda ids: pd.DataFrame({"id": ids})
    check_leakage(df(["1", "2"]), df(["3"]), ["4", "5"])  # no overlap: passes
    for train, val, ev in [(["1", "2"], ["2"], ["9"]), (["1"], ["2"], ["1"]), (["1"], ["2"], ["2"])]:
        try:
            check_leakage(df(train), df(val), ev)
        except ValueError as e:
            assert "overlap" in str(e)
        else:
            raise AssertionError(f"leakage not caught: {train} {val} {ev}")


def test_leakage_fires_in_load_data():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for f in ["val.csv", "variants_long.csv", "neg_control.csv"]:
            (d / f).write_bytes((SRC / f).read_bytes())
        eval_id = pd.read_csv(SRC / "variants_long.csv", dtype={"id": str}).id.iloc[0]
        pd.DataFrame({"id": ["x1", eval_id], "label": ["neutral", "positive"], "text": ["a b", "c d"]}).to_csv(
            d / "train.csv", index=False)
        try:
            load_data(d)
        except ValueError as e:
            assert eval_id in str(e)
        else:
            raise AssertionError("load_data did not stop on a train/eval overlap")


def test_macro_f1():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0
    # class 0: tp1 fp0 fn1 -> 2/3; class 1: tp1 fp1 fn0 -> 2/3; class 2: no rows -> 0
    assert abs(macro_f1([0, 0, 1], [0, 1, 1]) - (2 / 3 + 2 / 3 + 0) / 3) < 1e-12


def test_pred_frame():
    ev = pd.DataFrame({"id": ["1", "1"], "variant": ["A", "T1"], "y": [2, 2], "complete_tweet": [1, 1]})
    p = pred_frame(ev, [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]], [5, 6], "m", 13)
    assert p.pred.tolist() == [2, 0]
    assert np.allclose(p.confidence, [0.7, 0.5]) and np.allclose(p.margin, [0.5, 0.1])


def test_flips():
    # id 1: label pos, A right -> T1 flips (strict); id 2: label neg, A wrong (neu) -> T1 flips, not strict;
    # id 3: no change
    rows = [("1", "A", 2, 2, [.1, .2, .7]), ("1", "T1", 2, 0, [.6, .1, .3]),
            ("2", "A", 0, 1, [.3, .6, .1]), ("2", "T1", 0, 2, [.1, .4, .5]),
            ("3", "A", 1, 1, [.1, .8, .1]), ("3", "T1", 1, 1, [.1, .9, .0])]
    preds = pd.DataFrame([{"model": "m", "seed": 13, "id": i, "variant": v, "label": y, "pred": p,
                           "prob_neg": q[0], "prob_neu": q[1], "prob_pos": q[2],
                           "margin": sorted(q)[-1] - sorted(q)[-2], "complete_tweet": 1}
                          for i, v, y, p, q in rows])
    f = flips(preds).set_index(["id", "variant"])
    assert f.flip.tolist() == [False, True, False, True, False, False]
    assert f.strict_flip.tolist() == [False, True, False, False, False, False]
    assert f.pred_A.tolist() == [2, 2, 1, 1, 1, 1]
    assert np.allclose(f.conf_shift, [0, .3 - .7, 0, .4 - .6, 0, .9 - .8])
    assert np.allclose(f.margin_A, [.5, .5, .3, .3, .7, .7])


def test_real_eval_rows():
    train, val, ev = load_data()
    assert len(ev) == 8352, len(ev)
    assert ev.variant.value_counts().to_dict() == {**{v: 1000 for v in ["A", "T1", "T2", "T3", "LC", "NE", "UC", "D"]},
                                                  "NEG": 352}
    assert not ev.duplicated(["id", "variant"]).any()
    p = pred_frame(ev, np.full((len(ev), 3), 1 / 3), np.zeros(len(ev), int), "m", 13)
    assert len(p) == 8352 and not p.isna().any().any()
    assert len(flips(p)) == 8352


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")

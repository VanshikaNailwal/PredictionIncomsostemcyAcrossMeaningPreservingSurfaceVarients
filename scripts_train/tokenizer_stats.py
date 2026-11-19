"""Tokenizer statistics per (model, id, variant). Reads output_FROZEN/ only, writes results/.

Run from C:\\hinglish_thesis:  python scripts_train\\tokenizer_stats.py
"""
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")  # tokenizers only; also sidesteps the broken torch DLL in the global Python

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output_FROZEN"
OUT = ROOT / "results"
MODELS = ["bert-base-multilingual-cased", "xlm-roberta-base", "google/muril-base-cased"]
EXAMPLE_SEED = 2026


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "ab") == 2 and levenshtein("ab", "ab") == 0


def load():
    path = SRC / "variants_long.csv"
    long = pd.read_csv(path, dtype={"text": str}, keep_default_na=False)
    print(f"{path.name} columns: {list(long.columns)}")
    print(f"{path.name} variants:\n{long['variant'].value_counts().to_string()}")
    long = long[["id", "label", "variant", "text", "complete_tweet",
                 "n_changed_words", "char_edit_dist", "char_edit_dist_ci"]]

    neg_path = SRC / "neg_control.csv"
    if "NEG" not in set(long["variant"]) and neg_path.exists():
        neg = pd.read_csv(neg_path, dtype={"text_A": str, "text_NEG": str}, keep_default_na=False)
        print(f"NEG not in {path.name}; read from {neg_path.name} ({len(neg)} ids), columns: {list(neg.columns)}")
        a_text = long[long["variant"] == "A"].set_index("id")["text"]
        bad = (neg["text_A"] != neg["id"].map(a_text)).sum()
        print(f"  neg_control text_A mismatches vs variants_long A: {bad}")
        neg = neg.rename(columns={"text_NEG": "text"}).assign(variant="NEG")
        long = pd.concat([long, neg[["id", "label", "variant", "text", "complete_tweet"]]], ignore_index=True)
    print("complete_tweet: column 'complete_tweet' in variants_long.csv (and neg_control.csv for NEG)")
    return long


def text_stats(df):
    """Model-independent columns: word counts, edit distances, changed words."""
    a = df[df["variant"] == "A"].set_index("id")["text"]
    df["text_A"] = df["id"].map(a)
    words = df["text"].str.split()
    words_a = df["text_A"].str.split()
    df["n_words"] = words.str.len()
    df["n_words_A"] = words_a.str.len()
    df["char_edit_dist_new"] = [levenshtein(x, y) for x, y in zip(df["text_A"], df["text"])]
    df["char_edit_dist_ci_new"] = [levenshtein(x.lower(), y.lower()) for x, y in zip(df["text_A"], df["text"])]
    df["n_changed_words_new"] = [sum(p != q for p, q in zip(w, wa)) if len(w) == len(wa) else np.nan
                                 for w, wa in zip(words, words_a)]
    # cross-check against the values build_variants.py stored (NaN for NEG rows, which have none)
    for c in ["char_edit_dist", "char_edit_dist_ci", "n_changed_words"]:
        old = pd.to_numeric(df[c], errors="coerce")
        m = old.notna() & df[c + "_new"].notna() & (old != df[c + "_new"])
        print(f"  {c}: {m.sum()} rows differ from stored value")
        df[c] = df.pop(c + "_new")
    return df


def tok_stats(tok, texts):
    unk = tok.unk_token_id
    rows = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        words = t.split()
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        wids = [w for w in enc.word_ids() if w is not None]
        per_word = np.bincount(wids, minlength=len(words)) if words else np.array([])
        rows.append((len(ids), sum(i == unk for i in ids),
                     float((per_word >= 2).mean()) if len(words) else np.nan,
                     len(enc["input_ids"]) != len(ids)))
    return rows


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    base = text_stats(load())
    n_ids, n_var = base["id"].nunique(), base["variant"].nunique()

    frames, toks = [], {}
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m, use_fast=True)
        assert tok.is_fast, f"{m}: no fast tokenizer, word_ids unavailable"
        toks[m] = tok
        st = pd.DataFrame(tok_stats(tok, base["text"]), columns=["n_tokens", "unk_count", "frac_words_split", "_mismatch"])
        print(f"{m}: unk_token={tok.unk_token!r}; rows where word-split tokenization length "
              f"!= plain tokenization length: {st.pop('_mismatch').sum()}")
        frames.append(pd.concat([base.reset_index(drop=True), st], axis=1).assign(model=m))
    df = pd.concat(frames, ignore_index=True)

    df["unk_rate"] = df["unk_count"] / df["n_tokens"]
    df["fertility"] = df["n_tokens"] / df["n_words_A"]
    a = df[df["variant"] == "A"].set_index(["model", "id"])
    key = pd.MultiIndex.from_frame(df[["model", "id"]])
    df["n_tokens_A"] = a["n_tokens"].reindex(key).to_numpy()
    df["delta_tokens"] = df["n_tokens"] - df["n_tokens_A"]
    df["delta_fertility"] = df["fertility"] - a["fertility"].reindex(key).to_numpy()

    cols = ["model", "id", "variant", "label", "complete_tweet", "n_tokens", "n_words", "n_words_A",
            "fertility", "n_tokens_A", "delta_tokens", "delta_fertility", "unk_count", "unk_rate",
            "frac_words_split", "char_edit_dist", "char_edit_dist_ci", "n_changed_words"]
    df[cols].to_csv(OUT / "tokenizer_stats.csv", index=False)

    summ = df.groupby(["model", "variant"]).agg(
        count=("n_tokens", "size"),
        **{f"{c}_{s}": (c, s) for c in ["fertility", "delta_fertility", "unk_rate", "n_tokens"] for s in ["mean", "std"]})
    summ.to_csv(OUT / "tokenizer_summary.csv")

    import platform, tokenizers, transformers
    (OUT / "tokenizer_versions.txt").write_text(
        f"python {platform.python_version()}\ntransformers {transformers.__version__}\n"
        f"tokenizers {tokenizers.__version__}\npandas {pd.__version__}\n"
        + "".join(f"{m} vocab_size={t.vocab_size} class={type(t).__name__}\n" for m, t in toks.items()),
        encoding="utf-8")

    ex_ids = base.loc[base["variant"] == "A", "id"].sample(3, random_state=EXAMPLE_SEED).tolist()
    with open(OUT / "tokenizer_examples.txt", "w", encoding="utf-8") as f:
        for i in ex_ids:
            f.write(f"=== id {i} ===\n")
            for v in ["A", "T3", "D"]:
                t = base.loc[(base["id"] == i) & (base["variant"] == v), "text"].iloc[0]
                f.write(f"\n[{v}] {t}\n")
                for m, tok in toks.items():
                    pieces = tok.tokenize(t)
                    f.write(f"  {m} ({len(pieces)}): {pieces}\n")
            f.write("\n")

    print("\n--- CHECKS ---")
    expected = len(MODELS) * len(base)
    print(f"1. rows={len(df)}; 3 models x {n_ids} ids x {n_var} variants = {len(MODELS) * n_ids * n_var}; "
          f"expected from present (id, variant) pairs = {expected} -> {'OK' if len(df) == expected else 'MISMATCH'}")
    nans = df[["n_tokens", "fertility", "delta_fertility"]].isna().sum().to_dict()
    a_delta = df.loc[df["variant"] == "A", "delta_fertility"].abs().max()
    print(f"2. NaN counts: {nans}; max |delta_fertility| on A = {a_delta}")
    tbl = df[df["variant"].isin(["A", "T3", "D"])].pivot_table(index="model", columns="variant", values="fertility")
    print("3. mean fertility:\n" + tbl.round(3).to_string())
    for m, r in tbl.iterrows():
        if not r["T3"] > r["A"]:
            print(f"   NOTE {m}: T3 <= A")
    empty = df.loc[(df["text"].str.strip() == "") | (df["n_words_A"] == 0), ["id", "variant"]].drop_duplicates()
    print(f"4. empty text or n_words_A == 0: {len(empty)} pairs" + ("" if empty.empty else f"\n{empty.to_string()}"))
    print(f"   examples ids: {ex_ids}")
    print(f"5. runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

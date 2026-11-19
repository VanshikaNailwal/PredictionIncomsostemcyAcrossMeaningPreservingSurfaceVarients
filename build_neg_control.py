"""Meaning-changing control variant NEG (does not touch A, T1-T3, LC, NE, UC or D).

REMOVE: the sentence has exactly one negation (NEGATIONS words + contractions like "don ’ t"), and it is
        a NEGATIONS word -> delete it. Contractions are never deleted; 2+ negations -> left out.
INSERT: no negation at all (contractions included), but a Hin-tagged hai/h/hain right after a word -> insert "nahi" before
        the first one ("acha hai" -> "acha nahi hai"). Skipped if the word before is already a
        negation-like word (na, no, nai, ni, ...), a progressive/perfect participle (rha, gaya, ...)
        or a postposition/particle (ne, ko, se, ke, me, hi, bhi, ...), or if the sentence has any short
        negation spelling (nh, nhe, nahe, nai, ni, nah, na, no), or if the word before is a question word
        (kya, kaun, kahan, kha, ...) or hai is within 2 words of a "?".
Sentences that qualify for neither are left out.

Input:  output/base_1000.csv, output/variants_long.csv   (or $OUT_DIR instead of output/)
Output: output/neg_control.csv
        output/neg_validation_sample.csv  (40 sentences, A/NEG order shuffled, for blind rating)
        output/neg_validation_key.csv     (which of text_1/text_2 is NEG)
Run: python build_neg_control.py   (after build_variants.py)

Later analysis (not run here): compare the model flip rate on NEG with T1/T2/T3 on the same
control-set ids, and report the share of NEG flips that are strict positive <-> negative.
"""
import json
import random
from collections import Counter
import re

import pandas as pd

from build_variants import OUT, SEED, check  # same output folder, seed and loud checks as script 2

NEGATIONS = {"nahi", "nahin", "nhi", "nahii", "nhin", "mat", "not", "never"}  # unambiguous only
COPULAS = {"hai", "h", "hain"}  # not "he": in this data it is often "ही" (only/even)
# "ne nahi hai", "ko nahi hai": a postposition or particle right before hai means hai is not a plain copula slot
POSTPOSITIONS = {"ne", "ko", "se", "ka", "ki", "ke", "me", "mein", "mai", "pe", "par", "tak", "hi", "bhi", "to", "k"}
NEG_LIKE = NEGATIONS | {"na", "no", "nai", "ni", "nah", "nope"}
# INSERT only: a sentence with any of these short spellings may already be negated, so "nahi" is not inserted
SHORT_NEG = {"nh", "nhe", "nahe", "nai", "ni", "nah", "na", "no"}
# "vo kha hai" (where is he) -> "vo kha nahi hai" is a strange question, not a clean negation
QUESTION_WORDS = {"kya", "kaun", "kon", "kab", "kaise", "kese", "kyun", "kyu", "kyon", "kahan", "kaha", "kha",
                  "kidhar", "kitna", "kitne", "kitni", "kis", "kisne", "kiska"}
# Progressive/perfect participles: "ho rha hai" -> "ho rha nahi hai" is ungrammatical (nahi goes before the verb)
PARTICIPLE_RE = re.compile(r"(r[a]?h[aei]|rhe|gay[aie]|gy[aie]|gai|chuk[aie]|liya|diya|kiya|kia|hu[aie]|gya)")
# Contracted English negations count as negation but are never deleted: "don ’ t", "do n ’ t", "don't",
# and the apostrophe-less "dont"/"cant" spellings the tweets also use.
CONTRACTION_RE = re.compile(
    r"\b(?:don|doesn|didn|can|won|isn|couldn|wouldn|shouldn|aren|wasn)\s*['’]?\s*t\b"  # don ’ t, don't, dont
    r"|\b(?:do|does|did|ca|wo|is|could|would|should|are|was)\s+n\s*['’]\s*t\b",       # do n ’ t
    re.IGNORECASE)
INSERTED = "nahi"
N_SAMPLE = 40


def neg_op(tokens):
    """Return ((op, position, changed_word, new tokens), None) if the sentence qualifies,
    else (None, reason it was left out)."""
    words = [t for t, _ in tokens]
    neg_at = [i for i, w in enumerate(words) if w.lower() in NEGATIONS]
    n_contractions = len(CONTRACTION_RE.findall(" ".join(words)))
    if neg_at or n_contractions:
        if len(neg_at) + n_contractions > 1:
            return None, "remove: 2+ negations (incl. contractions)"
        if not neg_at:
            return None, "contraction is the only negation (no remove, no insert)"
        i = neg_at[0]
        return ("remove", i, words[i], words[:i] + words[i + 1:]), None
    for i, (w, tag) in enumerate(tokens):
        # "preceded by a word": the token right before must contain a letter (not ".", an emoji, ...)
        if i > 0 and tag == "Hin" and w.lower() in COPULAS and re.search(r"[A-Za-z]", words[i - 1]):
            prev = words[i - 1].lower()
            if any(w.lower() in SHORT_NEG for w in words):
                return None, "insert: sentence already has a short negation spelling"  # "pr nh utar ai nahi h"
            if prev in NEG_LIKE:
                return None, "insert: word before hai is negation-like"  # "na nahi hai"
            if PARTICIPLE_RE.fullmatch(prev):
                return None, "insert: word before hai is a participle"  # "ho rha nahi hai"
            if prev in POSTPOSITIONS:
                return None, "insert: word before hai is a postposition"  # "desh ke nahi hai"
            if prev in QUESTION_WORDS:
                return None, "insert: word before hai is a question word"  # "vo kha nahi hai"
            if any("?" in words[j] for j in range(max(0, i - 2), min(len(words), i + 3))):
                return None, "insert: hai is within 2 words of '?'"  # "kya problem nahi hai ?"
            return ("insert", i, INSERTED, words[:i] + [INSERTED] + words[i:]), None
    return None, "no negation and no Hin hai/h/hain after a word"


def main():
    base = pd.read_csv(OUT / "base_1000.csv", dtype={"id": str}, encoding="utf-8")
    rows, left_out = [], Counter()
    for r in base.itertuples():
        res, reason = neg_op(json.loads(r.tokens_json))
        if reason:
            left_out[reason] += 1
        else:
            op, pos, word, new = res
            text = " ".join(new)
            rows.append({"id": r.id, "label": r.label, "text_A": r.text, "text_NEG": text,
                         "neg_op": op, "changed_word": word, "position": pos,
                         "complete_tweet": int("…" not in text and "..." not in text),
                         "hin_share": r.hin_share})
    df = pd.DataFrame(rows)

    print("control set:")
    print(f"   sentences: {len(df)} of {len(base)}  by op: {df.neg_op.value_counts().to_dict()}")
    print(pd.crosstab(df.label, df.neg_op, margins=True).to_string())
    print("left out of the control set, by rule:")
    for reason, n in left_out.most_common():
        print(f"   {n:4d}  {reason}")
    if len(df) < 300:
        print(f"   NOTE: fewer than 300 sentences qualify ({len(df)}): "
              f"REMOVE gives {(df.neg_op == 'remove').sum()}, INSERT gives {(df.neg_op == 'insert').sum()}")

    print("checks:")
    a, n = df.text_A.str.split(), df.text_NEG.str.split()
    check(all(abs(len(x) - len(y)) == 1 for x, y in zip(a, n)), "word count differs from A by exactly 1 in every row")

    def only_change(x, y, op, pos):
        return (x[:pos] + x[pos + 1:] == y) if op == "remove" else (y[:pos] + y[pos + 1:] == x and y[pos] == INSERTED)

    check(all(only_change(x, y, op, p) for x, y, op, p in zip(a, n, df.neg_op, df.position)),
          "no other word changed (only the removed / inserted word differs)")
    check(all(x[p].lower() in NEGATIONS for x, op, p in zip(a, df.neg_op, df.position) if op == "remove"),
          "every removed word is on the negation list")
    n_neg = lambda text: sum(w.lower() in NEGATIONS for w in text.split()) + len(CONTRACTION_RE.findall(text))
    check(all(n_neg(t) == 1 for t, op in zip(df.text_A, df.neg_op) if op == "remove"),
          "REMOVE rows have exactly one negation in A (contractions counted)")
    check(all(n_neg(t) == 0 for t in df.text_NEG[df.neg_op == "remove"]), "REMOVE rows have no negation left in NEG")
    check(all(n_neg(t) == 0 for t in df.text_A[df.neg_op == "insert"]), "INSERT rows have no negation in A (contractions counted)")

    print("10 random INSERT examples:")
    ins = df[df.neg_op == "insert"]
    for _, e in ins.sample(min(10, len(ins)), random_state=SEED).iterrows():
        print(f"  id {e.id} ({e.label}, {e.neg_op} '{e.changed_word}' at {e.position})")
        print(f"    A    {e.text_A[:150]}")
        print(f"    NEG  {e.text_NEG[:150]}")

    long = pd.read_csv(OUT / "variants_long.csv", dtype={"id": str}, encoding="utf-8")
    t = long[long.id.isin(df.id) & long.variant.isin(["T1", "T2", "T3"])]
    print("typo variants on the same control-set ids (for the matched comparison later):")
    print(t.groupby("variant").agg(sentences=("id", "nunique"), words_changed=("n_changed_words", "sum"),
                                   mean_per_sentence=("n_changed_words", "mean"),
                                   unchanged_sentences=("n_changed_words", lambda s: (s == 0).sum())).round(2).to_string())

    # Blind validation sample: balanced REMOVE/INSERT, A/NEG order shuffled per row
    per_op = N_SAMPLE // 2
    ins, rem = df[df.neg_op == "insert"], df[df.neg_op == "remove"]
    k_ins = min(len(ins), max(per_op, N_SAMPLE - len(rem)))
    sample = pd.concat([ins.sample(k_ins, random_state=SEED),
                        rem.sample(min(len(rem), N_SAMPLE - k_ins), random_state=SEED)]).sample(frac=1, random_state=SEED)
    rng = random.Random(SEED)
    val, key = [], []
    for _, s in sample.iterrows():
        neg_first = rng.random() < 0.5
        t1, t2 = (s.text_NEG, s.text_A) if neg_first else (s.text_A, s.text_NEG)
        val.append({"id": s.id, "text_1": t1, "text_2": t2, "rater_sentiment_1": "", "rater_sentiment_2": "",
                    "sounds_natural_1": "", "sounds_natural_2": ""})  # per version, so NEG stays hidden
        key.append({"id": s.id, "neg_is": "text_1" if neg_first else "text_2", "neg_op": s.neg_op})
    check(len(val) == min(N_SAMPLE, len(df)), f"validation sample has {len(val)} rows {sample.neg_op.value_counts().to_dict()}")

    # Files are written only after every check above has passed
    df.to_csv(OUT / "neg_control.csv", index=False, encoding="utf-8")
    pd.DataFrame(val).to_csv(OUT / "neg_validation_sample.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(OUT / "neg_validation_key.csv", index=False, encoding="utf-8")
    print(f"wrote neg_control.csv ({len(df)} rows), neg_validation_sample.csv ({len(val)} rows), neg_validation_key.csv")


if __name__ == "__main__":
    main()

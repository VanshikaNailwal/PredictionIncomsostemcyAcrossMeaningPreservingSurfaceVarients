"""Script 2: build meaning-preserving variants of the base set.

Variants per base sentence (id and label kept):
  A      original text
  T1-T3  nested typo noise on 10% / 25% / 40% of editable words
  LC     lowercased
  NE     emojis removed
  UC     25% of editable words fully uppercased (independent of T1-T3)
  D      Devanagari, merged in from variants_D.csv (built by transliterate_devanagari.py)

Input:  output/base_1000.csv, output/variants_D.csv   (or $OUT_DIR instead of output/)
Output: output/variants_long.csv (one row per sentence and variant)
        output/variants_wide.csv (one row per sentence, one text column per variant)
Run: python build_variants.py   (after transliterate_devanagari.py build)
"""
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 2026
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent / "output"))
LEVELS = {"T1": 0.10, "T2": 0.25, "T3": 0.40}
UC_RATE = 0.25
VARIANTS = ["A", "T1", "T2", "T3", "LC", "NE", "UC", "D"]
NEGATIONS = {"nahi", "nahin", "nhi", "na", "mat", "not", "no", "never", "nai", "bina", "kabhi", "without"}
# Emoji / pictograph blocks, plus joiners and variation selectors that ride along with emojis
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍⃣⌀-⏿]")
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
VOWELS = set("aeiouAEIOU")
EDIT_TYPES = ["swap", "delete", "duplicate", "qwerty", "vowel_drop"]  # chosen by v in equal fifths


def qwerty_neighbours():
    """letter -> list of keys adjacent to it on a QWERTY keyboard (same row or the row above/below)."""
    pos = {c: (r, i) for r, row in enumerate(QWERTY) for i, c in enumerate(row)}
    return {c: [d for d, (r2, i2) in pos.items() if d != c and abs(r - r2) <= 1 and abs(i - i2) <= 1]
            for c, (r, i) in pos.items()}


NEIGHBOURS = qwerty_neighbours()


def rand(key, n=4):
    """n fixed numbers in [0, 1) from SHA-256 of `key`."""
    h = hashlib.sha256(key.encode()).digest()
    return [int.from_bytes(h[k:k + 8], "big") / 2**64 for k in range(0, 8 * n, 8)]


def rand4(sid, idx):
    """u, v, p, x for (sentence id, word index), from SHA-256 of "id|idx|2026".

    u decides IF the word is edited, v the edit type, p the position, x the replacement letter.
    Because they don't depend on the level, a word edited at T1 (u < 0.10) is also
    edited, in exactly the same way, at T2 and T3.
    """
    return rand(f"{sid}|{idx}|{SEED}")


def editable(tok, tag):
    """Only plain Hin/Eng words of 3+ ASCII letters that are not negations.

    isalpha() already excludes hashtags, emojis, numbers and punctuation.
    """
    return (tag in ("Hin", "Eng") and tok.isascii() and tok.isalpha()
            and len(tok) >= 3 and tok.lower() not in NEGATIONS)


def typo(word, v, p, x):
    """Apply one typo; return (new word, edit type actually applied).

    v picks the type (5 equal bins), p the position, x the replacement letter.
    Swaps of identical letters and words without a droppable vowel fall back to delete,
    so every selected word really changes.
    """
    n = len(word)
    kind = EDIT_TYPES[int(v * 5)]
    if kind == "swap":
        i = int(p * (n - 1))
        out = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if out != word:
            return out, "swap"
        kind = "swap->delete"
    if kind == "vowel_drop":
        # a vowel that is not the first letter; words have 3+ letters, so 2+ remain
        spots = [i for i in range(1, n) if word[i] in VOWELS]
        if spots:
            i = spots[int(p * len(spots))]
            return word[:i] + word[i + 1:], "vowel_drop"
        kind = "vowel_drop->delete"
    i = int(p * n)
    c = word[i]
    if kind == "duplicate":
        return word[:i] + c + word[i:], kind
    if kind == "qwerty":  # QWERTY neighbour, case preserved
        nb = NEIGHBOURS[c.lower()]
        r = nb[int(x * len(nb))]
        return word[:i] + (r.upper() if c.isupper() else r) + word[i + 1:], kind
    return word[:i] + word[i + 1:], kind  # delete (also the fallbacks)


def uc_selected(sid, idx, tok, tag):
    """UC: independent draw from "id|idx|UC|2026"; words already all uppercase are left alone."""
    return editable(tok, tag) and not tok.isupper() and rand(f"{sid}|{idx}|UC|{SEED}", 1)[0] < UC_RATE


def levenshtein(a, b):
    """Character edit distance (insert / delete / substitute, each cost 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def variants(sid, tokens):
    """Return {variant: (tokens, changed word indexes, {index: edit type})} for one sentence."""
    toks = [t for t, _ in tokens]
    out = {"A": (toks, set(), {})}
    for name, level in LEVELS.items():
        new, kinds = list(toks), {}
        for i, (tok, tag) in enumerate(tokens):
            u, v, p, x = rand4(sid, i)
            if u < level and editable(tok, tag):
                new[i], kinds[i] = typo(tok, v, p, x)
        out[name] = (new, set(kinds), kinds)
    lc = [t.lower() for t in toks]
    out["LC"] = (lc, {i for i, (a, b) in enumerate(zip(toks, lc)) if a != b}, {})
    # NE: strip emoji characters; tokens that were only emojis disappear
    ne = [EMOJI_RE.sub("", t) for t in toks]
    out["NE"] = ([t for t in ne if t], {i for i, (a, b) in enumerate(zip(toks, ne)) if a != b}, {})
    uc = [t.upper() if uc_selected(sid, i, t, g) else t for i, (t, g) in enumerate(tokens)]
    out["UC"] = (uc, {i for i, (a, b) in enumerate(zip(toks, uc)) if a != b}, {})
    return out


def check(ok, msg):
    """Print the check result and stop the script on failure."""
    print(f"   [{'OK' if ok else 'FAIL'}] {msg}")
    if not ok:
        raise SystemExit(f"check failed: {msg}")


def self_test():
    """Small fixed examples so a change in the edit logic is caught immediately."""
    assert typo("acha", 0.0, 0.0, 0.0) == ("caha", "swap")
    assert typo("aab", 0.0, 0.0, 0.0) == ("ab", "swap->delete")
    assert typo("acha", 0.5, 0.0, 0.0) == ("aacha", "duplicate")
    assert typo("nahi", 0.9, 0.0, 0.0) == ("nhi", "vowel_drop")
    assert typo("please", 0.9, 0.5, 0.0) == ("plese", "vowel_drop")  # droppable: e, a, e -> the middle one
    assert typo("xyz", 0.9, 0.0, 0.0) == ("yz", "vowel_drop->delete")
    assert levenshtein("kitten", "sitting") == 3
    assert EMOJI_RE.sub("", "ok😂❤️") == "ok"


def main():
    self_test()
    base = pd.read_csv(OUT / "base_1000.csv", dtype={"id": str}, encoding="utf-8")
    rows, changed, kinds = [], {}, {name: Counter() for name in LEVELS}
    n_editable = n_uc_eligible = 0
    protected_ok = selected_changed = True
    for r in base.itertuples():
        tokens = json.loads(r.tokens_json)
        orig = " ".join(t for t, _ in tokens)
        assert orig == r.text  # tokens_json and text must agree
        n_editable += sum(editable(t, g) for t, g in tokens)
        n_uc_eligible += sum(editable(t, g) and not t.isupper() for t, g in tokens)
        for name, (toks, chg, knd) in variants(r.id, tokens).items():
            changed[(r.id, name)] = chg
            if name in kinds:
                kinds[name].update(knd.values())
                selected_changed &= all(toks[i] != tokens[i][0] for i in chg)
            if name in ("T1", "T2", "T3", "UC"):
                # protected words (hashtags, emojis, numbers, punctuation, <3 letters, negations) never change
                protected_ok &= all(toks[i] == t for i, (t, g) in enumerate(tokens) if not editable(t, g))
            text = " ".join(toks)
            rows.append({"id": r.id, "label": r.label, "variant": name, "text": text,
                         "n_changed_words": len(chg),
                         "char_edit_dist": levenshtein(orig, text),
                         "complete_tweet": int("…" not in text and "..." not in text),
                         "hin_share": r.hin_share})
    df = pd.DataFrame(rows)

    # D comes from transliterate_devanagari.py; only valid if it was built from the current A text
    d = pd.read_csv(OUT / "variants_D.csv", dtype={"id": str}, encoding="utf-8")
    a_text = dict(zip(base.id, base.text))
    check(len(d) == len(base) and all(a_text.get(i) == t for i, t in zip(d.id, d.roman_text)),
          "variants_D.csv was built from the current A text (else rerun transliterate_devanagari.py build)")
    df = pd.concat([df, d[df.columns]], ignore_index=True)
    # case-insensitive distance: separates real edits from pure case changes (LC, UC)
    df.insert(df.columns.get_loc("char_edit_dist") + 1, "char_edit_dist_ci",
              [levenshtein(a_text[i].lower(), t.lower()) for i, t in zip(df.id, df.text)])
    df["variant"] = pd.Categorical(df.variant, VARIANTS)
    df = df.sort_values(["variant"], kind="stable").reset_index(drop=True)
    df["variant"] = df.variant.astype(str)

    print("checks:")
    counts = df.variant.value_counts()
    check(all(counts.get(v) == len(base) for v in VARIANTS), f"1,000 rows per variant {counts.reindex(VARIANTS).to_dict()}")
    check((df.text.str.strip() != "").all(), "no empty text")
    a = df[df.variant == "A"].set_index("id").text
    words = lambda text: len(text.split())
    has_word = lambda text: sum(bool(re.search(r"[^\W_]", t)) for t in text.split())  # tokens with a letter/digit
    same_count = all(words(t) == words(a[i]) for v, i, t in zip(df.variant, df.id, df.text) if v != "NE")
    same_ne = all(has_word(t) == has_word(a[i]) for v, i, t in zip(df.variant, df.id, df.text) if v == "NE")
    check(same_count and same_ne, "word count unchanged vs A (NE: tokens with a letter or digit)")
    for name, level in LEVELS.items():
        share = sum(len(changed[(i, name)]) for i in base.id) / n_editable
        check(abs(share - level) <= 0.01, f"{name}: {share:.2%} of {n_editable} editable words changed (target {level:.0%} ± 1%)")
    uc_share = sum(len(changed[(i, 'UC')]) for i in base.id) / n_uc_eligible
    check(abs(uc_share - UC_RATE) <= 0.01, f"UC: {uc_share:.2%} of {n_uc_eligible} editable, not-all-caps words uppercased (target 25% ± 1%)")
    check(all(changed[(i, "T1")] <= changed[(i, "T2")] <= changed[(i, "T3")] for i in base.id),
          "T1 subset of T2 subset of T3 changed positions, all 1,000 sentences")
    check(selected_changed, "every word selected in T1, T2, T3 actually changed")
    check(protected_ok,"no protected or negation word changed in T1, T2, T3, UC")
    check(not df.text.str.contains("HTTPURL|@USER").any(), "no HTTPURL / @USER placeholders")
    check((df[df.variant.isin(["LC", "UC"])].char_edit_dist_ci == 0).all(), "LC and UC: case-insensitive edit distance is 0")

    print("edit types applied per level (fallbacks shown as type->delete):")
    print(pd.DataFrame(kinds).fillna(0).astype(int).reindex(
        EDIT_TYPES + ["swap->delete", "vowel_drop->delete"]).fillna(0).astype(int).to_string())

    df["identical_to_A"] = df.text.values == a.reindex(df.id).values
    summary = df.groupby("variant").agg(mean_changed_words=("n_changed_words", "mean"),
                                        mean_char_edit_dist=("char_edit_dist", "mean"),
                                        mean_char_edit_dist_ci=("char_edit_dist_ci", "mean"),
                                        n_identical_to_A=("identical_to_A", "sum"),
                                        complete_tweets=("complete_tweet", "sum"))
    print(summary.reindex(VARIANTS).round(2).to_string())

    wide = df.pivot(index=["id", "label"], columns="variant", values="text")[VARIANTS].reset_index()
    print("10 examples (A / T1 / T2 / T3 / UC):")
    for _, w in wide.sample(10, random_state=SEED).iterrows():
        print(f"  id {w.id} ({w.label})")
        for v in ["A", "T1", "T2", "T3", "UC"]:
            print(f"    {v:2s}  {w[v][:150]}")

    # Files are written only after every check above has passed
    df.drop(columns="identical_to_A").to_csv(OUT / "variants_long.csv", index=False, encoding="utf-8")
    wide.to_csv(OUT / "variants_wide.csv", index=False, encoding="utf-8")
    print(f"wrote variants_long.csv ({len(df)} rows) and variants_wide.csv ({len(wide)} rows)")


if __name__ == "__main__":
    main()

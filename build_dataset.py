"""Script 1: build the frozen 1,000-sentence base set and a leak-free train/val split
from the SemEval-2020 Task 9 Hinglish CoNLL file.

URLs and @mentions are removed from the text (not replaced by placeholders).

Outputs (in output/, or in $OUT_DIR if set): base_1000.csv, train.csv, val.csv, README.txt
Run: python build_dataset.py
"""
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 2026
ROOT = Path(__file__).parent
SRC = ROOT / "data" / "Hinglish_train_14k_split_conll.txt"
OUT = Path(os.environ.get("OUT_DIR", ROOT / "output"))
LABELS = ["negative", "neutral", "positive"]
BASE_PER_LABEL = {"negative": 334, "neutral": 333, "positive": 333}
JACCARD_MAX = 0.8

# An "alphabetic word": Latin letters, optionally with an apostrophe part (e.g. "I'm").
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Pieces a t.co URL is split into in this file: https // t . co / CODE  (or "tco", "t co")
# "c" and ".…" occur in URLs cut off by tweet truncation: "https // t . c …", "https // t .…"
URL_PIECES = {"//", "t", ".", "co", "tco", "tc", "/", "c", ".…"}
DROP_REASONS = ["devanagari", "nan_token", "under_3_words", "over_50_words", "duplicate"]


# 1. Parse: lines become a tweet
def parse(path):
    """Read the CoNLL file into [{id, label, tokens: [[token, tag], ...]}, ...]."""
    sents, cur = [], None
    # splitlines() handles the CRLF line endings
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        # Real meta lines have 3 fields; tokens such as "metaphysist\tEng" also start
        # with "meta" but have only 2 fields, so they are kept as tokens.
        if len(parts) == 3 and parts[0] == "meta":
            cur = {"id": parts[1], "label": parts[2], "tokens": []}
            sents.append(cur)
        elif len(parts) == 2 and cur is not None and parts[0].strip():
            cur["tokens"].append(parts)
        # blank lines separate tweets; blank tokens (3 in the file) are skipped
    return sents


def handle_end(tokens, i, expect_piece=True):
    """Index just past the pieces of a split handle or hashtag body starting at i.

    Pieces alternate: name piece, "_" run, name piece, ...  e.g. "abhisar _ sharma", "__ schowdhury __".
    expect_piece=False when the first name piece was already consumed.
    """
    # shortcut: a name piece after a trailing "_" is always absorbed, so a handle ending in "_" may eat the next word
    while i < len(tokens):
        t = tokens[i][0]
        if re.fullmatch(r"_+", t):
            expect_piece = True
        elif expect_piece and re.fullmatch(r"[\w']+", t):
            expect_piece = False
        else:
            break
        i += 1
    return i


def glue_underscores(tokens):
    """A "_" left over was split out of one word by the tokenizer ("Dam _ Kham"): glue it back."""
    out, attach = [], False
    for tok, tag in tokens:
        if re.fullmatch(r"_+", tok):
            if out:  # a leading "_" with nothing before it is dropped
                out[-1] = [out[-1][0] + tok, out[-1][1]]
                attach = True
        elif attach:
            out[-1] = [out[-1][0] + tok, out[-1][1]]
            attach = False
        else:
            out.append([tok, tag])
    return out


def rebuild(tokens):
    """Delete URLs and @mentions (all their pieces); glue '#' + word (+ '_' pieces) into '#word'."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        tok, tag = tokens[i]
        nxt = tokens[i + 1][0] if i + 1 < n else ""
        at = re.fullmatch(r"([^@]*)@+(_*)", tok)  # "@", "@@", "@_", "(@", ".@", emoji + "@", ...
        at_glued = re.fullmatch(r"@+\w+", tok)  # handle glued to "@", e.g. "@919977012012"
        at_punct = re.fullmatch(r"@+([.…]+)", tok)  # truncated mention "@…": keep the punctuation

        # URL: starts at "http(s)", or at "//" when the scheme was lost ("// t . co / x")
        if tok in ("http", "https") or (tok == "//" and nxt in ("t", "tco", "tc")):
            i += 1 if tok != "//" else 0
            trailing = None
            while i < n:
                t = tokens[i][0]
                if t in URL_PIECES or t.startswith("/"):
                    if "…" in t:  # truncated URL like "//…": keep the tweet's truncation mark
                        trailing = "…"
                    i += 1
                    if t == "/" and i < n and tokens[i][0].isalnum():
                        i += 1  # the t.co code ends the URL
                        break
                else:
                    break
            if trailing:
                out.append([trailing, "O"])

        # Mention: "@" + handle pieces, all deleted; text glued before the "@" (e.g. "(") is kept
        elif at:
            if at.group(1):
                out.append([at.group(1), "O"])
            i = handle_end(tokens, i + 1)
        elif at_glued:
            i = handle_end(tokens, i + 1, expect_piece=False)
        elif at_punct:
            out.append([at_punct.group(1), "O"])
            i += 1

        # Hashtag: glue "#", its word and any "_" pieces so later steps can protect it
        elif tok == "#" and re.fullmatch(r"\w+", nxt) and nxt not in ("http", "https"):
            j = handle_end(tokens, i + 1)
            out.append(["#" + "".join(t for t, _ in tokens[i + 1:j]), tokens[i + 1][1]])
            i = j

        else:
            out.append([tok, tag])
            i += 1
    return glue_underscores(out)


def words(tokens):
    """Alphabetic words tagged Hin or Eng (hashtags, emojis, punctuation excluded)."""
    return [(t, g) for t, g in tokens if g in ("Hin", "Eng") and WORD_RE.fullmatch(t)]


def hin_share(tokens):
    """Hin-tagged words / (Hin + Eng words)."""
    w = words(tokens)
    return sum(g == "Hin" for _, g in w) / len(w)


def dedup_key(text):
    """Text used for duplicate checks: lowercased."""
    return text.lower()


def grams(text, k=5):
    """Set of lowercased character 5-grams."""
    s = text.lower()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def near_duplicates(query, ref):
    """Ids in `query` whose 5-gram Jaccard to any text in `ref` exceeds JACCARD_MAX.

    query, ref: DataFrames with id and text. An inverted index (5-gram -> ref rows)
    means each query is compared only with ref rows sharing at least one 5-gram.
    """
    ref_grams = [grams(t) for t in ref.text]
    index = defaultdict(list)
    for j, g in enumerate(ref_grams):
        for x in g:
            index[x].append(j)
    hits = []
    for qid, text in zip(query.id, query.text):
        g = grams(text)
        inter = defaultdict(int)  # ref row -> number of shared 5-grams
        for x in g:
            for j in index.get(x, ()):
                inter[j] += 1
        if any(c / (len(g) + len(ref_grams[j]) - c) > JACCARD_MAX for j, c in inter.items()):
            hits.append(qid)
    return hits


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Parse
    sents = parse(SRC)
    print(f"parsed sentences: {len(sents)}")
    assert all(s["label"] in LABELS for s in sents), "unexpected label"
    assert len({s["id"] for s in sents}) == len(sents), "duplicate UIDs"

    # 2. Remove URLs / mentions, glue hashtags; text = tokens joined by single spaces
    for s in sents:
        s["had_link"] = any(t in ("@", "http", "https") for t, _ in s["tokens"])
        s["tokens"] = rebuild(s["tokens"])
        s["text"] = " ".join(t for t, _ in s["tokens"])

    # Fragment check on every sentence (a superset of checking a sample)
    # (URL pieces, a lone "@", or a stray "_" piece of a split handle/hashtag)
    frag = re.compile(r"\bhttps?\b|// t\b|\bt \. co\b|\bt co /|\btco /|(^| )@( |$)|(^| )_+( |$)")
    bad = [s["text"] for s in sents if frag.search(s["text"])]
    print(f"sentences with URL/mention/'_' fragments left: {len(bad)}")
    for t in bad[:5]:
        print("   ", t[:160])
    assert not bad, "URL/mention fragments left"
    sample = pd.Series([s["text"] for s in sents if s["had_link"]]).sample(5, random_state=SEED)
    print("sample texts that had a URL or mention:")
    for t in sample:
        print("   ", t[:160])

    # 3. Filters, applied in order; each sentence is counted under its first failing reason
    drops = defaultdict(int)
    seen, kept = set(), []
    for s in sents:
        toks = [t for t, _ in s["tokens"]]
        n_words = len(words(s["tokens"]))
        key = dedup_key(s["text"])
        if any(DEVANAGARI_RE.search(t) for t in toks):
            drops["devanagari"] += 1
        elif "nan" in toks:
            drops["nan_token"] += 1
        elif n_words < 3:
            drops["under_3_words"] += 1
        elif n_words > 50:
            drops["over_50_words"] += 1
        elif key in seen:  # first occurrence is kept
            drops["duplicate"] += 1
        else:
            seen.add(key)
            kept.append(s)
    print("dropped per reason (applied in this order):")
    for r in DROP_REASONS:
        print(f"   {r:15s} {drops[r]}")
    print(f"kept: {len(kept)}")

    # 4. Hindi share per sentence
    df = pd.DataFrame({
        "id": [s["id"] for s in kept],
        "label": [s["label"] for s in kept],
        "text": [s["text"] for s in kept],
        "hin_share": [round(hin_share(s["tokens"]), 4) for s in kept],
        "tokens_json": [json.dumps(s["tokens"], ensure_ascii=False) for s in kept],
    })

    # 5. Freeze the base set: 334/333/333 from sentences with hin_share >= 0.20
    eligible = df[df.hin_share >= 0.20]
    print(f"hin_share < 0.20 (not eligible for base, stay in pool): {len(df) - len(eligible)}")
    print(f"eligible for base (hin_share >= 0.20): {len(eligible)} {eligible.label.value_counts().to_dict()}")
    base = pd.concat([eligible[eligible.label == l].sample(n, random_state=SEED) for l, n in BASE_PER_LABEL.items()])
    print(f"base: {len(base)} {base.label.value_counts().to_dict()}")

    # 6. Remove pool sentences that are near-duplicates of a base sentence
    pool = df[~df.id.isin(base.id)]
    near = near_duplicates(pool, base)
    pool = pool[~pool.id.isin(near)]
    print(f"pool after removing base: {len(df) - len(base)}; near-duplicates of base (5-gram Jaccard > {JACCARD_MAX}) removed: {len(near)}; remaining: {len(pool)}")

    # 7. Stratified 90/10 split: 10% of each label goes to val
    val = pool.groupby("label", group_keys=False).sample(frac=0.10, random_state=SEED)
    train = pool[~pool.id.isin(val.id)]
    # Leak guard: drop train sentences that are near-duplicates of a val sentence
    tv_near = near_duplicates(train, val)
    train = train[~train.id.isin(tv_near)]
    print(f"train sentences near-duplicate to val (Jaccard > {JACCARD_MAX}) removed from train: {len(tv_near)}")
    print(f"train: {len(train)} {train.label.value_counts().to_dict()}")
    print(f"val:   {len(val)} {val.label.value_counts().to_dict()}")

    # 8. Leak checks: ids, normalized text, and near-duplicates (recomputed independently)
    sets = {"base": base, "train": train, "val": val}
    for a, b in [("base", "train"), ("base", "val"), ("train", "val")]:
        assert not set(sets[a].id) & set(sets[b].id), f"id overlap {a}/{b}"
        assert not set(sets[a].text.map(dedup_key)) & set(sets[b].text.map(dedup_key)), f"text overlap {a}/{b}"
    assert not near_duplicates(pd.concat([train, val]), base), "near-duplicate of base left in train/val"
    assert not near_duplicates(train, val), "train/val near-duplicate left"
    print("leak checks: no shared ids, no shared normalized text, no near-duplicates between base/train/val")

    # Files are written only after every check above has passed
    cols = ["id", "label", "text"]
    base.to_csv(OUT / "base_1000.csv", index=False, encoding="utf-8")
    train[cols].to_csv(OUT / "train.csv", index=False, encoding="utf-8")
    val[cols].to_csv(OUT / "val.csv", index=False, encoding="utf-8")

    # README with seed, counts and SHA-256 of each output file
    files = ["base_1000.csv", "train.csv", "val.csv"]
    lines = [f"seed: {SEED}", f"source: {SRC.name}", f"parsed: {len(sents)}"]
    lines += [f"dropped_{r}: {drops[r]}" for r in DROP_REASONS]
    lines += [f"kept: {len(kept)}", f"hin_share_below_0.20: {len(df) - len(eligible)}",f"near_duplicates_of_base_removed: {len(near)}",
              f"train_near_duplicates_of_val_removed: {len(tv_near)}",
              f"base: {len(base)}", f"train: {len(train)}", f"val: {len(val)}", "", "sha256:"]
    lines += [f"{sha256(OUT / f)}  {f}" for f in files]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()

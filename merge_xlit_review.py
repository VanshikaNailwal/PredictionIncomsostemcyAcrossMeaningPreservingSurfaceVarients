"""Merge the hand-reviewed xlit_review.tsv into xlit_dictionary.tsv.

approved = yes -> add (tag, word, proposed)
approved = fix -> add (tag, word, correct_form)
approved = no  -> ignored; empty -> not reviewed yet, ignored
Existing dictionary entries are never changed. Nothing is written if any row is invalid.
The dictionary is backed up (xlit_dictionary.backup-<time>.tsv) before it is written.

Run (only after filling in the review):
  python merge_xlit_review.py                      # reads xlit_review.tsv
  python merge_xlit_review.py xlit_review.xlsx     # or the Excel copy
  python merge_xlit_review.py --dry-run            # report only, write nothing
Then rebuild D: .venv-xlit/Scripts/python transliterate_devanagari.py build
"""
import argparse
import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
STRAY_HALANT_RE = re.compile("्(?=[‌‍]|$)")
COLUMNS = ["word", "tag", "count", "indicxlit_output", "proposed", "confidence", "source", "sample_sentence",
           "approved", "correct_form"]


def read_review(path):
    """Read the review as strings. A TSV saved by Excel may be UTF-16 ("Unicode Text")."""
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding=enc)
        except (UnicodeError, pd.errors.ParserError):
            continue
    sys.exit(f"cannot read {path}: save it as UTF-8 (or Excel 'Unicode Text') TSV, or pass the .xlsx")


def bad_value(v):
    """Why a Devanagari value cannot go into the dictionary, or '' if it is fine."""
    if not v:
        return "empty"
    if re.search("[A-Za-z]", v):
        return "contains Latin letters"
    if "?" in v or "�" in v:
        return "contains '?' (lost characters? save the file as UTF-8)"
    if not DEVANAGARI_RE.search(v):
        return "no Devanagari"
    if STRAY_HALANT_RE.search(v):
        return "stray halant"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review", nargs="?", default=str(ROOT / "xlit_review.tsv"))
    ap.add_argument("--dictionary", default=str(ROOT / "xlit_dictionary.tsv"))
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()
    review_path, dict_path = Path(args.review), Path(args.dictionary)

    df = read_review(review_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"{review_path.name}: missing columns {missing}")
    df = df.apply(lambda s: s.str.strip())
    df["approved"] = df.approved.str.lower()

    if (df.approved == "").all():
        other = review_path.with_suffix(".xlsx" if review_path.suffix.lower() == ".tsv" else ".tsv")
        hint = f" Did you fill in {other.name} instead? Pass it as the argument." if other.exists() else ""
        sys.exit(f"{review_path.name}: no row has 'approved' filled in yet.{hint}")

    # Validate every row first; write nothing if anything is wrong
    errors = []
    for i, r in df.iterrows():
        line = i + 2  # header is line 1
        if r.approved not in ("yes", "no", "fix", ""):
            errors.append(f"line {line} {r.word}: approved must be yes/no/fix, got {r.approved!r}")
        elif r.approved == "yes" and bad_value(r.proposed):
            errors.append(f"line {line} {r.word}: proposed {r.proposed!r} {bad_value(r.proposed)}")
        elif r.approved == "fix" and bad_value(r.correct_form):
            errors.append(f"line {line} {r.word}: correct_form {r.correct_form!r} {bad_value(r.correct_form)}")
        if r.tag not in ("Hin", "Eng"):
            errors.append(f"line {line} {r.word}: tag must be Hin or Eng, got {r.tag!r}")
    dup = df[df.duplicated(["tag", "word"], keep=False)]
    if len(dup):
        errors.append(f"duplicate (tag, word) rows: {sorted(set(zip(dup.tag, dup.word)))[:10]}")
    if errors:
        print(f"{len(errors)} problem(s), nothing written:")
        print("\n".join("   " + e for e in errors[:50]))
        sys.exit(1)

    with open(dict_path, encoding="utf-8-sig", newline="") as f:
        existing = {(r["tag"], r["word"]): r["devanagari"] for r in csv.DictReader(f, delimiter="\t")}
    accept = df[df.approved.isin(["yes", "fix"])].copy()
    accept["devanagari"] = accept.proposed.where(accept.approved == "yes", accept.correct_form)
    already = accept[[(t, w) in existing for t, w in zip(accept.tag, accept.word)]]
    new = accept.drop(already.index)

    print(f"review rows: {len(df)}; approved yes {sum(df.approved == 'yes')}, fix {sum(df.approved == 'fix')}, "
          f"no {sum(df.approved == 'no')}, not reviewed {sum(df.approved == '')}")
    print(f"rows to merge: {len(new)} {new.tag.value_counts().to_dict()} "
          f"(yes {sum(new.approved == 'yes')}, fix {sum(new.approved == 'fix')})")
    if len(already):
        conflict = [(w, t, existing[(t, w)], d) for w, t, d in zip(already.word, already.tag, already.devanagari)
                    if existing[(t, w)] != d]
        print(f"skipped, already in the dictionary: {len(already)} (existing entries are never changed)")
        for w, t, old, d in conflict:
            print(f"   {t} {w}: dictionary has {old}, review says {d}  <- edit xlit_dictionary.tsv by hand if needed")
    spot = df[df.source.str.contains("spot_check") & (df.approved != "")]
    if len(spot):
        print(f"spot-check agreement: {sum(spot.approved == 'yes')}/{len(spot)} = "
              f"{sum(spot.approved == 'yes') / len(spot):.1%} of reviewed spot-check rows approved as proposed")
    else:
        print("spot-check agreement: no spot-check rows reviewed yet")

    if args.dry_run:
        print("dry run: nothing written")
        return
    if len(new) == 0:
        print("nothing to merge")
        return
    backup = dict_path.with_name(f"{dict_path.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}.tsv")
    shutil.copy2(dict_path, backup)
    needs_newline = not dict_path.read_bytes().endswith(b"\n")  # else the first new row joins the last old one
    with open(dict_path, "a", encoding="utf-8", newline="") as f:
        if needs_newline:
            f.write("\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for r in new.itertuples():
            w.writerow([r.tag, r.word, r.devanagari])
    print(f"merged {len(new)} rows into {dict_path.name} (backup: {backup.name})")
    print("next: .venv-xlit/Scripts/python transliterate_devanagari.py build, then build_variants.py, "
          "build_neg_control.py, build_review_workbook.py")


if __name__ == "__main__":
    main()

"""Excel review workbook for reading and hand-checking every variant (changes no existing file).

Reads:  output/base_1000.csv, variants_wide.csv, variants_long.csv, neg_control.csv
Writes: output/review_variants.xlsx, output/review_variants.log
Sheets: Review (1 row per base sentence), Summary (live formulas where possible), D_wordcheck
        (200 most frequent Roman words), D_longtail (150 non-dictionary words, frequency-weighted sample),
        NEG_control, Rater_sample (60 blind items) and a hidden Key.
Run: python build_review_workbook.py   (needs openpyxl; after build_neg_control.py)
"""
import json
import math
import random
import re
from collections import Counter, defaultdict

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from build_variants import OUT, SEED, VARIANTS, editable, variants  # same definitions as script 2
from transliterate_devanagari import Transliterator, latin_words  # same lookup that built D

XLSX = OUT / "review_variants.xlsx"
LONGTAIL_SAMPLE = OUT / "d_longtail_sample.tsv"  # the 150 frozen D_longtail words (delete to redraw)
LOG = OUT / "review_variants.log"
TEXT_FONT = "Nirmala UI"  # covers Devanagari; Windows falls back to Segoe UI Emoji for emoji glyphs
HIGHLIGHT = InlineFont(rFont=TEXT_FONT, b=True, color="C00000")
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
EXCEL_CELL_LIMIT = 32767
N = 1000
LABELS = ["negative", "neutral", "positive"]
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
log_lines = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    log_lines.append(line)


def rich_diff(a_text, v_text):
    """v_text as rich text with the tokens that differ from A in bold dark red (same token count)."""
    a, v = a_text.split(), v_text.split()
    assert len(a) == len(v)
    # shortcut: rich-text runs lose no-break spaces on save (1 sentence), so such cells stay plain, unhighlighted
    if a == v or re.search(r"[^\S ]", v_text):
        return v_text
    parts, plain = [], []
    for k, (x, y) in enumerate(zip(a, v)):
        sep = " " if k else ""
        if x == y:
            plain.append(sep + y)
        else:
            if plain or sep:
                parts.append("".join(plain) + sep)
                plain = []
            parts.append(TextBlock(HIGHLIGHT, y))
    if plain:
        parts.append("".join(plain))
    return CellRichText(parts)


def write_sheet(ws, df, text_cols, dropdowns=None, rich=None):
    """Header, rows, fonts, widths, wrap, row heights, frozen panes, filter and dropdowns."""
    cols = list(df.columns)
    ws.append(cols)
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(x, float) and math.isnan(x)) else x for x in row])
    for (col, fn) in (rich or {}).items():  # replace plain text by rich text where requested
        c = cols.index(col) + 1
        for r in range(2, len(df) + 2):
            ws.cell(r, c).value = fn(r - 2)
    for c, name in enumerate(cols, 1):
        letter = get_column_letter(c)
        ws.cell(1, c).font = Font(bold=True)
        ws.cell(1, c).fill = HEADER_FILL
        is_text = name in text_cols
        ws.column_dimensions[letter].width = 60 if is_text else max(10, min(24, len(name) + 2))
        if is_text:
            for r in range(2, len(df) + 2):
                ws.cell(r, c).font = Font(name=TEXT_FONT)
                ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical="top")
    # openpyxl cannot auto-fit; estimate lines from the longest text cell (~55 chars per line at width 60)
    present = [c for c in text_cols if c in cols]
    longest = df[present].fillna("").astype(str).apply(lambda s: s.str.len()).max(axis=1) if present else None
    for r in range(len(df)):
        lines = max(1, math.ceil(longest.iloc[r] / 55)) if present else 1
        ws.row_dimensions[r + 2].height = min(409, 15 * lines + 3)
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    for col, options in (dropdowns or {}).items():
        dv = DataValidation(type="list", formula1=f'"{",".join(options)}"', allow_blank=True)
        ws.add_data_validation(dv)
        letter = get_column_letter(cols.index(col) + 1)
        dv.add(f"{letter}2:{letter}{len(df) + 1}")


def main():
    base = pd.read_csv(OUT / "base_1000.csv", dtype={"id": str}, encoding="utf-8")
    wide = pd.read_csv(OUT / "variants_wide.csv", dtype={"id": str}, encoding="utf-8").set_index("id")
    long = pd.read_csv(OUT / "variants_long.csv", dtype={"id": str}, encoding="utf-8")
    neg = pd.read_csv(OUT / "neg_control.csv", dtype={"id": str}, encoding="utf-8").set_index("id")
    nchg = long.pivot(index="id", columns="variant", values="n_changed_words")
    edit = long.pivot(index="id", columns="variant", values="char_edit_dist")
    assert len(base) == N and (wide.reindex(base.id).A.values == base.text.values).all(), "wide file does not match base"

    # ---------- Sheet 1: Review ----------
    review = pd.DataFrame({
        "id": base.id, "label": base.label, "hin_share": base.hin_share,
        "complete_tweet": ["yes" if ("…" not in t and "..." not in t) else "no" for t in base.text],
        **{v: wide.reindex(base.id)[v].values for v in VARIANTS},
        "NEG": neg.reindex(base.id).text_NEG.values, "neg_op": neg.reindex(base.id).neg_op.values,
        "meaning_ok_T3": "", "meaning_ok_D": "", "D_errors": "", "notes": "",
        "T1_changed": nchg.reindex(base.id).T1.values, "T2_changed": nchg.reindex(base.id).T2.values,
        "T3_changed": nchg.reindex(base.id).T3.values, "T3_edit_dist": edit.reindex(base.id).T3.values,
    }).reset_index(drop=True)
    review_text_cols = VARIANTS + ["NEG", "D_errors", "notes"]

    # ---------- Sheet 2: Summary (computed in Python too, for the log) ----------
    n_editable = n_uc_eligible = 0
    nested_ok = True
    for r in base.itertuples():
        tokens = json.loads(r.tokens_json)
        n_editable += sum(editable(t, g) for t, g in tokens)
        n_uc_eligible += sum(editable(t, g) and not t.isupper() for t, g in tokens)
        v = variants(r.id, tokens)
        nested_ok &= v["T1"][1] <= v["T2"][1] <= v["T3"][1]

    # ---------- Sheet 3: D_wordcheck ----------
    count, tags, outs = Counter(), defaultdict(Counter), defaultdict(Counter)
    misaligned = 0
    for r in base.itertuples():
        tokens, d_toks = json.loads(r.tokens_json), wide.loc[r.id, "D"].split(" ")  # tokens are joined by plain spaces; split() would also cut at no-break spaces
        if len(tokens) != len(d_toks):
            misaligned += 1
            continue
        for (t, g), d in zip(tokens, d_toks):
            if g in ("Hin", "Eng") and re.fullmatch(r"[A-Za-z]+", t):
                w = t.lower()
                count[w] += 1
                tags[w][g] += 1
                outs[(w, g)][d] += 1
    wc_rows = []
    for w, n in count.most_common(200):
        tag = tags[w].most_common(1)[0][0]
        other = "Eng" if tag == "Hin" else "Hin"
        out = outs[(w, tag)].most_common(1)[0][0]
        other_out = outs[(w, other)].most_common(1)[0][0] if outs[(w, other)] else ""
        wc_rows.append({"roman_word": w, "count": n, "tagged_as": tag, "devanagari_output": out,
                        "correct": "", "correct_form": "",
                        "other_tag_output": f"{other}: {other_out}" if other_out and other_out != out else ""})
    wordcheck = pd.DataFrame(wc_rows)

    # ---------- Sheet 3b: D_longtail (words the dictionary does not cover, so IndicXlit decided) ----------
    sentences = [json.loads(tj) for tj in base.tokens_json]
    xl = Transliterator(w for toks in sentences for w in latin_words(toks))  # same lookup that built variant D
    # D must contain exactly what the transliterator produces for every token (proves the sheet matches D)
    d_mismatch, lt_per_sentence, d_output, tagged = 0, {}, defaultdict(Counter), Counter()
    for r, tokens in zip(base.itertuples(), sentences):
        d_toks = wide.loc[r.id, "D"].split(" ")  # tokens are joined by plain spaces; split() would also cut at no-break spaces
        n_lt = 0
        for (tok, used), d in zip(xl.sentence(tokens), d_toks):
            d_mismatch += d != tok
            for w, tag, _, dev, _, _ in used:
                tagged[(w, tag)] += 1
                d_output[(w, tag)][dev] += 1
                n_lt += (tag, w) not in xl.dict
        lt_per_sentence[r.id] = n_lt
    lt = pd.DataFrame([{"roman_word": w, "tag": t, "count": n} for (w, t), n in tagged.items() if (t, w) not in xl.dict])
    # one row per distinct word: its most frequent tag among the non-dictionary uses
    lt = lt.sort_values(["count", "tag"], ascending=[False, True]).drop_duplicates("roman_word")
    # The sample is frozen: it is drawn from non-dictionary words, so new dictionary entries would reshuffle it
    if LONGTAIL_SAMPLE.exists():
        frozen = pd.read_csv(LONGTAIL_SAMPLE, sep="\t", dtype=str, encoding="utf-8")
        longtail = frozen.assign(count=[tagged[(w, t)] for w, t in zip(frozen.roman_word, frozen.tag)])
    else:
        longtail = lt.sample(150, weights="count", random_state=SEED).sort_values("count", ascending=False)
        longtail[["roman_word", "tag"]].to_csv(LONGTAIL_SAMPLE, sep="\t", index=False, encoding="utf-8")
    longtail["devanagari_output"] = [d_output[(w, t)].most_common(1)[0][0] for w, t in zip(longtail.roman_word, longtail.tag)]
    longtail["correct"] = longtail["correct_form"] = ""
    longtail = longtail[["roman_word", "tag", "devanagari_output", "count", "correct", "correct_form"]].reset_index(drop=True)

    # ---------- Sheet 4: NEG_control ----------
    negc = neg.reset_index()[["id", "label", "neg_op", "changed_word", "text_A", "text_NEG"]]

    # ---------- Sheet 5: Rater_sample + hidden Key ----------
    multi = pd.concat([base[base.label == l].sample(15, random_state=SEED) for l in LABELS])
    pool = neg.reset_index()
    pool = pool[~pool.id.isin(multi.id)]
    neg_pick = []
    for k, l in enumerate(LABELS):  # 5 per label, alternating 3/2 between INSERT and REMOVE
        n_ins = 3 if k % 2 == 0 else 2
        for op, n in (("insert", n_ins), ("remove", 5 - n_ins)):
            neg_pick.append(pool[(pool.label == l) & (pool.neg_op == op)].sample(n, random_state=SEED))
    neg_pick = pd.concat(neg_pick)
    items = [(r.id, r.label, {v: wide.loc[r.id, v] for v in ["A", "D", "T3", "UC"]}) for r in multi.itertuples()]
    items += [(r.id, r.label, {"A": r.text_A, "NEG": r.text_NEG}) for r in neg_pick.itertuples()]
    rng = random.Random(SEED)
    rng.shuffle(items)
    rater, key = [], []
    for i, (sid, label, versions) in enumerate(items, 1):
        order = rng.sample(list(versions), len(versions))
        row = {"item": i, "id": sid}
        krow = {"item": i, "id": sid, "label": label, "type": "NEG pair" if "NEG" in versions else "A/D/T3/UC"}
        for k in range(1, 5):
            name = order[k - 1] if k <= len(order) else None
            row[f"Version {k}"] = versions[name] if name else ""
            krow[f"Version {k}"] = name or ""
        row["meaning_same"] = ""
        row.update({f"rater_sentiment_v{k}": "" for k in range(1, 5)})
        row["sounds_natural_1"] = row["sounds_natural_2"] = ""  # per version, so NEG stays hidden; 2-version rows only
        rater.append(row)
        key.append(krow)
    rater, key = pd.DataFrame(rater), pd.DataFrame(key)

    # ---------- Write workbook ----------
    wb = Workbook()
    ws = wb.active
    ws.title = "Review"
    a_texts = review.A.tolist()
    write_sheet(ws, review, review_text_cols,
                dropdowns={"meaning_ok_T3": ["yes", "no", "unsure"], "meaning_ok_D": ["yes", "no", "unsure"]},
                rich={v: (lambda i, v=v: rich_diff(a_texts[i], review[v][i])) for v in ["T1", "T2", "T3"]})

    col = {name: get_column_letter(i) for i, name in enumerate(review.columns, 1)}
    rng_ = lambda name: f"Review!{col[name]}2:{col[name]}{N + 1}"
    ws2 = wb.create_sheet("Summary")
    ws2.append(["Base set by label", "count"])
    for l in LABELS:
        ws2.append([l, f'=COUNTIF({rng_("label")},"{l}")'])
    ws2.append(["total", f'=COUNTA({rng_("label")})'])
    ws2.append([])
    ws2.append(["variant", "mean_words_changed", "mean_char_edit_dist", "pct_editable_words_changed",
                "sentences_identical_to_A", "source"])
    py_summary = {}
    for v in VARIANTS:
        sub = long[long.variant == v]
        changed_total = sub.n_changed_words.sum()
        pct = {"T1": changed_total / n_editable, "T2": changed_total / n_editable,
               "T3": changed_total / n_editable, "UC": changed_total / n_uc_eligible}.get(v)
        ident = int((sub.set_index("id").text.reindex(base.id).values == base.text.values).sum())
        py_summary[v] = (sub.n_changed_words.mean(), sub.char_edit_dist.mean(), pct, ident)
        mean_chg = f"=AVERAGE({rng_(v + '_changed')})" if v in ("T1", "T2", "T3") else round(sub.n_changed_words.mean(), 3)
        mean_ed = f"=AVERAGE({rng_('T3_edit_dist')})" if v == "T3" else round(sub.char_edit_dist.mean(), 3)
        ident_f = f"=SUMPRODUCT(--EXACT({rng_('A')},{rng_(v)}))"  # EXACT: case-sensitive, so LC/UC count right
        src = "formulas" if v in ("T1", "T2", "T3") else "formula (identical); values (means)"
        ws2.append([v, mean_chg, mean_ed, round(pct, 4) if pct is not None else "n/a", ident_f, src])
        if pct is not None:
            ws2.cell(ws2.max_row, 4).number_format = "0.00%"
    ws2.append([])
    ws2.append(["check", "result"])
    ws2.append(["T1 ⊂ T2 ⊂ T3 changed positions, all 1,000 sentences", "PASS" if nested_ok else "FAIL"])
    for v in ("T1", "T2", "T3"):
        ws2.append([f"{v}: sentences with zero edits", f'=COUNTIF({rng_(v + "_changed")},0)'])
    ws2.append(["editable words (T1-T3 denominator)", n_editable])
    ws2.append(["editable, not-all-caps words (UC denominator)", n_uc_eligible])
    for c, w in zip("ABCDEF", (52, 20, 20, 26, 24, 34)):
        ws2.column_dimensions[c].width = w
    for r in (1, 7, 18):
        for c in range(1, 7):
            if ws2.cell(r, c).value:
                ws2.cell(r, c).font = Font(bold=True)
                ws2.cell(r, c).fill = HEADER_FILL
    ws2.freeze_panes = "C2"
    ws2.auto_filter.ref = "A7:F15"

    write_sheet(wb.create_sheet("D_wordcheck"), wordcheck, ["devanagari_output", "correct_form", "other_tag_output"],
                dropdowns={"correct": ["yes", "no"]})
    write_sheet(wb.create_sheet("D_longtail"), longtail, ["devanagari_output", "correct_form"],
                dropdowns={"correct": ["yes", "no"]})
    write_sheet(wb.create_sheet("NEG_control"), negc, ["text_A", "text_NEG"])
    wsr = wb.create_sheet("Rater_sample")
    write_sheet(wsr, rater, [f"Version {k}" for k in range(1, 5)],
                dropdowns={"meaning_same": ["yes", "no"],
                           **{f"rater_sentiment_v{k}": ["negative", "neutral", "positive"] for k in range(1, 5)}})
    nat_cols = [get_column_letter(list(rater.columns).index(f"sounds_natural_{k}") + 1) for k in (1, 2)]
    dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    wsr.add_data_validation(dv)
    two_version_rows = [r for r, v3 in enumerate(rater["Version 3"], 2) if not v3]
    for r in range(2, len(rater) + 2):
        for c in nat_cols:
            if r in two_version_rows:
                dv.add(f"{c}{r}")
            else:  # grey out on the 4-version rows, where these columns do not apply
                wsr[f"{c}{r}"].fill = PatternFill("solid", fgColor="EEEEEE")
    wsk = wb.create_sheet("Key")
    write_sheet(wsk, key, [])
    wsk.sheet_state = "hidden"
    wb.save(XLSX)

    # ---------- Checks ----------
    log(f"wrote {XLSX.name}")
    log("rows per sheet (excluding header):")
    expected = {"Review": review, "D_wordcheck": wordcheck, "D_longtail": longtail, "NEG_control": negc,
                "Rater_sample": rater, "Key": key}
    wb2 = load_workbook(XLSX)  # rich text reads back as plain strings
    for name, df in expected.items():
        n_rows = wb2[name].max_row - 1
        log(f"   {name:13s} {n_rows:5d}  {'OK' if n_rows == len(df) else 'FAIL'}")
        assert n_rows == len(df)
    log(f"   Summary        {wb2['Summary'].max_row} lines (tables)")
    log(f"   Key sheet hidden: {wb2['Key'].sheet_state == 'hidden'}")
    nat_letters = {get_column_letter(list(rater.columns).index(f"sounds_natural_{k}") + 1) for k in (1, 2)}
    nat_cells = {f"{get_column_letter(c)}{r}" for dv in wb2["Rater_sample"].data_validations.dataValidation
                 for rng_ in dv.sqref.ranges for r, c in rng_.cells if get_column_letter(c) in nat_letters}
    n_pairs = sum(key.type == "NEG pair")
    log(f"   sounds_natural_1/_2 dropdown cells: {len(nat_cells)} (2 x {n_pairs} two-version rows)")
    assert len(nat_cells) == 2 * n_pairs
    hdr = [c.value for c in wb2["Rater_sample"][1]]
    assert not any("NEG" in str(h) for h in hdr), "a Rater_sample header reveals NEG"
    log(f"   Rater_sample headers reveal nothing about NEG: {hdr}")
    log(f"   Rater_sample: {sum(key.type == 'A/D/T3/UC')} A/D/T3/UC items "
        f"{key[key.type == 'A/D/T3/UC'].label.value_counts().to_dict()}, {sum(key.type == 'NEG pair')} NEG pairs "
        f"{key[key.type == 'NEG pair'].label.value_counts().to_dict()}, ids overlapping: {key.id.duplicated().sum()}")

    empty = {v: int(review[v].isna().sum() + (review[v].astype(str).str.strip() == "").sum()) for v in VARIANTS}
    log(f"empty cells in A and variant columns: {empty}")
    assert not any(empty.values())
    longest = max(len(str(c.value)) for wsx in wb2 for row in wsx.iter_rows() for c in row if c.value is not None)
    log(f"longest cell: {longest} characters (Excel limit {EXCEL_CELL_LIMIT})")
    assert longest <= EXCEL_CELL_LIMIT

    # Round trip: every text cell must read back exactly as written
    mismatch, n_dev, n_emo, n_cells = 0, 0, 0, 0
    for name, df in expected.items():
        sheet = wb2[name]
        for c_idx, c_name in enumerate(df.columns, 1):
            for r_idx, value in enumerate(df[c_name], 2):
                if isinstance(value, str) and value:
                    got = sheet.cell(r_idx, c_idx).value
                    n_cells += 1
                    mismatch += got != value
                    n_dev += bool(DEVANAGARI_RE.search(value))
                    n_emo += bool(EMOJI_RE.search(value))
    log(f"round trip: {n_cells} text cells re-read, {mismatch} differ; "
        f"{n_dev} contain Devanagari, {n_emo} contain emojis")
    assert mismatch == 0

    log("summary values (computed in Python; the sheet uses live formulas where noted):")
    log(f"   nested T1 ⊂ T2 ⊂ T3: {'PASS' if nested_ok else 'FAIL'}; zero-edit sentences: "
        + ", ".join(f"{v} {int((review[v + '_changed'] == 0).sum())}" for v in ('T1', 'T2', 'T3')))
    for v, (mc, me, pct, ident) in py_summary.items():
        log(f"   {v:3s} mean_changed {mc:6.2f}  mean_edit_dist {me:6.2f}  "
            f"pct_editable {'n/a' if pct is None else f'{pct:.2%}':>7s}  identical_to_A {ident}")
    if misaligned:
        log(f"   WARNING: {misaligned} sentences skipped in D_wordcheck (A and D token counts differ)")

    in_dict = sum((t, w) in xl.dict for w, t in zip(longtail.roman_word, longtail.tag))
    log(f"D_longtail: {len(longtail)} words (frozen in {LONGTAIL_SAMPLE.name}; {in_dict} are now in the dictionary), "
        f"{longtail['count'].sum()} tokens; tags {longtail.tag.value_counts().to_dict()}")
    log(f"D_longtail outputs match variant D: {d_mismatch} mismatching tokens")
    assert d_mismatch == 0
    log("10 sentences with the most long-tail (non-dictionary) words, A next to D:")
    for sid in sorted(lt_per_sentence, key=lambda i: (-lt_per_sentence[i], i))[:10]:
        log(f"  id {sid}: {lt_per_sentence[sid]} long-tail words")
        log(f"    A  {wide.loc[sid, 'A']}")
        log(f"    D  {wide.loc[sid, 'D']}")

    log("5 example rows from Review:")
    for _, r in review.sample(5, random_state=SEED).iterrows():
        log(f"  id {r.id} ({r.label}, hin_share {r.hin_share}, complete {r.complete_tweet}, "
            f"T1/T2/T3 changed {r.T1_changed}/{r.T2_changed}/{r.T3_changed})")
        for v in VARIANTS + ["NEG"]:
            if isinstance(r[v], str):
                log(f"    {v:3s} {r[v][:120]}")
    LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

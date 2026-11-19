"""Builds manual_review_sheet.xlsx: a random, label-balanced review sample.
Run from C:\\hinglish_thesis:   python make_manual_review_sheet.py
Reads output_FROZEN\\variants_wide.csv (falls back to output\\variants_wide.csv). Needs openpyxl only."""
import csv, random, sys, os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

SEED, PER_LABEL, SENT_PER_LABEL = 2026, 50, 20
src = next((p for p in ("output_FROZEN/variants_wide.csv", "output/variants_wide.csv") if os.path.exists(p)), None)
if not src: sys.exit("variants_wide.csv not found in output_FROZEN or output")
rows = list(csv.DictReader(open(src, encoding="utf-8-sig", newline="")))
cols = {c.lower(): c for c in rows[0]}
need = ["id", "label", "a", "t3", "d"]
miss = [n for n in need if n not in cols]
if miss: sys.exit(f"Missing columns {miss}. Found: {list(rows[0])}")
ctw = cols.get("complete_tweet")
g = lambda r, k: r[cols[k]]

rng = random.Random(SEED)
by = {}
for r in rows: by.setdefault(g(r, "label"), []).append(r)
sample = []
for lab in sorted(by):
    pool = sorted(by[lab], key=lambda r: g(r, "id"))
    sample += rng.sample(pool, min(PER_LABEL, len(pool)))
rng.shuffle(sample)

wb = Workbook()
hdr = Font(bold=True, color="FFFFFF"); fill = PatternFill("solid", fgColor="305496")
wrap = Alignment(wrap_text=True, vertical="top")
def head(ws, names, widths):
    ws.append(names)
    for i, (c, w) in enumerate(zip(ws[1], widths)):
        c.font, c.fill, c.alignment = hdr, fill, wrap
        ws.column_dimensions[c.column_letter].width = w
    ws.freeze_panes = "A2"

ws = wb.active; ws.title = "Instructions"
for line in [
 "MANUAL REVIEW SHEET - read every sentence yourself; do not use scripts or templates.",
 "",
 "Sheet Review (150 random sentences):",
 " meaning_ok_T3: compare A with T3 (typo version). yes = a Hindi/Hinglish reader gets the same meaning and sentiment.",
 "   no = a typo creates a different word, breaks a negation/name/key word, or the sentence is unreadable.",
 " meaning_ok_D: compare A with D (Devanagari). yes only if EVERY word is a correct spelling of the same word.",
 " D_wrong_tokens: list each wrong token as  roman -> devanagari -> correct form.",
 " note: quote the exact tokens involved. Leave empty if fine. Never write generic notes.",
 " Be strict: if unsure, write 'unsure' and say why.",
 "",
 "Sheet Blind_sentiment (60 items): read ONE text, give sentiment (positive/neutral/negative) from the text alone,",
 " and say whether it sounds natural (yes/no). Do not look at other sheets while doing this one.",
 "",
 f"Sampling: seed {SEED}, {PER_LABEL} per label from the frozen 1,000-sentence base. The hidden Key sheet is for the researcher only."]:
    ws.append([line])
ws.column_dimensions["A"].width = 120

rv = wb.create_sheet("Review")
head(rv, ["id", "complete_tweet", "A", "T3", "D", "meaning_ok_T3", "meaning_ok_D", "D_wrong_tokens", "note"],
     [9, 10, 50, 50, 50, 14, 14, 30, 40])
for r in sample:
    rv.append([g(r, "id"), r[ctw] if ctw else "", g(r, "a"), g(r, "t3"), g(r, "d"), None, None, None, None])
    for c in rv[rv.max_row]: c.alignment = wrap
dv = DataValidation(type="list", formula1='"yes,no,unsure"', allow_blank=True)
rv.add_data_validation(dv); dv.add(f"F2:G{rv.max_row}")

bs = wb.create_sheet("Blind_sentiment")
head(bs, ["item", "text", "sentiment", "sounds_natural"], [7, 90, 14, 14])
key = wb.create_sheet("Key"); key.append(["item", "id", "variant", "gold_label", "label_split_in_review"])
sent = []
for lab in sorted(by):
    sent += [r for r in sample if g(r, "label") == lab][:SENT_PER_LABEL]
rng.shuffle(sent)
for i, r in enumerate(sent, 1):
    v = rng.choice(["a", "t3", "d"])
    bs.append([i, g(r, v), None, None]); bs[bs.max_row][1].alignment = wrap
    key.append([i, g(r, "id"), v.upper(), g(r, "label")])
d1 = DataValidation(type="list", formula1='"positive,neutral,negative"', allow_blank=True)
d2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
bs.add_data_validation(d1); bs.add_data_validation(d2)
d1.add(f"C2:C{bs.max_row}"); d2.add(f"D2:D{bs.max_row}")
key.sheet_state = "hidden"
wb.save("manual_review_sheet.xlsx")
print("source:", src, "| review rows:", len(sample), "| blind items:", len(sent))
print("label counts:", {l: sum(g(r,'label')==l for r in sample) for l in by})
print("IMPORTANT: before sending to the rater, delete the hidden 'Key' sheet from a COPY.")
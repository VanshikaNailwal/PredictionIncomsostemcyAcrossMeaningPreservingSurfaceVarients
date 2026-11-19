"""Script 3: build variant D (full Devanagari) from the A text of the base set.

Every Latin-letter word is transliterated, using its Hin/Eng tag from tokens_json:
  1. Hand-made dictionary first (xlit_dictionary.tsv: tag, word, devanagari).
     It covers the 200 most frequent Hin-tagged and the 200 most frequent Eng-tagged words.
  2. Words not in the dictionary (both tags): IndicXlit's top candidate; a later candidate only
     if the top one is unusable (empty, Latin letters, or a stray halant like तक्‌).
  3. If IndicXlit has no usable candidate: a letter-by-letter map, so no word stays Roman.
"UP" and "US" (exact case) are always यूपी and यूएस; lowercase "up"/"us" go through the normal rules.
Context rules (before the dictionary, whole-word tokens only): "r" after u/you/we/they/ur -> आर;
yrs/yr -> ईयर्स/ईयर; they -> दे; us -> अस after an English preposition or when Eng-tagged, else उस;
"hi" after hy/hai/h/hain -> ही; "ati" before kar/kr/karna or sentence-initial -> अति.
ALL-CAPS rule: in sentences where fewer than 50% of the words (2+ letters) are ALL-CAPS, an ALL-CAPS
word takes the English reading even when tagged Hin (THE -> द, ME -> मी). Not for RT/BJP, and not for
abbreviation-like words (no vowel, or IndicXlit spells them out: TK, HM, EVM).
Abbreviation guard (list only): a 2-4 letter, non-ALL-CAPS, non-dictionary word that IndicXlit spells
out letter by letter (jis -> जेआईएस) is listed in output/short_unresolved.tsv; D still uses IndicXlit.
Hashtags, numbers, emojis and punctuation are left unchanged.
IndicXlit candidates are cached per word in output/xlit_candidates.json.

Needs the separate Python 3.10 venv (fairseq does not run on 3.11+), see requirements-xlit.txt.
Run:
  .venv-xlit/Scripts/python transliterate_devanagari.py test     # dictionary entries, check by hand
  .venv-xlit/Scripts/python transliterate_devanagari.py build    # writes output/variants_D.csv
  python transliterate_devanagari.py review                      # writes xlit_review.tsv/.xlsx to fill in
"""
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_variants import OUT, SEED, levenshtein  # reuse script 2's definitions (OUT honours $OUT_DIR)

ROOT = Path(__file__).parent
CANDIDATES = OUT / "xlit_candidates.json"
DICTIONARY = ROOT / "xlit_dictionary.tsv"
PROPOSED = ROOT / "xlit_dictionary_proposed.tsv"      # next 600 words per tag, hand proposals (not merged)
SHORT_PROPOSED = ROOT / "xlit_short_proposed.tsv"     # hand proposals for the guarded short words (not merged)
REVIEW = ROOT / "xlit_review.tsv"                     # filled in by hand, merged by merge_xlit_review.py
TOPK = 4  # IndicXlit candidates kept per word (= beam width)
N_TOP = 200  # the dictionary must cover the 200 most frequent words of each tag
N_SPOT_CHECK = 60
LATIN_RE = re.compile(r"[A-Za-z]+")
STRAY_HALANT_RE = re.compile("्(?=[‌‍]|$)")  # halant ending a word or followed by a joiner

# Last-resort map, one Latin letter -> Devanagari (vowels as independent letters).
# Only used when IndicXlit gives no usable candidate; every use is printed in the build log.
LETTER_MAP = dict(zip("abcdefghijklmnopqrstuvwxyz",
                      ["अ", "ब", "क", "द", "ए", "फ़", "ग", "ह", "इ", "ज", "क", "ल", "म",
                       "न", "ओ", "प", "क़", "र", "स", "त", "उ", "व", "व", "क्स", "य", "ज़"]))
# English letter names in Devanagari, to spot IndicXlit spelling a short word out as an abbreviation
LETTER_NAMES = {"a": ["ए"], "b": ["बी"], "c": ["सी"], "d": ["डी"], "e": ["ई"], "f": ["एफ", "एफ़"], "g": ["जी"],
                "h": ["एच"], "i": ["आई"], "j": ["जे"], "k": ["के"], "l": ["एल"], "m": ["एम"], "n": ["एन"],
                "o": ["ओ"], "p": ["पी"], "q": ["क्यू"], "r": ["आर"], "s": ["एस"], "t": ["टी"], "u": ["यू"],
                "v": ["वी"], "w": ["डब्ल्यू", "डबल्यू"], "x": ["एक्स"], "y": ["वाई"], "z": ["ज़ेड", "जेड"]}
SHORT_GUARD = (2, 4)  # word lengths the abbreviation guard applies to (1 letter cannot be "spelled out")
CAPS_KEEP = {"rt", "bjp"}  # ALL-CAPS words that keep their own tag
CAPS_MAX_SHARE = 0.5  # the ALL-CAPS rule applies only in sentences with fewer ALL-CAPS words than this
CASE_SENSITIVE = {"UP": "यूपी", "US": "यूएस"}  # exact case, checked before everything else ("up"/"us" use the rules)


def is_protected(tok):
    """Tokens kept as-is: hashtags."""
    return tok.startswith("#")


def lang(tag):
    """Hin stays Hin; every other tag goes through the English rules."""
    return "Hin" if tag == "Hin" else "Eng"


def is_caps(run):
    """ALL-CAPS word of 2+ letters (RT and BJP excepted)."""
    return len(run) >= 2 and run.isupper() and run.lower() not in CAPS_KEEP


def latin_words(tokens):
    """Lowercased Latin-letter runs of all unprotected tokens."""
    return [m.group().lower() for t, _ in tokens if not is_protected(t) for m in LATIN_RE.finditer(t)]


def caps_share(tokens):
    """Share of ALL-CAPS words among the words of 2+ letters (RT/BJP not counted)."""
    runs = [m.group() for t, _ in tokens if not is_protected(t) for m in LATIN_RE.finditer(t)
            if len(m.group()) >= 2 and m.group().lower() not in CAPS_KEEP]
    return sum(r.isupper() for r in runs) / len(runs) if runs else 0.0


def m_whole(t):
    """True if the token is a single Latin-letter word (context rules only apply to whole words)."""
    return LATIN_RE.fullmatch(t) is not None


US_AS_ENGLISH_AFTER = {"for", "with", "to", "give", "let", "about", "among", "between", "from", "by", "on", "in",
                       "join", "tell", "show", "help", "thank", "follow", "support", "send", "bring", "make"}


def context_rule(low, i, tag):
    """Devanagari for short ambiguous words decided by their neighbours, or None.

    low: the sentence's tokens lowercased; i: position of the word; tag: Hin/Eng."""
    w = low[i]
    prev = low[i - 1] if i > 0 else ""
    nxt = low[i + 1] if i + 1 < len(low) else ""
    if w == "r" and prev in {"u", "you", "we", "they", "ur"}:
        return "आर"  # "u r", "we r" = are, not और
    if w in ("yrs", "yr"):
        return {"yrs": "ईयर्स", "yr": "ईयर"}[w]
    if w == "they":
        return "दे"  # always English, even when tagged Hin
    if w == "us":
        return "अस" if prev in US_AS_ENGLISH_AFTER or tag == "Eng" else "उस"
    if w == "hi" and prev in {"hy", "hai", "h", "hain"}:
        return "ही"  # "hai hi" = emphatic ही, not the greeting
    if w == "ati" and (nxt in {"kar", "kr", "karna"} or i == 0):
        return "अति"  # "ati kar di" = excess, not आती
    return None


def usable(dev):
    return bool(dev) and not LATIN_RE.search(dev) and not STRAY_HALANT_RE.search(dev)


def is_letter_spelling(w, dev):
    """True if `dev` spells `w` out letter by letter (jis -> जेआईएस, tm -> टीएम, ata -> एटीए)."""
    pattern = "".join("(?:" + "|".join(LETTER_NAMES[c]) + ")" for c in w)
    return re.fullmatch(pattern, dev) is not None


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_dictionary():
    rows = read_tsv(DICTIONARY)
    d = {(r["tag"], r["word"]): r["devanagari"] for r in rows}
    assert len(d) == len(rows), "duplicate (tag, word) in dictionary"
    assert all(usable(v) for v in d.values()), "dictionary value with Latin letters or a stray halant"
    return d


def load_proposals(path):
    """(tag, word) -> {proposed, confidence} from a proposal TSV, or {} if the file is missing."""
    return {(r["tag"], r["word"]): r for r in read_tsv(path)} if path.exists() else {}


class Transliterator:
    def __init__(self, words):
        """words: the lowercased Latin-letter words that occur (their IndicXlit candidates are fetched)."""
        self.dict = load_dictionary()
        self.cands = json.loads(CANDIDATES.read_text(encoding="utf-8")) if CANDIDATES.exists() else {}
        self.fetch(set(words))
        # Devanagari form -> Hindi dictionary words with that form; only used by the test mode to flag
        # Eng dictionary entries spelled like a different Hindi word (Eng "to" vs Hin "to" -> तो).
        self.hindi_forms = defaultdict(set)
        for (tag, w), v in self.dict.items():
            if tag == "Hin":
                self.hindi_forms[v].add(w)

    def is_other_hindi_word(self, form, w):
        """True if `form` spells a Hindi dictionary word and `w` is not one of its romanizations."""
        return form in self.hindi_forms and w not in self.hindi_forms[form]

    def fetch(self, words):
        """Run IndicXlit for words not in the candidate cache."""
        missing = sorted(w for w in words if w not in self.cands)
        if not missing:
            return
        from ai4bharat.transliteration import XlitEngine  # heavy import, only when needed
        engine = XlitEngine("hi", beam_width=TOPK, src_script_type="roman")
        for i, w in enumerate(missing, 1):
            try:
                self.cands[w] = engine.translit_word(w, topk=TOPK)["hi"]
            except Exception as e:  # keep going; the letter map covers this word
                print(f"   IndicXlit error on {w!r}: {e}")
                self.cands[w] = []
            if i % 500 == 0:
                self.save()  # checkpoint so an interrupted run resumes quickly
                print(f"   IndicXlit: {i}/{len(missing)} words")
        self.save()

    def save(self):
        CANDIDATES.write_text(json.dumps(self.cands, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")

    def top1(self, w):
        c = self.cands.get(w, [])
        return c[0] if c and usable(c[0]) else ""

    def is_abbrev(self, w):
        """Abbreviation-like: no vowel (tk, hm) or IndicXlit spells it out letter by letter (evm)."""
        return not re.search("[aeiou]", w) or (bool(self.top1(w)) and is_letter_spelling(w, self.top1(w)))

    def guarded(self, w, caps=False):
        """Abbreviation guard: a short, non-ALL-CAPS word whose IndicXlit top-1 spells it out letter by letter."""
        return (SHORT_GUARD[0] <= len(w) <= SHORT_GUARD[1] and not caps and bool(self.top1(w))
                and is_letter_spelling(w, self.top1(w)))

    def choose(self, w, tag):
        """Return (devanagari, source) for one lowercased word with its Hin/Eng tag.

        Not in the dictionary: IndicXlit top-1 for both tags; a later candidate only if top-1 is unusable."""
        if (tag, w) in self.dict:
            return self.dict[(tag, w)], "dictionary"
        for k, c in enumerate(self.cands.get(w, [])):
            if usable(c):
                return c, "indicxlit" if k == 0 else "indicxlit_alt"
        return "".join(LETTER_MAP[ch] for ch in w), "letter_map"

    def sentence(self, tokens):
        """Transliterate one sentence: [(new token, [(word, tag, caps, devanagari, source, caps_rule)])]."""
        caps_ok = caps_share(tokens) < CAPS_MAX_SHARE
        low = [t.lower() for t, _ in tokens]
        out = []
        for i, (t, g) in enumerate(tokens):
            if is_protected(t):
                out.append((t, []))
                continue
            used = []

            def repl(m, g=g, used=used, i=i, whole=m_whole(t)):
                run = m.group()
                w, caps = run.lower(), is_caps(run)
                if run in CASE_SENSITIVE:  # "UP" (the state), never the English reading
                    used.append((w, lang(g), caps, CASE_SENSITIVE[run], "dictionary", False))
                    return CASE_SENSITIVE[run]
                ctx = context_rule(low, i, lang(g)) if whole else None
                if ctx:
                    used.append((w, lang(g), caps, ctx, "context_rule", False))
                    return ctx
                caps_rule = caps_ok and caps and not self.is_abbrev(w)
                tag = "Eng" if caps_rule else lang(g)
                dev, src = self.choose(w, tag)
                used.append((w, tag, caps, dev, src, caps_rule and lang(g) == "Hin"))
                return dev

            out.append((LATIN_RE.sub(repl, t), used))
        return out

    def tagged_words(self, tokens):
        """(word, tag actually used) for every Latin-letter run of the sentence."""
        return [(u[0], u[1]) for _, used in self.sentence(tokens) for u in used]


def load_base_and_transliterator():
    base = pd.read_csv(OUT / "base_1000.csv", dtype={"id": str}, encoding="utf-8")
    sentences = [json.loads(tj) for tj in base.tokens_json]
    xl = Transliterator(w for toks in sentences for w in latin_words(toks))
    return base, sentences, xl


def run_test(base, xl, words):
    """Print every dictionary entry next to IndicXlit's top candidate for a manual check."""
    for tag in ("Hin", "Eng"):
        top = [w for (w, _), _ in Counter({k: n for k, n in words.items() if k[1] == tag}).most_common(N_TOP)]
        missing = [w for w in top if (tag, w) not in xl.dict]
        print(f"top-{N_TOP} {tag}-tagged words missing from the dictionary: {missing or 'none'}")
    for tag in ("Hin", "Eng"):
        entries = [(w, v) for (t, w), v in xl.dict.items() if t == tag]
        print(f"\n{tag} dictionary ({len(entries)} words)")
        print(f"{'#':>4}  {'count':>5}  {'word':16s} {'dictionary':14s} IndicXlit top-1")
        for i, (w, v) in enumerate(entries, 1):
            flag = "" if v == xl.top1(w) else "  *"
            print(f"{i:4d}  {words[(w, tag)]:5d}  {w:16s} {v:14s} {xl.top1(w)}{flag}")
    clash = [f"{w}->{v} (= Hin {'/'.join(sorted(xl.hindi_forms[v] - {w}))})"
             for (t, w), v in xl.dict.items() if t == "Eng" and xl.is_other_hindi_word(v, w)]
    print("\n* = dictionary differs from IndicXlit")
    print(f"Eng dictionary entries spelled like a different Hin-tagged word (check by hand): {clash or 'none'}")


def run_build(base, sentences, xl):
    rows, sources, by_dict_changed = [], Counter(), 0
    alt_used, fallback, caps_flipped = Counter(), Counter(), 0
    unresolved, sample = Counter(), {}
    word_uses = []  # (word, tag, devanagari) for every Latin-letter run, for the coverage report
    cause = {}  # (id, token index) -> why the token may differ from the previous D
    for r, tokens in zip(base.itertuples(), sentences):
        out = xl.sentence(tokens)
        new = [tok for tok, _ in out]
        for k, (_, used) in enumerate(out):
            for w, tag, caps, dev, src, caps_rule in used:
                sources[src] += 1
                word_uses.append((w, tag, dev))
                caps_flipped += caps_rule
                cause.setdefault((r.id, k), "context rule" if src == "context_rule"
                                 else "dictionary" if src == "dictionary"
                                 else "Eng word: IndicXlit top-1" if tag == "Eng" and src == "indicxlit"
                                 else "ALL-CAPS rule" if caps_rule or caps else f"other ({src})")
                if src == "dictionary" and dev != xl.top1(w):
                    by_dict_changed += 1
                if src == "indicxlit_alt":
                    alt_used[(w, xl.top1(w), dev)] += 1
                if src == "letter_map":
                    fallback[(w, dev)] += 1
                if (tag, w) not in xl.dict and xl.guarded(w, caps):
                    unresolved[(w, tag, dev)] += 1
                    sample.setdefault((w, tag), r.text)
        text = " ".join(new)
        rows.append({"id": r.id, "label": r.label, "variant": "D", "text": text,
                     "n_changed_words": sum(a != b for (a, _), b in zip(tokens, new)),
                     "char_edit_dist": levenshtein(r.text, text),
                     "complete_tweet": int("…" not in text and "..." not in text),
                     "hin_share": r.hin_share,
                     "roman_text": r.text})
    df = pd.DataFrame(rows)

    total = sum(sources.values())
    print(f"words transliterated: {total}, by source: {dict(sources)}")
    print("share of alphabetic word tokens in D by source:")
    for name, keys in [("dictionary", ["dictionary"]), ("context rules", ["context_rule"]), ("IndicXlit alone (top-1)", ["indicxlit"]),
                       ("IndicXlit next candidate (top-1 unusable)", ["indicxlit_alt"]), ("letter map", ["letter_map"])]:
        n = sum(sources[k] for k in keys)
        print(f"   {name:42s} {n:6d}  {n / total:6.2%}")
    print(f"ALL-CAPS rule: {caps_flipped} Hin-tagged ALL-CAPS tokens read as English ({caps_flipped / total:.2%}); "
          f"sentences where it is off (>= {CAPS_MAX_SHARE:.0%} ALL-CAPS): "
          f"{sum(caps_share(t) >= CAPS_MAX_SHARE for t in sentences)}")

    # Tokens that differ from the previous D, with the likely cause
    old_path = OUT / "variants_D.csv"
    if old_path.exists():
        old = pd.read_csv(old_path, dtype={"id": str}, encoding="utf-8").set_index("id").text
        changes = []
        for r, tokens, row in zip(base.itertuples(), sentences, rows):
            words = [a for a, _ in tokens]
            for k, ((a, g), o, n) in enumerate(zip(tokens, old.get(r.id, "").split(" "), row["text"].split(" "))):
                if o != n:
                    changes.append({"id": r.id, "position": k, "roman_word": a, "tag": g, "old_devanagari": o,
                                    "new_devanagari": n, "cause": "stray halant removed" if STRAY_HALANT_RE.search(o)
                                    else cause.get((r.id, k), "other"),
                                    "context": " ".join(words[max(0, k - 3):k] + [f"[{a}]"] + words[k + 1:k + 4])})
        ch = pd.DataFrame(changes, columns=["id", "position", "roman_word", "tag", "old_devanagari", "new_devanagari",
                                            "cause", "context"])
        ch.to_csv(OUT / "d_changes.tsv", sep="\t", index=False, encoding="utf-8")
        print(f"tokens that differ from the previous variants_D.csv: {len(ch)} -> {OUT.name}/d_changes.tsv "
              f"{ch.cause.value_counts().to_dict()}")
        for c in ch.itertuples():
            print(f"   {c.id:>6s} {c.position:3d}  {c.roman_word:12s} {c.tag:4s} {c.old_devanagari} -> {c.new_devanagari}"
                  f"   | {c.context}")

    short_prop = load_proposals(SHORT_PROPOSED)
    rows_u = [{"word": w, "tag": tag, "count": n, "indicxlit_output": xl.top1(w), "used_in_D": dev,
               "proposed": short_prop.get((tag, w), {}).get("proposed", ""),
               "confidence": short_prop.get((tag, w), {}).get("confidence", ""),
               "sample_sentence": sample[(w, tag)]} for (w, tag, dev), n in unresolved.most_common()]
    pd.DataFrame(rows_u, columns=["word", "tag", "count", "indicxlit_output", "used_in_D", "proposed", "confidence",
                                  "sample_sentence"]).to_csv(OUT / "short_unresolved.tsv", sep="\t", index=False,
                                                             encoding="utf-8")
    n_unres = sum(unresolved.values())
    print(f"abbreviation guard (list only): {len(rows_u)} words, {n_unres} tokens ({n_unres / total:.2%}) "
          f"-> {OUT.name}/short_unresolved.tsv; with a proposal: {sum(bool(u['proposed']) for u in rows_u)}")
    prop = load_proposals(PROPOSED)
    if prop:
        hits = [(w, tag, dev) for w, tag, dev in word_uses if (tag, w) in prop]
        n_chg = sum(prop[(tag, w)]["proposed"] != dev for w, tag, dev in hits)
        print(f"proposed dictionary ({PROPOSED.name}, not merged): {len(hits)} tokens ({len(hits) / total:.2%}); "
              f"would change {n_chg} ({n_chg / total:.2%})")
    print(f"words changed by the dictionary (dictionary output differs from IndicXlit top-1): {by_dict_changed}")
    print(f"words whose IndicXlit top-1 was unusable, next candidate used: {sum(alt_used.values())}")
    for (w, top, dev), n in alt_used.most_common(15):
        print(f"   {w:15s} {top or '(unusable)'} -> {dev}  (x{n})")
    print(f"letter-map fallbacks: {dict(fallback) or 0}")

    # Checks (the file is written only if all pass)
    stayed_roman = [t for text in df.text for t in text.split(" ") if not is_protected(t) and LATIN_RE.search(t)]
    print(f"words that stayed in Roman (excluding hashtags): {len(stayed_roman)}")
    assert not stayed_roman, stayed_roman[:20]
    halant = [t for text in df.text for t in text.split(" ") if STRAY_HALANT_RE.search(t)]
    print(f"words with a stray halant: {len(halant)}")
    assert not halant, halant[:20]
    assert len(df) == len(base) and (df.text.str.strip() != "").all()
    assert not df.text.str.contains("HTTPURL|@USER").any()
    hashtags = lambda text: [t for t in text.split() if re.match(r"#\w", t)]
    assert all(hashtags(a) == hashtags(d) for a, d in zip(df.roman_text, df.text))
    print("1,000 rows, no empty text, no placeholders, hashtags unchanged: OK")
    print(df[["n_changed_words", "char_edit_dist", "complete_tweet"]].mean().round(2).to_string())
    df.to_csv(OUT / "variants_D.csv", index=False, encoding="utf-8")
    print(f"wrote variants_D.csv: {len(df)} rows")


def run_review(base, sentences, xl):
    """Write xlit_review.tsv (+ .xlsx with dropdowns): the proposal rows that need a human decision."""
    counts, sample, sample_any = Counter(), {}, {}
    for r, tokens in zip(base.itertuples(), sentences):
        for wt in xl.tagged_words(tokens):
            counts[wt] += 1
            sample.setdefault(wt, r.text)
            sample_any.setdefault(wt[0], r.text)  # for proposals whose (word, tag) no longer occurs (count 0)
    prop, short = load_proposals(PROPOSED), load_proposals(SHORT_PROPOSED)
    rows = {}

    def add(key, p, source):
        tag, w = key
        row = rows.setdefault(key, {"word": w, "tag": tag, "count": counts[(w, tag)], "indicxlit_output": xl.top1(w),
                                    "proposed": p["proposed"], "confidence": p["confidence"], "source": [],
                                    "sample_sentence": sample.get((w, tag), sample_any.get(w, "")),
                                    "approved": "", "correct_form": ""})
        row["source"].append(source)

    for key, p in short.items():  # (c) every guarded short word
        add(key, p, "short_word")
    for key, p in prop.items():
        if p["proposed"] != xl.top1(key[1]):
            add(key, p, "differs_from_indicxlit")  # (a)
        if p["confidence"] == "?":
            add(key, p, "unsure")  # (b)
        if key in short and short[key]["proposed"] != p["proposed"]:
            rows[key]["source"].append(f"600-list proposed {p['proposed']}")
    agree = sorted(k for k, p in prop.items() if k not in rows)  # proposals that equal IndicXlit, sure, not short
    for key in random.Random(SEED).sample(agree, min(N_SPOT_CHECK, len(agree))):
        add(key, prop[key], "spot_check")  # (d)
    df = pd.DataFrame(rows.values())
    group = df.source.map(lambda s: {"short_word": 1, "spot_check": 2}.get(s[0], 0))  # 600-list rows first
    df["source"] = df.source.map("; ".join)
    df = df.assign(_g=group).sort_values(["_g", "count"], ascending=[True, False]).drop(columns="_g")
    df.to_csv(REVIEW, sep="\t", index=False, encoding="utf-8-sig")  # BOM so Excel shows Devanagari

    from openpyxl import Workbook  # main environment only; the review mode does not need IndicXlit
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "xlit_review"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    for c, name in enumerate(df.columns, 1):
        ws.cell(1, c).font = Font(bold=True)
        letter = ws.cell(1, c).column_letter
        ws.column_dimensions[letter].width = {"sample_sentence": 70, "source": 30}.get(name, 16)
        if name in ("indicxlit_output", "proposed", "correct_form", "sample_sentence"):
            for r in range(2, len(df) + 2):
                ws.cell(r, c).font = Font(name="Nirmala UI")
                ws.cell(r, c).alignment = Alignment(wrap_text=name == "sample_sentence", vertical="top")
    dv = DataValidation(type="list", formula1='"yes,no,fix"', allow_blank=True)
    ws.add_data_validation(dv)
    col = ws.cell(1, list(df.columns).index("approved") + 1).column_letter
    dv.add(f"{col}2:{col}{len(df) + 1}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(REVIEW.with_suffix(".xlsx"))

    groups = Counter(s for v in df.source for s in v.split("; ") if not s.startswith("600-list"))
    print(f"wrote {REVIEW.name} and {REVIEW.with_suffix('.xlsx').name}: {len(df)} rows "
          f"(a word in several groups appears once); rows per group: {dict(groups)}")
    print(f"   tokens covered: {df['count'].sum()}; '?' rows: {(df.confidence == '?').sum()}; "
          f"short words whose proposal differs from the 600-list: {df.source.str.contains('600-list').sum()}")


def self_test():
    """Fixed examples for the context rules, so a change in them is caught immediately."""
    t = lambda s, i, tag="Hin": context_rule(s.lower().split(), i, tag)
    assert t("u r great", 1) == "आर" and t("ram r shyam", 1) is None
    assert t("5 yrs", 1) == "ईयर्स" and t("1 yr", 1) == "ईयर" and t("they said", 0) == "दे"
    assert t("fight for us", 2) == "अस" and t("join us", 1) == "अस" and t("us din", 0) == "उस" and t("us din", 0, "Eng") == "अस"
    assert t("ye hai hi", 2) == "ही" and t("hi bhai", 0) is None
    assert t("Ati kar di", 0) == "अति" and t("sab ati kr", 1) == "अति" and t("nahi ati hai", 1) is None


def main():
    self_test()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("test", "build", "review"):
        sys.exit("usage: transliterate_devanagari.py test|build|review")
    base, sentences, xl = load_base_and_transliterator()
    if mode == "test":
        run_test(base, xl, Counter(wt for toks in sentences for wt in xl.tagged_words(toks)))
    elif mode == "build":
        run_build(base, sentences, xl)
    else:
        run_review(base, sentences, xl)


if __name__ == "__main__":
    main()

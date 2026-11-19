"""Export the CSV outputs to Excel files for browsing (not used by the pipeline).

Writes output/excel/: train.xlsx, val.xlsx, base_1000.xlsx, variants.xlsx
variants.xlsx has a side_by_side sheet (one row per sentence, one text column per variant),
one sheet per variant with all metric columns, and a NEG sheet (meaning-changing control) if built.
Run after scripts 1-3: python export_excel.py   (needs openpyxl)
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "output"
XL = OUT / "excel"


def write(df, path, sheets=None):
    """Write one or more sheets, freeze the header row and set readable column widths."""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, d in (sheets or {"data": df}).items():
            d.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                header = col[0].value
                ws.column_dimensions[col[0].column_letter].width = 90 if "text" in str(header) else 14
    print(f"wrote {path.relative_to(OUT.parent)}")


def main():
    XL.mkdir(exist_ok=True)
    read = lambda name: pd.read_csv(OUT / name, dtype={"id": str}, encoding="utf-8")
    write(read("train.csv"), XL / "train.xlsx")
    write(read("val.csv"), XL / "val.xlsx")
    write(read("base_1000.csv").drop(columns="tokens_json"), XL / "base_1000.xlsx")

    long = read("variants_long.csv")
    sheets = {"side_by_side": read("variants_wide.csv")}
    sheets.update({v: g for v, g in long.groupby("variant", sort=False)})
    if (OUT / "neg_control.csv").exists():  # meaning-changing control, from build_neg_control.py
        sheets["NEG"] = read("neg_control.csv")
    write(None, XL / "variants.xlsx", sheets)


if __name__ == "__main__":
    main()

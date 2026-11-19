"""Run train_eval.py for 3 models x 3 seeds, one subprocess per run (GPU memory freed between runs).

Resume-safe: a run whose preds file exists is skipped, so after a Colab disconnect just run it again.
Logs one line per run to OUT/run_log.csv: model, seed, status, seconds, val macro-F1, best epoch, peak GPU MB.
Run: python scripts_train/run_all.py [--out_dir DIR] [--save_ckpt]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from train_eval import MODELS, ROOT, SEEDS, preds_path, val_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    ap.add_argument("--save_ckpt", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run_log.csv"
    failed = []
    for model in MODELS:
        for seed in SEEDS:
            if preds_path(out, model, seed).exists():
                print(f"skip {model} {seed}: preds file exists")
                continue
            print(f"=== {model} seed {seed} ===", flush=True)
            t0 = time.time()
            cmd = [sys.executable, str(Path(__file__).with_name("train_eval.py")), "--model", model,
                   "--seed", str(seed), "--out_dir", str(out)] + (["--save_ckpt"] if args.save_ckpt else [])
            code = subprocess.run(cmd).returncode
            ok = code == 0 and preds_path(out, model, seed).exists()
            vp = val_path(out, model, seed)
            info = json.loads(vp.read_text(encoding="utf-8")) if ok and vp.exists() else {}
            row = {"model": model, "seed": seed, "status": "ok" if ok else f"failed (exit {code})",
                   "seconds": round(time.time() - t0), "val_macro_f1": info.get("val_macro_f1"),
                   "best_epoch": info.get("best_epoch"), "peak_gpu_mem_mb": info.get("peak_gpu_mem_mb")}
            new = not log.exists()
            with open(log, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(row, flush=True)
            if not ok:
                failed.append((model, seed))
    if failed:
        sys.exit(f"failed runs: {failed} (rerun this script to retry them)")
    print("all runs done")


if __name__ == "__main__":
    main()

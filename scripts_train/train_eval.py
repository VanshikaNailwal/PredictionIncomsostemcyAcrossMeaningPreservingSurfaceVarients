"""Fine-tune one model with one seed on train.csv, then predict every eval variant in the same process.

Run from the project root (the folder that holds output_FROZEN/ and scripts_train/):
  python scripts_train/train_eval.py --model xlm-roberta-base --seed 13 [--pilot] [--out_dir DIR] [--save_ckpt]
Writes: DIR/preds/{model}_{seed}.csv, DIR/val_{model}_{seed}.json, DIR/versions.txt
        ({model} = name after the last "/", e.g. muril-base-cased)
--pilot: 1 epoch on 200 train rows, default DIR results_pilot/ (so run_all.py never skips a real run).
torch and transformers are imported inside run(), so the data and metric helpers work without torch.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output_FROZEN"
MODELS = ["bert-base-multilingual-cased", "xlm-roberta-base", "google/muril-base-cased"]
# Hugging Face commit of each model repo: the snapshots used by tokenizer_stats.py (see results/tokenizer_versions.txt)
REVISIONS = {"bert-base-multilingual-cased": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
             "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
             "google/muril-base-cased": "afd9f36c7923d54e97903922ff1b260d091d202f"}
SEEDS = [13, 42, 2026]
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
PROB_COLS = ["prob_neg", "prob_neu", "prob_pos"]
LR, BATCH, MAX_LEN, MAX_EPOCHS, PATIENCE = 2e-5, 32, 256, 5, 2
WEIGHT_DECAY, WARMUP_SHARE = 0.01, 0.10
PILOT_ROWS = 200
PRED_COLS = ["model", "seed", "id", "variant", "pred", "label", *PROB_COLS, "confidence", "margin",
             "n_tokens", "complete_tweet"]


def short(model):
    return model.split("/")[-1]


def preds_path(out_dir, model, seed):
    return Path(out_dir) / "preds" / f"{short(model)}_{seed}.csv"


def val_path(out_dir, model, seed):
    return Path(out_dir) / f"val_{short(model)}_{seed}.json"


def read(path):
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False, encoding="utf-8")
    print(f"{path.name}: {len(df)} rows, columns {list(df.columns)}")
    return df


def to_ids(labels):
    bad = set(labels) - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels {bad}")
    return labels.map(LABEL2ID).to_numpy()


def check_leakage(train, val, eval_ids):
    """Stop if any id is shared between train, val and the eval set."""
    tr, va, ev = set(train.id), set(val.id), set(eval_ids)
    overlap = {"train/val": tr & va, "train/eval": tr & ev, "val/eval": va & ev}
    bad = {k: sorted(v)[:10] for k, v in overlap.items() if v}
    if bad:
        raise ValueError(f"id overlap (first 10 per pair): {bad}")


def load_data(src=SRC):
    """train, val, eval frames. eval = every variant of variants_long.csv + NEG from neg_control.csv."""
    train, val = read(src / "train.csv"), read(src / "val.csv")
    long, neg = read(src / "variants_long.csv"), read(src / "neg_control.csv")
    neg = neg.rename(columns={"text_NEG": "text"}).assign(variant="NEG")
    cols = ["id", "label", "variant", "text", "complete_tweet"]
    ev = pd.concat([long[cols], neg[cols]], ignore_index=True)
    check_leakage(train, val, ev.id)
    for name, df in [("train", train), ("val", val), ("eval", ev)]:
        df["y"] = to_ids(df.label)
    if (ev.text.str.strip() == "").any():
        raise ValueError("empty eval text")
    print(f"label map: {LABEL2ID}")
    print(f"eval rows: {len(ev)} {ev.variant.value_counts().to_dict()}")
    return train, val, ev


def macro_f1(y_true, y_pred, n_classes=3):
    """Mean F1 over all classes; a class with no true and no predicted rows scores 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1))


def pred_frame(ev, probs, n_tokens, model, seed):
    """One row per eval text: argmax pred, its probability (confidence) and top1 - top2 (margin)."""
    probs = np.asarray(probs, dtype=float)
    top = np.sort(probs, axis=1)
    df = pd.DataFrame({"model": model, "seed": seed, "id": ev.id.to_numpy(), "variant": ev.variant.to_numpy(),
                       "pred": probs.argmax(1), "label": ev.y.to_numpy()})
    df[PROB_COLS] = probs
    df["confidence"] = top[:, -1]
    df["margin"] = top[:, -1] - top[:, -2]
    df["n_tokens"] = n_tokens
    df["complete_tweet"] = ev.complete_tweet.to_numpy()
    return df[PRED_COLS]


def run(args):
    import torch
    import tokenizers
    import transformers
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              get_linear_schedule_with_warmup)

    t0 = time.time()
    out = Path(args.out_dir or ROOT / ("results_pilot" if args.pilot else "results"))
    (out / "preds").mkdir(parents=True, exist_ok=True)
    train, val, ev = load_data()
    if args.pilot:
        train = train.sample(PILOT_ROWS, random_state=args.seed)
    epochs = 1 if args.pilot else MAX_EPOCHS

    transformers.set_seed(args.seed)  # python, numpy, torch (CPU and CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    print(f"device {device}, fp16 {fp16}, train rows {len(train)}, epochs <= {epochs}")

    revision = REVISIONS[args.model]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, revision=revision)
    collate = DataCollatorWithPadding(tok)

    def features(df, with_labels):
        enc = tok(df.text.tolist(), truncation=True, max_length=MAX_LEN)
        feats = [{"input_ids": i, "attention_mask": m} for i, m in zip(enc["input_ids"], enc["attention_mask"])]
        if with_labels:
            for f, y in zip(feats, df.y):
                f["labels"] = int(y)
        return feats

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=revision, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL2ID).to(device)

    @torch.inference_mode()
    def predict(df):
        model.eval()
        probs = []
        for batch in DataLoader(features(df, False), batch_size=BATCH * 2, collate_fn=collate):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=fp16):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(probs)

    train_dl = DataLoader(features(train, True), batch_size=BATCH, shuffle=True, collate_fn=collate,
                          generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # schedule spans all `epochs` even if early stopping ends training sooner
    total_steps = epochs * len(train_dl)
    warmup_steps = int(WARMUP_SHARE * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    print(f"schedule: {total_steps} total steps, {warmup_steps} warmup, then linear decay to 0")
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    if fp16:
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, best_state, history, bad = -1.0, 0, None, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=fp16):
                loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        f1 = macro_f1(val.y, predict(val).argmax(1))
        history.append({"epoch": epoch, "val_macro_f1": f1, "last_train_loss": float(loss),
                        "lr": sched.get_last_lr()[0]})
        print(f"epoch {epoch}: val macro-F1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop: no improvement for {PATIENCE} epochs")
                break
    model.load_state_dict(best_state)

    probs = predict(ev)
    n_tokens = [len(x) for x in tok(ev.text.tolist(), add_special_tokens=False)["input_ids"]]
    preds = pred_frame(ev, probs, n_tokens, args.model, args.seed)

    info = {"model": args.model, "revision": revision, "seed": args.seed, "pilot": args.pilot, "best_epoch": best_epoch,
            "val_macro_f1": best, "history": history, "train_rows": len(train), "device": device,
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20) if fp16 else None,
            "seconds": round(time.time() - t0)}
    val_path(out, args.model, args.seed).write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "versions.txt").write_text(
        f"python {platform.python_version()}\ntorch {torch.__version__}\ntransformers {transformers.__version__}\n"
        f"tokenizers {tokenizers.__version__}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
        f"cuda {torch.version.cuda}\ngpu {torch.cuda.get_device_name(0) if fp16 else 'none'}\n"
        + "".join(f"{m} revision={r}\n" for m, r in REVISIONS.items()), encoding="utf-8")
    if args.save_ckpt:
        ckpt = out / "ckpt" / f"{short(args.model)}_{args.seed}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
    # preds file last, via rename: its existence is what run_all.py treats as "run finished"
    path = preds_path(out, args.model, args.seed)
    tmp = path.with_suffix(".tmp")
    preds.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    print(f"wrote {path} ({len(preds)} rows); best epoch {best_epoch}, val macro-F1 {best:.4f}; "
          f"{info['seconds']}s, peak GPU {info['peak_gpu_mem_mb']} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pilot", action="store_true", help=f"1 epoch, {PILOT_ROWS} train rows")
    ap.add_argument("--out_dir", help="default results/ (results_pilot/ with --pilot)")
    ap.add_argument("--save_ckpt", action="store_true", help="save the best model (off by default)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

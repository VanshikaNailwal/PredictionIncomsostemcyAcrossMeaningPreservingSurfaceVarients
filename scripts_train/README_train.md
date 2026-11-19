# Training on Colab

3 models x 3 seeds, trained on `output_FROZEN/train.csv`, early stopping on `val.csv`, then every run
predicts all 8 variants of the 1,000 base sentences plus the 352 NEG texts (8,352 rows per run).

| script | what it does |
|---|---|
| `train_eval.py` | one model + one seed: train, pick best epoch on val macro-F1, predict, write preds |
| `run_all.py` | all 9 runs, one subprocess each; skips runs whose preds file exists (resume after disconnect) |
| `make_flips.py` | `flips_long.csv`: each variant's prediction vs the A prediction |
| `test_train_logic.py` | data, leakage, metric and flip checks; needs no torch or GPU |

Settings: AdamW lr 2e-5 (weight decay 0.01), linear warmup 10% then linear decay (to 0 over 5 epochs' worth
of steps, even if early stopping ends sooner; the pilot uses its own 1-epoch total), batch 32, max_len 256, up to 5 epochs, patience 2 on val macro-F1, fp16 on GPU.
Labels: negative=0, neutral=1, positive=2. Model files are pinned to fixed Hugging Face commits
(`REVISIONS` in `train_eval.py`), the same ones used for `results/tokenizer_stats.csv`.

## Before you start

Upload `output_FROZEN/` and `scripts_train/` to Google Drive as `MyDrive/hinglish_thesis/`.
In Colab: Runtime > Change runtime type > T4 GPU.

## Cells

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. Copy the frozen data and scripts to the fast local disk; results go to Drive (survives disconnects)
!mkdir -p /content/hinglish_thesis
!cp -r /content/drive/MyDrive/hinglish_thesis/output_FROZEN /content/drive/MyDrive/hinglish_thesis/scripts_train /content/hinglish_thesis/
%cd /content/hinglish_thesis
OUT = '/content/drive/MyDrive/hinglish_thesis/results'
```

```python
# 3. Install pinned versions and check the GPU
!pip install -q -r scripts_train/requirements_train.txt
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# 4. Logic checks (no GPU needed, a few seconds). Must end with "7 tests passed"
!python scripts_train/test_train_logic.py
```

```python
# 5. Pilot: 1 epoch on 200 rows, ~1-2 min. Check that it ends with "wrote ... (8352 rows)"
!python scripts_train/train_eval.py --model xlm-roberta-base --seed 13 --pilot --out_dir /content/drive/MyDrive/hinglish_thesis/results_pilot
```

```python
# 6. All 9 runs. If Colab disconnects, run cells 1-3 and this one again: finished runs are skipped
!python scripts_train/run_all.py --out_dir {OUT}
```

```python
# 7. Flip table
!python scripts_train/make_flips.py --out_dir {OUT}
```

## Expected runtime (estimate for a T4, not measured yet)

About 1-1.5 min per epoch (11,335 rows, short tweets with dynamic padding), plus about 30 s to predict the
8,352 eval texts. So roughly 4-8 min per run, depending on when early stopping hits, and about 1 hour for
all 9 runs. Use the real numbers from `results/run_log.csv` after the pilot and the first run.

## Outputs (in `results/` on Drive)

- `preds/{model}_{seed}.csv`: model, seed, id, variant, pred, label, prob_neg, prob_neu, prob_pos,
  confidence, margin, n_tokens, complete_tweet. `{model}` is the name after the last `/`
  (`muril-base-cased`). `n_tokens` excludes special tokens, as in `tokenizer_stats.csv`.
- `val_{model}_{seed}.json`: best epoch, best val macro-F1, per-epoch history, seconds, peak GPU memory.
- `run_log.csv`: one line per run from `run_all.py`.
- `versions.txt`: Python, torch, transformers, tokenizers, CUDA, GPU and model revisions of the last run.
  Cite this Python version in the paper.
- `flips_long.csv`: from `make_flips.py`.
- Checkpoints only with `--save_ckpt` (about 0.7-1.1 GB each, in `ckpt/`).

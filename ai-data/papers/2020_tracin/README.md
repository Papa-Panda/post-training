# TracIn - Training Data Influence via Tracing Gradient Descent

Minimal, practical implementation of TracIn (Pruthi et al., NeurIPS 2020) for data cleaning.

This repo is the companion to the paper reading note:
**Estimating Training Data Influence by Tracing Gradient Descent**  
arXiv: https://arxiv.org/abs/2002.08484 | Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/

## Why this repo

Influence Functions needs Hessian inverse ($H^{-1}$) - slow and unstable.  
TracIn replaces it with checkpoint gradient dot products:

```
TracIn(z, z') = sum_t  eta_t * grad L(z', theta_t) · grad L(z, theta_t)
```

3 checkpoints are enough to find mislabeled / outlier code data.

## What's inside

- `NOTES.md` - 1-page paper summary (motivation / pipeline / 3 tricks / results / transferable ideas)
- `tracin_demo.py` - Minimal 80-line toy demo, CPU-only, finds 10 noisy labels via self-influence
- `requirements.txt`

## Quickstart

```bash
pip install torch
python tracin_demo.py
```

Output: Top 10 self-influence indices -> these are likely dirty data.

## Transfer to coding data

1. Train a 1B proxy for 3 epochs, save 3 checkpoints
2. Compute self-influence for all training code files
3. Top 2% export for human review -> delete outdated API / wrong labels / copy-paste Q&A

See NOTES.md for full 2-point transfer plan.

## References

- Paper PDF: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
- Official TF code: https://github.com/frederick0329/TracIn

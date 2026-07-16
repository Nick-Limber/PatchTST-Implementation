# Implementing PatchTST

A training-focused project: Implement PatchTST from
the published paper (https://arxiv.org/abs/2211.14730) and 
train it from scratch on sales time series data,
starting with synthetic data and moving to the M5 Forecasting
dataset.

## Current status

- **Phase 1** -- synthetic data, prove the training loop and base PatchTST
  architecture work at all (in progress)

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```

## Key decisions

- **Why synthetic data first** -- Synthetic data with a known, 
controllable structure makes it possible to distinguish 
the difference between "the training loop has a bug" and
"real data is messy" -- a model that can't learn a clean synthetic
pattern has a pipeline bug, instead of a data-quality problem. 
- **No pretrained weights** -- the model is trained entirely from scratch,
  so the architecture, training loop, and evaluation are all visible.

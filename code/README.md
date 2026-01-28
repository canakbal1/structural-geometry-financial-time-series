# Code Overview

This directory contains a reference implementation of the methodology described in the accompanying unpublished working paper.

The code is organized to emphasize conceptual clarity and reproducibility rather than production optimization.

## Structure

- `main.py`  
  End-to-end pipeline that downloads data, constructs time-delay embeddings, computes rolling covariance matrices, and evaluates structural diagnostics.

## Notes

- The implementation is diagnostic and descriptive only; no predictive modeling is performed.
- Parameters are intentionally shared across assets to avoid asset-specific tuning.

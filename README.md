# Drug Likeness Model (DLM)

## Overview
The Drug Likeness Model (DLM) is a four-layer Self-Normalizing Neural Network trained on bioactivity-grounded ChEMBL 34 data for drug-likeness prediction across both classical (Ro5) and Beyond Rule of Five (bRo5) chemical space, including macrocycles, PROTACs, and stapled peptides.

Unlike existing models that use FDA-approved drugs as positives and  screening libraries as negatives, the DLM grounds its labels in  experimentally validated bioactivity: confirmed high-affinity binding (IC50 < 50 nM) or clinical progression (Phases 1-4).

## Features
- 93.76% accuracy on random holdout (ROC-AUC 0.9771)
- 87.63% accuracy on scaffold-split out-of-distribution generalization
- 94.43% accuracy on blind clinical candidate validation
- bRo5-aware piecewise scoring function for interpretable triage

## Requirements
[dependencies]

## Installation
[installation instructions]

## Usage
[usage instructions]

## Citation
TBD

## License
[license]

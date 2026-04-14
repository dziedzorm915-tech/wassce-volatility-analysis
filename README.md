# WASSCE Volatility Analysis

## Overview
This repository contains the complete Python code for the study: **"Variation in WASSCE Performance in Core Subjects Using Volatility Models"**

**Author:** Roland Ankudze, AIMS Ghana (2026)  
**Supervisor:** Dr. Benedict Mbeah-Baiden

## Methodology
- AR(1)-ARCH(1) model with QMLE
- Parametric bootstrap (B = 1000)
- Engle LM test with bootstrap inference
- Block bootstrap for cross-subject analysis (B = 1000)

## Mathematical Properties
| Property | Formula |
|----------|---------|
| Conditional variance | σ²_t = ω + α₁ε²_{t-1} |
| Unconditional variance | σ² = ω/(1-α₁) |
| Kurtosis | κ = 3(1-α₁²)/(1-3α₁²) |
| Half-life | h = ln(0.5)/ln(α₁) |

## Requirements
- Python 3.8+
- pandas, numpy, scipy, matplotlib, statsmodels, seaborn

## Installation & Usage
```bash
pip install -r requirements.txt
python wassce_analysis.py
# T-RLPM: Transformer-Based Reinforcement Learning for Portfolio Management

This repository contains the code implementation of **T-RLPM**, my final project for **ISE 617 – Reinforcement Learning**, taught by [**Dr. Ye Hu**](https://scholar.google.com/citations?user=TZ_qFpAAAAAJ&hl=en) at the University of Miami.  

## 📘 Project Overview

T-RLPM is a transformer-based reinforcement learning framework for dynamic portfolio optimization. It builds directly upon the EarnMore framework by incorporating key improvements:

- Using transformer-style encoders for both the actor and critic networks,
- Improving risk-adjusted returns over state-of-the-art baselines such as EarnMore and DeepTrader in the GSP (global stock pool) setting.

We evaluate the framework using the DJIA (Dow Jones Industrial Average) dataset and compare it with classical, ML-based and RL-based baselines. The model achieved a **59.87% annual return**, outperforming all previous benchmarks in GSP evaluation.

However, when evaluated on Customizable Stock Pools (CSPs), T-RLPM struggles to generalize and its performance falls significantly below state-of-the-art levels, highlighting key limitations in masked learning dynamics when combined with transformer architectures.

---

```
python setup.py install
cd ..
pip install -r requirements.txt
```

# RUN
```
PYTHONUNBUFFERED=1 python tools/train.py --config configs/transformer_attn_mask_sac_pm.py
```

# Reference code
ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL

RL-Adventure: https://github.com/higgsfield/RL-Adventure

Qlib: https://github.com/microsoft/qlib

EarnMore: https://github.com/DVampire/EarnMore



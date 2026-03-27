# Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies
This repository contains the code used for the experiments in ["Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies"](https://openreview.net/forum?id=0ZkWWxcHKV&noteId=h2mwZwoWrG) by Koichi Tanaka, Kazuki Kawamura, Takanori Muroi, Yusuke Narita, Yuki Sasamoto, Kei Tateno, Takuma Udagawa, Wei-Wei Du, Yuta Saito. This paper was accepted at [ICLR 2026](https://iclr.cc/Conferences/2026).

## Abstract
Off-Policy Evaluation (OPE) is an important practical problem in algorithmic ranking systems, where the goal is to estimate the expected performance of a new ranking policy using only offline logged data collected under a different, logging policy. Existing estimators, such as the ranking-wise and position-wise inverse propensity score (IPS) estimators, require the data collection policy to be sufficiently stochastic and suffer from severe bias when the logging policy is deterministic. In this paper, we propose novel estimators, Click-based Inverse Propensity Score (CIPS) and Click-based Doubly Robust (CDR), which exploit the intrinsic stochasticity of user click behavior to address this challenge. Unlike existing methods that rely on the stochasticity of the logging policy, our approach uses click probability as a new form of importance weighting, enabling low-bias OPE even under deterministic logging policies where existing methods incur substantial bias. We provide theoretical analyses of the bias and variance properties of the proposed estimators and show, through synthetic and real-world experiments, that our estimators achieve significantly lower bias compared to strong baselines, particularly in settings with completely deterministic logging policies.

## Citation
```
@inproceedings{
    tanaka2026offpolicy,
    title={Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies},
    author={Koichi Tanaka and Kazuki Kawamura and Takanori Muroi and Yusuke Narita and Yuki Sasamoto and Kei Tateno and Takuma Udagawa and Wei-Wei Du and Yuta Saito},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=0ZkWWxcHKV}
}
```

## Setup
The Python environment is built using uv. You can build the same environment as in our experiments by cloning the repository and running uv sync directly under the folder.
```
# build the environment with uv
uv sync

# activate the environment
source .venv/bin/activate
```

## Runing the code
### Section 5.1: SYNTHETIC DATA
The commands needed to reproduce the experiments are summarized below. Please move under the `src` directly first and then run the commands.
```
cd src

# How does CIPS perform when varying the logged data size? 
uv run python main_num_data.py setting.run_file="main_num_data"

# How does CIPS perform when varying the ranking length?
uv run python ./src/main_len_list.py setting.run_file="main_len_list"

# How does CIPS perform when violating the independence of potential rewards condition? 
uv run python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion"

# How does CIPS perform with different levels of logging policy stochasticity?
uv run python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio"

```

### Section 5.2: REAL-WORLD DATA
We use [KuaiRec](https://kuairec.com) in our real-world experiments. Please download the above datasets from the repository and put them under `./real/data/`. Then, run the following.


```
cd real

# Figure 5: varying logged data sizes
uv run python ./real/kuairec.py setting.run_file="kuairec"

# Figure 6: varying deterministic user thresholds
uv run python ./real/main_deterministic_ratio.py setting.run_file="kuairec_deterministic_ratio"
```

# Authors of the paper
* Koichi Tanaka (Hanjuku-kaso Co., Ltd. / Keio University)
* Kazuki Kawamura (Sony Group Corporation)
* Takanori Muroi (Sony Group Corporation)
* Yusuke Narita (Hanjuku-kaso Co., Ltd. / Yale University)
* Yuki Sasamoto (Sony Group Corporation)
* Kei Tateno (Sony Group Corporation)
* Takuma Udagawa (Sony Group Corporation)
* Weiwei Du (Sony Group Corporation)
* Yuta Saito (Hanjuku-kaso Co., Ltd.)

# Contact
For any question about the paper and code, feel free to contact: Takuma.Udagawa@sony.com

# Licence
This software is released under the MIT License, see LICENSE for the detail.

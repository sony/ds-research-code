# Off-Policy Evaluation and Learning for the Future under Non-Stationarity

This repository contains the code used for the experiments performed in the paper "[Off-Policy Evaluation and Learning for the Future
under Non-Stationarity](https://dl.acm.org/doi/abs/10.1145/3690624.3709237)", accepted at [KDD2025](https://kdd2025.kdd.org/).

## abstract

We study the novel problem of future off-policy evaluation (F-OPE) and learning (F-OPL) for estimating and optimizing the future value of policies in non-stationary environments, where distributions vary over time. In e-commerce recommendations, for instance, our goal is often to estimate and optimize the policy value for the upcoming month using data collected by an old policy in the previous month. A critical challenge is that data related to the future environment is not observed in the historical data. Existing methods assume stationarity or depend on restrictive reward-modeling assumptions, leading to significant bias. To address these limitations, we propose a novel estimator named Off-Policy Estimator for the Future Value (OPFV), designed for accurately estimating policy values at any future time point. The key feature of OPFV is its ability to leverage the useful structure within time-series data. While future data might not be present in the historical log, we can leverage, for example, seasonal, weekly, or holiday effects that are consistent in both the historical and future data. Our estimator is the first to exploit these time-related structures via a new type of importance weighting, enabling effective F-OPE. Theoretical analysis identifies the conditions under which OPFV becomes low-bias. In addition, we extend our estimator to develop a new policy-gradient method to proactively learn a good future policy using only historical data. Empirical results show that our methods substantially outperform existing methods in estimating and optimizing the future policy value under non-stationarity for various experimental setups.


## Requirements and Setup

The versions of Python and necessary packages are specified as follows.

```bash
python = "3.8.18"
matplotlib = "^3.7.2"
numpy = "^1.23.5"
obp = "^0.5.5"
pandas = "^2.0.3"
scikit-learn = "^1.3.0"
scipy = "^1.10.1"
seaborn = "^0.12.2"
```

## Running the Code

### Section 4: Synthetic Data

```
# How does OPFV perform with varying target time?
synthetic/F-OPE/non-stationary-reward/standard/main_target_time.ipynb

# How does OPFV perform with varying informativeness of the time feature effect?
synthetic/F-OPE/non-stationary-reward/standard/main_lambda.ipynb

# How does OPFV perform with varying number of time feature for OPFV?
synthetic/F-OPE/non-stationary-reward/standard/main_num_time_feature.ipynb

# How does OPFV perform with varying logged data sizes?
synthetic/F-OPE/non-stationary-reward/standard/main_n_trains.ipynb

# How does OPFV-PG perform with varying target time?
synthetic/F-OPL/main_target_time.ipynb

# How does OPFV-PG perform with varying logged data sizes?
synthetic/F-OPL/main_n_trains.ipynb
```

### Section 4: Real-World Data

We use "KuaiRec" from [A Fully-observed Dataset for Recommender Systems](https://kuairec.com/). Please download the datasets from the repository and put them in the following way.

```
KuaiRec
   └ data
      ├ big_matrix.csv
      ├ item_categories.csv
      ├ item_daily_features.csv
      ├ small_matrix.csv
      ├ social_network.csv
      ├ user_features.csv

neurips2024-opfv/
   └ src/
      ├ synthetic/
      ├ real/
          ├ F-OPL/
              ├ main.ipynb
              ├ main-opfv-tune-phi.ipynb
```

Then, run the following notebook.

```
main.ipynb
main-opfv-tune-phi.ipynb
```

# Authors of the paper
* Tatsuhiro Shimizu (Yale University / Hanjuku-kaso Co., Ltd.)
* Kazuki Kawamura (Sony Group Corporation)
* Takanori Muroi (Sony Group Corporation)
* Yusuke Narita (Hanjuku-kaso Co., Ltd. / Yale University)
* Kei Tateno (Sony Group Corporation)
* Takuma Udagawa (Sony Group Corporation)
* Yuta Saito (Cornell University / Hanjuku-kaso Co., Ltd.)

# Contact
For any question about the paper and code, feel free to contact: Takuma.Udagawa@sony.com

# Licence
This software is released under the MIT License, see LICENSE for the detail.

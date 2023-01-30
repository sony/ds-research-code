# Policy Adaptive Estimator Selection for Off-Policy Evaluation


## About

This code accompanies the experiments conducted in the paper ["Policy-Adaptive Estimator Selection for Off-Policy Evaluation"](https://arxiv.org/abs/2211.13904), which has been accepted at AAAI'23.

If you find this paper and code useful in your research, then please cite:
```
@inproceedings{PolicyAdaptiveEstimatorSelection2023,
  title={Policy-Adaptive Estimator Selection for Off-Policy Evaluation},
  author={Udagawa, Takuma and Kiyohara, Haruka and Narita, Yusuke and Saito, Yuta and Tateno, Kei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={xxx},
  year={2023}
}
```

## Dependencies

- **python>=3.8.12**
- matplotlib>=3.4.2
- numpy>=1.21.2
- obp>=0.5.5
- pandas>=1.3.2
- scikit-learn>=1.0.0
- scipy>=1.7.1
- torch>=1.9.0


## Running the code

To run the simulation with synthetic data, navigate to the `src/evaluation` directory and run the command

```
python evaluation_with_synthetic_data.py \
  --n_actions 10 \
  --dim_context 10 \
  --reward_type binary \
  --reward_function 0 \
  --beta_1 -2 \
  --beta_2 2 \
  --n1 1000 \
  --n2 1000 \
  --random_state 1 \
  --beta_list_for_pi_e -10 -5 0 5 10 \
  --model_list_for_pi_e 0 0 0 0 0 \
  --metric mse \
  --n_bootstrap 10 \
  --n_data_generation 100 \
  --proposed_method pasif \
  --pasif_k 0.2 0.2 0.2 0.2 0.2 \
  --pasif_regularization_weight -997 -997 -997 -997 -997 \
  --pasif_batch_size 2000 2000 2000 2000 2000 \
  --pasif_n_epochs 5000 5000 5000 5000 5000 \
  --pasif_optimizer 1 1 1 1 1 \
  --pasif_lr 0.001 0.001 0.001 0.001 0.001 \
  --gt_n_sampling 100
```

This will run synthetic experiments with $(\beta_{1}, \beta_{2})=(-2,2)$ and $\beta_{e} \in \{-10, -5, 0, 5, 10\}$. If you want to know the detailed meaning of the command, please read the explanation of argument in source code. If you refer to the paper and change the value of the argument, you can perform the same experiment as the paper.


## Authors of the paper
- Takuma Udagawa (Sony Group Corporation)
- Haruka Kiyohara (Tokyo Institute of Technology / Hanjuku-kaso Co., Ltd.)
- Yusuke Narita (Hanjuku-kaso Co., Ltd. / Yale University)
- Yuta Saito (Cornell University / Hanjuku-kaso Co., Ltd.)
- Kei Tateno (Sony Group Corporation)

## Contact
For any question about the paper and code, feel free to contact: Takuma.Udagawa@sony.com

## Licence
This software is released under the MIT License, see LICENSE for the detail.


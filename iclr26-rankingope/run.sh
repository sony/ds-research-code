# 5.1 SYNTHETIC DATA

# uv run python ./src/main_num_data.py \
#   setting.run_file="main_num_data" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"

# uv run python ./src/main_len_list.py \
#   setting.run_file="main_len_list" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"

# uv run python ./src/main_effect_conversion.py \
#   setting.run_file="main_effect_conversion" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"

# uv run python ./src/main_deterministic_ratio.py \
#   setting.run_file="main_deterministic_ratio" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"


# 5.2 REAL-WORLD DATA

# uv run python ./real/kuairec.py \
#   setting.run_file="kuairec" \
#   setting.real.reward_type_conversion=continuous \
#   setting.real.len_list=6 \
#   setting.real.evaluation_policy_logit=linear_behavior_policy_logit \
#   setting.real.deterministic_user_threshold=inf

# uv run python ./real/main_deterministic_ratio.py \
#   setting.run_file="kuairec_deterministic_ratio" \
#   setting.real.reward_type_conversion=continuous \
#   setting.real.len_list=6 \
#   setting.real.evaluation_policy_logit=linear_behavior_policy_logit \
#   setting.real.deterministic_user_threshold=inf


# Appendix

# uv run python ./src/main_estimation_noise.py \
#   setting.run_file="main_estimation_noise" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"

# uv run python ./src/main_eps.py \
#   setting.run_file="main_eps" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"


# selection

# uv run python ./src_selection/main_num_data.py \
#   setting.run_file="main_num_data" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"

# uv run python ./src_selection/main_deterministic_ratio.py \
#   setting.run_file="main_deterministic_ratio" \
#   setting.reward_type_conversion="continuous" \
#   setting.len_list=6 \
#   setting.evaluation_policy_logit="linear_behavior_policy_logit" \
#   setting.deterministic_user_threshold="inf"
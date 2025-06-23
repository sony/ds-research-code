# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import warnings

warnings.filterwarnings("ignore")
import datetime
import time

import conf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from regression_model_time import RegressionModelTimeTrue
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from utils_time import (
    obtain_num_time_structure,
    unix_time_to_date,
    unix_time_to_day_of_week,
    unix_time_to_day_of_week_and_date,
    unix_time_to_day_of_week_and_hour,
    unix_time_to_day_of_week_and_hour_and_date,
    unix_time_to_hour,
    unix_time_to_hour_and_date,
)

NUM_HOUR = 24
NUM_OFF_DAY_OR_NOT = 2


def obtain_column_names_with_NA(df):
    # Identify columns with NA values
    columns_with_na = df.columns[df.isna().any()].tolist()

    # Display columns with NA values
    if conf.flag_show_parameters == True:
        print("Columns with NA values:", columns_with_na)


def create_new_column_whose_element_is_list(
    data_frame,
    column_list_to_be_excluded_for_list,
    new_column_name,
    flag_drop_original_column,
):
    # Copy the DataFrame
    df = data_frame.copy()

    # Obtain the list of all columns
    all_columns_list = df.columns

    # Create the list of columns to be included in new column
    column_list_to_be_included_for_list = [
        x for x in all_columns_list if x not in column_list_to_be_excluded_for_list
    ]

    # Create the elements of the new column
    for i in range(len(column_list_to_be_included_for_list)):
        if i == 0:
            elem = df[column_list_to_be_included_for_list[i]]
        else:
            elem = np.column_stack([elem, df[column_list_to_be_included_for_list[i]]])

    # Apply the function to create a new column as a list
    df[new_column_name] = elem.tolist()

    # Drop the original columns if needed
    if flag_drop_original_column == True:
        df = df.drop(columns=column_list_to_be_included_for_list)

    return df


def preprocess_item_categories(item_categories):
    time_preprocess_item_categories_start = time.time()

    # Copy the DataFrame
    item_categories_preprocessed = item_categories.copy()

    # Create a set of unique values from all lists in 'Column_with_lists'
    unique_values = set(
        value for sublist in item_categories_preprocessed["feat"] for value in sublist
    )

    # Create a new column for each unique value and initialize with 0
    for value in unique_values:
        item_categories_preprocessed[value] = item_categories_preprocessed[
            "feat"
        ].apply(lambda x: 1 if value in x else 0)

    # Drop the original 'Column_with_lists'
    item_categories_preprocessed = item_categories_preprocessed.drop("feat", axis=1)

    # Create a new column that is a list of 0 or 1
    item_categories_preprocessed["tag_context"] = item_categories_preprocessed.apply(
        lambda row: [int(row[col]) for col in item_categories_preprocessed.columns[1:]],
        axis=1,
    )

    column_list_for_item_categories_preprocessed = ["video_id", "tag_context"]
    item_categories_preprocessed = item_categories_preprocessed[
        column_list_for_item_categories_preprocessed
    ]

    time_preprocess_item_categories_end = time.time()

    if conf.flag_show_execution_time_for_preprocess == True:
        execution_time = (
            time_preprocess_item_categories_end - time_preprocess_item_categories_start
        )
        print(
            f"Execution time of preprocess_item_categories = {execution_time / 60:.3f} mins"
        )

    return item_categories_preprocessed


def create_one_hot_embedding_columns(df, string_columns):
    dataframe = df.copy()
    # Create one-hot encoding for string columns
    one_hot_encoded = pd.get_dummies(dataframe[string_columns])

    # Convert the resulting DataFrame to int datatype
    one_hot_encoded = one_hot_encoded.astype(int)

    # Concatenate the new one-hot encoded columns with the original DataFrame
    dataframe = pd.concat([dataframe, one_hot_encoded], axis=1)

    # Drop the original string columns
    dataframe = dataframe.drop(string_columns, axis=1)
    return dataframe


def preprocess_item_daily_features(item_daily_features):
    if conf.flag_show_execution_time_for_preprocess == True:
        print(f"Executing preprocess_item_daily_features")

    time_preprocess_item_daily_features_start = time.time()
    ### Descriptive Analysis ###
    if conf.flag_show_parameters == True:
        print(
            f"The number of rows in the DataFrame of item daily features = {len(item_daily_features)}"
        )
        print(
            f"The number of unique videos = {len(item_daily_features['video_id'].unique())}\n"
        )

        print(
            f"The number of unique video type = {len(item_daily_features['video_type'].unique())}"
        )
        print(
            f"The number of unique upload date of the video = {len(item_daily_features['upload_dt'].unique())}"
        )
        print(
            f"The number of unique upload type of the video = {len(item_daily_features['upload_type'].unique())}"
        )
        print(
            f"The number of unique visible status = {len(item_daily_features['visible_status'].unique())}"
        )
        print(
            f"The number of unique authors = {len(item_daily_features['author_id'].unique())}"
        )
        print(
            f"The number of unique music ID = {len(item_daily_features['music_id'].unique())}"
        )
        print(
            f"The number of unique video tag ID = {len(item_daily_features['video_tag_id'].unique())}"
        )
        print(
            f"The number of unique video tag name = {len(item_daily_features['video_tag_name'].unique())}\n"
        )

    ### Copy the item_daily_features DataFrame ###
    item_daily_features_preprocessed = item_daily_features.copy()

    ### Drop the columns which contain NA values: ['video_duration', 'video_tag_name', 'collect_cnt', 'collect_user_num', 'cancel_collect_cnt', 'cancel_collect_user_num'] ###
    item_daily_features_preprocessed = item_daily_features_preprocessed.dropna(axis=1)

    ### Chnage the datatype of the columns to be conisdered as categories rather than quantities ###
    item_daily_features_preprocessed["author_id"] = item_daily_features_preprocessed[
        "author_id"
    ].astype(str)
    item_daily_features_preprocessed["music_id"] = item_daily_features_preprocessed[
        "music_id"
    ].astype(str)
    item_daily_features_preprocessed["video_tag_id"] = item_daily_features_preprocessed[
        "video_tag_id"
    ].astype(str)

    ### Drop ["auhtor_id", "music_id"] because these columns will create too sparse one hot embeddings ###
    columns_list_to_be_dropped = ["author_id", "music_id"]
    item_daily_features_preprocessed = item_daily_features_preprocessed.drop(
        columns=columns_list_to_be_dropped
    )

    ### Convert the upload_dt to unix_time ###
    # Convert the 'upload_dt' to datetime objects
    item_daily_features_preprocessed["upload_dt"] = pd.to_datetime(
        item_daily_features_preprocessed["upload_dt"]
    )

    # Create a new column with Unix timestamp
    item_daily_features_preprocessed["upload_ut"] = (
        item_daily_features_preprocessed["upload_dt"].astype(int) / 10**9
    )  # Convert nanoseconds to seconds

    # Drop the upload_dt
    item_daily_features_preprocessed = item_daily_features_preprocessed.drop(
        columns="upload_dt"
    )

    ### Create item_features DataFrame that is not dependent on the time
    column_list_to_be_included_in_new_df = [
        "video_id",
        "video_type",
        "upload_ut",
        "upload_type",
        "visible_status",
        "video_width",
        "video_height",
        "video_tag_id",
    ]
    item_features_preprocessed = item_daily_features_preprocessed[
        column_list_to_be_included_in_new_df
    ]

    ### Create One Hot embedding for categorical variables ###

    # Identify string columns
    string_columns = item_daily_features_preprocessed.select_dtypes(
        include="object"
    ).columns
    if conf.flag_show_parameters == True:
        print(f"The columns to be converted to one hot embedding = {string_columns}")

    # Create one hot embedding
    item_daily_features_preprocessed = create_one_hot_embedding_columns(
        df=item_daily_features_preprocessed, string_columns=string_columns
    )
    item_features_preprocessed = create_one_hot_embedding_columns(
        df=item_features_preprocessed, string_columns=string_columns
    )

    # Take the mean for each video_id
    item_features_preprocessed = (
        item_features_preprocessed.groupby(by="video_id").mean().reset_index()
    )

    ### Create a new column whose element is a list of the elements in the other columns
    column_list_to_be_excluded_for_list_for_daily = ["video_id", "date"]
    column_list_to_be_excluded_for_list = ["video_id"]

    item_daily_features_preprocessed = create_new_column_whose_element_is_list(
        data_frame=item_daily_features_preprocessed,
        column_list_to_be_excluded_for_list=column_list_to_be_excluded_for_list_for_daily,
        new_column_name="item_daily_feature_context",
        flag_drop_original_column=True,
    )
    item_features_preprocessed = create_new_column_whose_element_is_list(
        data_frame=item_features_preprocessed,
        column_list_to_be_excluded_for_list=column_list_to_be_excluded_for_list,
        new_column_name="item_daily_feature_context",
        flag_drop_original_column=True,
    )

    time_preprocess_item_daily_features_end = time.time()
    total_execution_time = (
        time_preprocess_item_daily_features_end
        - time_preprocess_item_daily_features_start
    )
    if conf.flag_show_execution_time_for_preprocess == True:
        print(
            f"Execution time for preprocess_item_daily_features = {total_execution_time / 60:.3f} mins"
        )

    return item_daily_features_preprocessed, item_features_preprocessed


def create_df_action_context(
    item_categories_preprocessed,
    item_features_preprocessed,
    item_daily_features_preprocessed,
):
    df_action_context = pd.merge(
        left=item_categories_preprocessed,
        right=item_features_preprocessed,
        how="inner",
        on="video_id",
    )

    df_action_context["action_context"] = (
        df_action_context["tag_context"]
        + df_action_context["item_daily_feature_context"]
    )
    df_action_context = df_action_context.drop(
        columns=["tag_context", "item_daily_feature_context"]
    )

    df_action_context_daily = pd.merge(
        left=item_daily_features_preprocessed,
        right=item_categories_preprocessed,
        how="left",
        on="video_id",
    )

    df_action_context_daily["action_context"] = (
        df_action_context_daily["tag_context"]
        + df_action_context_daily["item_daily_feature_context"]
    )
    df_action_context_daily = df_action_context_daily.drop(
        columns=["tag_context", "item_daily_feature_context"]
    )
    return df_action_context, df_action_context_daily


def preprocess_user_features(user_features):
    if conf.flag_show_execution_time_for_preprocess == True:
        print(f"Executing preprocess_user_features")
    time_preprocess_user_features_start = time.time()

    user_features_preprocessed = user_features.copy()
    ### Drop the columns that contain NA values: ['onehot_feat4', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17'] ###
    user_features_preprocessed = user_features_preprocessed.dropna(axis=1)
    user_features_preprocessed

    ### Drop the columns which are just the categorical version of the other columns
    # Define the list of the columns that are duplicate
    columns_list_to_be_dropped_as_duplicate = [
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]
    # Drop the columns in the dupliate list
    user_features_preprocessed = user_features_preprocessed.drop(
        columns=columns_list_to_be_dropped_as_duplicate
    )

    ### One hot embedding for the categorical variables ###
    # Define the list of all columns
    all_columns_list = user_features_preprocessed.columns.tolist()

    # Define the list of the name of the columns that are not to be embedded
    columns_list_not_to_be_embedded = [
        "user_id",
        "follow_user_num",
        "fans_user_num",
        "friend_user_num",
        "register_days",
        "onehot_feat3",
        "onehot_feat8",
    ]

    # Define the list of the columns that should be one hot embedded
    columns_list_to_be_embedded = [
        x for x in all_columns_list if x not in columns_list_not_to_be_embedded
    ]

    # Change the datatypes of the columns that should be emebedded to str
    for column_name in columns_list_to_be_embedded:
        user_features_preprocessed[column_name] = user_features_preprocessed[
            column_name
        ].astype(str)

    # Create one hot embedding for the specified columns
    user_features_preprocessed = create_one_hot_embedding_columns(
        df=user_features_preprocessed, string_columns=columns_list_to_be_embedded
    )

    ### Create a new column whose element is a list of the elements in the other columns
    column_list_to_be_excluded_for_list = ["user_id"]
    user_features_preprocessed = create_new_column_whose_element_is_list(
        data_frame=user_features_preprocessed,
        column_list_to_be_excluded_for_list=column_list_to_be_excluded_for_list,
        new_column_name="context",
        flag_drop_original_column=True,
    )
    time_preprocess_user_features_end = time.time()
    total_execution_time = (
        time_preprocess_user_features_end - time_preprocess_user_features_start
    )
    if conf.flag_show_execution_time_for_preprocess == True:
        print(
            f"Execution time for preprocess_user_features = {total_execution_time / 60:.3f} mins"
        )

    return user_features_preprocessed


def create_df_train_and_test(
    small_matrix, big_matrix, random_state, n_actions=conf.n_actions
):
    ###### Preprocess the small_matrix ######

    ### Create the dataframe without the missing values ###
    small_matrix_preprocessed = small_matrix.copy()
    small_matrix_preprocessed = small_matrix_preprocessed.dropna()
    small_matrix_preprocessed = small_matrix_preprocessed.reset_index()
    small_matrix_preprocessed = small_matrix_preprocessed.drop("index", axis=1)

    ### Create the dataframe without the missing values ###
    big_matrix_preprocessed = big_matrix.copy()
    big_matrix_preprocessed = big_matrix_preprocessed.dropna()
    big_matrix_preprocessed = big_matrix_preprocessed.reset_index()
    big_matrix_preprocessed = big_matrix_preprocessed.drop("index", axis=1)

    ### Extract the training and evaluation data ###

    # define the last time of the training data
    dt_end_of_training = datetime.datetime(
        year=2020, month=8, day=4, hour=23, minute=59, second=59
    )
    ut_end_of_training = datetime.datetime.timestamp(dt_end_of_training)

    # Define the smallest and biggest unix time to be included in the data
    smallest_ut = datetime.datetime.timestamp(datetime.datetime(2020, 7, 5, 0, 0, 0, 0))
    biggest_ut = datetime.datetime.timestamp(
        datetime.datetime(2020, 9, 5, 23, 59, 59, 59)
    )

    # Extract the training and test data
    big_matrix_preprocessed = big_matrix_preprocessed[
        big_matrix_preprocessed["timestamp"] <= ut_end_of_training
    ][big_matrix_preprocessed["timestamp"] >= smallest_ut].reset_index(drop=True)
    small_matrix_preprocessed = small_matrix_preprocessed[
        small_matrix_preprocessed["timestamp"] > ut_end_of_training
    ][small_matrix_preprocessed["timestamp"] <= biggest_ut].reset_index(drop=True)

    ###### Preprocess the big_matrix ######

    ### Extract the rows whose video is included in the small matrix ###
    # Extract the unique video list in the small matrix
    list_of_unique_video_in_small_matrix = small_matrix_preprocessed[
        "video_id"
    ].unique()

    # Extract the unique video list in the big matrix
    list_of_unique_video_in_big_matrix = big_matrix_preprocessed["video_id"].unique()

    # Extract the unique video list whose elements are in small and big matirx
    list_of_unqiue_videos_that_are_in_both_matrix = np.intersect1d(
        list_of_unique_video_in_small_matrix, list_of_unique_video_in_big_matrix
    )

    print(
        f"Maximum number of unique actions that we can use for training and test data = {len(list_of_unqiue_videos_that_are_in_both_matrix)}"
    )

    # Extract the sub DataFrame from the small matrix and big matrix
    small_matrix_preprocessed = small_matrix_preprocessed[
        small_matrix_preprocessed["video_id"].isin(
            list_of_unqiue_videos_that_are_in_both_matrix
        )
    ]
    big_matrix_preprocessed = big_matrix_preprocessed[
        big_matrix_preprocessed["video_id"].isin(
            list_of_unqiue_videos_that_are_in_both_matrix
        )
    ]

    ### It does not affect the number of training data so much if we eliminate the outliers  in timestamp outside (July 5th, September 5th) ###

    if conf.flag_show_parameters == True:
        # There is outliers in timestamp outside (July 5th, September 5th)
        print(
            f"The earliest date in the big matrix is {datetime.datetime.fromtimestamp(big_matrix_preprocessed['timestamp'].min())}"
        )
        print(
            f"the oldest date in the big matrix is {datetime.datetime.fromtimestamp(big_matrix_preprocessed['timestamp'].max())}\n"
        )

        # There is no outliers in timestamp outside (July 5th, September 5th)
        print(
            f"The earliest date in the small matrix is {datetime.datetime.fromtimestamp(small_matrix_preprocessed['timestamp'].min())}"
        )
        print(
            f"The oldest date in the small matrix is {datetime.datetime.fromtimestamp(small_matrix_preprocessed['timestamp'].max())}\n"
        )

    # Check how manny datapoints are outliers
    timestamp_big_list = list(big_matrix_preprocessed["timestamp"])
    timestamp_big_list.sort()
    timestamp_big_list = np.array(timestamp_big_list)
    if conf.flag_show_parameters == True:
        print(
            f"Samples whose time is earlier than July 5th = {(timestamp_big_list < smallest_ut).sum()}"
        )
        print(
            f"Samples whose time is older than September 5th = {(timestamp_big_list > biggest_ut).sum()}"
        )
        print(f"Total data = {len(timestamp_big_list)}\n")

    ###### Preprocess both the small_matrix and gig_matrix ######

    ### Drop the columns that we do not use ###
    small_matrix_preprocessed = small_matrix_preprocessed.drop(
        ["play_duration", "video_duration", "time", "date"], axis=1
    )
    big_matrix_preprocessed = big_matrix_preprocessed.drop(
        ["play_duration", "video_duration", "time", "date"], axis=1
    )

    ### limit the number of actions to compute fast ###

    # Sample the unique actions to be considered
    random_ = check_random_state(random_state)
    unique_video_array_sampled = random_.choice(
        list_of_unqiue_videos_that_are_in_both_matrix, size=n_actions, replace=False
    )

    # Extract the sub Data Frame according to the sampled actions
    small_matrix_preprocessed = small_matrix_preprocessed[
        small_matrix_preprocessed["video_id"].isin(unique_video_array_sampled)
    ]
    big_matrix_preprocessed = big_matrix_preprocessed[
        big_matrix_preprocessed["video_id"].isin(unique_video_array_sampled)
    ]

    # Sort the rows in DataFrame by timestamp from earilest to oldest
    small_matrix_preprocessed = small_matrix_preprocessed.sort_values(by="timestamp")
    big_matrix_preprocessed = big_matrix_preprocessed.sort_values(by="timestamp")

    # Reset the index
    df_train = big_matrix_preprocessed.reset_index(drop=True)
    df_test = small_matrix_preprocessed.reset_index(drop=True)

    return df_train, df_test


def preprocess_df_train_and_test(df_train, df_test, df_context, df_action_context):
    ### Merge df_train and df_test and df_context and df_action_context ###
    df_train_preprocessed = pd.merge(
        left=df_train, right=df_context, how="inner", on="user_id"
    )

    df_train_preprocessed = pd.merge(
        left=df_train_preprocessed, right=df_action_context, how="inner", on="video_id"
    )
    df_train_preprocessed = df_train_preprocessed.rename(
        columns={"video_id": "action", "timestamp": "time", "watch_ratio": "reward"}
    )
    df_train_preprocessed = df_train_preprocessed.drop(columns=["user_id"])

    df_test_preprocessed = pd.merge(
        left=df_test, right=df_context, how="inner", on="user_id"
    )

    df_test_preprocessed = pd.merge(
        left=df_test_preprocessed, right=df_action_context, how="inner", on="video_id"
    )
    df_test_preprocessed = df_test_preprocessed.rename(
        columns={"video_id": "action", "timestamp": "time", "watch_ratio": "reward"}
    )
    df_test_preprocessed = df_test_preprocessed.drop(columns=["user_id"])

    ### Craete the list of unique action

    # Create the unique aciton list in training and test data
    unique_action_list_train = df_train_preprocessed["action"].unique().tolist()
    unique_action_list_test = df_test_preprocessed["action"].unique().tolist()

    if set(unique_action_list_train) != set(unique_action_list_test):
        print(f"unique_action_list_train = {unique_action_list_train}")
        print(f"unique_action_list_test = {unique_action_list_test}")
        print(f"len(unique_action_list_train) = {len(unique_action_list_train)}")
        print(f"len(unique_action_list_test) = {len(unique_action_list_test)}")
        raise ValueError(
            "unique_action_list_train should be same as unique_action_list_test"
        )

    # Create the list whose elements are in either two lists
    unique_action_list = list(
        set(unique_action_list_train) | set(unique_action_list_test)
    )

    # Sort the list
    unique_action_list.sort()

    ### change the maximum number of action to the conf_small.n_actions - 1 ###
    df_train_preprocessed["action"] = df_train_preprocessed["action"].replace(
        unique_action_list, range(len(unique_action_list))
    )
    df_test_preprocessed["action"] = df_test_preprocessed["action"].replace(
        unique_action_list, range(len(unique_action_list))
    )

    # Sort the rows by time in the training and test data
    df_train_preprocessed = (
        df_train_preprocessed.sort_values(by="time")
        .reset_index()
        .drop(columns=["index"])
    )
    df_test_preprocessed = (
        df_test_preprocessed.sort_values(by="time")
        .reset_index()
        .drop(columns=["index"])
    )

    return df_train_preprocessed, df_test_preprocessed


def reduce_the_dim_context(
    df_train_preprocessed,
    df_test_preprocessed,
    random_state=conf.random_state,
    dim_context=conf.dim_context,
):
    # Conduct PCA to reduce the dimension of context if needed
    if conf.flag_show_execution_time_for_preprocess == True:
        print(f"Executing reduce_the_dim_context")
    time_reduce_the_dim_context_start = time.time()
    if len(df_train_preprocessed["context"][0]) == len(
        df_test_preprocessed["context"][0]
    ):
        dim_context_original = len(df_train_preprocessed["context"][0])
        if conf.flag_show_num_original_dim_context == True:
            print(f"The dimension of the original context = {dim_context_original}")
            print(f"The dimension of the context after PCA = {dim_context}")
    else:
        raise ValueError(
            "The dimensions of context in training and test data are different"
        )

    # Extract the context from the training data
    context_train = df_train_preprocessed["context"].to_list()
    context_train = np.array(context_train)

    # # Normalize the context data
    scaler = StandardScaler()
    context_train = scaler.fit_transform(context_train)

    # Do PCA for the context in the training data
    pca = PCA(
        n_components=dim_context, svd_solver="randomized", random_state=random_state
    )
    context_train = pca.fit_transform(context_train)

    # Plot explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    if conf.flag_show_context_PCA == True:
        # Plotting
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            alpha=0.8,
            align="center",
        )
        plt.title("Explained Variance Ratio")
        plt.xlabel("Principal Components")
        plt.ylabel("Variance Ratio")

        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(cumulative_explained_variance) + 1),
            cumulative_explained_variance,
            marker="o",
            linestyle="--",
        )
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Variance Ratio")

        plt.tight_layout()
        plt.show()

    print(
        f"The cumularive explained variance for the dimension of context is {cumulative_explained_variance[-1]:.3f}"
    )

    ## Extract the context from the test data
    context_test = df_test_preprocessed["context"].to_list()
    context_test = np.array(context_test)

    # Do PCA for the context in the test data
    context_test = scaler.transform(context_test)
    context_test = pca.transform(context_test)

    ### Create new context by PCA and drop the original context for training data ###
    new_column_train = pd.DataFrame(
        {"context_after_PCA": [np.array(row) for row in context_train]}
    )

    # Concatenate the new DataFrame with the original DataFrame
    df_train_preprocessed = pd.concat([df_train_preprocessed, new_column_train], axis=1)

    df_train_preprocessed = df_train_preprocessed.drop(columns=["context"])
    df_train_preprocessed = df_train_preprocessed.rename(
        columns={"context_after_PCA": "context"}
    )

    ### Create new context by PCA and drop the original context for test data ###
    new_column_test = pd.DataFrame(
        {"context_after_PCA": [np.array(row) for row in context_test]}
    )

    # Concatenate the new DataFrame with the original DataFrame
    df_test_preprocessed = pd.concat([df_test_preprocessed, new_column_test], axis=1)

    df_test_preprocessed = df_test_preprocessed.drop(columns=["context"])
    df_test_preprocessed = df_test_preprocessed.rename(
        columns={"context_after_PCA": "context"}
    )

    time_reduce_the_dim_context_end = time.time()
    total_execution_time = (
        time_reduce_the_dim_context_end - time_reduce_the_dim_context_start
    )
    if conf.flag_show_execution_time_for_preprocess == True:
        print(
            f"Execution time for educe_the_dim_context = {total_execution_time / 60:.3f} mins"
        )

    return df_train_preprocessed, df_test_preprocessed


def reduce_the_dim_action_context(
    df_train_preprocessed,
    df_test_preprocessed,
    random_state=conf.random_state,
    dim_action_context=conf.dim_action_context,
):
    if conf.flag_show_execution_time_for_preprocess == True:
        print(f"Executing reduce_the_dim_action_context")
    # Conduct PCA to reduce the dimension of the action context if needed
    time_reduce_the_dim_action_context_start = time.time()

    if len(df_train_preprocessed["action_context"][0]) == len(
        df_test_preprocessed["action_context"][0]
    ):
        dim_action_context_original = len(df_train_preprocessed["action_context"][0])
        if conf.flag_show_num_original_dim_action_context == True:
            print(
                f"The dimension of the original action context = {dim_action_context_original}"
            )
            print(
                f"The dimension of the action context after PCA = {dim_action_context}"
            )
    else:
        raise ValueError(
            "The dimensions of action context in training and test data are different"
        )

    # Extract the context from the training data
    action_context_train = df_train_preprocessed["action_context"].to_list()
    action_context_train = np.array(action_context_train)

    # Normalize the context data
    scaler = StandardScaler()
    action_context_train = scaler.fit_transform(action_context_train)

    # Do PCA for the context in the training data
    pca = PCA(
        n_components=dim_action_context,
        svd_solver="randomized",
        random_state=random_state,
    )
    action_context_train = pca.fit_transform(action_context_train)

    # Plot explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    if conf.flag_show_action_context_PCA == True:
        # Plotting
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            alpha=0.8,
            align="center",
        )
        plt.title("Explained Variance Ratio")
        plt.xlabel("Principal Components")
        plt.ylabel("Variance Ratio")

        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(cumulative_explained_variance) + 1),
            cumulative_explained_variance,
            marker="o",
            linestyle="--",
        )
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Variance Ratio")

        plt.tight_layout()
        plt.show()

    print(
        f"The cumularive explained variance for the dimension of action context is {cumulative_explained_variance[-1]:.3f}"
    )

    ## Extract the context from the test data
    action_context_test = df_test_preprocessed["action_context"].to_list()
    action_context_test = np.array(action_context_test)

    # Do PCA for the context in the test data
    action_context_test = scaler.transform(action_context_test)
    action_context_test = pca.transform(action_context_test)

    ### Create new context by PCA and drop the original context for training data ###
    new_column_train = pd.DataFrame(
        {"action_context_after_PCA": [np.array(row) for row in action_context_train]}
    )

    # Concatenate the new DataFrame with the original DataFrame
    df_train_preprocessed = pd.concat([df_train_preprocessed, new_column_train], axis=1)

    df_train_preprocessed = df_train_preprocessed.drop(columns=["action_context"])
    df_train_preprocessed = df_train_preprocessed.rename(
        columns={"action_context_after_PCA": "action_context"}
    )

    ### Create new context by PCA and drop the original context for test data ###
    new_column_test = pd.DataFrame(
        {"action_context_after_PCA": [np.array(row) for row in action_context_test]}
    )

    # Concatenate the new DataFrame with the original DataFrame
    df_test_preprocessed = pd.concat([df_test_preprocessed, new_column_test], axis=1)

    df_test_preprocessed = df_test_preprocessed.drop(columns=["action_context"])
    df_test_preprocessed = df_test_preprocessed.rename(
        columns={"action_context_after_PCA": "action_context"}
    )

    time_reduce_the_dim_action_context_end = time.time()
    total_execution_time = (
        time_reduce_the_dim_action_context_end
        - time_reduce_the_dim_action_context_start
    )
    if conf.flag_show_execution_time_for_preprocess == True:
        print(
            f"Execution time for reduce_the_dim_action_context = {total_execution_time / 60:.3f} mins"
        )

    return df_train_preprocessed, df_test_preprocessed


def create_time_structure_one_fold(time_train, time_test, unix_time_to_time_feature):
    # Create the one hot matirx
    unix_time_to_time_feature_vec = np.vectorize(unix_time_to_time_feature)
    num_time_feature = obtain_num_time_structure(unix_time_to_time_feature)
    time_structure_train_ = unix_time_to_time_feature_vec(time_train)
    time_structure_one_hot_matrix_train = np.eye(num_time_feature)[
        time_structure_train_
    ]
    time_structure_test_ = unix_time_to_time_feature_vec(time_test)
    time_structureone_hot_matrix_test = np.eye(num_time_feature)[time_structure_test_]

    return time_structure_one_hot_matrix_train, time_structureone_hot_matrix_test


def create_time_structure_day_of_week(time_train, time_test):
    return create_time_structure_one_fold(
        time_train, time_test, unix_time_to_day_of_week
    )


def create_time_structure_hour(time_train, time_test):
    return create_time_structure_one_fold(time_train, time_test, unix_time_to_hour)


def create_time_structure_date(time_train, time_test):
    return create_time_structure_one_fold(time_train, time_test, unix_time_to_date)


def create_time_structure_two_folds(
    time_train, time_test, unix_time_to_time_feature1, unix_time_to_time_feature2
):
    time_structure_one_hot_matrix_train1, time_structureone_hot_matrix_test1 = (
        create_time_structure_one_fold(
            time_train, time_test, unix_time_to_time_feature1
        )
    )
    time_structure_one_hot_matrix_train2, time_structureone_hot_matrix_test2 = (
        create_time_structure_one_fold(
            time_train, time_test, unix_time_to_time_feature2
        )
    )
    time_structure_train = np.concatenate(
        (time_structure_one_hot_matrix_train1, time_structure_one_hot_matrix_train2),
        axis=1,
    )
    time_structure_test = np.concatenate(
        (time_structureone_hot_matrix_test1, time_structureone_hot_matrix_test2), axis=1
    )
    return time_structure_train, time_structure_test


def create_time_structure_day_of_week_and_hour(time_train, time_test):
    return create_time_structure_two_folds(
        time_train, time_test, unix_time_to_day_of_week, unix_time_to_hour
    )


def create_time_structure_day_of_week_and_date(time_train, time_test):
    return create_time_structure_two_folds(
        time_train, time_test, unix_time_to_day_of_week, unix_time_to_date
    )


def create_time_structure_hour_and_date(time_train, time_test):
    return create_time_structure_two_folds(
        time_train, time_test, unix_time_to_hour, unix_time_to_date
    )


def create_time_structure_three_folds(
    time_train,
    time_test,
    unix_time_to_time_feature1,
    unix_time_to_time_feature2,
    unix_time_to_time_feature3,
):
    time_structure_one_hot_matrix_train1, time_structureone_hot_matrix_test1 = (
        create_time_structure_one_fold(
            time_train, time_test, unix_time_to_time_feature1
        )
    )
    time_structure_one_hot_matrix_train2, time_structureone_hot_matrix_test2 = (
        create_time_structure_one_fold(
            time_train, time_test, unix_time_to_time_feature2
        )
    )
    time_structure_one_hot_matrix_train3, time_structureone_hot_matrix_test3 = (
        create_time_structure_one_fold(
            time_train, time_test, unix_time_to_time_feature3
        )
    )
    time_structure_train = np.concatenate(
        (
            time_structure_one_hot_matrix_train1,
            time_structure_one_hot_matrix_train2,
            time_structure_one_hot_matrix_train3,
        ),
        axis=1,
    )
    time_structure_test = np.concatenate(
        (
            time_structureone_hot_matrix_test1,
            time_structureone_hot_matrix_test2,
            time_structureone_hot_matrix_test3,
        ),
        axis=1,
    )
    return time_structure_train, time_structure_test


def create_time_structure_day_of_week_and_hour_and_date(time_train, time_test):
    return create_time_structure_three_folds(
        time_train,
        time_test,
        unix_time_to_day_of_week,
        unix_time_to_hour,
        unix_time_to_date,
    )


def create_time_structure(time_train, time_test):
    if conf.phi_scalar_func_for_OPFV == unix_time_to_day_of_week:
        return create_time_structure_day_of_week(time_train, time_test)
    elif conf.phi_scalar_func_for_OPFV == unix_time_to_hour:
        return create_time_structure_hour(time_train, time_test)
    elif conf.phi_scalar_func_for_OPFV == unix_time_to_date:
        return create_time_structure_date(time_train, time_test)
    elif conf.phi_scalar_func_for_OPFV == unix_time_to_day_of_week_and_hour:
        return create_time_structure_day_of_week_and_hour(time_train, time_test)
    elif conf.phi_scalar_func_for_OPFV == unix_time_to_day_of_week_and_date:
        return create_time_structure_day_of_week_and_date(time_train, time_test)
    elif conf.phi_scalar_func_for_OPFV == unix_time_to_hour_and_date:
        return create_time_structure_hour_and_date(time_train, time_test)
    elif conf.phi_scalar_func_for_OPFV == unix_time_to_day_of_week_and_hour_and_date:
        return create_time_structure_day_of_week_and_hour_and_date(
            time_train, time_test
        )
    else:
        raise ValueError(
            "phi_scalar_func_for_OPFV should be either unix_time_to_day_of_week, unix_time_to_hour, unix_time_to_date, unix_time_to_day_of_week_and_hour, unix_time_to_day_of_week_and_date, unix_time_to_hour_and_date, unix_time_to_day_of_week_and_hour_and_date"
        )


def create_behavior_and_evaluation_policy_and_pscore(
    df_train_preprocessed,
    df_test_preprocessed,
    random_state=conf.random_state,
    n_actions=conf.n_actions,
):
    ### Extract the context, time, and action in training and test data
    if conf.flag_show_execution_time_for_preprocess == True:
        print(f"Executing create_behavior_and_evaluation_policy_and_pscore")
    time_create_behavior_and_evaluation_policy_and_pscore_start = time.time()

    # Extract the context in the training data
    context_train = df_train_preprocessed["context"].to_list()
    context_train = np.array(context_train)

    # Extract the time in the training data
    time_train = df_train_preprocessed["time"].to_list()
    time_train = np.array(time_train)

    # Extract the context in the test data
    context_test = df_test_preprocessed["context"].to_list()
    context_test = np.array(context_test)

    # Extract the time in the test data
    time_test = df_test_preprocessed["time"].to_list()
    time_test = np.array(time_test)

    # Craete the list of action from training data
    action_train = df_train_preprocessed["action"].to_list()
    action_train = np.array(action_train)

    # Craete the list of action from test data
    action_test = df_test_preprocessed["action"].to_list()
    action_test = np.array(action_test)

    ### Normalize the time ###
    time_train_test = np.concatenate((time_train, time_test))

    def normalize_time(time):
        min_time = time_train_test.min()
        max_time = time_train_test.max()
        return (time - min_time) / max_time

    time_train_normalized = normalize_time(time=time_train)
    time_test_normalized = normalize_time(time=time_test)

    # Create the time structure matrix for training and test data
    time_structure_train, time_structure_test = create_time_structure(
        time_train=time_train, time_test=time_test
    )

    ### Create the input for the neural network for the estimation of the behavior policy ###
    # Concatenate the matrix to create input for neural network
    input_for_nn_train = np.concatenate(
        (context_train, time_train_normalized.reshape(-1, 1), time_structure_train),
        axis=1,
    )
    input_for_nn_test = np.concatenate(
        (context_test, time_test_normalized.reshape(-1, 1), time_structure_test), axis=1
    )

    # Get unique elements and their counts
    _, counts_occurrences_same_ation_train = np.unique(action_train, return_counts=True)
    min_occurrences_same_ation_train = np.min(counts_occurrences_same_ation_train)

    behavior_pl_train = SGDClassifier(n_jobs=-1, random_state=random_state)

    # Wrap the classifier with CalibratedClassifierCV
    if min_occurrences_same_ation_train < 2:
        calibrated_behavior_pl_train = CalibratedClassifierCV(behavior_pl_train)
    elif min_occurrences_same_ation_train == 2:
        calibrated_behavior_pl_train = CalibratedClassifierCV(behavior_pl_train, cv=2)
    else:
        calibrated_behavior_pl_train = CalibratedClassifierCV(behavior_pl_train, cv=3)

    # Fit the calibrated classifier on the training data
    calibrated_behavior_pl_train.fit(X=input_for_nn_train, y=action_train)

    # Predict probabilities on the test set
    pi_b = calibrated_behavior_pl_train.predict_proba(X=input_for_nn_train)

    _, counts_occurrences_same_ation_test = np.unique(action_test, return_counts=True)
    min_occurrences_same_ation_test = np.min(counts_occurrences_same_ation_test)

    behavior_pl_test = SGDClassifier(n_jobs=-1, random_state=random_state)

    # Wrap the classifier with CalibratedClassifierCV
    if min_occurrences_same_ation_test < 2:
        calibrated_behavior_pl_test = CalibratedClassifierCV(behavior_pl_test)
    elif min_occurrences_same_ation_test == 2:
        calibrated_behavior_pl_test = CalibratedClassifierCV(behavior_pl_test, cv=2)
    else:
        calibrated_behavior_pl_test = CalibratedClassifierCV(behavior_pl_test, cv=3)

    # Fit the calibrated classifier on the training data
    calibrated_behavior_pl_test.fit(X=input_for_nn_test, y=action_test)

    # Predict probabilities on the test set
    pi_e = calibrated_behavior_pl_test.predict_proba(X=input_for_nn_test)

    ### Extract the propensity score from the behavior policy ###
    num_trains = pi_b.shape[0]
    pscore_pi_b = pi_b[range(num_trains), action_train]

    ### Extract the propensity score from the behavior policy ###
    num_tests = pi_e.shape[0]
    pscore_pi_e = pi_e[range(num_tests), action_test]

    ### Reshape the behavior and evaluation policies
    pi_b = pi_b.reshape(pi_b.shape[0], pi_b.shape[1], 1)
    pi_e = pi_e.reshape(pi_e.shape[0], pi_e.shape[1], 1)

    time_create_behavior_and_evaluation_policy_and_pscore_end = time.time()
    total_execution_time = (
        time_create_behavior_and_evaluation_policy_and_pscore_end
        - time_create_behavior_and_evaluation_policy_and_pscore_start
    )
    if conf.flag_show_execution_time_for_preprocess == True:
        print(
            f"Execution time for create_behavior_and_evaluation_policy_and_pscore = {total_execution_time / 60:.3f} mins"
        )

    return pi_b, pi_e, pscore_pi_b, pscore_pi_e


def create_dataset_dataset_train_dataset_test(
    df_train_preprocessed,
    df_test_preprocessed,
    pi_b,
    pi_e,
    pscore_pi_b,
    pscore_pi_e,
    n_actions=conf.n_actions,
    dim_context=conf.dim_context,
):
    ### Extract the action context from training and test data ###
    for i in range(n_actions):
        if (
            len(
                df_train_preprocessed[df_train_preprocessed["action"] == i][
                    "action_context"
                ]
            )
            != 0
        ):
            if i == 0:
                action_context = (
                    df_train_preprocessed[df_train_preprocessed["action"] == i][
                        "action_context"
                    ]
                    .to_numpy()[0]
                    .reshape(1, -1)
                )
            else:
                action_context = np.append(
                    action_context,
                    df_train_preprocessed[df_train_preprocessed["action"] == i][
                        "action_context"
                    ]
                    .to_numpy()[0]
                    .reshape(1, -1),
                    axis=0,
                )
        elif (
            len(
                df_test_preprocessed[df_test_preprocessed["action"] == i][
                    "action_context"
                ]
            )
            != 0
        ):
            if i == 0:
                action_context = (
                    df_test_preprocessed[df_test_preprocessed["action"] == i][
                        "action_context"
                    ]
                    .to_numpy()[0]
                    .reshape(1, -1)
                )
            else:
                action_context = np.append(
                    action_context,
                    df_test_preprocessed[df_test_preprocessed["action"] == i][
                        "action_context"
                    ]
                    .to_numpy()[0]
                    .reshape(1, -1),
                    axis=0,
                )
        else:
            raise ValueError(f"There is no action context found for action{i}")

    ### Create dataset, dataset_train, and dataset_test except for expected reward###
    dataset = {}
    dataset_train = {}
    dataset_test = {}
    dataset_list = [dataset_train, dataset_test]
    df_preprocessed_list = [df_train_preprocessed, df_test_preprocessed]

    # Create dataset_traina and dataset_test
    for i in range(len(dataset_list)):
        dataset_list[i]["n_rounds"] = df_preprocessed_list[i].shape[0]
        dataset_list[i]["n_actions"] = n_actions
        dataset_list[i]["time"] = df_preprocessed_list[i]["time"].to_numpy()
        dataset_list[i]["context"] = np.array(
            df_preprocessed_list[i]["context"].to_list()
        )
        dataset_list[i]["action_context"] = action_context
        dataset_list[i]["action"] = df_preprocessed_list[i]["action"].to_numpy()
        dataset_list[i]["reward"] = df_preprocessed_list[i]["reward"].to_numpy()
        if i == 0:
            dataset_list[i]["pi_b"] = pi_b
            dataset_list[i]["pscore"] = pscore_pi_b
        else:
            dataset_list[i]["pi_b"] = pi_e
            dataset_list[i]["pscore"] = pscore_pi_e

    # Create dataset
    dataset["n_actions"] = n_actions
    dataset["action_context"] = action_context
    dataset["dim_context"] = dim_context
    dataset["t_oldest"] = int(
        datetime.datetime.timestamp(
            datetime.datetime(year=2020, month=7, day=5, hour=0, minute=0, second=0)
        )
    )
    dataset["t_now"] = int(
        datetime.datetime.timestamp(
            datetime.datetime(year=2020, month=8, day=4, hour=23, minute=59, second=59)
        )
    )
    dataset["t_future"] = int(
        datetime.datetime.timestamp(
            datetime.datetime(year=2020, month=9, day=5, hour=23, minute=59, second=59)
        )
    )

    if (
        dataset_train["n_actions"] != dataset_test["n_actions"]
        or dataset_train["n_actions"] != dataset["n_actions"]
    ):
        print(f'dataset_train["n_actions"] = {dataset_train["n_actions"]}')
        print(f'dataset_test["n_actions"] = {dataset_test["n_actions"]}')
        print(f'dataset["n_actions"] = {dataset["n_actions"]}')
        raise ValueError("n_actions should be same for dataset_train and dataset_test")

    if conf.flag_show_num_trains_and_tests == True:
        print(f"The number of the training data = {dataset_train['n_rounds']}")
        print(f"The number of the test data = {dataset_test['n_rounds']}")

    print(f"Number of actions in dataset |A| = {dataset['n_actions']}\n")
    if conf.flag_show_parameters == True:
        print(f"The dimension of context in dataset d_x = {dataset['dim_context']}")

        print(
            f"The oldest date in the training data = {datetime.datetime.fromtimestamp(dataset_train['time'][0])}"
        )
        print(
            f"The earliest date in the training data = {datetime.datetime.fromtimestamp(dataset_train['time'][-1])}\n"
        )

        print(
            f"The oldest date in the test data = {datetime.datetime.fromtimestamp(dataset_test['time'][0])}"
        )
        print(
            f"The earliest date in the test data = {datetime.datetime.fromtimestamp(dataset_test['time'][-1])}\n"
        )

        print(
            f'dataset["t_oldest"] = {datetime.datetime.fromtimestamp(dataset["t_oldest"])}'
        )
        print(f'dataset["t_now"] = {datetime.datetime.fromtimestamp(dataset["t_now"])}')
        print(
            f'dataset["t_future"] = {datetime.datetime.fromtimestamp(dataset["t_future"])}'
        )

    return dataset, dataset_train, dataset_test


# Normalize time from 0 (t_oldest) to 1 (t_future)
def normalize_time(unix_time, t_oldest, t_future):
    return (unix_time - t_oldest) / (t_future - t_oldest)


def create_expected_reward(
    dataset, dataset_train, dataset_test, random_state=conf.random_state
):
    if conf.flag_show_execution_time_for_preprocess == True:
        print(f"Executing create_expected_reward")
    ### Create expected reward and set the estimated one in the dataset_train and dataset_test ###
    time_create_expected_reward_start = time.time()

    normalized_time_train = normalize_time(
        unix_time=dataset_train["time"],
        t_oldest=dataset["t_oldest"],
        t_future=dataset["t_future"],
    )
    normalized_time_test = normalize_time(
        unix_time=dataset_test["time"],
        t_oldest=dataset["t_oldest"],
        t_future=dataset["t_future"],
    )

    time_structure_train, time_structure_test = create_time_structure(
        time_train=dataset_train["time"], time_test=dataset_test["time"]
    )

    time_start_regression_model_time_true_train = time.time()
    # Instantiate the regression model
    true_reg_model_time_train = RegressionModelTimeTrue(
        n_actions=dataset["n_actions"],
        action_context=dataset["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=random_state, n_jobs=-1
        ),
    )

    estimated_rewards_for_train_true = true_reg_model_time_train.fit_predict(
        context=dataset_train["context"],
        time=normalized_time_train,
        time_structure=time_structure_train,
        action=dataset_train["action"],
        reward=dataset_train["reward"],
        n_folds=3,
        random_state=random_state,
    )

    time_end_regression_model_time_true_train = time.time()
    execution_time_regression_model_time_true_train = (
        time_end_regression_model_time_true_train
        - time_start_regression_model_time_true_train
    )
    print(
        f"RegressionModelTimeTrue fitting and predition for training data time = {execution_time_regression_model_time_true_train / 60:.3f} mins"
    )

    time_start_regression_model_time_true_test = time.time()

    true_reg_model_time_test = RegressionModelTimeTrue(
        n_actions=dataset["n_actions"],
        action_context=dataset["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=random_state, n_jobs=-1
        ),
    )

    estimated_rewards_for_test_true = true_reg_model_time_test.fit_predict(
        context=dataset_test["context"],
        time=normalized_time_test,
        time_structure=time_structure_test,
        action=dataset_test["action"],
        reward=dataset_test["reward"],
        n_folds=3,
        random_state=random_state,
    )

    time_end_regression_model_time_true_test = time.time()
    execution_time_regression_model_time_true_test = (
        time_end_regression_model_time_true_test
        - time_start_regression_model_time_true_test
    )
    print(
        f"RegressionModelTimeTrue fitting and predition for test data time = {execution_time_regression_model_time_true_test / 60:.3f} mins"
    )

    # Reshape the expected reward
    estimated_rewards_for_train_true = np.squeeze(
        estimated_rewards_for_train_true, axis=2
    )
    estimated_rewards_for_test_true = np.squeeze(
        estimated_rewards_for_test_true, axis=2
    )

    # Define the expected reward for train and test dataset
    dataset_train["expected_reward"] = estimated_rewards_for_train_true
    dataset_test["expected_reward"] = estimated_rewards_for_test_true

    time_create_expected_reward_end = time.time()
    total_execution_time = (
        time_create_expected_reward_end - time_create_expected_reward_start
    )
    if conf.flag_show_execution_time_for_preprocess == True:
        print(
            f"Execution time for create_expected_reward = {total_execution_time / 60:.3f} mins"
        )


def pre_process(
    small_matrix,
    big_matrix,
    item_categories,
    item_daily_features,
    user_features,
    social_network,
    random_state=conf.random_state,
    n_actions=conf.n_actions,
    dim_context=conf.dim_context,
    dim_action_context=conf.dim_action_context,
):
    if conf.flag_show_block_preprocessing == True:
        print(f"\n#################### START of preprocessing ####################")
    ###### Descriptive Analysis of the small_matrix and big_matrix ######
    # Identify the columns that include NA values for each DataFrame
    list_of_all_data_frame = [
        small_matrix,
        big_matrix,
        item_categories,
        item_daily_features,
        user_features,
        social_network,
    ]
    list_of_all_data_frame_name = [
        "small_matrix",
        "big_matrix",
        "item_categories",
        "item_daily_features",
        "user_features",
        "social_network",
    ]

    if conf.flag_show_parameters == True:
        for i in range(len(list_of_all_data_frame)):
            print(f"{list_of_all_data_frame_name[i]}")
            obtain_column_names_with_NA(list_of_all_data_frame[i])
            print("")

    ### Preprocess the small and big matrix to create df_train and df_test
    df_train, df_test = create_df_train_and_test(
        small_matrix=small_matrix,
        big_matrix=big_matrix,
        random_state=random_state,
        n_actions=n_actions,
    )

    if conf.flag_show_parameters == True:
        print(f"Finished create_df_train_and_test\n")

    ### Create the DataFrame whose columns are action and action context (31 unique tag)
    item_categories_preprocessed = preprocess_item_categories(
        item_categories=item_categories
    )

    if conf.flag_show_parameters == True:
        print(f"Finished preprocess_item_categories\n")

    # Preprocess item_daily_features
    item_daily_features_preprocessed, item_features_preprocessed = (
        preprocess_item_daily_features(item_daily_features=item_daily_features)
    )

    if conf.flag_show_parameters == True:
        print(f"Finished preprocess_item_daily_features\n")

    # Merge the two DataFrames to construct new DataFrame on action context
    df_action_context, df_action_context_daily = create_df_action_context(
        item_categories_preprocessed=item_categories_preprocessed,
        item_features_preprocessed=item_features_preprocessed,
        item_daily_features_preprocessed=item_daily_features_preprocessed,
    )
    if conf.flag_show_parameters == True:
        print(f"Finished create_df_action_context\n")

    # Preprocess user_features
    df_context = preprocess_user_features(user_features=user_features)

    if conf.flag_show_parameters == True:
        print(f"Finished preprocess_user_features\n")

    # Preprocess DataFrame for training and test data
    df_train_preprocessed, df_test_preprocessed = preprocess_df_train_and_test(
        df_train=df_train,
        df_test=df_test,
        df_context=df_context,
        df_action_context=df_action_context,
    )
    if conf.flag_show_parameters == True:
        print(f"Finished preprocess_df_train_and_test\n")

    # Reduce the dimension of context and action context
    df_train_preprocessed, df_test_preprocessed = reduce_the_dim_context(
        df_train_preprocessed=df_train_preprocessed,
        df_test_preprocessed=df_test_preprocessed,
        random_state=random_state,
        dim_context=dim_context,
    )
    if conf.flag_show_parameters == True:
        print(f"Finished reduce_the_dim_context\n")

    df_train_preprocessed, df_test_preprocessed = reduce_the_dim_action_context(
        df_train_preprocessed=df_train_preprocessed,
        df_test_preprocessed=df_test_preprocessed,
        random_state=random_state,
        dim_action_context=dim_action_context,
    )
    if conf.flag_show_parameters == True:
        print(f"Finished reduce_the_dim_action_context\n")

    # Estimate the behavior and evaluation policies with propensity scores
    pi_b, pi_e, pscore_pi_b, pscore_pi_e = (
        create_behavior_and_evaluation_policy_and_pscore(
            df_train_preprocessed,
            df_test_preprocessed,
            random_state=random_state,
            n_actions=n_actions,
        )
    )

    if conf.flag_show_parameters == True:
        print(f"Finished create_behavior_and_evaluation_policy_and_pscore\n")

    # Create dataset, dataset_train, and dataset_test except for expected reward
    dataset, dataset_train, dataset_test = create_dataset_dataset_train_dataset_test(
        df_train_preprocessed,
        df_test_preprocessed,
        pi_b,
        pi_e,
        pscore_pi_b,
        pscore_pi_e,
        n_actions=n_actions,
        dim_context=dim_context,
    )
    if conf.flag_show_parameters == True:
        print(f"Finished create_dataset_dataset_train_dataset_test\n")

    ### Create expected reward and set the estimated one in the dataset_train and dataset_test ###
    create_expected_reward(
        dataset=dataset,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        random_state=random_state,
    )
    if conf.flag_show_parameters == True:
        print(f"Finished create_expected_reward\n")

    if conf.flag_show_block_preprocessing == True:
        print(f"#################### END of preprocessing ####################\n")

    return dataset, dataset_train, dataset_test

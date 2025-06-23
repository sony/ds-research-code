# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime
from typing import Callable, Tuple

import numpy as np

SECONDS_PER_DAY = 24 * 60 * 60
BIG_NUM = int(1e5)
NUM_DAY_OF_WEEK = 7
NUM_WEEKDAY_WEEKEND = 2

NUM_TIME_FEATURES_DAY_OF_WEEK = 7
NUM_TIME_FEATURES_HOUR = 24
NUM_TIME_FEATURES_FOUR = 4
NUM_TIME_FEATURES_AM_PM = 2
NUM_TIME_FEATURES_DATE = 31

NUM_TIME_FEATURES_DAY_OF_WEEK_HOUR = 24 * 7
NUM_TIME_FEATURES_DAY_OF_WEEK_DATE = 31 * 7
NUM_TIME_FEATURES_HOUR_DATE = 24 * 31

NUM_TIME_FEATURES_DAY_OF_WEEK_HOUR_DATE = 24 * 7 * 31


coef_func_signature = Callable[
    [np.ndarray, np.ndarray, np.random.RandomState],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


def unix_time_to_season(unix_time):
    # Convert Unix timestamp to a datetime object
    datetime_converted = datetime.datetime.fromtimestamp(unix_time)
    month = datetime_converted.month
    # Get the season as an integer (0 = Spring, 1 = Summer, 2 = Fall, 3 = Winter)
    if 1 <= month <= 3:
        return 0
    elif 4 <= month <= 6:
        return 1
    elif 7 <= month <= 9:
        return 2
    elif 10 <= month <= 12:
        return 3


def unix_time_to_day_of_week(unix_time):
    unix_time = int(unix_time)
    # Convert Unix timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(unix_time)
    # Get the day of the week as an integer (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    weekday = dt_object.weekday()
    return weekday


def unix_time_to_weekday_weekend(unix_time):
    weekday = unix_time_to_day_of_week(unix_time)
    if weekday in set(range(0, 5)):
        return 0  # Weekday
    else:
        return 1  # Weekends


def unix_time_to_hour(unix_time):
    unix_time = int(unix_time)
    # Convert Unix timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(unix_time)
    # Get the hour as an integer (0 = 0:00~0:59, 1 = 1:00~1:59, ..., 23 = 23:00~23:59)
    hour = dt_object.hour
    return hour


def unix_time_to_four_paritions_a_day(unix_time):
    hour = unix_time_to_hour(unix_time)
    if hour in set(range(0, 6)):
        return 0  # Night
    elif hour in set(range(6, 12)):
        return 1  # Morning
    elif hour in set(range(12, 18)):
        return 2  # Afternoon
    else:
        return 3  # Evning


def unix_time_to_AM_PM(unix_time):
    hour = unix_time_to_hour(unix_time)
    if hour in set(range(0, 12)):
        return 0  # AM
    else:
        return 1  # PM


def unix_time_to_date(unix_time):
    unix_time = int(unix_time)
    # Convert Unix timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(unix_time)
    # Get the day (from 1 to 31)
    date = dt_object.date().day
    return date - 1  # (from 0 to 30)


def unix_time_to_weekday_weekend_and_AM_PM(unix_time):
    return unix_time_to_weekday_weekend(
        unix_time
    ) * NUM_TIME_FEATURES_AM_PM + unix_time_to_AM_PM(unix_time)


def unix_time_to_weekday_weekend_and_four_paritions_a_day(unix_time):
    return unix_time_to_weekday_weekend(
        unix_time
    ) * NUM_TIME_FEATURES_FOUR + unix_time_to_four_paritions_a_day(unix_time)


def unix_time_to_weekday_weekend_and_hour(unix_time):
    return unix_time_to_weekday_weekend(
        unix_time
    ) * NUM_TIME_FEATURES_HOUR + unix_time_to_hour(unix_time)


def unix_time_to_day_of_week_and_AM_PM(unix_time):
    return unix_time_to_day_of_week(
        unix_time
    ) * NUM_TIME_FEATURES_AM_PM + unix_time_to_AM_PM(unix_time)


def unix_time_to_day_of_week_and_four_paritions_a_day(unix_time):
    return unix_time_to_day_of_week(
        unix_time
    ) * NUM_TIME_FEATURES_FOUR + unix_time_to_four_paritions_a_day(unix_time)


def unix_time_to_day_of_week_and_hour(unix_time):
    return unix_time_to_day_of_week(
        unix_time
    ) * NUM_TIME_FEATURES_HOUR + unix_time_to_hour(unix_time)


def unix_time_to_day_of_week_and_hour(unix_time):
    return unix_time_to_day_of_week(
        unix_time
    ) * NUM_TIME_FEATURES_HOUR + unix_time_to_hour(unix_time)  # (0 ~ 24 * 7 - 1)


def unix_time_to_day_of_week_and_date(unix_time):
    return unix_time_to_day_of_week(
        unix_time
    ) * NUM_TIME_FEATURES_DATE + unix_time_to_date(unix_time)  # (0 ~ 31 * 7 - 1)


def unix_time_to_hour_and_date(unix_time):
    return unix_time_to_hour(unix_time) * NUM_TIME_FEATURES_DATE + unix_time_to_date(
        unix_time
    )  # (0 ~ 24 * 31 - 1)


def unix_time_to_day_of_week_and_hour_and_date(unix_time):
    return (
        unix_time_to_day_of_week(unix_time)
        * NUM_TIME_FEATURES_HOUR
        * NUM_TIME_FEATURES_DATE
        + unix_time_to_hour(unix_time) * NUM_TIME_FEATURES_DATE
        + unix_time_to_date(unix_time)
    )  # (0 ~ 31 * 24 * 7 - 1)


def days_passed_to_time_structure_tree(days_passed, num_time_structure):
    max_K_th_power_of_two = 1
    for i in range(BIG_NUM):
        max_K_th_power_of_two *= 2
        if num_time_structure < max_K_th_power_of_two:
            max_K_th_power_of_two /= 2
            max_K_th_power_of_two = int(max_K_th_power_of_two)
            break
    remainder = int(num_time_structure % max_K_th_power_of_two)
    if remainder == 0:
        ref_days = 366 / max_K_th_power_of_two
        ref_intervals = 366 / max_K_th_power_of_two
        for i in range(num_time_structure):
            if days_passed <= ref_days:
                return i
            ref_days += ref_intervals

        return num_time_structure - 1
    else:
        ref_days = 366 / (max_K_th_power_of_two * 2)
        ref_intervals = 366 / (max_K_th_power_of_two * 2)
        for i in range(num_time_structure):
            if days_passed <= ref_days:
                return i
            if i + 1 < 2 * remainder:
                ref_days += ref_intervals
            else:
                ref_days += ref_intervals * 2

        return num_time_structure - 1


def unix_time_to_time_structure_n_tree(unix_time, num_time_structure):
    # Convert the Unix timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(unix_time)

    # Get the year from the datetime object
    year = dt_object.year

    # Get the first day of the year for the given year
    first_day_of_year = datetime.datetime(year, 1, 1)

    # Calculate the number of days passed in the year
    number_of_days_passed = (dt_object - first_day_of_year).days + 1

    return days_passed_to_time_structure_tree(number_of_days_passed, num_time_structure)


def obtain_num_time_structure(time_structure_func):
    if time_structure_func == unix_time_to_season:
        return 4
    elif time_structure_func == unix_time_to_day_of_week:
        return 7
    elif time_structure_func == unix_time_to_hour:
        return 24
    elif time_structure_func == unix_time_to_date:
        return 31
    elif time_structure_func == unix_time_to_day_of_week_and_hour:
        return 7 * 24
    elif time_structure_func == unix_time_to_day_of_week_and_date:
        return 7 * 31
    elif time_structure_func == unix_time_to_hour_and_date:
        return 24 * 31
    elif time_structure_func == unix_time_to_day_of_week_and_hour_and_date:
        return 7 * 24 * 31


def obtain_num_episodes_for_Prognosticator(time_structure_func):
    return int(31 / obtain_num_time_structure(time_structure_func))


def obtain_num_time_structures_in_a_week(time_structure_func):
    if time_structure_func == unix_time_to_day_of_week:
        return 7
    elif time_structure_func == unix_time_to_hour:
        return 24 * 7
    elif time_structure_func == unix_time_to_date:
        return 7
    elif time_structure_func == unix_time_to_day_of_week_and_hour:
        return 7 * 24
    elif time_structure_func == unix_time_to_day_of_week_and_date:
        return 7
    elif time_structure_func == unix_time_to_hour_and_date:
        return 24 * 7
    elif time_structure_func == unix_time_to_day_of_week_and_hour_and_date:
        return 24 * 7


def obtain_num_days_in_one_cycle(time_structure_func):
    if time_structure_func == unix_time_to_season:
        return 365
    elif time_structure_func == unix_time_to_day_of_week:
        return 7
    elif time_structure_func == unix_time_to_hour:
        return 1
    elif time_structure_func == unix_time_to_time_structure_n_tree:
        return 365


# Normalize time to [0, 1] (or [0, scale])
def normalize_time(time, t_oldest, t_future, scale=1):
    return scale * (time - t_oldest) / (t_future - t_oldest)

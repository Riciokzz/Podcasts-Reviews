import numpy as np


def remove_outliers(df, column_name):
    """
    Function removes rows from given dataframe which have outliers.
    :param df: Main dataframe to work with.
    :param column_name: Integer column in which function will search outliers.
    :return: Dataframe without outliers.
    """

    q1, q3 = np.percentile(df[column_name], [25, 75])
    iqr = q3 - q1

    # All outliers
    all_outliers = (df[column_name] > q3 + iqr * 1.5) | (df[column_name] < q1 - iqr * 1.5)

    # Extreme outliers
    extreme_outliers = (df[column_name] > q3 + iqr * 3) | (df[column_name] < q1 - iqr * 3)

    f"All outliers: {all_outliers.sum()}, Extreme outliers: {extreme_outliers.sum()}"

    return df[~all_outliers]


def degrees_of_freedom(std1, std2, n1, n2):
    """
    Calculate degrees of freedom for Welch's t-test
    :param std1 and std2: STD for both samples
    :param n1, n2: Sample sizes
    :return df: Returning degree of freedom
    """
    numerator = (std1 ** 2 / n1 + std2 ** 2 / n2) ** 2
    denominator = (std1 ** 2 / n1) ** 2 / (n1 - 1) + (std2 ** 2 / n2) ** 2 / (n2 - 1)
    df = numerator / denominator

    return df

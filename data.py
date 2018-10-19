""" Generate time series data.
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller
import doctest


def __get_safe_value(var, index):
    return var[index] if index >= 0 else 0


def __sum_items(coef, var):
    """ Calculate the expression:
        sum(coef[i]*var[t-i]) over i,
    where i = 1, 2, ..., len(coef) and t = len(var).
    :param coef: list of numbers
    :param var: list of numbers
    :return: summation result

    >>> __sum_items([1, 2, 3, 4], [4, 3, 2, 1])  # 1*1 + 2*2 + 3*3 + 4*4
    30
    >>> __sum_items([1, 2, 3, 4], [2, 1]) # 1*1 + 2*2 + 3*0 + 4*0
    5
    >>> __sum_items([1, 2, 3, 4], []) # 0
    0
    >>> __sum_items([], []) # 0
    0
    """
    t = len(var) - 1
    order = len(coef)
    x = [__get_safe_value(var, t - i) for i in range(order)]
    return sum(map(lambda a, b: a * b, coef, x))


def arma(data_size, phi, theta, delta):
    x = [1] * data_size
    for t in range(data_size):
        e = np.random.normal(0, 1, t+1)
        x[t] = delta + __sum_items(phi, x[0:t]) + __sum_items(theta, e[0:t]) + e[t]
    return x


def test_stationarity(data, alpha=0.05, print_detail=True):
    """ Test stationarity of time series data.
    :param data: time series data, formatted as list
    :param alpha: significance level.
    :param print_detail: if True print additional information.
    """
    result = adfuller(data)
    is_stationary = True if result[1] <= alpha else False
    if print_detail:
        print('ADF statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('critical values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    return is_stationary


def season(m):
    """ Generates seasonal data with s=18.

    :param m: number of seasons.
    :return: list
    """
    k = 10
    x = [i for i in range(1, k+1)] + [k-i for i in range(1, k-1)]
    x = x * m
    noise = np.random.normal(0, 1, 2 * m * k)
    return [i for i in map(lambda a, b: a+b, noise, x)]


def season_diff(x, s):
    """ Make seasonal differential.

    :param x: data (list)
    :param s: length of a season (int)
    :return: list
    """
    size = len(x)
    y = [0] * size
    for i in range(size):
        y[i] = x[i] - x[i-s] if i-s >= 0 else x[i]
    return y


def garch(data_size, alpha=None, beta=None, alpha0=5):

    if alpha is None:
        alpha = []
    if beta is None:
        beta = []

    x = [0] * data_size
    x2 = [0] * data_size
    var = [1] * data_size

    for t in range(data_size):
        var[t] = alpha0 + __sum_items(alpha, x2[0:t]) + __sum_items(beta, var[0:t])
        x[t] = np.sqrt(var[t]) * np.random.normal(0, 1)
        x2[t] = x[t] * x[t]
    return x


if __name__ == '__main__':
    doctest.testmod()





import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics import tsaplots as tp

import data


class TestARMA(object):

    def __init__(self, data_size, phi=None, theta=None, delta=2):
        """ Initialize training data.
        :param data_size: data size
        :param phi: list, coefficients of lag X variables (X is observed data)
        :param theta: list, coefficients of lag W variables (W is error)
        :param delta: constant (do not set it zero)
        """
        if phi is None:
            phi = []
        if theta is None:
            theta = []
        # generate data via ARMA(p, q) process
        self.x = data.arma(data_size, phi, theta, delta)

    def plot_x(self):
        """ Plot training data.
        """
        plt.plot(self.x)
        plt.show()

    def plot_acf_pacf(self, lags=None):
        if not lags:
            lags = len(self.x) - 1
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(211)
        tp.plot_acf(self.x, ax=ax1, lags=lags, title='ACF')
        ax2 = fig.add_subplot(212)
        tp.plot_pacf(self.x, ax=ax2, lags=lags, title='PACF')
        plt.show()

    def test_stationarity(self, print_detail=False):
        return data.test_stationarity(self.x, print_detail=print_detail)

    def predict_and_plot(self, order, start=0, end=None):
        """ Make predictions, then plot.

        :param order: tuple, e.g. (p, q)
        :param start: int, start point of forecast
        :param end: int, end point of forecast
        """
        if not self.test_stationarity():
            raise ValueError("Data is not stationary!")
        if end is None:
            end = len(self.x) + len(self.x) // 2
        # fit model
        model = ARMA(self.x, order=order)
        result = model.fit(disp=-1)
        # predict
        y = result.predict(start=start, end=end)
        # plot
        plt.plot(range(start, end+1), list(y), color='red', label='ARMA(%d,%d)' % order)
        plt.plot(self.x)
        plt.legend()
        plt.show()


data_size = 100


def predict_ar():
    phi = [0.3]  # p = 1
    t = TestARMA(data_size, phi=phi)
    p = len(phi)
    t.predict_and_plot((p, 0))


def predict_ma():
    theta = [0.7, 0.3]
    t = TestARMA(data_size, theta=theta)
    q = len(theta)
    t.predict_and_plot((0, q))


def predict_arma():
    phi = [0.4, 0.3]
    theta = [0.3, 0.2]
    t = TestARMA(data_size, phi=phi, theta=theta)
    p = len(phi)
    q = len(theta)
    t.predict_and_plot((p, q))


def plot_acf_pacf_ar():
    phi = [0.5]
    t = TestARMA(data_size, phi=phi)
    t.plot_acf_pacf(20)


def plot_acf_pacf_ma():
    theta = [0.5]
    t = TestARMA(data_size, theta=theta)
    t.plot_acf_pacf(20)


def plot_acf_pacf_arma():
    phi = [0.5]
    theta = [0.5]
    t = TestARMA(data_size, phi=phi, theta=theta)
    t.plot_acf_pacf(20)


if __name__ == '__main__':
    predict_ar()
    predict_ma()
    predict_arma()

    plot_acf_pacf_ar()
    plot_acf_pacf_ma()
    plot_acf_pacf_arma()

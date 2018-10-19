import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics import tsaplots as tp

import data


class TestSARIMAX(object):

    def __init__(self, m):
        """
        :param m: number of seasons.
        """
        self.x = data.season(m)

    def plot_x(self):
        """ Plot training data.
        """
        plt.plot(self.x)
        plt.show()

    def plot_acf_pacf(self, lags=None):
        if not lags:
            lags = len(self.x) - 1
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        tp.plot_acf(self.x, ax=ax1, lags=lags, title='ACF')
        ax2 = fig.add_subplot(212)
        tp.plot_pacf(self.x, ax=ax2, lags=lags, title='PACF')
        plt.show()

    def test_stationarity(self, print_detail=False):
        return data.test_stationarity(self.x, print_detail=print_detail)

    def predict_and_plot(self, order, seasonal_order, start=0, end=None):
        """ Make predictions with ARIMA and seasonal ARIMA resp., then plot.

        :param order: tuple, i.e. (p, d, q)
        :param seasonal_order: tuple, i.e. (P,Q,D,s)
        :param start: int, start point of forecast
        :param end: int, end point of forecast
        """
        if not self.test_stationarity():
            raise ValueError("Data is not stationary!")
        if end is None:
            end = len(self.x) + len(self.x) // 2

        # -----------ARIMA------------
        # fit model
        model1 = ARIMA(self.x, order=order)
        result1 = model1.fit(disp=-1)
        # predict
        y1 = result1.predict(start=start, end=end)
        # plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(start, end + 1), list(y1), color='gray', label='ARIMA(%d,%d,%d)' % order)
        plt.plot(self.x)

        # -------Seasonal ARIMA-------
        # fit model
        model2 = SARIMAX(self.x, order=order, seasonal_order=seasonal_order)
        result2 = model2.fit(disp=-1)
        # predict
        y2 = result2.predict(start=start, end=end)
        # plot
        plt.plot(range(start, end+1), list(y2), color='red',
                 label='ARIMA(%d,%d,%d)*(%d,%d,%d, %d)' % tuple(list(order) + list(seasonal_order)))
        plt.legend()
        plt.show()


def test_sarimax():
    t = TestSARIMAX(5)
    t.predict_and_plot((1, 0, 0), (0, 1, 0, 18), start=90)


if __name__ == '__main__':
    test_sarimax()



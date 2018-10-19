import matplotlib.pyplot as plt

import data


class TestGarch(object):

    def __init__(self, data_size, alpha=None, beta=None, alpha0=5):
        self.x = data.garch(data_size, alpha, beta, alpha0)

    def plot_x(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, label='GARCH(1,1)')
        plt.show()

    def test_stationarity(self, print_detail=False):
        return data.test_stationarity(self.x, print_detail=print_detail)


if __name__ == '__main__':
    t = TestGarch(500, alpha=[0.5], beta=[0.5])
    if t.test_stationarity():
        t.plot_x()
    else:
        raise ValueError("data is not stationary!")

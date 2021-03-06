# Linear predictor system to predict the data of a 1-D function

import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
    # number of inputs
    n = 75
    # number of values to predict
    s = 500
    # learning rate
    lr = 0.001

    t = np.arange(0, 100+0.05, 0.05)
    nn = LinearPredictor(t, f(t), n, s, lr, True)

    nn.linear_system_training()
    nn.online_prediction()


class LinearPredictor:
    def __init__(self, t, x, n, s, alpha, graphs_on):
        """t is a vector with the time in which every element of x was sampled
           x is a vector with inputs
           n is the number of input to use
           s is the number of values to predict
           alpha is the learning rate for the training stage
           sr is the sampling rate of the input signal
        """
        self.t = t
        self.t_old = self.t.copy()
        self.start_time = t[0]
        self.end_time = t[-1]
        samples = x.size

        # in the last moment there are only n + s samples remaining
        self.stop_training = samples - n - s

        if self.stop_training < 0:
            sys.exit("Insufficient amount of input signal samples")

        self.x = x[:, np.newaxis]
        self.x_old = self.x.copy()
        self.y = self.x[n:n+s, 0:1]
        self.w = np.zeros((s, n+1))
        self.n = n
        self.s = s

        # training rate
        self.alpha = alpha

        # prediction
        self.x2 = None

        # graphics
        self.graphs_on = graphs_on

        if graphs_on:
            self.fig = plt.figure()
            plt.axes()
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

    def update_training_set(self, t, x, n, s, alpha):
        """t is a vector with the time in which every element of x was sampled
           x is a vector with inputs
           n is the number of input to use
           s is the number of values to predict
           alpha is the learning rate for the training stage
           sr is the sampling rate of the input signal
        """
        self.t = t
        self.x = x
        self.y = self.x[n:n+s, 0:1]
        self.n = n
        self.s = s

        # training rate
        self.alpha = alpha

        # prediction
        self.x2 = None

    def one_step_training(self):
        """Update all weights of the linear predictor"""
        input_x = np.ones((self.s, self.n+1))

        for i in range(self.s):
            input_x[i:i+1, 1:self.n+1] = self.x[i:self.n+i, 0:1].transpose().copy()

        # ----- partial training -----
        # dot product row wise to get predictions
        yp = np.einsum('ij, ij->i', self.w, input_x)
        yp = yp[:, np.newaxis]

        # weights update
        self.w = self.w + self.alpha * (self.y - yp) * input_x
        # print(self.w)

        # print('\nend of adaline training\n')

    def linear_system_training(self):
        print('starting linear system training')

        for offset in np.arange(0, self.stop_training + 1):
            t = self.t_old[offset:offset+self.n+self.s]
            x = self.x_old[offset:offset+self.n+self.s]

            self.update_training_set(t, x, self.n, self.s, self.alpha)

            self.one_step_training()
            data_x = x[0:self.n]
            self.predict(data_x)

            if self.graphs_on:
                if offset == 0:
                    plt.ion()
                    plt.show()

                self.plot()

        # check if all data was used
        # print(self.t[-1] == self.t_old[-1])

    def predict(self, data_x):
        input_x = np.ones((self.s, self.n+1))
        input_x[0:1, 1:self.n+1] = data_x.transpose().copy()
        x2 = data_x.copy()

        # dot product to get new prediction
        yp = np.matmul(self.w[0:1], input_x[0:1].transpose())
        x2 = np.append(x2, yp, axis=0)

        for i in range(1, self.w.shape[0]):
            input_x[i:i+1, 1:self.n+1] = x2[i:self.n+i, 0:1].transpose().copy()

            # dot product to get new prediction
            yp = np.matmul(self.w[i:i+1], input_x[i:i+1].transpose())

            # update x2
            x2 = np.append(x2, yp, axis=0)

        self.x2 = x2

    def online_prediction(self):
        print('starting linear system online predictions')

        for offset in np.arange(0, self.stop_training + 1):
            t = self.t_old[offset:offset+self.n+self.s]
            x = self.x_old[offset:offset+self.n+self.s]

            self.update_training_set(t, x, self.n, self.s, self.alpha)

            data_x = x[0:self.n]
            self.predict(data_x)

            if self.graphs_on:
                self.plot()

    def plot(self):
        a = self.fig.axes[0]
        a.plot(self.t_old, self.x_old, 'm')
        a.plot(self.t[0:self.n], self.x[0:self.n], '.b')
        a.plot(self.t[self.n:self.n+self.s], self.x2[self.n:], '.r')
        plt.ylim(-2, 2)
        self.fig.canvas.draw()
        plt.pause(0.0001)
        a.clear()


def f(t):
    return np.sin(t) + np.sin(2*t)


if __name__ == "__main__":
    main()

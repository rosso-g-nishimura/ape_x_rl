import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class XLogger:
    def __init__(self, save_dir):
        self.x_plot = os.path.join(save_dir, 'x_plot.jpg')
        self.x = []
        self.y = []
        self.loss_plot = os.path.join(save_dir, 'loss_plot.jpg')
        self.loss = []
        self.q_plot = os.path.join(save_dir, 'q_plot.jpg')
        self.q = []
        self.reward_plot = os.path.join(save_dir, 'reward_plot.jpg')
        self.reward = []

    def plot(self, x, loss, Q, reward, cnt):
        self.x.append(cnt)
        self.y.append(x)
        self.loss.append(loss)
        self.q.append(Q)
        self.reward.append(reward)
        plt.plot(self.x, self.y)
        plt.legend('x', loc='lower right')
        plt.savefig(self.x_plot)
        plt.close()
        plt.plot(self.x, self.loss)
        plt.legend('loss', loc='lower right')
        plt.savefig(self.loss_plot)
        plt.close()
        plt.plot(self.x, self.q)
        plt.legend('q', loc='lower right')
        plt.savefig(self.q_plot)
        plt.close()
        plt.plot(self.x, self.reward)
        plt.legend('reward', loc='lower right')
        plt.savefig(self.reward_plot)
        plt.close()

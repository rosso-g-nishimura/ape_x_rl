import torch

from sonic_env import make_env
from neural import Network


class DemoActor:
    def __init__(self, state_dim, shared_weight):
        self.state_dim = state_dim
        self.total_step = 0

        self.env = make_env()
        self.action_dim = self.env.action_space.n

        self.net = Network(self.state_dim, self.action_dim)
        self.net.load_state_dict(shared_weight)

    def demo(self, video=True):
        state = self.env.reset()
        done = False
        frames = []
        while not done:
            if video:
                frames.append(self.env.render(mode='rgb_array'))
            else:
                self.env.render()
            with torch.no_grad():
                q_est = self.net(torch.tensor(state).float().unsqueeze(0))
                action = torch.argmax(q_est, dim=1).item()
            state, reward, done, info = self.env.step(action)
            # print(reward)
            self.total_step += 1

        print(f'total: {self.total_step} steps')

        if video:
            return frames
        else:
            return None

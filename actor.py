from collections import namedtuple
import torch
import random
import ray

from sonic_env import make_env
from neural import Network


Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'done', 'next_state', 'target_Q'])
N_Step_Transition = namedtuple('N_Step_Transition',
                               ['state', 'action', 'reward', 'done', 'next_state', 'target_Q', 'next_target_Q'])


class LocalBuffer:
    def __init__(self, n_step, gamma=0.99):
        self.n_step = n_step
        self.short_buffer = []
        self.n_step_buffer = []
        self.gamma = gamma
        self.gamma_list = [self.gamma ** i for i in range(self.n_step)]
        self.gamma_n = self.gamma ** self.n_step

    def add(self, tr, done):
        self.short_buffer.append(tr)
        if len(self.short_buffer) == self.n_step:
            rewards = [x.reward for x in self.short_buffer]
            n_reward = sum([r * g for (r, g) in zip(rewards, self.gamma_list)])
            self.n_step_buffer.append(
                N_Step_Transition(
                    self.short_buffer[0].state, self.short_buffer[0].action,
                    n_reward, self.short_buffer[0].done, self.short_buffer[-1].next_state,
                    self.short_buffer[0].target_Q, self.short_buffer[-1].target_Q))
            if done:
                self.short_buffer.clear()
            else:
                self.short_buffer = self.short_buffer[1:]

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.size
        elif self.size < batch_size:
            return None

        sample_buffer = self.n_step_buffer[:batch_size]
        del self.n_step_buffer[:batch_size]
        return sample_buffer

    @property
    def size(self):
        return len(self.n_step_buffer)


@ray.remote
class Actor:
    def __init__(self, actor_id, state_dim, weight_path,
                 steps=20_000, n_step=3, epsilon=0.1, gamma=0.99,
                 n_step_batch_size=10, test_mode=False):
        self.actor_id = actor_id
        self.state_dim = state_dim
        self.steps = steps
        self.total_step = 0

        self.env = make_env()
        self.state = self.env.reset()
        self.action_dim = self.env.action_space.n
        self.time_out = 3_000

        self.net = Network(self.state_dim, self.action_dim)
        self.weight_path = weight_path
        self.load_weight_path(weight_path)
        self.net.cpu()

        self.local_buffer = LocalBuffer(n_step, gamma)
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_gamma = gamma ** n_step
        self.n_step_batch_size = n_step_batch_size

        self.is_test = test_mode

    def load_weight_path(self, weight_path):
        if weight_path != '.' and self.weight_path != weight_path:
            self.weight_path = weight_path
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            self.net.load_state_dict(state_dict)

    # dqn td error (not ddqn)
    def compute_priorities(self, batch):
        priorities = []
        for tr in batch:
            priority = tr.reward + \
                       (1 - tr.done) * self.n_gamma * torch.argmax(tr.next_target_Q, dim=1) - \
                       torch.argmax(tr.target_Q, dim=1)
            priority = torch.abs(priority).float().item()
            priorities.append(priority)
        return priorities

    def run(self, weight_path='.'):
        self.load_weight_path(weight_path)
        state = self.state
        for step in range(self.steps):
            # estimate
            with torch.no_grad():
                q_est = self.net(torch.tensor(state).float().unsqueeze(0))
            # epsilon-greedy
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = torch.argmax(q_est, dim=1).item()
            # step
            next_state, reward, done, info = self.env.step(action)
            self.local_buffer.add(Transition(state, action, reward, done, next_state, q_est), done)

            # if done
            if done:
                state = self.env.reset()
                self.total_step = 0
            else:
                state = next_state
                self.total_step += 1

        self.state = state
        batch = self.local_buffer.sample()
        priorities = self.compute_priorities(batch)
        return batch, priorities, self.actor_id

    def test(self, weight_path):
        self.load_weight_path(weight_path)
        state = self.env.reset()
        x = 0
        r = 0
        for _ in range(self.time_out):
            with torch.no_grad():
                q_est = self.net(torch.tensor(state).float().unsqueeze(0))
                action = torch.argmax(q_est, dim=1).item()
            state, reward, done, info = self.env.step(action)
            r += reward
            if done:
                break
            else:
                x = info['x']
        return x, r

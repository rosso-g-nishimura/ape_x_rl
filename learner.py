import copy
import numpy as np
import os
import torch
import ray
from collections import namedtuple

from neural import Network

N_Step_Transition = namedtuple('N_Step_Transition',
                               ['state', 'action', 'reward', 'done', 'next_state', 'target_Q', 'next_target_Q'])


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, state_dim, action_dim,
                 batch_size=32, n_step=3, gamma=0.99, target_sync=2_500,
                 save_dir='.', load_path=None, sync_cnt=0):
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.n_gamma = gamma ** n_step
        self.target_sync = target_sync
        self.sync_cnt = sync_cnt
        self.main_Q = Network(self.state_dim, action_dim).float().cuda()

        self.save_dir = save_dir
        self.weight_path = '.'
        if load_path is not None:
            state_dict = torch.load(load_path, map_location=('cuda'))
            self.main_Q.load_state_dict(state_dict)

        # LOAD WEIGHT
        self.shared_weight = self.main_Q.state_dict()
        self.optimizer = torch.optim.Adam(self.main_Q.parameters(), lr=0.0001)

        self.target_Q = copy.deepcopy(self.main_Q)
        for p in self.target_Q.parameters():
            p.requires_grad = False

    def get_weight_path(self):
        return self.weight_path

    def learn(self, batches):
        priorities = []
        losses = []
        Qs = []
        for batch in batches:
            loss, priority, Q = self.compute(batch)
            priorities.append(priority)
            self.update(loss)
            losses.append(loss)
            Qs.append(Q)
        losses = torch.stack(losses).mean().to('cpu').detach().numpy()
        Qs = torch.cat(Qs, dim=0).mean().to('cpu').detach().numpy()
        return priorities, self.weight_path, self.sync_cnt // self.target_sync, losses, Qs

    def compute(self, batch):
        tr = N_Step_Transition(*zip(*batch))
        states = torch.from_numpy(np.array(tr.state).astype(np.float32)).cuda()
        next_states = torch.from_numpy(np.array(tr.next_state).astype(np.float32)).cuda()
        rewards = torch.from_numpy(np.array(tr.reward).astype(np.float32)).cuda()
        dones = torch.from_numpy(np.array(tr.done).astype(np.int32)).cuda()
        actions = np.array(tr.action)

        estimate_Q = self.main_Q(states)[np.arange(0, self.batch_size), actions]
        with torch.no_grad():
            targets = self.target_Q(next_states)
            best_actions = torch.argmax(self.main_Q(next_states), dim=1)
        td_err = rewards\
            + (1 - dones) * self.n_gamma * targets[np.arange(0, self.batch_size), best_actions]\
            - estimate_Q

        loss = ((td_err ** 2) / 2).mean()
        priority = torch.abs(td_err)
        return loss, priority, estimate_Q

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.sync_cnt += 1
        if self.sync_cnt % self.target_sync == 0:
            self.target_Q.load_state_dict(self.main_Q.state_dict())
            self.weight_path = os.path.join(self.save_dir,
                                            f'sync_{str(self.sync_cnt // self.target_sync).zfill(6)}.chkpt')
            torch.save(self.main_Q.state_dict(), self.weight_path)

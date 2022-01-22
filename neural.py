import torch.nn as nn


# dueling net
class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        c, h, w = input_dim
        self.output_dim = output_dim

        # 4x84x84 -> 32x20x20 -> 64x9x9 -> 64x7x7
        ch_1, ks_1, st_1 = 64, 8, 4
        ch_2, ks_2, st_2 = 64, 4, 2
        ch_3, ks_3, st_3 = 64, 3, 1
        fc_1, fc_2 = 3136, 512

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=ch_1, kernel_size=ks_1, stride=st_1),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch_1, out_channels=ch_2, kernel_size=ks_2, stride=st_2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch_2, out_channels=ch_3, kernel_size=ks_3, stride=st_3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_a = nn.Sequential(
            nn.Linear(fc_1, fc_2),
            nn.ReLU(),
            nn.Linear(fc_2, self.output_dim)
        )
        self.fc_v = nn.Sequential(
            nn.Linear(fc_1, fc_2),
            nn.ReLU(),
            nn.Linear(fc_2, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        adv = self.fc_a(x)
        val = self.fc_v(x)
        duel = val + adv - adv.mean(1, keepdim=True).expand(-1, self.output_dim)
        return duel


# # not dueling network
# class SimpleNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SimpleNetwork, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_dim)
#         )
#
#     def forward(self, x):
#         return self.net(x)

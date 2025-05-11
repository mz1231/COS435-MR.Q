import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# Architecture dimensions
zs_dim = 512
za_dim = 256
zsa_dim = 512
reward_bins = 65

def ln_activ(x, activ):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)

# weight and bias initialization 
def init_weights(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

class StateEncoder(nn.Module):
    def __init__(self, image_observation_space, state_channels):
        super().__init__()
        self.image_observation_space = image_observation_space
        self.activ = F.elu

        if self.image_observation_space:
            self.zs_cnn1 = nn.Conv2d(state_channels, 32, 3, stride=2)
            self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            self.zs_lin = nn.Linear(1568, zs_dim)  # assumes 84x84 input
        else:
            self.zs_mlp1 = nn.Linear(state_channels, 512)
            self.zs_mlp2 = nn.Linear(512, 512)
            self.zs_mlp3 = nn.Linear(512, zs_dim)

        self.apply(init_weights)

    def cnn_forward(self, state):
        state = state / 255.0 - 0.5
        x = self.activ(self.zs_cnn1(state))
        x = self.activ(self.zs_cnn2(x))
        x = self.activ(self.zs_cnn3(x))
        x = self.activ(self.zs_cnn4(x))
        x = x.reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(x), self.activ)
    
    def mlp_forward(self, state):
        x = ln_activ(self.zs_mlp1(state), self.activ)
        x = ln_activ(self.zs_mlp2(x), self.activ)
        x = ln_activ(self.zs_mlp3(x), self.activ)
        return x

    def forward(self, state):
        if self.image_observation_space:
            return self.cnn_forward(state)
        else:
            return self.mlp_forward(state)

class StateActionEncoder(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.activ = F.elu
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa1 = nn.Linear(zs_dim + za_dim, 512)
        self.zsa2 = nn.Linear(512, 512)
        self.zsa3 = nn.Linear(512, zsa_dim)
        output_dim = reward_bins + zs_dim + 1
        self.model = nn.Linear(zsa_dim, output_dim)

        self.apply(init_weights)

    def forward(self, zs, action):
        za = self.activ(self.za(action))
        zsa = torch.cat([zs, za], dim=1)
        zsa = ln_activ(self.zsa1(zsa), self.activ)
        zsa = ln_activ(self.zsa2(zsa), self.activ)
        zsa = self.zsa3(zsa)
        return self.model(zsa), zsa

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim: int, discrete_action_space: bool):
        super().__init__()
        self.l1 = nn.Linear(zs_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)
        if discrete_action_space:
            self.final_activ = partial(F.gumbel_softmax, tau=10)
        else:
            self.final_activ = torch.tanh

    def forward(self, zs: torch.Tensor):
        pre_activ = F.relu(self.l1(zs))
        pre_activ = F.relu(self.l2(pre_activ))
        
        x = self.final_activ(self.l3(pre_activ))
        return x, pre_activ


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # layers for q1
        self.l1 = nn.Linear(zsa_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

        # layers for q2
        self.l4 = nn.Linear(zsa_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)

        self.apply(init_weights)

    def forward(self, zsa: torch.Tensor):
        q1 = F.relu(self.l1(zsa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(zsa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    def Q1(self, zsa: torch.Tensor):
        q1 = F.relu(self.l1(zsa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1
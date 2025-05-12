import numpy as np
import torch
import torch.nn.functional as F
import buffer
import networks
import copy

class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool, multi_discrete: bool,
        device: torch.device, history):

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.max_action = max_action
        self.pixel_obs = pixel_obs
        self.discrete = discrete
        self.multi_discrete = multi_discrete
        self.device = device
        self.history = history

        # Hyperparameters (page 23 of original paper)
        self.dyn_loss_weight = 1
        self.reward_loss_weight = 0.1
        self.terminal_loss_weight = 0.1
        self.pre_activ_weight = 1e-5
        self.encoder_horizon = 5
        self.multi_horizon_Q = 3

        # TD3 
        self.target_policy_noise = 0.2 * (0.5 if discrete else 1.0)
        self.target_policy_noise_clip = 0.3 * (0.5 if discrete else 1.0)

        # LAP
        self.prob_smoothing = 0.4
        self.min_priority = 1

        # Exploration
        self.init_rand_exploration = 10000
        self.exploration_noise = 0.2 * (0.5 if discrete else 1.0)

        # Common
        self.discount = 0.99
        self.buffer_cap = 1e6
        self.batch_size = 256
        self.target_update_freq = 250
        self.replay_ratio = 1

        # Initialize buffer, encoders, value, and policy
        self.replay_buffer = buffer.ReplayBuffer(
            self.obs_shape, self.action_dim, self.max_action, self.pixel_obs, self.device,
            self.history, self.encoder_horizon, self.buffer_cap, self.batch_size,
            True, initial_priority=self.min_priority)
        
        self.state_shape = self.replay_buffer.state_shape

        self.state_encoder = networks.StateEncoder(self.pixel_obs, self.obs_shape[0] * self.history).to(self.device)
        self.state_action_encoder = networks.StateActionEncoder(self.action_dim).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(
            list(self.state_encoder.parameters()) + list(self.state_action_encoder.parameters()),
            lr=1e-4, weight_decay=1e-4)
        self.state_encoder_target = copy.deepcopy(self.state_encoder)
        self.state_action_encoder_target = copy.deepcopy(self.state_action_encoder)

        self.value = networks.ValueNetwork().to(self.device)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=3e-4, weight_decay=1e-4)
        self.value_target = copy.deepcopy(self.value)

        self.policy = networks.PolicyNetwork(action_dim=self.action_dim, discrete_action_space=self.discrete).to(self.device)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=3e-4, weight_decay=1e-4)
        self.policy_target = copy.deepcopy(self.policy)

        self.reward_scale = 1
        self.target_reward_scale = 0
        self.training_steps = 0
        self.two_hot = TwoHot(self.device, -10, 10, 65)

    def select_action(self, state: np.array, use_exploration: bool=True):
        # Sample random action if in exploration phase (first 10k steps and during evaluation)
        if use_exploration and self.replay_buffer.size <= self.init_rand_exploration:
            return None 

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            state = state.reshape(-1, *self.state_shape)
            zs = self.state_encoder.forward(state)
            action = self.policy.forward(zs)[0]
            if use_exploration:
                action += torch.randn_like(action) * self.exploration_noise
            if self.discrete:
                return int(action.argmax())
            elif self.multi_discrete:
                action_probs = torch.softmax(action, dim=-1).cpu().data.numpy()
                action_probs /= action_probs.sum(axis=-1, keepdims=True)
                return np.array([int(action.argmax()) for i in range(self.action_dim)])
            else:
                return action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action

    # Update MR.Q
    def train(self):
        # still exploring random actions
        if self.replay_buffer.size <= self.init_rand_exploration: 
            return

        if self.training_steps % self.target_update_freq == 0:
            self.target_networks_and_reward_scaling()

            for t in range(self.target_update_freq):
                # Encoder Update
                self.encoder_update()

        Q, Q_target = self.policy_and_value_update(self.replay_buffer)

        # use LAP to sample transitions with priority according to TD errors
        TD_error_priority = (Q - Q_target.expand(-1,2)).abs().max(1).values
        priority = TD_error_priority.clamp(min=self.min_priority).pow(self.prob_smoothing)
        self.replay_buffer.update_priority(priority)

        self.training_steps += 1

    def target_networks_and_reward_scaling(self):
        # Target Networks
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.value_target.load_state_dict(self.value.state_dict())
        self.state_encoder_target.load_state_dict(self.state_encoder.state_dict())
        self.state_action_encoder_target.load_state_dict(self.state_action_encoder.state_dict())
        # Reward Scaling
        self.target_reward_scale = self.reward_scale
        self.reward_scale = self.replay_buffer.reward_scale()
    
    # Encoder Training (Section 4.2.1 of original Paper)
    def encoder_update(self):
        state, action, next_state, reward, not_done = self.replay_buffer.sample(
        self.encoder_horizon, include_intermediate=True
        )
        state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, use_augs=True)
        
        with torch.no_grad():
            target_zs_next = self.state_encoder_target(next_state.reshape(-1, *self.state_shape))
            target_zs_next = target_zs_next.reshape(state.shape[0], -1, 512)
        
        pred_zs = self.state_encoder(state[:, 0])
        encoder_loss = 0
        prev_mask = torch.ones((state.shape[0], 1), device=state.device)

        for t in range(self.encoder_horizon):
            loss_t = self.encoder_loss_step(
                pred_zs, action[:, t], reward[:, t], not_done[:, t], target_zs_next[:, t], prev_mask, self.replay_buffer.env_terminates
            )
            encoder_loss += loss_t
            prev_mask = prev_mask * not_done[:, t].reshape(-1, 1)

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()

    def encoder_loss_step(self, zs, a_t, r_t, not_done_t, target_zs_next_t, mask, env_terminates):
        pred_all, _ = self.state_action_encoder(zs, a_t)
        pred_d, pred_zs_next, pred_r = pred_all[:, :1], pred_all[:, 1:513], pred_all[:, 513:]

        loss_r = self.reward_loss(pred_r, r_t, mask)
        loss_d = self.dynamics_loss(pred_zs_next, target_zs_next_t, mask)
        loss_t = self.terminal_loss(pred_d, not_done_t, mask) if env_terminates else 0

        # Encoder Loss Formula
        return (
            self.reward_loss_weight * loss_r
            + self.dyn_loss_weight * loss_d
            + self.terminal_loss_weight * loss_t
        )

    # Reward Loss Formula (L_Reward)
    def reward_loss(self, pred_r, r_t, mask):
        return (self.two_hot.cross_entropy_loss(pred_r, r_t) * mask).mean()
    
    # Dynamics Loss Formula (L_Dynamics)
    def dynamics_loss(self, pred_zs_next, target_zs_next, mask):
        return (F.mse_loss(pred_zs_next, target_zs_next, reduction='none') * mask).mean()

    # Terminal Loss Formula (L_Terminal)
    def terminal_loss(self, pred_d, not_done_t, mask):
        target_done = 1. - not_done_t.reshape(-1, 1)
        return (F.mse_loss(pred_d, target_done, reduction='none') * mask).mean()

    # Policy and Value Update (Sections 4.2.2 and 4.2.3 of original Paper) - based on TD3 from COS 435 HW 6 
    def policy_and_value_update(self, replay_buffer):
        # Sample replay buffer - predict multi-step returns over a horizon H_Q
        state, action, next_state, reward, not_done = replay_buffer.sample(self.multi_horizon_Q, include_intermediate=False)
        state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, True)
        reward, term_discount = multi_step_reward(reward, not_done, self.discount)

        with torch.no_grad():
            next_zs = self.state_encoder_target.forward(next_state)

            # Select action according to policy and add clipped noise.
            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(-self.target_policy_noise_clip, self.target_policy_noise_clip)
            next_action = self.policy_target.forward(next_zs)[0] + noise
            if self.discrete:
                next_action = F.one_hot(next_action.argmax(1), next_action.shape[1]).float()
            else:
                next_action = next_action.clamp(-1,1)

            # Compute the target Q value
            next_zsa = self.state_action_encoder_target(next_zs, next_action)[1]
            next_Q1, next_Q2 = self.value_target(next_zsa)
            Q_target = torch.min(next_Q1, next_Q2)
            Q_target = (reward + term_discount * Q_target * self.target_reward_scale)/self.reward_scale

            zs = self.state_encoder.forward(state) 
            zsa = self.state_action_encoder.forward(zs, action)[1]

        # Get current Q estimates
        current_Q1, current_Q2 = self.value(zsa)
        Q = torch.cat([current_Q1, current_Q2], 1)

        # Compute critic loss
        value_loss = F.huber_loss(Q, Q_target.expand(-1,2), delta=1.0)

        # Optimize the critic
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 20)
        self.value_optimizer.step()

        # Compute actor loss
        policy_action, pre_activ = self.policy(zs)
        zsa = self.state_action_encoder(zs, policy_action)[1]
        Q_policy = self.value.Q1(zsa)
        policy_loss = -Q_policy.mean() + self.pre_activ_weight * pre_activ.pow(2).mean()

        # Optimize the actor
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        return Q, Q_target

# Taken from original MR.Q algorithm (Fujimoto et. al)
class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins

    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot

    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)

# Taken from original MR.Q algorithm (Fujimoto et. al)
def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float):
    ms_reward = 0
    scale = 1
    for i in range(reward.shape[1]):
        ms_reward += scale * reward[:,i]
        scale *= discount * not_done[:,i]
    
    return ms_reward, scale

# Taken from original MR.Q algorithm (Fujimoto et. al)
def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state

# Taken from original MR.Q algorithm (Fujimoto et. al)
# Random shift.
def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)
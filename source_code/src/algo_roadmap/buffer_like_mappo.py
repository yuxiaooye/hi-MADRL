import numpy as np
import math

class SeparatedReplayBuffer():

    def __init__(self, T_horizon, n_rollout_threads, obs_dim, act_dim, is_uav):
        self.episode_length = math.ceil(T_horizon / n_rollout_threads)  # threadsT_horizon
        self.n_rollout_threads = n_rollout_threads
        self.is_uav = is_uav

        # episode_length1ceil
        # obsobs_prime
        self.obs = np.zeros((self.episode_length, self.n_rollout_threads, obs_dim))
        self.obs_prime = np.zeros((self.episode_length, self.n_rollout_threads, obs_dim))
        act_d = act_dim if is_uav else 1
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_d))
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_d))
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1))
        self.dones = np.ones((self.episode_length, self.n_rollout_threads, 1))
        self.available_actions = None if is_uav else np.ones((self.episode_length, self.n_rollout_threads, act_dim))
        self.nei_r = np.zeros((self.episode_length, self.n_rollout_threads, 1))
        self.uav_r = np.zeros((self.episode_length, self.n_rollout_threads, 1))
        self.car_r = np.zeros((self.episode_length, self.n_rollout_threads, 1))
        self.global_r = np.zeros((self.episode_length, self.n_rollout_threads, 1))

        self.step = 0

    def buffer_insert(self, obs, actions, action_log_probs, rewards, obs_prime, dones, nei_r, uav_r, car_r, global_r, available_actions=None):
        self.obs[self.step] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.rewards[self.step] = rewards.copy()
        self.obs_prime[self.step] = obs_prime.copy()
        self.dones[self.step] = dones.copy()
        self.nei_r[self.step] = nei_r.copy()
        self.uav_r[self.step] = uav_r.copy()
        self.car_r[self.step] = car_r.copy()
        self.global_r[self.step] = global_r.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length















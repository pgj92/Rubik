"""
PPO for the Rubik's Cube environment.

Adapted from CleanRL's single-file ppo.py:
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    (kept locally as `train/ppo.py` for reference / diffing)

The three substantive changes from upstream:
  1. Env factory: replaced `gym.make(env_id)` with our RubikEnv.
  2. Network: observations are int color indices in 0..5; we one-hot them
     to a 324-dim float vector at the network input.
  3. Hyperparams: cube-specific defaults (shorter episodes, sparser reward).

Usage:
    python -m train.ppo_cube --scramble-depth 1 --total-timesteps 200000
"""
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cube.gym_env import RubikEnv
from cube.cube import N_COLORS, N_STICKERS


@dataclass
class Args:
    exp_name: str = "ppo_cube"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # --- Cube-specific args ---
    scramble_depth: int = 1
    """fixed scramble depth (curriculum will plug in here later)"""
    max_steps: int = 30
    """episode step limit; should be a bit larger than scramble_depth"""

    # --- PPO standard args ---
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.95               # shorter horizon for the cube
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.05            # higher entropy bonus — sparse reward needs more exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ============================================================================
# TODO A: env factory
# ----------------------------------------------------------------------------
# Replace the upstream `make_env(env_id, idx, capture_video, run_name)` with
# a version that creates our RubikEnv. Wrap with RecordEpisodeStatistics so
# CleanRL's logging picks up returns and episode lengths automatically.
#
# Signature should be:
#     def make_env(scramble_depth: int, max_steps: int) -> Callable[[], gym.Env]
#
# ============================================================================
def make_env(scramble_depth: int, max_steps: int):
    def thunk():
        env = RubikEnv(scramble_depth=scramble_depth, max_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# TODO B: one-hot the observation
# ----------------------------------------------------------------------------
# Our env's observation is shape (54,), dtype int8, values in {0,1,2,3,4,5}.
# A small MLP doesn't learn well from raw integer color indices — distance
# between color 1 and color 2 is meaningless. Standard fix: one-hot encode
# each sticker, giving a (54, 6) → flatten to (324,) float vector.
#
# Implement _onehot below so that:
#     x:  (batch, 54)        int tensor of values 0..5
#     out: (batch, 324)      float tensor with N_STICKERS * N_COLORS one-hots
# ============================================================================
def _onehot(x: torch.Tensor) -> torch.Tensor:
    return F.one_hot(x.long(), num_classes=N_COLORS).reshape(-1).float()


# Input dimension to the MLPs after one-hot
OBS_DIM = N_STICKERS * N_COLORS   # 54 * 6 = 324


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(_onehot(x))

    def get_action_and_value(self, x, action=None):
        h = _onehot(x)
        logits = self.actor(h)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(h)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__d{args.scramble_depth}__s{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ============================================================================
    # TODO C: wire up the parallel envs
    # ----------------------------------------------------------------------------
    # Use gym.vector.SyncVectorEnv with `args.num_envs` copies of make_env(...).
    # This is a one-liner once TODO A is done.
    # ============================================================================
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.scramble_depth, args.max_steps) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage. Observations are stored as int (we'll one-hot at network input).
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.long).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.long, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    # Rolling success tracker for logging
    recent_returns: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.long, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            # RecordEpisodeStatistics: gymnasium 1.0+ puts finished-episode stats
            # under infos["episode"] as a dict-of-arrays, with infos["_episode"]
            # as a boolean mask over the sub-envs.
            if "episode" in infos:
                mask = infos.get("_episode", np.ones(args.num_envs, dtype=bool))
                for i in range(args.num_envs):
                    if mask[i]:
                        ep_r = float(infos["episode"]["r"][i])
                        ep_l = int(infos["episode"]["l"][i])
                        recent_returns.append(ep_r)
                        if len(recent_returns) > 100:
                            recent_returns.pop(0)
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)

        # bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if recent_returns:
            mean_ret = float(np.mean(recent_returns))
            print(f"iter {iteration:4d}  step {global_step:8d}  "
                  f"mean_ret(last 100)={mean_ret:.3f}  SPS={sps}")

    envs.close()
    writer.close()

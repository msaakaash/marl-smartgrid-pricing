import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from wrapper_citylearn_marl import CityLearnParallelEnv
from torch.utils.tensorboard import SummaryWriter


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        self.mean = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.mean(x), self.log_std.exp(), self.value(x)


def train(schema="citylearn/data/citylearn_challenge_2022/climate_zone_1.json"):
    env = CityLearnParallelEnv(schema, n_consumers=5)
    writer = SummaryWriter()

    obs, _ = env.reset()
    consumer_obs_dim = len(obs["consumer_0"])
    cons_model = ActorCritic(consumer_obs_dim, 1)
    agg_model = ActorCritic(len(obs["aggregator"]), 2)
    grid_model = ActorCritic(len(obs["grid"]), 1)

    opt = optim.Adam(list(cons_model.parameters()) + list(agg_model.parameters()) + list(grid_model.parameters()), lr=1e-3)

    for update in range(10):
        obs, _ = env.reset()
        ep_reward = {a: 0.0 for a in env.agents}
        for step in range(50):
            actions = {}
            for a in env.agents:
                x = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                if "consumer" in a:
                    mean, std, val = cons_model(x)
                    dist = Normal(mean, std)
                    act = dist.sample()
                    actions[a] = act.squeeze().detach().numpy()
                elif a == "aggregator":
                    mean, std, val = agg_model(x)
                    dist = Normal(mean, std)
                    actions[a] = dist.sample().squeeze().detach().numpy()
                elif a == "grid":
                    mean, std, val = grid_model(x)
                    dist = Normal(mean, std)
                    actions[a] = dist.sample().squeeze().detach().numpy()

            obs, rew, term, trunc, info = env.step(actions)
            for k, v in rew.items():
                ep_reward[k] += v
            if all(term.values()):
                break

        print(f"Update {update}: { {k: round(v,2) for k,v in ep_reward.items()} }")
        writer.add_scalar("reward/consumer_avg", np.mean([ep_reward[a] for a in ep_reward if "consumer" in a]), update)
        writer.add_scalar("reward/aggregator", ep_reward["aggregator"], update)
        writer.add_scalar("reward/grid", ep_reward["grid"], update)

    writer.close()


if __name__ == "__main__":
    train()

from wrapper_citylearn_marl import CityLearnParallelEnv

def evaluate(schema="citylearn/data/citylearn_challenge_2022/climate_zone_1.json"):
    env = CityLearnParallelEnv(schema, n_consumers=5)
    obs, _ = env.reset()
    total_demand = []
    for t in range(100):
        actions = {a: [0.0] for a in env.agents}  # dummy no-op
        obs, rew, term, trunc, info = env.step(actions)
        total_demand.append(sum(env.env.net_electricity_consumption))
        if all(term.values()):
            break
    print("Average demand:", sum(total_demand) / len(total_demand))

if __name__ == "__main__":
    evaluate()

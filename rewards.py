def consumer_reward(consumption, price, comfort_dev, emergency, incentive=0.0):
    """Shaped reward for consumers."""
    cost_penalty = -price * consumption
    comfort_penalty = -5.0 * comfort_dev
    emergency_penalty = -10.0 * emergency
    incentive_bonus = 2.0 * incentive
    return cost_penalty + comfort_penalty + emergency_penalty + incentive_bonus


def aggregator_reward(peak_demand, total_incentive):
    """Aggregator tries to reduce peak demand but also minimize incentive spending."""
    return -peak_demand - 0.5 * total_incentive


def grid_reward(total_demand, grid_limit):
    """Grid penalizes overload beyond limit."""
    overload = max(0.0, total_demand - grid_limit)
    return -overload * 10.0

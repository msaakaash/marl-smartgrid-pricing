from wrapper_citylearn_marl import MARLWrapper
import numpy as np

def train_prototype():
    wrapper = MARLWrapper()
    
    # A simple dictionary to store a "policy" for each agent
    # Dynamically initialize based on the number of consumers from the wrapper
    agent_policies = {i: {} for i in range(wrapper.num_consumers)}

    num_episodes = 2
    for episode in range(num_episodes):
        wrapper.reset()
        print(f"--- Training Episode {episode + 1}/{num_episodes} ---")
        
        # A 24-hour simulation
        for t in range(24):
            state, consumer_rewards, grid_reward, dr_type, new_peak = wrapper.step()
            
            # Simplified learning update (conceptual)
            # The agent conceptually updates its policy based on the reward
            # This is a placeholder for a real learning algorithm
            for i in range(wrapper.num_consumers):
                agent_policies[i][t] = consumer_rewards[i]
            
            print(f"Time Step {t}: DR Signal = {dr_type}, New Peak Demand = {new_peak:.2f}, Grid Reward = {grid_reward:.2f}")

    print("\nTraining complete.")
    print("Simplified 'Learned' Policy (Average Reward per agent at each step):")
    for agent_id, policy in agent_policies.items():
        print(f"Agent {agent_id}: {policy}")

    return agent_policies

if __name__ == "__main__":
    trained_policy = train_prototype()
    # Save the 'policy' to be used by the evaluation script
    np.save('trained_policy.npy', trained_policy, allow_pickle=True)
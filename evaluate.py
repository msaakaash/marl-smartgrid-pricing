from wrapper_citylearn_marl import MARLWrapper
import numpy as np

def evaluate_prototype():
    try:
        # Load the 'trained' policy from the training script
        trained_policy = np.load('trained_policy.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print("Please run train_marl_a2c.py first to train the model.")
        return

    wrapper = MARLWrapper()
    wrapper.reset()

    new_peak_demands = []

    print("--- Running Evaluation Simulation ---")

    for t in range(24):
        # In a real evaluation, you would load the trained policy to choose an action.
        # For this prototype, the wrapper's step function is deterministic, so we just run it.
        state, consumer_rewards, grid_reward, dr_type, new_peak = wrapper.step()

        print(f"Time Step {t}: DR Signal = {dr_type}, New Peak Demand = {new_peak:.2f}, Grid Reward = {grid_reward:.2f}")
        new_peak_demands.append(new_peak)

    avg_new_peak = np.mean(new_peak_demands)
    print("\nEvaluation Complete.")
    print(f"Average Peak Demand during evaluation: {avg_new_peak:.2f}")

if __name__ == "__main__":
    evaluate_prototype()
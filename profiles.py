import numpy as np

class ConsumerProfile:
    """Represents a consumer's profile with key attributes."""
    def __init__(self, building_id, flexibility, willingness):
        self.id = building_id
        # Represents how much the consumer can shift their load (0.0 to 1.0)
        self.flexibility = flexibility
        # Represents the consumer's willingness to participate in DR (0.0 to 1.0)
        self.willingness = willingness

def generate_profiles(num_consumers):
    """Generates a list of random consumer profiles."""
    profiles = []
    for i in range(num_consumers):
        # Generate random values for flexibility and willingness
        flexibility = np.random.uniform(0.1, 0.9)
        willingness = np.random.uniform(0.1, 0.9)
        profiles.append(ConsumerProfile(i, flexibility, willingness))
    return profiles
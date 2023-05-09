import numpy as np
import matplotlib.pyplot as plt

# Load the rewards from the .npy file
rewards = np.load('reward_array.npy')

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards over time')

# Save the plot to a file
plt.savefig('rewards_plot.png')

# Display the plot
plt.show()

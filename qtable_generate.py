import numpy as np
import pandas as pd

# Load the Q-table from the .npy file
qtable = np.load('qtable.npy')

# Convert the Q-table to a DataFrame
df = pd.DataFrame(qtable)

# Print the DataFrame
print(df)

# CARLA-RL

reward_array.npy: default
reward_array1.npy: high reward

reward_array2.npy: (high reward + extreme collision penalty)
it converges into finishing early

reward_array3.npy + qtable3.npy: (low speed, low steering) + (high reward + extreme collision penalty + distance bonus)
learns to rotate and go straight

import numpy as np
import matplotlib.pyplot as plt

data = np.zeros((500,))
data1 = np.load(f"./save/qlearning.npy")
data2 = np.load("./save/sarsa.npy")
plt.plot(data1, label='Q-Learning')
plt.plot(data2, label='Sarsa')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.ylim(-100, 0)
plt.title("Q-Learning vs. Sarsa")
plt.legend()
plt.show()
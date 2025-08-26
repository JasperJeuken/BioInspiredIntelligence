import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', size=18)

# Manually found by adjusting main.py:
elite_fraction = [2.5, 5, 7.5, 10]  # mutation rate = 0.09
elite_generation = [42, 41, 44, 46]

mutation_rate = [0.03, 0.06, 0.09, 0.12]  # elite fractinon = 0.05
mutation_generation = [47, 42, 41, 45]

# Create plots
fig, ax = plt.subplots(2, 1)

ax[0].plot(elite_fraction, elite_generation)
ax[0].set_xlabel('Elite group size [%]')
ax[0].set_ylabel('# of generations')
ax[0].grid()

ax[1].plot(mutation_rate, mutation_generation)
ax[1].set_xlabel('Mutation rate [-]')
ax[1].set_ylabel('# of generations')
ax[1].grid()

fig.tight_layout()
plt.show()
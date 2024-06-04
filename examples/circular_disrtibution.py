import numpy as np
import neurotool as nrt
import matplotlib.pyplot as plt

#Generate samples
phases = np.random.uniform(low=-np.pi, high=np.pi, size=100)
amples = np.ones_like(phases)

bins, distr = nrt.circstat.circular_distribution(amples, phases, 0.2, nkernel=15, density=True)


plt.plot(bins, distr)
plt.show()
import numpy as np
import neurotools as nrt


epoches = [[0, 40], [60, 100]]
spike_train = np.random.uniform(0, 100, 100)

spike_rates = nrt.spikes.get_mean_spike_rate_by_epoches(epoches, spike_train)

print(spike_rates)
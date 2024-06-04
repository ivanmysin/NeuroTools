import numpy as np
import neurotool as nrt
import matplotlib.pyplot as plt

# Generate signal
t, dt = np.linspace(0, 1, 1000, retstep=True)
signal = np.cos(2*np.pi*10*t)

# Add artifact
signal[300:400] = 10 * (np.random.rand(100)-0.5)

# Remove articacts
threshold_std = 4.0 # threshold for artifacts detection, means number of std difference from average level
filtered_signal = nrt.lfp.clear_articacts(signal, threshold_std=threshold_std)


# Plots
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
axes[0].plot(t, signal)
axes[0].set_title(f"Raw signal")
axes[1].plot(t, filtered_signal)
axes[1].set_title(f"Removed artifacts")
plt.show()
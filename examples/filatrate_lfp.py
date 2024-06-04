import numpy as np
import neurotool as nrt
import matplotlib.pyplot as plt

# Generate signal
t, dt = np.linspace(0, 1, 1000, retstep=True)
w1 = 5.0
w2 = 23.0
signal = np.cos(2*np.pi*w1*t) + 0.2*np.cos(2*np.pi*w2*t)

# Filtrate
fs = 1 / dt
filter = nrt.lfp.Butter_bandpass_filter(lowcut=3, highcut=8, fs=fs, order=3)
filtered_signal = filter.filtrate(signal)

# Plots
fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(t, signal)
axes[0].set_title(f"Signal with {w1} Hz and {w2} Hz")
axes[1].plot(t, filtered_signal)
axes[1].set_title(f"Filtered signal")
plt.show()
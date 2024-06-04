import numpy as np
from scipy.signal import hilbert
import neurotool as nrt
import matplotlib.pyplot as plt

# Generate signal
t, dt = np.linspace(0, 10, 20000, retstep=True)
signal = np.exp(300 * np.cos(2*np.pi*0.4*t + 0.75*np.pi)) * np.cos(2*np.pi*150*t)
signal = signal / np.max(signal) + np.random.normal(0, 0.1, t.size)

# Search ripples
fs = 1 / dt
signal = hilbert(signal)
threshold = 4 # threshold for ripples detection, means number of std difference from average level
accept_win = 0.02 # minimal length of ripple in sec
ripples_epoches = nrt.lfp.get_ripples_episodes_indexes(signal, fs, threshold=threshold, accept_win=accept_win)

# Plots
fig, axes = plt.subplots(nrows=1, sharex=True, sharey=True)
axes.plot(t, signal.real)
for idx in range(ripples_epoches.shape[1]):
    axes.vlines(t[ripples_epoches[:, idx]], ymin=-1, ymax=1, color='red')

axes.set_xlabel("Time (sec)")


plt.show()
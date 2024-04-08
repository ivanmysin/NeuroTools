import numpy as np
import neurotools as nrt
import matplotlib.pyplot as plt

# Generate signal
t, dt = np.linspace(0, 1, 1000, retstep=True)
w1 = 5.0
w2 = 32.0
low_rhythms_phases = 2*np.pi*w1*t
peak_phase_of_high_rhythm = 0.5*np.pi
lfp_signal = np.cos(low_rhythms_phases) + 0.2*np.exp(1.5 * np.cos(low_rhythms_phases - peak_phase_of_high_rhythm)) * np.cos(2*np.pi*w2*t)


# Filtrate
fs = 1 / dt
phase_signal = nrt.lfp.Butter_bandpass_filter(lowcut=3, highcut=7, fs=fs, order=3).filtrate(lfp_signal)
amps_signal = nrt.lfp.Butter_bandpass_filter(lowcut=25, highcut=45, fs=fs, order=3).filtrate(lfp_signal)

coupling, bins = nrt.lfp.cossfrequency_phase_amp_coupling(phase_signal,amps_signal)


fig, axes = plt.subplots(nrows=2)
axes[0].plot(t, lfp_signal)
axes[0].set_title("Signal with phase amplitude modulation")
axes[0].set_xlabel("Time (sec)")
axes[1].plot(bins, coupling[0, :])
axes[1].vlines(peak_phase_of_high_rhythm, ymin=np.min(coupling), ymax=np.max(coupling), color="red")
axes[1].set_title("Phase-amplitude coupling")
axes[1].set_xlabel("Phase (rad)")
axes[1].set_ylabel("Coupling value")

plt.show()
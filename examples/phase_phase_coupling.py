import numpy as np
import neurotools as nrt
import matplotlib.pyplot as plt

# Generate signal
t, dt = np.linspace(0, 1, 1000, retstep=True)
w1 = 5.0
w2 = 30.0
w3 = 85.0

theta_band = np.cos(2*np.pi*w1*t)
slow_gamma_band = 0.5 * np.cos(2*np.pi*w2*t)
middle_gamma_band = 0.2 * np.cos(2*np.pi*w3*t)
lfp_signal = theta_band + slow_gamma_band + middle_gamma_band
nmarray = np.arange(1, 21)

coupling_theta_sgamma = nrt.lfp.cossfrequency_phase_phase_coupling(theta_band, slow_gamma_band, nmarray, thresh_std=None, circ_distr=False)
coupling_theta_smgamma = nrt.lfp.cossfrequency_phase_phase_coupling(theta_band, middle_gamma_band, nmarray, thresh_std=None, circ_distr=False)


fig, axes = plt.subplots(nrows=2)
axes[0].plot(t, lfp_signal)
axes[0].set_title("Signal with phase-phase coupling")
axes[0].set_xlabel("Time (sec)")
axes[1].plot(nmarray, coupling_theta_sgamma, color='blue', label=f"{w1} Hz vs {w2} Hz")
axes[1].plot(nmarray, coupling_theta_smgamma, color='green', label=f"{w1} Hz vs {w3} Hz")
axes[1].legend(loc='upper right')
axes[1].set_title("Phase-phase coupling")
axes[1].set_xlabel("coefficients for slow phases")
axes[1].set_ylabel("Coupling value")

plt.show()
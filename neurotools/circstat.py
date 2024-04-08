import numpy as np
import scipy.stats as stat
import scipy.signal as sig
from scipy.ndimage.filters import convolve1d
from scipy.signal.windows import parzen
# from numba import jit

def get_angles_in_range(angles):
    """
    return angles from -pi to pi
    """
    two_pi = 2 * np.pi
    anles_in_range = angles % (two_pi)
    anles_in_range[anles_in_range < -np.pi] += two_pi
    anles_in_range[anles_in_range >= np.pi] -= two_pi

    return anles_in_range

################################################################
#@jit(nopython=True)
def get_circular_mean_R(filtered_lfp, spike_train, mean_calculation = 'uniform'):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param spike_train: времена импульсов
    :mean_calculation: способ вычисления циркулярного среднего и R
    :return: циркулярное среднее и R
    """
    if len(spike_train) == 0:
        return np.nan, np.nan

    if mean_calculation == 'uniform':
        angles = np.angle(np.take(filtered_lfp, spike_train))
        mean = np.mean(np.exp(angles * 1j))
        circular_mean = np.angle(mean)
        R = np.abs(mean)
        return circular_mean, R

    elif mean_calculation == 'normalized':
        phase_signal = np.take(filtered_lfp, spike_train)
        phase_signal = phase_signal / np.sum(np.abs(phase_signal))
        mean = np.sum(phase_signal)
        circular_mean = np.angle(mean)
        R = np.abs(mean)
        return circular_mean, R
    else:
        raise ValueError("This mean_calculation is not acceptable")


#####################################################################
def circular_distribution(amples, angles, angle_step, nkernel=15, density=True):
    """
    return circular distribution smoothed by the parsen kernel
    """
    kernel = parzen(nkernel)
    bins = np.arange(-np.pi, np.pi + angle_step, angle_step)
    distr, _ = np.histogram(angles, bins=bins, weights=amples, density=density)

    distr = convolve1d(distr, kernel, mode="wrap")
    bins = np.convolve(bins, [0.5, 0.5], mode="valid")

    return bins, distr

###############################################################


###################################################################

###################################################################

###################################################################






########################################################################

def __slice_by_bound_values(arr, left_bound, right_bound):
    sl = np.s_[np.argmin(np.abs(arr - left_bound)): np.argmin(np.abs(arr - right_bound))]

    return sl
import numpy as np
import scipy.stats as stat
import scipy.signal as sig
from scipy.ndimage.filters import convolve1d
from scipy.signal.windows import parzen
# from numba import jit

def get_angles_in_range(angles):
    """
    The function gives the angle values in the range from -pi to pi

    Parameters
    ----------
    angles: numpy.ndarray
        Angles in rads

    Returns
    -------
    anles_in_range: numpy.ndarray
         Angles in range from -pi to pi

    """
    two_pi = 2 * np.pi
    anles_in_range = angles % (two_pi)
    anles_in_range[anles_in_range < -np.pi] += two_pi
    anles_in_range[anles_in_range >= np.pi] -= two_pi

    return anles_in_range

################################################################
#@jit(nopython=True)
def get_circular_mean_R(filtered_lfp, spike_train, fs, mean_calculation = 'uniform'):
    """
    The function calculates the circular mean and the ray length (R) for spikes and by phase of filtered LFP signal

    Parameters
    ----------
    filtered_lfp: numpy.ndarray
        filtered lfp signal

    spike_train: numpy.ndarray
        times of spikes

    fs: float
        sampling rate

    mean_calculation: string, optional
        If "uniform" all spikes have equal weight.
        If "normalized" each spike has a weight in accordance with the amplitude of the lfp signal.

    Returns
    -------
    circular_mean: float
         Circular mean

    R: float
         Ray length

    """
    if len(spike_train) == 0:
        return np.nan, np.nan

    spike_train = np.floor(spike_train * fs).astype(np.int64)

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
    The function calculates circular distribution smoothed by the parsen kernel

    Parameters
    ----------
    amples: numpy.ndarray
        amplitudes of vectors

    angles: numpy.ndarray
        angles in radians

    nkernel: int, optional
        Number points in parzen window to smooth distribution
        Default 15

    density: bool, optional

    Returns
    -------
    bins: numpy.ndarray
         bins for circular distribution

    distr: numpy.ndarray
         smoothed circular distribution

    """
    kernel = parzen(nkernel)
    bins = np.arange(-np.pi, np.pi + angle_step, angle_step)
    distr, _ = np.histogram(angles, bins=bins, weights=amples, density=density)

    distr = convolve1d(distr, kernel, mode="wrap")
    bins = np.convolve(bins, [0.5, 0.5], mode="valid")

    return bins, distr

########################################################################







########################################################################

def __slice_by_bound_values(arr, left_bound, right_bound):
    sl = np.s_[np.argmin(np.abs(arr - left_bound)): np.argmin(np.abs(arr - right_bound))]

    return sl
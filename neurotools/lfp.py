"""NeuroTools core functions for lfp signals processing"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import parzen
from scipy.ndimage.filters import convolve1d
from scipy import stats
from scipy.special import rel_entr
from . import circstat
# from numba import jit

class Butter_bandpass_filter:
    """ Implements the Butterword filter class.

    Attributes
    ----------
    lowcut: float
        Low bound for frequency in Hz
    highcut: float
        Upper bound for frequency in Hz
    fs: float
        Sampling rate in Hz
    order: int
        Order of Butterword filter

    Methods
    ----------
    filtrate(numpy.ndarray)
        Apply filer to signal
    """
    def __init__(self, lowcut, highcut, fs, order):
        """
        Parameters
        ----------
        lowcut: float
            Low bound for frequency in Hz
        highcut: float
            Upper bound for frequency in Hz
        fs: float
            Sampling rate in Hz
        order: int
            Order of Butterword filter
        """
        self.fs = fs
        self.nyq = self.fs * 0.5
        self.lowcut = lowcut / self.nyq
        self.highcut = highcut / self.nyq
        self.order = order
        self.b, self.a = butter(N=self.order, Wn=[self.lowcut, self.highcut], btype='bandpass')

    def filtrate(self, lfp):
        """ Apply filter to the signal

        Parameters
        ----------
        lfp: numpy.ndarray
            Raw LFP signal for filtration

        Returns
        -------
        filtered: numpy.ndarray
            Filtered LFP signal
        """

        filtered = filtfilt(self.b, self.a, lfp)
        return filtered
########################################################################################
# @jit(nopython=True)
def __clear_artifacts(lfp, win, threshold_std, threshold):
    mean_lfp = np.mean(lfp)
    lfp = lfp - mean_lfp
    lfp_std = np.std(lfp)
    is_large = np.logical_or((lfp > threshold_std * lfp_std), (lfp < -threshold_std * lfp_std))
    is_large = is_large.astype(np.float64)
    is_large = np.convolve(is_large, win)
    is_large = is_large[win.size // 2:-win.size // 2 + 1]
    is_large = is_large > threshold
    lfp[is_large] = np.random.normal(0, 0.001 * lfp_std, np.sum(is_large)) + mean_lfp
    return lfp
def clear_articacts(lfp, win_size=101, threshold_std=10,  tht=0.1):
    """ Remove artifacts from lfp signals.

    Parameters
    ----------
    lfp: numpy.ndarray, list
        Input signal array.
    win_size: int, optional
        A length of window in which threshold crossings are defined as a single artifact
        Default value is 101
    threshold_std: float, optional
        Threshold for artifacts detection, means number of std difference from average level
        Default value is 10
    tht: float in range (0, 1), optional
        Threshold for detect start and end of artifacts.
        High value will lead to separate management of artifacts.
        At a low value, close artifacts will be combined into one.
        Default value is 0.1

    Returns
    -------
    lfp: numpy.ndarray
        A signal in which segments with artifacts are replaced by low-amplitude noise
    """
    win = parzen(win_size)
    lfp = __clear_artifacts(lfp, win, threshold_std, tht)
    return lfp
########################################################################################

#@jit(nopython=True)
def get_ripples_episodes_indexes(ripples_lfp, fs, threshold=4, accept_win=0.02):
    """ Find ripples events in LFP signal

    Parameters
    ----------
    ripples_lfp: numpy.ndarray
        the lfp signal filtered in the ripple range after the Hilbert transform
    fs: float
        Sampling rate
    threshold: float, optional
        Threshold for ripples detection, means number of std difference from average level. Default value is 4.
    accept_win: float, optional
        Minimal ripple length in seconds. Default value is 0.02 sec.

    Returns
    -------
    ripples_epoches: numpy.ndarray with shape (2, n)
        The array of indexes of the starts and ends of ripple events.
        n is number of ripples events, first values along 0 axis are starts, second values are ends.
    """

    ripples_lfp_th = threshold * np.std(ripples_lfp.real)
    ripples_abs = np.abs(ripples_lfp)
    is_up_threshold = ripples_abs > ripples_lfp_th
    is_up_threshold = is_up_threshold.astype(np.int32)
    diff = np.diff(is_up_threshold)
    diff = np.append(is_up_threshold[0], diff)

    start_idx = np.ravel(np.argwhere(diff == 1))
    end_idx = np.ravel(np.argwhere(diff == -1))

    if start_idx[0] > end_idx[0]:
        end_idx = end_idx[1:]

    if start_idx[-1] > end_idx[-1]:
        start_idx = start_idx[:-1]

    accept_intervals = (end_idx - start_idx) > accept_win * fs
    start_idx = start_idx[accept_intervals]
    end_idx = end_idx[accept_intervals]

    ripples_epoches = np.append(start_idx, end_idx).reshape((2, start_idx.size))
    return ripples_epoches

########################################################################################
#@jit(nopython=True)
def get_theta_non_theta_epoches(theta_lfp, delta_lfp, fs, theta_threshold=2, accept_win=2):
    """ Find theta and non-theta epoches in LFP signal

    Parameters
    ----------
    theta_lfp: numpy.ndarray
        the lfp signal filtered in the theta range after the Hilbert transform
    delta_lfp: numpy.ndarray
        the lfp signal filtered in the delta range after the Hilbert transform
    fs: float
        Sampling rate
    theta_threshold: float, optional
        Threshold for theta epoches detection. In theta epochs, the ratio of theta power to delta power is greater than the threshold.
        Default value is 2.
    accept_win: float, optional
        Minimal theta epoches length in seconds. Default value is 2 sec.

    Returns
    -------
    theta_epoches: numpy.ndarray with shape (2, n)
        The array of indexes of the starts and ends of theta epochs.
        n is epochs, first values along 0 axis are starts, second values are ends.
    non_theta_epoches: numpy.ndarray with shape (2, n)
        The array of indexes of the starts and ends of non-theta epochs.
        n is number of epochs, first values along 0 axis are starts, second values are ends.
    """
    theta_amplitude = np.abs(theta_lfp)
    delta_amplitude = np.abs(delta_lfp)

    relation = theta_amplitude / (delta_amplitude + 0.0000001)
    #     relation = relation[relation < 20]
    is_up_threshold = relation > theta_threshold
    #     is_up_threshold = stats.zscore(relation) > theta_threshold
    # в случе z-scoe необходимо изменение порога
    is_up_threshold = is_up_threshold.astype(np.int32)
    relation = theta_amplitude / delta_amplitude
    is_up_threshold = relation > theta_threshold
    is_up_threshold = is_up_threshold.astype(np.int32)

    diff = np.diff(is_up_threshold)

    start_idx = np.ravel(np.argwhere(diff == 1))
    end_idx = np.ravel(np.argwhere(diff == -1))

    if start_idx[0] > end_idx[0]:
        start_idx = np.append(0, start_idx)

    if start_idx[-1] > end_idx[-1]:
        end_idx = np.append(end_idx, relation.size - 1)

    # игнорируем небольшие пробелы между тета-эпохами
    large_intervals = (start_idx[1:] - end_idx[:-1]) > accept_win * fs
    large_intervals = np.append(True, large_intervals)
    start_idx = start_idx[large_intervals]
    end_idx = end_idx[large_intervals]

    # игнорируем небольшие тета-эпохи
    large_th_epochs = (end_idx - start_idx) > accept_win * fs
    start_idx = start_idx[large_th_epochs]
    end_idx = end_idx[large_th_epochs]

    # Все готово, упаковываем в один массив
    theta_epoches = np.append(start_idx, end_idx).reshape((2, start_idx.size))

    # Инвертируем тета-эпохи, чтобы получить дельта-эпохи
    non_theta_start_idx = end_idx[:-1]
    non_theta_end_idx = start_idx[1:]

    # Еще раз обрабатываем начало и конец сигнала
    if start_idx[0] > 0:
        non_theta_start_idx = np.append(0, non_theta_start_idx)
        non_theta_end_idx = np.append(start_idx[0], non_theta_end_idx)

    if end_idx[-1] < relation.size - 1:
        non_theta_start_idx = np.append(non_theta_start_idx, end_idx[-1])
        non_theta_end_idx = np.append(non_theta_end_idx, relation.size - 1)

    # Все готово, упаковываем в один массив
    non_theta_epoches = np.append(non_theta_start_idx, non_theta_end_idx).reshape((2, non_theta_start_idx.size))

    return theta_epoches, non_theta_epoches
########################################################################################
def cossfrequency_phase_amp_coupling(phase_signal, amps, phasebins=20, nkernel=15):
    """ Compute disribution amplitudes by phases of phase_signal
    Belluscio, M. A., Mizuseki, K., Schmidt, R., Kempter, R. & Buzsáki, G. Cross-Frequency Phase–Phase Coupling between Theta and Gamma Oscillations in the Hippocampus. J. Neurosci. 32, 423–435 (2012).

    Parameters
    ----------
    phase_signal: numpy.ndarray
        The lfp signal filtered in the range of low frequency rhythm. Phase of it are used.
        The array can contain complex values of the analytical signal.
    amps: numpy.ndarray
        The lfp signal filtered in the range of high frequency rhythms.
    phasebins: int, optional
        Number of bins for histogram. Default value is 20.
    nkernel: int, optional
        Number points in parzen window. It used for smoothing.
        Default value is 15.

    Returns
    -------
    coupling: numpy.ndarray
        Density of power rhythms (amps) by phases of phase_signal along each axis of amps
    bins: numpy.ndarray
        Array of bins.
    """

    if not np.iscomplex(phase_signal).all():
        phase_signal = hilbert(phase_signal)

    if not np.iscomplex(amps).all():
        amps = hilbert(amps)

    if len(amps.shape) == 1:
        amps = np.reshape(amps, (1, -1))

    phase_signal = np.angle(phase_signal, deg=False)
    coefAmp = np.abs(amps)

    coefAmp = stats.zscore(coefAmp, axis=1)
    coupling = np.empty(shape=(coefAmp.shape[0], phasebins), dtype=np.float64)

    kernel = parzen(nkernel)
    for freq_idx in range(coefAmp.shape[0]):
        coup, bins = np.histogram(phase_signal, bins=phasebins, weights=coefAmp[freq_idx, :], range=[-np.pi, np.pi])
        coup = convolve1d(coup, kernel, mode="wrap")
        coupling[freq_idx, :] = coup

    coupling = coupling / (coupling.max() - coupling.min())
    bins = np.convolve(bins, [0.5, 0.5], mode='valid')

    return coupling, bins

########################################################################################
def cossfrequency_phase_phase_coupling(low_fr_signal, high_fr_signal, nmarray, thresh_std=None, circ_distr=False):
    """ Compute phase-phase coupling (n:m test)
    Belluscio, M. A., Mizuseki, K., Schmidt, R., Kempter, R. & Buzsáki, G. Cross-Frequency Phase–Phase Coupling between Theta and Gamma Oscillations in the Hippocampus. J. Neurosci. 32, 423–435 (2012).

    Parameters
    ----------
    low_fr_signal: numpy.ndarray
        The lfp signal filtered in the range of low frequency rhythm.
        The array can contain complex values of the analytical signal.
    high_fr_signal: numpy.ndarray
        The lfp signal filtered in the range of high frequency rhythms.
        The array can contain complex values of the analytical signal.
    nmarray: numpy.ndarray
        Coefficients for speed up phases of low frequency signal
    thresh_std: float or None, optional
        Means number of std difference from average level.
        When calculating the coefficient, only points whose amplitude is above the threshold are taken into account
        If None threshold is not applied. Default value is None.
    circ_distr: bool, optional
        If True the function returns circular distributions for each coefficient in nmarray
        If True the function returns empty list distrs and bins

    Returns
    -------
    coupling: numpy.ndarray
        Coherence of phases for each coefficient in nmarray
    bins: numpy.ndarray, optional
        Only returned if circ_distr is True
        Array of bins.
    distrs: list of numpy.ndarray, optional
        Only returned if circ_distr is True
        Circular distributions of phase differencies for each coefficient in nmarray
    """

    if not np.iscomplex(low_fr_signal).all():
        low_fr_analitic_signal = hilbert(low_fr_signal)
    else:
        low_fr_analitic_signal = low_fr_signal

    if not np.iscomplex(high_fr_signal).all():
        high_fr_analitic_signal = hilbert(high_fr_signal)
    else:
        high_fr_analitic_signal = high_fr_signal

    coupling = np.zeros(nmarray.size)

    if not thresh_std is None:

        abs_signal = np.abs(low_fr_analitic_signal * high_fr_analitic_signal)

        signif_poits = abs_signal > (np.mean(abs_signal) + thresh_std * np.std(abs_signal))

        low_fr_analitic_signal = low_fr_analitic_signal[signif_poits]
        high_fr_analitic_signal = high_fr_analitic_signal[signif_poits]

        if low_fr_analitic_signal.size == 0:
            return coupling

    low_fr_angles = np.angle(low_fr_analitic_signal, deg=False)
    high_fr_angles = np.angle(high_fr_analitic_signal, deg=False)

    distrs = []
    bins = []
    for i in range(nmarray.size):

        vects_angle = low_fr_angles * nmarray[i] - high_fr_angles
        x_vect = np.cos(vects_angle)
        y_vect = np.sin(vects_angle)
        mean_resultant_length = np.sqrt(np.sum(x_vect)**2 + np.sum(y_vect)**2) / vects_angle.size
        coupling[i] = mean_resultant_length

        if circ_distr:
            vects_angle = circstat.get_angles_in_range(vects_angle)
            bins_anles, distr = circstat.circular_distribution(np.ones_like(vects_angle), vects_angle, angle_step=0.1,
                                                      nkernel=15)
            distrs.append(distr)
            bins.append(bins_anles)

    if circ_distr:
        return coupling, bins, distrs
    else:
        return coupling
########################################################################################
def get_modulation_index(disribution, norm=False):
    """ Сompule modulation index for phase-amplitude coupling.
    Kullback–Leibler divergence between uniform and given disribution.

    Parameters
    ----------
    disribution: numpy.ndarray
        Density of circular distribution
    norm: bool, optional
        If True density is normed to sum = 1

    Returns
    ----------
    modulation_index: float
        Modulation index
    """

    if norm:
        disribution = disribution - np.min(disribution)
        disribution = disribution / np.sum(disribution)

    unif = 1.0 / (2 * np.pi)
    modulation_index = np.sum(rel_entr(disribution, unif))
    return modulation_index
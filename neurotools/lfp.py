import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import parzen
# from numba import jit

class Butter_bandpass_filter():
    def __init__(self, lowcut, highcut, fs, order):
        """
        :param lowcut: float low bound for frequency
        :param highcut: float upper bound for frequency
        :param fs: sampling rate
        :param order: int order of Butterword filter
        """
        self.fs = fs
        self.nyq = self.fs * 0.5
        self.lowcut = lowcut / self.nyq
        self.highcut = highcut / self.nyq
        self.order = order
        self.b, self.a = butter(N=self.order, Wn=[self.lowcut, self.highcut], btype='bandpass')

    def filtrate(self, lfp):
        """
        :param lfp: numpy array with raw LFP signal
        :return: numpy array with filtered LFP signal
        """
        filtered = filtfilt(self.b, self.a, lfp)
        return filtered

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
    """
    :param lfp:
    :param win_size:
    :param threshold_std:
    :param tht:
    :return:
    """
    win = parzen(win_size)
    lfp = __clear_artifacts(lfp, win, threshold_std, tht)
    return lfp




#@jit(nopython=True)
def get_ripples_episodes_indexes(ripples_lfp, fs, threshold=4, accept_win=0.02):
    """
    :param ripples_lfp: сигнал lfp, отфильтрованный в риппл-диапазоне после преобразования Гильберта
    :param fs: частота дискретизации
    :param threshold: порог для определения риппла
    :param accept_win: минимальная длина риппла в сек
    :return:  списки начал и концов риппл событий в единицах, указанных в частоте дискретизации (fs)
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


#@jit(nopython=True)
def get_theta_non_theta_epoches(theta_lfp, delta_lfp, fs, theta_threshold=2, accept_win=2):
    """
    :param theta_lfp: отфильтрованный в тета-диапазоне LFP после преобразования Гильберта
    :param delta_lfp: отфильтрованный в дельа-диапазоне LFP после преобразования Гильберта
    :param theta_threshold : порог для отделения тета- от дельта-эпох
    :param accept_win : порог во времени, в котором переход не считается.
    :return: массив индексов начала и конца тета-эпох, другой массив для нетета-эпох
    """
    theta_amplitude = np.abs(theta_lfp)
    delta_amplitude = np.abs(delta_lfp)

    relation = theta_amplitude / delta_amplitude
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


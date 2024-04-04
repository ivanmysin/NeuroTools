import numpy as np
# from numba import jit

#@jit(nopython=True)
def get_circular_mean_R(filtered_lfp, spike_train, mean_calculation = 'uniform'):
    """
    :param filtered_lfp: отфильтрованный в нужном диапазоне LFP
    :param spike_train: времена импульсов
    :mean_calculation: способ вычисления циркулярного среднего и R
    :return: циркулярное среднее и R
    """
    #fs - не нужно, т.к. спайки указаны в частоте записи лфп
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
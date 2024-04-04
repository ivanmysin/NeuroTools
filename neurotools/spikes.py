import numpy as np
# from numba import jit


#@jit(nopython=True)
def __get_for_one_epoch(limits, spikes):
    x = spikes[(spikes >= limits[0]) & (spikes < limits[1])]
    return x


#@jit(nopython=True)
def get_mean_spike_rate_by_epoches(epoches_indexes, spike_train, fs):
    """
    :param epoches_indexes: массив начал и концов тета эпох в формате
                          [[start, stop], [start, stop]]
    :param spike_train: индексы импульсов
    :param fs: частота дискретизации
    :return: среднее для  эпох, ст.откл.
    """
    if epoches_indexes.size == 0:
        return 0, 0
    spikes = np.zeros(epoches_indexes.shape[1], dtype=np.float64)

    for idx in range(epoches_indexes.shape[1]):
        start_idx = epoches_indexes[0, idx]
        end_idx = epoches_indexes[1, idx]
        spikes_in_epoches = __get_for_one_epoch((start_idx, end_idx), spike_train)

        # print(end_idx - start_idx)
        spikes_rate = spikes_in_epoches.size / (end_idx - start_idx) * fs
        spikes[idx] = spikes_rate


    spike_rate = np.mean(spikes)
    spike_rate_std = np.std(spikes)

    return spike_rate, spike_rate_std
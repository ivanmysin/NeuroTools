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
##########################################################################################
def get_phase_disrtibution(train, lfp, fs):
    """
    compute disrtibution of spikes by phases of LFP
    """
    if train.size == 0:
        return np.empty(0, dtype=np.float), np.empty(0, dtype=np.float), np.empty(0, dtype=np.float)

    nkernel = 15

    analitic_signal = sig.hilbert(lfp)
    lfp_phases = np.angle(analitic_signal, deg=False)
    lfp_ampls = np.abs(analitic_signal)

    train = np.floor(train * fs * 0.001).astype(np.int) # multiply on 0.001 because train in ms, fs in Hz

    train = train[train < lfp.size-1]

    train_phases = lfp_phases[train]
    train_ampls = lfp_ampls[train]

    R = np.abs( np.mean(analitic_signal[train]) )

    count, bins = np.histogram(train_phases, bins=50, density=True, range=[-np.pi, np.pi], weights=train_ampls )

    kernel = parzen(nkernel)

    # distr, _ = np.histogram(angles, bins=bins, weights=amples, density=density)

    count = convolve1d(count, kernel, mode="wrap")

    bins = np.convolve(bins, [0.5, 0.5], mode="valid")
    return bins, count, R
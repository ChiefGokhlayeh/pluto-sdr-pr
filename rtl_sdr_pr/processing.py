import numpy as np
import numpy.typing as npt


def direct_ambiguity(
    delay_bins: npt.ArrayLike,
    doppler_bins: npt.ArrayLike,
    echo_signal: npt.ArrayLike,
    reference_signal: npt.ArrayLike,
    sampling_freq: float,
):
    delay_bins = np.asarray(delay_bins)
    doppler_bins = np.asarray(doppler_bins)
    num_samples = echo_signal.shape[0]

    echo_signal = np.pad(echo_signal, (0, num_samples))

    n = np.arange(num_samples)
    n_lag = np.int32(np.outer(delay_bins, np.ones((num_samples))) + n)
    lag_prod = reference_signal[n] * echo_signal[n_lag].conjugate()
    dop_exp = np.exp(-2j * np.pi * np.outer(n, doppler_bins) / sampling_freq)
    return np.matmul(lag_prod, dop_exp)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

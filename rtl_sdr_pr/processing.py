import numpy as np
import numpy.typing as npt


def direct_ambiguity(
    delay_bins: npt.ArrayLike,
    doppler_bins: npt.ArrayLike,
    echo_signal: np.ndarray,
    reference_signal: np.ndarray,
    sampling_freq: float,
) -> np.ndarray:
    """Direct (slow) implementation of cross ambiguity function (CAF).

    Parameters
    ----------
    delay_bins : npt.ArrayLike
        Delay bin(s) by which to shift the echo signal in time-domain and
        calculate the CAF. Can be scalar or array-like.
    doppler_bins : npt.ArrayLike
        Frequency bin(s) by which to shift the echo signal in frequency-domain
        and calculate the CAF. Can be scalar or array-like.
    echo_signal : np.ndarray
        Samples representing the echo signal received via the indirect paths.
    reference_signal : np.ndarray
        Samples representing the reference signal received via the direct path.
    sampling_freq : float
        Sampling frequency of both echo and reference samples.

    Returns
    -------
    np.ndarray
        Cross ambiguity of echo and reference signal.

    Examples
    --------
    Calculate the CAF of a real-valued rectangular pulse shifted by 50 samples.
    This will produce a triangular CAF with its peak at the 50th element.
    >>> reference = np.pad(np.ones(10), (0, 1000))
    >>> echo = np.roll(reference, 50)
    >>> amb = direct_ambiguity(
    ...     delay_bins=range(100),
    ...     doppler_bins=0,
    ...     echo_signal=echo,
    ...     reference_signal=reference,
    ...     sampling_freq=1e3)
    >>> np.argmax(amb)
    50
    """
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

import numpy as np
import numpy.typing as npt

import scipy.signal
import scipy.fft


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


def fast_ambiguity(
    num_delay_bins: int,
    num_doppler_bins: int,
    reference_signal: np.ndarray,
    surveillance_signal: np.ndarray,
) -> np.ndarray:

    """Fast implementation of cross ambiguity function (CAF), using Fourier
    Transform of Lag Product approach.

    Parameters
    ----------
    num_delay_bins : int
        Number of delay bins by which to shift the surveillance signal in
        time-domain and calculate the CAF. Can be scalar or array-like. Must be
        `>= 1`.
    num_doppler_bins : int
        Number of frequency bins by which to shift the surveillance signal in
        frequency-domain and calculate the CAF. Can be scalar or array-like.
        Must be `>= 1`.
    reference_signal : np.ndarray
        Samples from the surveillance channel, which contains the direct path
        signal.
    surveillance_signal : np.ndarray
        Samples from the surveillance channel, which contains target echos.

    Returns
    -------
    np.ndarray
        Cross ambiguity of surveillance and reference signal as 2D array with
        size `num_delay_bins` x `num_doppler_bins`.

    Examples
    --------
    Calculate the CAF of a real-valued rectangular pulse shifted by 50 samples.
    This will produce a triangular CAF with its peak at the 50th element.
    >>> reference = np.pad(np.ones(10), (0, 1000))
    >>> surveillance = np.roll(reference, 50)
    >>> amb = fast_ambiguity(
    ...     num_delay_bins=100,
    ...     num_doppler_bins=1,
    ...     reference_signal=reference,
    ...     surveillance_signal=surveillance)
    >>> np.argmax(amb)
    50
    """

    num_taps = reference_signal.shape[0] // num_doppler_bins

    fir = scipy.signal.dlti(np.ones(num_taps), 1)

    amb = np.empty((num_delay_bins, num_doppler_bins), dtype=np.complex64)

    surv_cmplx_conj = surveillance_signal.conjugate()
    surv_cmplx_conj = np.pad(surv_cmplx_conj, pad_width=(0, num_delay_bins))

    for delay_bin, lag in enumerate(-np.arange(num_delay_bins)):
        lag_product = (
            reference_signal
            * np.roll(surv_cmplx_conj, lag)[: reference_signal.shape[0]]
        )
        amb[delay_bin, :] = scipy.signal.decimate(
            lag_product, num_taps, ftype=fir
        )[:num_doppler_bins]

    amb = scipy.fft.fftshift(scipy.fft.fft(amb, axis=1), axes=1)

    return amb


if __name__ == "__main__":
    import doctest

    doctest.testmod()

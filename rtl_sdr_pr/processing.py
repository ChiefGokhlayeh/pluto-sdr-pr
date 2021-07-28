import numpy as np
from numpy.typing import ArrayLike

import typing

try:
    import cupy as cp
    import cusignal

    NO_CUPY = False
except ImportError:
    NO_CUPY = True

import scipy.signal
import scipy.fft


def direct_ambiguity(
    delay_bins: ArrayLike,
    doppler_bins: ArrayLike,
    surveillance_signal: np.ndarray,
    reference_signal: np.ndarray,
    sampling_freq: float,
) -> np.ndarray:
    """Direct (slow) implementation of cross ambiguity function (CAF).

    Parameters
    ----------
    delay_bins : ArrayLike
        Delay bin(s) by which to shift the surveillance signal in time-domain
        and calculate the CAF. Can be scalar or array-like.
    doppler_bins : ArrayLike
        Frequency bin(s) by which to shift the surveillance signal in
        frequency-domain and calculate the CAF. Can be scalar or array-like.
    reference_signal : np.ndarray
        Samples from the surveillance channel, which contains the direct path
        signal.
    surveillance_signal : np.ndarray
        Samples from the surveillance channel, which contains target echos.
    sampling_freq : float
        Sampling frequency of both surveillance and reference samples.

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
    >>> amb = direct_ambiguity(
    ...     delay_bins=range(100),
    ...     doppler_bins=0,
    ...     surveillance_signal=surveillance,
    ...     reference_signal=reference,
    ...     sampling_freq=1e3)
    >>> np.argmax(amb)
    50
    """
    delay_bins = np.asarray(delay_bins)
    doppler_bins = np.asarray(doppler_bins)
    num_samples = surveillance_signal.shape[0]

    surveillance_signal = np.pad(surveillance_signal, (0, num_samples))

    n = np.arange(num_samples)
    n_lag = np.int32(np.outer(delay_bins, np.ones((num_samples))) + n)
    lag_prod = reference_signal[n] * surveillance_signal[n_lag].conjugate()
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
        time-domain and calculate the CAF. Must be
        `>= 1`.
    num_doppler_bins : int
        Number of frequency bins by which to shift the surveillance signal in
        frequency-domain and calculate the CAF.
        Must be `>= 1`.
    reference_signal : np.ndarray
        Samples from the reference channel, which contains the direct path
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
    num_samples_per_cpi = reference_signal.shape[0]

    num_taps = num_samples_per_cpi // num_doppler_bins

    fir = scipy.signal.dlti(np.ones(num_taps), 1)

    amb = np.empty((num_delay_bins, num_doppler_bins), dtype=np.complex64)

    surv_cmplx_conj = surveillance_signal.conjugate()
    surv_cmplx_conj = np.pad(surv_cmplx_conj, pad_width=(0, num_delay_bins))

    lag_product = np.empty((num_samples_per_cpi,), dtype=amb.dtype)
    decimation = np.empty((num_doppler_bins,), dtype=amb.dtype)
    for delay_bin, lag in enumerate(-np.arange(num_delay_bins)):
        np.multiply(
            reference_signal,
            np.roll(surv_cmplx_conj, lag)[:num_samples_per_cpi],
            out=lag_product,
        )
        decimation[:] = scipy.signal.decimate(lag_product, num_taps, ftype=fir)[
            :num_doppler_bins
        ]
        amb[delay_bin, :] = decimation

    amb = scipy.fft.fftshift(scipy.fft.fft(amb, axis=1), axes=1)

    return amb


if not NO_CUPY:

    def gpu_ambiguity(
        num_delay_bins: int,
        num_doppler_bins: int,
        reference_signal: np.ndarray,
        surveillance_signal: np.ndarray,
        num_samples_per_cpi: int,
        batch_size: typing.Optional[int] = 1,
    ):
        """Fast implementation of cross ambiguity function (CAF) on GPU, using
        Fourier Transform of Lag Product approach.

        Requires CuPy to be installed.

        Parameters
        ----------
        num_delay_bins : int
            Number of delay bins by which to shift the surveillance signal in
            time-domain and calculate the CAF. Must be `>= 1`.
        num_doppler_bins : int
            Number of frequency bins by which to shift the surveillance signal
            in frequency-domain and calculate the CAF. Must be `>= 1`.
        reference_signal : np.ndarray
            Samples from the reference channel, which contains the direct path
            signal.
        surveillance_signal : np.ndarray
            Samples from the surveillance channel, which contains target echos.
        num_samples_per_cpi : int
            Number of samples per Coherent Processing Intervall (CPI), i.e. how
            many samples shall be correlated.
        batch_size : Optional[int]
            Number of consecutive CPIs to be processed. Inputs
            `reference_signal` and `surveillance_signal` must contain at least
            `num_samples_per_cpi * batch_size` samples.

        Returns
        -------
        np.ndarray
            Cross ambiguity of surveillance and reference signal as 2D array
            with size `num_delay_bins` x `num_doppler_bins` x `batch_size`.

        Examples
        --------
        Calculate the CAF of a real-valued rectangular pulse shifted by 50
        samples.
        This will produce a triangular CAF with its peak at the 50th element.
        >>> reference = np.pad(np.ones(10), (0, 1000))
        >>> surveillance = np.roll(reference, 50)
        >>> amb = gpu_ambiguity(
        ...     num_delay_bins=100,
        ...     num_doppler_bins=1,
        ...     reference_signal=reference,
        ...     surveillance_signal=surveillance,
        ...     num_samples_per_cpi=reference.shape[0])
        >>> np.argmax(amb)
        50
        """

        reference_signal = cp.asarray(
            reference_signal[: num_samples_per_cpi * batch_size].reshape(
                num_samples_per_cpi, batch_size
            )
        )
        surveillance_signal = cp.asarray(
            surveillance_signal[: num_samples_per_cpi * batch_size].reshape(
                num_samples_per_cpi, batch_size
            )
        )

        num_taps = num_samples_per_cpi // num_doppler_bins

        amb = cp.empty(
            (num_delay_bins, num_doppler_bins, batch_size),
            dtype=cp.complex64,
        )

        surv_cmplx_conj = surveillance_signal.conjugate()
        surv_cmplx_conj = cp.pad(
            surv_cmplx_conj, pad_width=((0, num_delay_bins), (0, 0))
        )

        lag_product = cp.empty(
            (num_samples_per_cpi, batch_size), dtype=amb.dtype
        )
        decimation = cp.empty_like(lag_product)
        for delay_bin, lag in enumerate(-np.arange(num_delay_bins)):
            cp.multiply(
                reference_signal,
                cp.roll(surv_cmplx_conj, lag, axis=0)[:num_samples_per_cpi],
                out=lag_product,
            )

            decimation[:] = cusignal.decimate(lag_product, num_taps, axis=1)
            amb[delay_bin, :, :] = decimation[:num_doppler_bins, :]

        amb = cp.fft.fftshift(cp.fft.fft(amb, axis=1), axes=1)

        return amb.get()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

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

    Comprehensive example in which an amplitude modulated square wave is shifted
    in time and frequency.
    >>> import matplotlib.pyplot as plt

    >>> duration = 1.0
    >>> duty_cycle = 0.5
    >>> delay = 0.3
    >>> doppler = -0.2
    >>> sample_rate = 4000
    >>> carrier_freq = 100
    >>> num_samples = int(duration * sample_rate)
    >>> t = np.arange(num_samples) / sample_rate

    >>> waveform_prototype = np.concatenate(
    ...     [
    ...         np.ones(int(num_samples * duty_cycle)),
    ...         np.zeros(int(num_samples * (1 - duty_cycle))),
    ...     ]
    ... )

    >>> waveform_prototype *= 1 + 0.5 * np.sin(2 * np.pi * carrier_freq * t)

    >>> waveform = scipy.signal.hilbert(waveform_prototype)

    >>> ref_waveform = waveform
    >>> surv_waveform = np.roll(
    ...     ref_waveform, int(delay * waveform.shape[0])
    ... ) * np.exp(-2j * doppler * np.pi * sample_rate * t)

    >>> _, axs = plt.subplots(2, 1, figsize=(10, 7))
    >>> _ = axs[0].plot(np.real(ref_waveform))
    >>> _ = axs[0].set_title("Reference Channel")
    >>> _ = axs[0].set_ylabel("Real");
    >>> axs[0].grid(True)
    >>> _ = axs[1].plot(np.real(surv_waveform));
    >>> _ = axs[1].set_title("Surveillance Channel");
    >>> _ = axs[1].set_ylabel("Real");
    >>> axs[1].grid(True)

    >>> amb = fast_ambiguity(
    ...     num_samples,
    ...     sample_rate,
    ...     ref_waveform,
    ...     surv_waveform
    ... )
    >>> peak = np.unravel_index(np.argmax(amb), amb.shape)

    >>> _, ax = plt.subplots(figsize=(10, 5))
    >>> _ = ax.set_title("Cross Ambiguity (log-scale)")
    >>> _ = ax.annotate(
    ...     f"Peak ({peak[0]},{peak[1] - sample_rate // 2:.0f})",
    ...     peak,
    ...     xytext=(10, 10),
    ...     xycoords="data",
    ...     textcoords="offset pixels",
    ...     arrowprops={"arrowstyle": "wedge"},
    ... )
    >>> _ = ax.set_xlabel("Delay [Samples]")
    >>> _ = ax.set_ylabel("Doppler [Hz]")
    >>> _ = ax.set_yticks(np.linspace(0, sample_rate, 8, endpoint=False))
    >>> _ = ax.set_yticklabels(
    ...     map(lambda y: f"{y - sample_rate // 2:.0f}", ax.get_yticks())
    ... )
    >>> _ = ax.imshow(10 * np.log10(np.abs(amb.T)))

    >>> assert np.allclose(
    ...     peak,
    ...     np.array(
    ...         [
    ...             num_samples * delay,
    ...             sample_rate * doppler + sample_rate // 2,
    ...         ]
    ...     ),
    ...     atol=200,
    ... )
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

"""Signal Generator Module

Used to generate synchronization sequences and modulate data points into OFDM
symbols.

Implementation based on 3GPP TS 36.211 version 16.4.0 Release 16 as described in
ETSI TS 136 211 V16.4.0 (2021-02).
"""
import collections

import numpy as np

from numpy.typing import ArrayLike
from typing import Optional, Tuple

ROOTS = np.array([25, 29, 34])

PSS_SEQUENCE_LENGTH = 62
SSS_SEQUENCE_LENGTH = 62
CELL_ID_GROUPS = 168


def generate_pss_sequence(cell_id: ArrayLike) -> np.ndarray:
    """Generate frequency-domain PSS sequence consisting of 62-length Zadoff-Chu
    sequence.

    Parameters
    ----------
    cell_id : ArrayLike
        Physical layer cell identity deciding which Zadoff-Chu roots to use
        during generation. If given an array of `N` cell IDs, output will be
        `Nx62` matrix, each row containing the Zadoff-Chu sequence for the given
        cell ID.

    Returns
    -------
    np.ndarray
        A `Nx62` matrix where each row contains a Zadoff-Chu sequence. `N`
        representing the number of `cell_id`s passed to the function.

    Examples
    --------
    Generate PSS sequence for cell ID 0:
    >>> generate_pss_sequence(0)
    array([[ 1.        -0.00000000e+00j, -0.79713251-6.03804410e-01j,
             0.36534102-9.30873749e-01j, -0.73305187-6.80172738e-01j,
             0.98017249+1.98146143e-01j,  0.95557281+2.94755174e-01j,
            -0.5       -8.66025404e-01j,  0.76604444-6.42787610e-01j,
            -0.22252093-9.74927912e-01j,  0.6234898 +7.81831482e-01j,
             0.45621066+8.89871809e-01j,  0.36534102-9.30873749e-01j,
             0.95557281+2.94755174e-01j,  0.76604444-6.42787610e-01j,
            -0.5       +8.66025404e-01j, -0.73305187+6.80172738e-01j,
             0.98017249+1.98146143e-01j, -0.22252093+9.74927912e-01j,
             0.6234898 +7.81831482e-01j, -0.79713251-6.03804410e-01j,
            -0.5       -8.66025404e-01j, -0.5       +8.66025404e-01j,
            -0.79713251-6.03804410e-01j, -0.98883083+1.49042266e-01j,
             0.95557281-2.94755174e-01j,  0.98017249+1.98146143e-01j,
            -0.22252093-9.74927912e-01j,  1.        -6.27365790e-14j,
             0.76604444-6.42787610e-01j, -0.73305187+6.80172738e-01j,
            -0.98883083+1.49042266e-01j, -0.98883083+1.49042266e-01j,
            -0.73305187+6.80172738e-01j,  0.76604444-6.42787610e-01j,
             1.        -6.66653525e-14j, -0.22252093-9.74927912e-01j,
             0.98017249+1.98146143e-01j,  0.95557281-2.94755174e-01j,
            -0.98883083+1.49042266e-01j, -0.79713251-6.03804410e-01j,
            -0.5       +8.66025404e-01j, -0.5       -8.66025404e-01j,
            -0.79713251-6.03804410e-01j,  0.6234898 +7.81831482e-01j,
            -0.22252093+9.74927912e-01j,  0.98017249+1.98146143e-01j,
            -0.73305187+6.80172738e-01j, -0.5       +8.66025404e-01j,
             0.76604444-6.42787610e-01j,  0.95557281+2.94755174e-01j,
             0.36534102-9.30873749e-01j,  0.45621066+8.89871809e-01j,
             0.6234898 +7.81831482e-01j, -0.22252093-9.74927912e-01j,
             0.76604444-6.42787610e-01j, -0.5       -8.66025404e-01j,
             0.95557281+2.94755174e-01j,  0.98017249+1.98146143e-01j,
            -0.73305187-6.80172738e-01j,  0.36534102-9.30873749e-01j,
            -0.79713251-6.03804410e-01j,  1.        +1.02115525e-12j]])

    Generate a PSS sequence for multiple cell IDs:
    >>> pss = generate_pss_sequence([0, 1, 2])
    >>> pss.shape
    (3, 62)
    """
    n, u = np.meshgrid(np.arange(PSS_SEQUENCE_LENGTH), ROOTS[cell_id])
    zadoff_chu = np.zeros_like(n, dtype=np.complex128)
    zadoff_chu[:, : PSS_SEQUENCE_LENGTH // 2] = np.exp(
        -1j
        * (
            np.pi
            * u[:, : PSS_SEQUENCE_LENGTH // 2]
            * n[:, : PSS_SEQUENCE_LENGTH // 2]
            * (n[:, : PSS_SEQUENCE_LENGTH // 2] + 1)
        )
        / (PSS_SEQUENCE_LENGTH + 1)
    )
    zadoff_chu[:, PSS_SEQUENCE_LENGTH // 2 :] = np.exp(
        -1j
        * (
            np.pi
            * u[:, PSS_SEQUENCE_LENGTH // 2 :]
            * (n[:, PSS_SEQUENCE_LENGTH // 2 :] + 1)
            * (n[:, PSS_SEQUENCE_LENGTH // 2 :] + 2)
        )
        / (PSS_SEQUENCE_LENGTH + 1)
    )
    return zadoff_chu


def _generate_s_sequence(n: int):
    if n < 0:
        raise ValueError(f"Function undefined for given n: {n} < 0")

    if int(n) in _generate_s_sequence._CACHE:
        return _generate_s_sequence._CACHE[int(n)]

    result = (_generate_s_sequence(n - 3) + _generate_s_sequence(n - 5)) % 2
    _generate_s_sequence._CACHE[int(n)] = result

    return result


def _generate_c_sequence(n: int):
    if n < 0:
        raise ValueError(f"Function undefined for given n: {n} < 0")

    if int(n) in _generate_c_sequence._CACHE:
        return _generate_c_sequence._CACHE[int(n)]

    result = (_generate_c_sequence(n - 2) + _generate_c_sequence(n - 5)) % 2
    _generate_c_sequence._CACHE[int(n)] = result

    return result


def _generate_z_sequence(n: int):
    if n < 0:
        raise ValueError(f"Function undefined for given n: {n} < 0")

    if int(n) in _generate_z_sequence._CACHE:
        return _generate_z_sequence._CACHE[int(n)]

    result = (
        _generate_z_sequence(n - 1)
        + _generate_z_sequence(n - 2)
        + _generate_z_sequence(n - 4)
        + _generate_z_sequence(n - 5)
    ) % 2
    _generate_z_sequence._CACHE[int(n)] = result

    return result


_generate_s_sequence._CACHE = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
}

_generate_c_sequence._CACHE = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
}

_generate_z_sequence._CACHE = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
}


def _generate_m_sequence(
    n: int, m_generator: collections.abc.Callable[[int], int]
):
    return 1 - 2 * m_generator(n)


def generate_sss_sequence(
    cell_id_in_group: int, subframe_index: Optional[int] = 0
) -> np.ndarray:
    """Generate frequency-domain SSS sequence consisting of two scrambled
    interleaved concatinated length-31 binary sequences.

    Parameters
    ----------
    cell_id_in_group : int
        Cell ID in group as determined by PSS sequence for seeding the
        scrambler.
    subframe_index : int, optional
        Subframe index, range 0 to 9, determining which m-sequence is being
        generated, by default 0

    Returns
    -------
    np.ndarray
        A `168x62` matrix containing the 168 different SSS variants for each of
        the 62 sub-carriers.

    Raises
    ------
    ValueError
        If invalid subframe index is out of valid range.

    Examples
    --------
    Generate all possible SSS sequences for cell ID in group 1 located in
    subframe 0:
    >>> generate_sss_sequence(1, 0)
    array([[ 1, -1,  1, ...,  1, -1, -1],
           [ 1, -1,  1, ..., -1,  1,  1],
           [ 1, -1,  1, ...,  1,  1,  1],
           ...,
           [ 1,  1,  1, ..., -1, -1, -1],
           [ 1, -1,  1, ..., -1,  1, -1],
           [ 1,  1,  1, ..., -1,  1,  1]])
    """
    cell_id_groups = np.arange(CELL_ID_GROUPS)

    q_prime = cell_id_groups // 30
    q = (cell_id_groups + q_prime * (q_prime + 1) / 2) // 30
    m_prime = cell_id_groups + q * (q + 1) / 2
    m0 = m_prime % 31
    m1 = (m0 + m_prime // 31 + 1) % 31

    d = np.empty((cell_id_groups.shape[0], SSS_SEQUENCE_LENGTH), dtype=int)

    if subframe_index % 10 in range(0, 5):
        for idx in np.arange(d.shape[0]):
            d[idx, 0::2] = generate_sss_sequence.generate_m_sequence(
                (np.arange(31) + m0[idx]) % 31, _generate_s_sequence
            ) * generate_sss_sequence.generate_m_sequence(
                (np.arange(31) + cell_id_in_group) % 31, _generate_c_sequence
            )
            d[idx, 1::2] = (
                generate_sss_sequence.generate_m_sequence(
                    (np.arange(31) + m1[idx]) % 31, _generate_s_sequence
                )
                * generate_sss_sequence.generate_m_sequence(
                    (np.arange(31) + cell_id_in_group + 3) % 31,
                    _generate_c_sequence,
                )
                * generate_sss_sequence.generate_m_sequence(
                    (np.arange(31) + (m0[idx] % 8)) % 31, _generate_z_sequence
                )
            )
    elif subframe_index % 10 in range(5, 10):
        for idx in np.arange(d.shape[0]):
            d[idx, 0::2] = generate_sss_sequence.generate_m_sequence(
                (np.arange(31) + m1[idx]) % 31, _generate_s_sequence
            ) * generate_sss_sequence.generate_m_sequence(
                (np.arange(31) + cell_id_in_group) % 31, _generate_c_sequence
            )
            d[idx, 1::2] = (
                generate_sss_sequence.generate_m_sequence(
                    (np.arange(31) + m0[idx]) % 31, _generate_s_sequence
                )
                * generate_sss_sequence.generate_m_sequence(
                    (np.arange(31) + cell_id_in_group + 3) % 31,
                    _generate_c_sequence,
                )
                * generate_sss_sequence.generate_m_sequence(
                    (np.arange(31) + (m1[idx] % 8)) % 31, _generate_z_sequence
                )
            )
    else:
        raise ValueError("Invalid subframe index")

    return d


generate_sss_sequence.generate_m_sequence = np.frompyfunc(
    _generate_m_sequence, 2, 1
)


def get_num_fft_bins(num_sc: int) -> int:
    return int(
        2 ** np.ceil(np.log2(num_sc))
    )  # find closest power of 2 fitting all sub-carriers


def get_cyclic_prefix_lengths(
    num_fft_bins: int,
    extended: Optional[bool] = False,
    num_sc_rb: Optional[int] = 0,
) -> np.ndarray:
    if not extended:
        cyc_pref = np.tile(np.array([160, 144, 144, 144, 144, 144, 144]), 2)
    else:
        if num_sc_rb <= 0:
            raise ValueError(
                "When extended=True cyclic prefix num_sc_rb must be a positive"
                "integer."
            )
        if num_sc_rb == 12:
            return np.full((6,), 512)
        elif num_sc_rb == 24:
            return np.full((3,), 1024)
        elif num_sc_rb == 72:
            return np.full((1,), 3072)
        elif num_sc_rb == 144:
            return np.full((1,), 6144)
        elif num_sc_rb == 486:
            return np.full((1,), 9216)
        else:
            raise ValueError(
                "Unrecognized number of sub-carriers per resource"
                f" block, num_sc_rb={num_sc_rb}"
            )

    return (cyc_pref / np.sum(cyc_pref) * num_fft_bins).astype(int)


def ofdm_modulate_subframe(grid: np.ndarray) -> Tuple[np.ndarray, float]:
    """OFDM modulate grid of subframes into time-domain waveforms.

    Parameters
    ----------
    grid : np.ndarray
        A grid of dimensions `NUMSUBF x NUMSC x SYMPERSUBF`, where `NUMSUBF`
        defines the number of subframes, `NUMSC` the number of sub-carriers in
        the channel, `SYMPERSUBF` then umber of OFDM symbols per subframe.

    Returns
    -------
    np.ndarray, float
        First: A `NUMSUBF x M` matrix, where each row (`0,...,NUMSUBF`) contains
        `M` time-domain samples representing one subframe.
        Second: Sampling rate used to generate the time-domain waveform. This
        might be higher than the channel bandwidth. Cyclic prefixing requires a
        two's power number of samples per symbol, therefore the generator
        sampling rate (per second) is given by the formula `(2^ceil(log2(num_sc)) * symbols_per_frame + total_cyclic_prefix_length) * 1000`.
    """  # noqa E501
    num_subframes = grid.shape[0]
    num_sc = grid.shape[1]
    num_fft_bins = get_num_fft_bins(num_sc)
    first_sc_idx = (
        num_fft_bins // 2 - num_sc // 2 - 1
    )  # offset of first sub-carrier in larger-than-necessary set of FFT-bins
    cyc_pref = get_cyclic_prefix_lengths(num_fft_bins)
    symbols_per_slot = 7
    slots_per_subframe = 2
    symbols_per_subframe = symbols_per_slot * slots_per_subframe
    samples_per_subframe = int(
        num_fft_bins * symbols_per_subframe + np.sum(cyc_pref)
    )
    subframes_per_frame = 10
    frames_per_second = 100
    samples_per_frame = samples_per_subframe * subframes_per_frame
    generator_sample_rate = samples_per_frame * frames_per_second

    ifft_in = np.zeros(
        (num_subframes, num_fft_bins, symbols_per_subframe), dtype=np.complex128
    )
    ifft_in[:, first_sc_idx : first_sc_idx + num_sc // 2 + 1, :] = grid[
        :, np.arange(num_sc // 2 + 1), :
    ]
    ifft_in[
        :,
        first_sc_idx + 1 + (num_sc // 2 + 1) : first_sc_idx + num_sc + 1,
        :,
    ] = grid[:, np.arange(num_sc // 2 + 1, num_sc), :]

    ifft_out = np.fft.ifft(np.fft.fftshift(ifft_in, axes=1), axis=1)

    subframe_waveforms = np.empty(
        (num_subframes, samples_per_subframe), dtype=np.complex128
    )

    for sym_idx in np.arange(symbols_per_subframe):
        subframe_waveforms[
            :,
            sym_idx * num_fft_bins
            + np.sum(cyc_pref[:sym_idx]) : sym_idx * num_fft_bins
            + np.sum(cyc_pref[: sym_idx + 1]),
        ] = ifft_out[
            :, -cyc_pref[sym_idx] :, sym_idx
        ]  # copy cyclic prefix to begining of each time-slot
        subframe_waveforms[
            :,
            sym_idx * num_fft_bins
            + np.sum(cyc_pref[: sym_idx + 1]) : (sym_idx + 1) * num_fft_bins
            + np.sum(cyc_pref[: sym_idx + 1]),
        ] = ifft_out[
            :, :, sym_idx
        ]  # fill rest of time-slot with OFDM symbol

    return (subframe_waveforms, generator_sample_rate)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

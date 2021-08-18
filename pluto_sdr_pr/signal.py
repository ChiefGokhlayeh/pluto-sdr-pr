"""Signal Generator Module

Used to generate synchronization sequences and modulate data points into OFDM
symbols.

Implementation based on 3GPP TS 36.211 version 16.4.0 Release 16 as described in
ETSI TS 136 211 V16.4.0 (2021-02).
"""
import collections
from io import SEEK_CUR
import logging
from enum import Enum, auto
from functools import cache
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from .ioutils import SampleIO

LOGGER = logging.getLogger(__name__)

try:
    from cupy import asarray as cupy_asarray
    from cupy import correlate as cupy_correlate

    NO_CUPY = False

    cupy_asarray(np.array([1, 2, 3])).get()  # Check GPU hardware availability

    def correlate(a: np.ndarray, b: np.ndarray, mode: str):
        return cupy_correlate(cupy_asarray(a), cupy_asarray(b), mode=mode).get()


except Exception as e:
    LOGGER.warning("Failed to import CUPY! Fallback to SciPy.", exc_info=e)

    from scipy.signal import correlate as scipy_correlate

    NO_CUPY = True

    def correlate(a: np.ndarray, b: np.ndarray, mode: str):
        return scipy_correlate(a, b, mode=mode, method="fft")


ROOTS = np.array([25, 29, 34])

PSS_SEQUENCE_LENGTH = 62
SSS_SEQUENCE_LENGTH = 62
CELL_ID_GROUPS = 168


class CyclicPrefixMode(Enum):
    NORMAL = auto()
    EXTENDED = auto()

    def get_lengths(
        self,
        num_fft_bins: int,
        num_sc_rb: int = 0,
    ) -> np.ndarray:
        if self is CyclicPrefixMode.NORMAL:
            cyc_pref = np.tile(np.array([160, 144, 144, 144, 144, 144, 144]), 2)
        elif self is CyclicPrefixMode.EXTENDED:
            if num_sc_rb <= 0:
                raise ValueError(
                    "For CyclicPrefixMode.EXTENDED num_sc_rb must be a positive"
                    " integer."
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
        else:
            raise Exception("Unknown cyclic prefix mode.")

        return (cyc_pref / np.sum(cyc_pref) * num_fft_bins).astype(int)


class ENodeB:
    T_S = 1 / (15000 * 2048)
    T_FRAME = T_S * 307200

    @classmethod
    def bandwidth_to_num_resource_blocks(cls, channel_bandwidth: int):
        return int(0.9 * channel_bandwidth / 180e3)

    def __init__(
        self,
        num_resource_blocks: int,
        cyclic_prefix_mode: CyclicPrefixMode = CyclicPrefixMode.NORMAL,
        num_sc_per_resource_block: int = 12,
        num_slots_per_symbol: int = 7,
        num_slots_per_subframe: int = 2,
        num_pss_rb: int = 6,
        num_sss_rb: int = 6,
        pss_symbol_idx_in_slot: int = 6,
        sss_symbol_idx_in_slot: int = 4,
        sss_subframe_index: int = 0,
        num_subframes_per_frame=10,
        num_frames_per_second=100,
    ) -> None:
        self._num_resource_blocks = num_resource_blocks
        self._cyclic_prefix_mode = cyclic_prefix_mode
        self._num_sc_per_resource_block = num_sc_per_resource_block
        self._num_symbols_per_slot = num_slots_per_symbol
        self._num_slots_per_subframe = num_slots_per_subframe
        self._num_pss_rb = num_pss_rb
        self._num_sss_rb = num_sss_rb
        self._pss_symbol_idx_in_slot = pss_symbol_idx_in_slot
        self._sss_symbol_idx_in_slot = sss_symbol_idx_in_slot
        self._sss_subframe_index = sss_subframe_index
        self._num_subframes_per_frame = num_subframes_per_frame
        self._num_frames_per_second = num_frames_per_second

    @property
    def num_resource_blocks(self) -> int:
        return self._num_resource_blocks

    @property
    def cyclic_prefix_mode(self) -> CyclicPrefixMode:
        return self._cyclic_prefix_mode

    @property
    def num_sc_per_resource_block(self) -> int:
        return self._num_sc_per_resource_block

    @property
    @cache
    def num_sc(self) -> int:
        return self.num_resource_blocks * self.num_sc_per_resource_block

    @property
    def num_symbols_per_slot(self) -> int:
        return self._num_symbols_per_slot

    @property
    def num_slots_per_subframe(self) -> int:
        return self._num_slots_per_subframe

    @property
    @cache
    def num_symbols_per_subframe(self) -> int:
        return self.num_symbols_per_slot * self.num_slots_per_subframe

    @property
    def num_pss_rb(self) -> int:
        return self._num_pss_rb

    @property
    def num_sss_rb(self) -> int:
        return self._num_sss_rb

    @property
    def pss_symbol_idx_in_slot(self) -> int:
        return self._pss_symbol_idx_in_slot

    @property
    @cache
    def pss_sc_offset_in_symbol(self) -> int:
        return int(
            ((self.num_resource_blocks - self.num_pss_rb) / 2)
            * self.num_sc_per_resource_block
        )

    @property
    def sss_symbol_idx_in_slot(self) -> int:
        return self._sss_symbol_idx_in_slot

    @property
    @cache
    def sss_sc_offset_in_symbol(self) -> int:
        return int(
            ((self.num_resource_blocks - self.num_sss_rb) / 2)
            * self.num_sc_per_resource_block
        )

    @property
    def sss_subframe_index(self) -> int:
        return self._sss_subframe_index

    @property
    @cache
    def num_fft_bins(self) -> int:
        return int(
            2 ** np.ceil(np.log2(self.num_sc))
        )  # find closest power of 2 fitting all sub-carriers

    @property
    def num_subframes_per_frame(self) -> int:
        return self._num_subframes_per_frame

    @property
    def num_frames_per_second(self) -> int:
        return self._num_frames_per_second

    @property
    @cache
    def num_samples_per_subframe(self) -> int:
        return self.num_fft_bins * self.num_symbols_per_subframe + np.sum(
            self.cyclic_prefix_mode.get_lengths(self.num_fft_bins, self.num_sc)
        )

    @property
    @cache
    def ofdm_sample_rate(self) -> int:
        return (
            self.num_samples_per_subframe
            * self.num_subframes_per_frame
            * self.num_frames_per_second
        )


class CorrelationResult:
    def __init__(self, magnitudes: np.ndarray, sample_rate: int) -> None:
        self._magnitudes = magnitudes
        self._sample_rate = sample_rate

    @property
    def magnitudes(self) -> np.ndarray:
        return self._magnitudes

    @magnitudes.setter
    def magnitudes(self, value: np.ndarray) -> None:
        self._magnitudes = value

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value

    @property
    @cache
    def max_peak_index(self) -> Tuple[int, int]:
        return np.unravel_index(
            np.argmax(self.magnitudes, axis=None), self.magnitudes.shape
        )

    @property
    @cache
    def peak_time_offset(self):
        return self.max_peak_index[1] / self.sample_rate

    @property
    @cache
    def max_magnitude(self):
        return self.magnitudes[self.max_peak_index[0]]


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


@cache
def _generate_s_sequence(n: int):
    if n < 0:
        raise ValueError(f"Function undefined for given n: {n} < 0")

    if 0 <= n <= 3:
        return 0
    elif n == 4:
        return 1
    else:
        return (_generate_s_sequence(n - 3) + _generate_s_sequence(n - 5)) % 2


@cache
def _generate_c_sequence(n: int):
    if n < 0:
        raise ValueError(f"Function undefined for given n: {n} < 0")

    if 0 <= n <= 3:
        return 0
    elif n == 4:
        return 1
    else:
        return (_generate_c_sequence(n - 2) + _generate_c_sequence(n - 5)) % 2


@cache
def _generate_z_sequence(n: int):
    if n < 0:
        raise ValueError(f"Function undefined for given n: {n} < 0")

    if 0 <= n <= 3:
        return 0
    elif n == 4:
        return 1
    else:
        return (
            _generate_z_sequence(n - 1)
            + _generate_z_sequence(n - 2)
            + _generate_z_sequence(n - 4)
            + _generate_z_sequence(n - 5)
        ) % 2


def _generate_m_sequence(n: int, m_generator: collections.abc.Callable):
    return 1 - 2 * m_generator(n)


def generate_sss_sequence(
    cell_id_in_group: int, subframe_index: int = 0
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


def ofdm_modulate_subframe(
    grid: np.ndarray, enb: ENodeB
) -> Tuple[np.ndarray, float]:
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
    num_fft_bins = enb.num_fft_bins

    assert num_fft_bins >= num_sc

    first_sc_idx = (
        num_fft_bins // 2 - num_sc // 2 - 1
    )  # offset of first sub-carrier in larger-than-necessary set of FFT-bins
    cyc_pref = enb.cyclic_prefix_mode.get_lengths(num_fft_bins, enb.num_sc)

    ifft_in = np.zeros(
        (num_subframes, num_fft_bins, enb.num_symbols_per_subframe),
        dtype=np.complex128,
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
        (num_subframes, enb.num_samples_per_subframe), dtype=np.complex128
    )

    for sym_idx in np.arange(enb.num_symbols_per_subframe):
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

    return subframe_waveforms


class MultiSignalStream:
    def __init__(self) -> None:
        self._inputs: List[SampleIO] = None

    @property
    def inputs(self) -> List[SampleIO]:
        return self._inputs

    @property
    def curr_enb(self) -> ENodeB:
        return self._enb

    def start_unsynchronized(self, *inputs: SampleIO) -> None:
        self._inputs = inputs

    def start_synchronized(
        self, *inputs: SampleIO, enb: ENodeB, **kwargs
    ) -> Tuple[int, List[CorrelationResult], List[CorrelationResult]]:
        """Start a new stream with multiple synchronized inputs.

        Examples
        --------
        >>> from .ioutils import SdriqSampleIO

        >>> enb = ENodeB(6)
        >>> mss = MultiSignalStream()

        >>> ref_fp = "tests/data/two_frames_unaligned.sdriq"
        >>> obsrv_fp = "tests/data/two_frames_unaligned_shifted.sdriq"

        >>> with SdriqSampleIO(ref_fp) as ref, SdriqSampleIO(obsrv_fp) as obsrv:
        ...     cell_id, _, _ = mss.start_synchronized(
        ...         ref, obsrv, enb=enb, num_frames=8
        ...     )
        ...     cell_id
        36
        """

        if len(inputs) <= 0:
            raise ValueError("Parameter inputs must not be empty.")

        checksPassed = all(
            input.sample_rate == inputs[0].sample_rate
            and input.center_frequency == inputs[0].center_frequency
            for input in inputs
        )
        if not checksPassed:
            raise ValueError(
                "Input sample_rate or center_frequency attributes not equal"
            )

        sample_rate = inputs[0].sample_rate

        earliest = min(input.start for input in inputs)

        time_diffs = np.fromiter(
            [(input.start - earliest).total_seconds() for input in inputs],
            float,
        )
        estim_sample_shifts = time_diffs * sample_rate

        refine_grace_period = 1

        for idx, estim_sample_shift in enumerate(estim_sample_shifts):
            inputs[idx].seek(
                max(
                    int(estim_sample_shift) - refine_grace_period * sample_rate,
                    0,
                ),
                SEEK_CUR,
            )

        cell_id, pss_correlations, sss_correlations = self.find_cell(
            *inputs, enb=enb, num_frames=kwargs.get("num_frames", 8)
        )

        self._inputs = inputs
        self._enb = enb

        return cell_id, pss_correlations, sss_correlations

    def find_cell(
        self, *inputs: SampleIO, enb: ENodeB, num_frames: int
    ) -> Tuple[int, List[CorrelationResult], List[CorrelationResult]]:
        def throw_if_index_mismatch(
            correlations: List[CorrelationResult],
            reference_peak_index: int,
            type: str,
        ):
            if any(
                1
                for corr in correlations[1:]
                if corr.max_peak_index[0] != reference_peak_index
            ):
                raise ValueError(
                    f"Mismatching {type}: "
                    f"{[c.max_peak_index[0] for c in correlations]}"
                )

        start_offsets = [input.tell() for input in inputs]

        pss_correlations = self.determine_pss_offsets(
            *inputs, enb=enb, num_frames=num_frames
        )
        pss_index = pss_correlations[0].max_peak_index[0]
        throw_if_index_mismatch(pss_correlations, pss_index, "PSS index")
        LOGGER.debug(f"Matching PSS indices: {pss_index}")

        for input, offset in zip(inputs, start_offsets):
            input.seek(offset)

        sss_correlations = self.determine_sss_offsets(
            *inputs,
            cell_id_in_group=pss_index,
            enb=enb,
            num_frames=num_frames,
        )
        sss_index = sss_correlations[0].max_peak_index[0]
        throw_if_index_mismatch(sss_correlations, sss_index, "SSS index")

        cell_id = 3 * sss_index + pss_index

        for input, offset in zip(inputs, start_offsets):
            input.seek(offset)

        return cell_id, pss_correlations, sss_correlations

    def determine_pss_offsets(
        self, *inputs: SampleIO, enb: ENodeB, num_frames: int
    ) -> List[CorrelationResult]:
        pss_sequences = generate_pss_sequence([0, 1, 2])

        grid = np.zeros(
            (pss_sequences.shape[0], enb.num_sc, enb.num_symbols_per_subframe),
            dtype=np.complex128,
        )

        pss_indices = np.arange(
            enb.pss_sc_offset_in_symbol,
            enb.pss_sc_offset_in_symbol + pss_sequences.shape[1],
        )

        grid[:, pss_indices, enb.pss_symbol_idx_in_slot] = pss_sequences[:, :]

        pss_waveforms = ofdm_modulate_subframe(grid, enb)

        return [
            self.find_correlation(
                input, pss_waveforms, enb.ofdm_sample_rate, num_frames
            )
            for input in inputs
        ]

    def determine_sss_offsets(
        self,
        *inputs: SampleIO,
        enb: ENodeB,
        cell_id_in_group: int,
        num_frames: int,
    ) -> List[CorrelationResult]:
        sss_sequences = generate_sss_sequence(
            cell_id_in_group, enb.sss_subframe_index
        )

        grid = np.zeros(
            (sss_sequences.shape[0], enb.num_sc, enb.num_symbols_per_subframe),
            dtype=np.complex128,
        )

        sss_indices = np.arange(
            enb.sss_sc_offset_in_symbol,
            enb.sss_sc_offset_in_symbol + sss_sequences.shape[1],
        )

        grid[:, sss_indices, enb.sss_symbol_idx_in_slot] = sss_sequences[:, :]

        sss_waveforms = ofdm_modulate_subframe(grid, enb)

        return [
            self.find_correlation(
                input, sss_waveforms, enb.ofdm_sample_rate, num_frames
            )
            for input in inputs
        ]

    def find_correlation(
        self,
        input: SampleIO,
        waveforms: np.ndarray,
        ofdm_sample_rate: int,
        num_frames: int,
    ) -> CorrelationResult:
        obsrv_samples = input.read(
            int(input.sample_rate * ENodeB.T_FRAME * num_frames)
        )

        if obsrv_samples.shape[0] < int(
            input.sample_rate * ENodeB.T_FRAME * num_frames
        ):
            raise EOFError(f"Not enough samples in input {input}.")

        sample_rate_diff = ofdm_sample_rate - input.sample_rate
        if sample_rate_diff > 0:
            LOGGER.warning(
                f"Input sample rate at {input.sample_rate} S/sec lower than "
                "needed, padding frequency-space with zeros to achieve "
                f"{ofdm_sample_rate} S/sec"
            )
            obsrv_samples = self.pad_freqs_of_waveform(
                obsrv_samples,
                int(sample_rate_diff * ENodeB.T_FRAME * num_frames),
            )
        elif sample_rate_diff < 0:
            LOGGER.info(
                f"Input sample rate at {input.sample_rate} S/sec higher than "
                f"needed, downsampling to {ofdm_sample_rate} S/sec"
            )
            obsrv_samples = signal.resample(obsrv_samples, ofdm_sample_rate)

        corr_mags = np.vstack(
            [
                np.abs(correlate(obsrv_samples, waveform, mode="valid"))
                for waveform in waveforms
            ]
        )

        return CorrelationResult(corr_mags, ofdm_sample_rate)

    def pad_freqs_of_waveform(self, waveform, sample_rate_diff):
        fftout = np.fft.fftshift(np.fft.fft(waveform))
        padded_fftout = np.pad(
            fftout,
            pad_width=(
                sample_rate_diff // 2,
                sample_rate_diff // 2,
            ),
        )
        return np.fft.ifft(np.fft.fftshift(padded_fftout))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

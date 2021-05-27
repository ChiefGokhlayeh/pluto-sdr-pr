import numpy as np
import struct
import typing

SDRIQ_IQ_STRUCT = struct.Struct("<IQQIII")  # cSpell:disable-line


def read_samples(
    fid: typing.BinaryIO, num_samples: int, offset: typing.Optional[int] = 0
) -> np.ndarray:
    """Read specified number of IQ samples from IO-like.

    Parameters
    ----------
    fid : typing.BinaryIO
        IO-like from which to read
    num_samples : int
        Number of samples to read.
    offset : typing.Optional[int], optional
        Offset (in samples) to skip before reading, by default 0.

    Returns
    -------
    np.ndarray
        A 1-D array with the given number of complex IQ samples or less if EOF
        reached.

    Examples
    --------
    Read 100 complex IQ samples from a file.
    >>> file_path = 'tests/data/200samples_625000000_2880000.raw'
    >>> with open(file_path, 'rb') as fid:
    ...     samples = read_samples(fid, 100)
    >>> len(samples)
    100
    >>> type(samples[0])
    <class 'numpy.complex64'>
    """

    iq_as_i64 = np.fromfile(
        fid, dtype=np.int64, count=num_samples, offset=offset
    )
    return iq_as_i64.view(np.complex64)


def read_sdriq_samples(
    fid: typing.BinaryIO, num_samples: int, offset: typing.Optional[int] = 0
):
    """
    Read specified number of samples from sdriq file.

    :param fid: IO-like from which to read.
    :param num_samples: Number of samples to read.
    :param offset: Offset (in samples) to skip before reading.
    :returns An array-like with the given number of samples or less if EOF
    reached.
    """

    header = np.fromfile(fid, dtype=np.uint8, count=32)

    (
        sample_rate,
        center_frequency,
        start_time_stamp,
        sample_size,
        _,  # filler
        _,  # crc
    ) = SDRIQ_IQ_STRUCT.unpack(header.tobytes())

    header_dict = {
        "sample_rate": sample_rate,
        "center_frequency": center_frequency,
        "start_time_stamp": start_time_stamp,
        "sample_size": sample_size,
    }

    if sample_size == 16:
        config = {"dtype": np.int16, "scale": 0xFFFF}
    elif sample_size == 24:
        config = {"dtype": np.int32, "scale": 0xFFFFFF}
    else:
        raise Exception(f"Sample size {sample_size} is not supported")

    data = np.fromfile(
        fid,
        dtype=config["dtype"],
        offset=SDRIQ_IQ_STRUCT.size + offset * 2,
        count=num_samples * 2,
    )
    float_data = data.astype(np.float32) / config["scale"]
    n = float_data[0::2] + 1j * float_data[1::2]

    return n, header_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()

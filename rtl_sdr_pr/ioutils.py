import struct
import numpy as np
import numpy.typing as npt
import typing

IQ_STRUCT = struct.Struct("ff")


def read_samples(
    fid: typing.BinaryIO, num_samples: int, offset: typing.Optional[int] = 0
) -> npt.ArrayLike:
    """
    Read specified number of samples from IO-like.

    :param fid: IO-like from which to read.
    :param num_samples: Number of samples to read.
    :param offset: Offset (in samples) to skip before reading.
    :returns An array-like with the given numer of samples or less if EOF
    reached.
    """

    if offset:
        fid.seek(offset * IQ_STRUCT.size)
    gen_n = (
        complex(iq[0], iq[1])
        for iq in IQ_STRUCT.iter_unpack(fid.read(IQ_STRUCT.size * num_samples))
    )
    n = np.fromiter(gen_n, np.complex64)
    return n


def read_sdriq_samples(
    fid: typing.BinaryIO, num_samples: int, offset: typing.Optional[int] = 0
):
    """
    Read specified number of samples from sdriq file.

    :param fid: IO-like from which to read.
    :param num_samples: Number of samples to read.
    :param offset: Offset (in samples) to skip before reading.
    :returns An array-like with the given numer of samples or less if EOF
    reached.
    """

    header = np.fromfile(fid, dtype=np.uint8, count=32)

    (
        sample_rate,
        center_frequency,
        start_time_stamp,
        sample_size,
        filler,
        crc32,
    ) = struct.unpack("<IQQIII", header.tobytes())

    if sample_size == 16:
        config = {"dtype": np.int16, "scale": 0xFFFF}
    elif sample_size == 24:
        config = {"dtype": np.int32, "scale": 0xFFFFFF}
    else:
        raise Exception(f"Sample rate {sample_size} is not supported")

    data = np.fromfile(
        fid, dtype=config["dtype"], offset=32 + offset * 2, count=num_samples * 2
    )
    float_data = data.astype(np.float32) / config["scale"]
    n = float_data[0::2] + 1j * float_data[1::2]

    return n


if __name__ == "__main__":
    import doctest

    doctest.testmod()

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

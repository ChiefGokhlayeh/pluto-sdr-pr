import struct
import numpy as np
import numpy.typing as npt
import typing

IQ_STRUCT = struct.Struct("ff")


def read_samples(
    fid: typing.BinaryIO, num_samples: int, offset: typing.Optional[int] = 0
) -> npt.ArrayLike:
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

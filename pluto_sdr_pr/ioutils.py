from __future__ import annotations

import binascii
import datetime
import io
import struct
import typing
from abc import ABC, abstractmethod, abstractproperty

import numpy as np

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
    fid: typing.Union[typing.BinaryIO, str],
    num_samples: int,
    offset: typing.Optional[int] = 0,
) -> typing.Tuple[np.ndarray, typing.Dict[str, int]]:
    """Read specified number of IQ samples from IO-like including `.sdriq`
    header.

    Parameters
    ----------
    fid : typing.BinaryIO
        file-like object or file path from which to read.
    num_samples : int
        Number of samples to read.
    offset : typing.Optional[int], optional
        Offset (in samples) to skip before reading, by default 0.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        A 1-D array with the given number of complex IQ samples or less if EOF
        reached and a dictionary containing information from `.sdriq` header.

    Examples
    --------
    Read header from a file object.
    >>> file_path = 'tests/data/header-only.sdriq'
    >>> with open(file_path, 'rb') as fid:
    ...     samples, header = read_sdriq_samples(fid, 0)
    >>> len(samples)
    0
    >>> type(samples.dtype)
    <class 'numpy.dtype[complex128]'>
    >>> header["sample_rate"]
    5000000
    >>> header["center_frequency"]
    626000000
    >>> header["start"].isoformat()
    '2021-05-23T18:35:07'
    >>> header["pcm_sample_size"]
    24

    Read 100 complex IQ samples from file path.
    >>> file_path = 'tests/data/300samples.sdriq'
    >>> samples, _ = read_sdriq_samples(file_path, 100)
    >>> len(samples)
    100
    """

    sdriq = SdriqSampleIO(fid)

    header_dict = {
        "sample_rate": sdriq.sample_rate,
        "center_frequency": sdriq.center_frequency,
        "start": sdriq.start,
        "pcm_sample_size": sdriq.pcm_sample_size,
    }

    return sdriq.read(num_samples, offset), header_dict


class SampleIO(ABC):
    @abstractproperty
    def sample_rate(self) -> int:
        pass

    @abstractproperty
    def center_frequency(self) -> int:
        pass

    @abstractproperty
    def start(self) -> datetime.datetime:
        pass

    @abstractmethod
    def read(
        self,
        num_samples: typing.Optional[int] = -1,
        offset: typing.Optional[int] = 0,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int = 0) -> int:
        pass

    @abstractmethod
    def seekable(self) -> bool:
        pass

    @abstractmethod
    def readable(self) -> bool:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> SampleIO:
        pass

    @abstractmethod
    def __exit__(
        self,
        t: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[typing.TracebackType],
    ) -> typing.Optional[bool]:
        pass


class CRCError(Exception):
    def __init__(
        self,
        message: str,
        expected_crc: typing.Optional[int] = None,
        calculated_crc: typing.Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self._expected_crc = expected_crc
        self._calculated_crc = calculated_crc

    @property
    def expected_crc(self) -> typing.Optional[int]:
        return self._expected_crc

    @property
    def calculated_crc(self) -> typing.Optional[int]:
        return self._calculated_crc


class SdriqSampleIO(SampleIO):
    """Represents an `.sdriq` stream of samples.

    Can be used to read samples and metadata from an underlying file-like
    object.

    Examples
    --------
    >>> with SdriqSampleIO(open("tests/data/header-only.sdriq")) as sdriq:
    ...     sdriq.sample_rate
    ...     sdriq.center_frequency
    ...     sdriq.start.isoformat()
    ...     sdriq.pcm_sample_size
    5000000
    626000000
    '2021-05-23T18:35:07'
    24
    >>> sdriq.closed
    True
    """

    def __init__(
        self, fid: typing.Union[str, typing.BinaryIO], ignore_crc: bool = False
    ) -> None:
        super().__init__()

        if type(fid) is str:
            self._fid = open(fid)
        else:
            self._fid = fid

        header = np.fromfile(
            self._fid, dtype=np.byte, count=SDRIQ_IQ_STRUCT.size
        )

        (
            self._sample_rate,
            self._center_frequency,
            self._start_time_stamp,
            self._pcm_sample_size,
            _,  # filler
            self.crc,  # crc
        ) = SDRIQ_IQ_STRUCT.unpack(header.tobytes())

        if not ignore_crc:
            calc_crc32 = binascii.crc32(header[: SDRIQ_IQ_STRUCT.size - 4])
            if calc_crc32 != self.crc:
                raise CRCError(
                    f"CRC check of {str(fid)} failed!", self.crc, calc_crc32
                )

        if self._pcm_sample_size == 16:
            self._pcm_dtype = np.dtype(np.int16)
        elif self._pcm_sample_size == 24:
            self._pcm_dtype = np.dtype(np.int32)
        else:
            raise Exception(f"Sample size {self.sample_size} is not supported")

        self._start = datetime.datetime.utcfromtimestamp(
            self._start_time_stamp
            * (1 if self._start_time_stamp > 2 ^ 32 else 1000)
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def center_frequency(self) -> int:
        return self._center_frequency

    @property
    def start(self) -> datetime.datetime:
        return self._start

    @property
    def pcm_sample_size(self) -> int:
        return self._pcm_sample_size

    def __enter__(self) -> SdriqSampleIO:
        return self

    def __exit__(
        self,
        t: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[typing.TracebackType],
    ) -> typing.Optional[bool]:
        return self._fid.__exit__(t, value, traceback)

    def read(
        self,
        num_samples: typing.Optional[int] = -1,
        offset: typing.Optional[int] = 0,
    ) -> np.ndarray:
        """Read samples from SDR-IQ file.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to read. If negative, all samples will be read
            (only applicable if underlying file-object is seekable).
        offset : int, optional
            Offset (in samples) to skip before reading, by default 0.

        Returns
        -------
        np.ndarray
            A 1-D array with the given number of complex IQ samples or less if
            EOF reached.

        Examples
        --------
        Read all samples from `.sdriq` file.
        >>> with SdriqSampleIO(open("tests/data/300samples.sdriq")) as sdriq:
        ...     samples = sdriq.read()
        ...     samples.shape
        ...     samples[0:10]
        (300,)
        array([ 0.0534668 +0.03881836j,  0.05786133+0.11987305j,
               -0.0300293 -0.03588867j, -0.04980469-0.03271485j,
               -0.01464844+0.08813477j,  0.02294922-0.02319336j,
               -0.01220703-0.11254883j, -0.05590821-0.09741212j,
                0.07983399-0.09838868j,  0.07397461+0.02246094j])

        Read 50 samples from `.sdriq` file.
        >>> with SdriqSampleIO(open("tests/data/300samples.sdriq")) as sdriq:
        ...     samples = sdriq.read(50)
        ...     samples.shape
        ...     samples[0:10]
        (50,)
        array([ 0.0534668 +0.03881836j,  0.05786133+0.11987305j,
               -0.0300293 -0.03588867j, -0.04980469-0.03271485j,
               -0.01464844+0.08813477j,  0.02294922-0.02319336j,
               -0.01220703-0.11254883j, -0.05590821-0.09741212j,
                0.07983399-0.09838868j,  0.07397461+0.02246094j])

        Read 50 samples from `.sdriq` file starting at sample offset 5.
        >>> with SdriqSampleIO(open("tests/data/300samples.sdriq")) as sdriq:
        ...     samples = sdriq.read(50, 5)
        ...     samples.shape
        ...     samples[0:5]
        (50,)
        array([ 0.02294922-0.02319336j, -0.01220703-0.11254883j,
               -0.05590821-0.09741212j,  0.07983399-0.09838868j,
                0.07397461+0.02246094j])
        """
        pcm_bytes = np.fromfile(
            self._fid,
            dtype=self._pcm_dtype,
            offset=offset * self._pcm_dtype.itemsize * 2,
            count=num_samples * 2,
        )

        available_num_samples = pcm_bytes.shape[0] // 2

        return (
            pcm_bytes[: available_num_samples * 2].astype(np.float64)
            / (2 ** self._pcm_sample_size - 1)
        ).view(np.complex128)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Change stream position to the given sample `offset`. `offset` is
        interpreted relative to the position indicated in `whence`.

        Parameters
        ----------
        offset : int
            Offset (in samples) relative to position indicated by `whence`.
        whence : int, optional
            Base from which to view offset from, by default SEEK_SET (0). Uses
            constants as defined by `io` package:
             - `io.SEEK_SET` or `0`
             - `io.SEEK_CUR` or `1`
             - `io.SEEK_END` or `2`

        Returns
        -------
        int
            New absolute position (in samples).

        Raises
        ------
        ValueError
            For unknown values of `whence`.
        """
        byte_offset = offset * np.dtype(self._pcm_dtype).itemsize
        curr_byte_pos = self._fid.tell()
        if whence == io.SEEK_SET:
            byte_pos = self._fid.seek(
                byte_offset + SDRIQ_IQ_STRUCT.size, whence
            )
        elif whence == io.SEEK_CUR:
            byte_pos = self._fid.seek(
                max(byte_offset, SDRIQ_IQ_STRUCT.size - curr_byte_pos), whence
            )
        elif whence == io.SEEK_END:
            byte_pos = self._fid.seek(
                max(byte_offset, SDRIQ_IQ_STRUCT.size - curr_byte_pos),
                whence,
            )
        else:
            raise ValueError(f"Unknown value for whence parameter: {whence}")

        return (byte_pos - SDRIQ_IQ_STRUCT.size) // np.dtype(
            self._pcm_dtype
        ).itemsize

    def seekable(self) -> bool:
        return self._fid.seekable()

    def readable(self) -> bool:
        return self._fid.readable()

    def tell(self) -> int:
        """Return the current stream position (in samples).

        Returns
        -------
        int
            Absolute position (in samples).
        """
        return (self._fid.tell() - SDRIQ_IQ_STRUCT.size) // np.dtype(
            self._pcm_dtype
        ).itemsize

    def close(self) -> None:
        return self._fid.close()

    @property
    def closed(self) -> bool:
        return self._fid.closed


if __name__ == "__main__":
    import doctest

    doctest.testmod()

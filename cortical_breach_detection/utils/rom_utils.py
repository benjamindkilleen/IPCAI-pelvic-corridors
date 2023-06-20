import struct
import numpy as np


def parse_rom_data(rom_data: bytes):
    """
    Parses marker coordinates (X/Y/Z, in mm) from NDI .rom file data
    Parameters
    ----------
    rom_data : bytes
            The raw data of the ROM file to be parsed

    Returns
    -------
    list
            An n-by-3 matrix (i.e., a length-n list of length-3 lists) containing the X/Y/Z
            coordinates (in mm) of each of the body's n markers; or None if reading failed
    """
    num_markers = int.from_bytes(
        rom_data[28:29], byteorder="little"
    )  # number of markers is stored in byte 28
    pos = 72  # marker coordinates are stored beginning at byte 72; each coordinate is a 32-bit (4-byte) float
    markers = list()
    for _ in range(num_markers):
        raw_xyz = tuple(
            round(f, 2) for f in struct.unpack("<3f", rom_data[pos : pos + 12])
        )  # read 3x 4-byte floats, rounding to 2 decimal places
        markers.append(
            tuple(f if f != 0.0 else 0.0 for f in raw_xyz)
        )  # set "-0.0" to "+0.0"
        pos += 12
    return markers


def parse_rom_file(rom_file: str) -> np.ndarray:
    """
    Parses marker coordinates (X/Y/Z, in mm) from NDI .rom file
    Parameters
    ----------
    rom_file : str
            The path to the ROM file to be parsed

    Returns
    -------
    list
            An n-by-3 matrix (i.e., a length-n list of length-3 lists) containing the X/Y/Z
            coordinates (in mm) of each of the body's n markers; or None if reading failed
    """
    with open(rom_file, "rb") as f:
        rom_data = f.read()
    return np.array(parse_rom_data(rom_data))

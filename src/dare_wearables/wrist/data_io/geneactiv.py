from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import BinaryIO, Optional, Tuple

import numpy as np

# ----------------------------
# Constants
# ----------------------------
GN_SAMPLES = 300          # 3600 hex chars / 12 chars-per-sample
DATA_STR_LEN = 3600       # the line has 3600 hex chars for accel+light


# ----------------------------
# Fast hex decoding
# ----------------------------
_HEX_LUT = np.zeros(256, dtype=np.uint8)
_HEX_LUT[ord("0"):ord("9") + 1] = np.arange(0, 10, dtype=np.uint8)
_HEX_LUT[ord("A"):ord("F") + 1] = np.arange(10, 16, dtype=np.uint8)
_HEX_LUT[ord("a"):ord("f") + 1] = np.arange(10, 16, dtype=np.uint8)

_GN_ARANGE = np.arange(GN_SAMPLES, dtype=np.float64)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class GNInfo:
    fs: float = np.nan
    gain: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    volts: float = np.nan
    lux: float = np.nan
    npages: int = -1

    max_n: int = -1
    fs_err: int = 0


@dataclass
class GNData:
    ts: np.ndarray          # (n_samples,)
    acc: np.ndarray         # (n_samples, 3)
    temp: np.ndarray        # (n_samples,)
    light: np.ndarray       # (n_samples,)


# ----------------------------
# Text/binary helpers
# ----------------------------
def _readline_text(fp: BinaryIO) -> Optional[str]:
    b = fp.readline()
    if not b:
        return None
    return b.decode("ascii", errors="ignore").rstrip("\r\n")


def _parseline(fp: BinaryIO) -> Tuple[str, str]:
    line = _readline_text(fp)
    if line is None:
        raise EOFError("Unexpected EOF while parsing key:value line")
    if ":" not in line:
        return line.strip(), ""
    key, val = line.split(":", 1)
    return key.strip(), val.strip()


def _parse_first_number(s: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else float("nan")


def _parse_first_int(s: str) -> int:
    m = re.search(r"[-+]?\d+", s)
    return int(m.group(0)) if m else -1


def _parse_time_line_to_epoch_seconds_utc(time_line: str) -> float:
    """
    Equivalent intent to C's timegm(&tm0) + msec/1000 (UTC).
    This is a permissive parser; it looks for YYYY-MM-DD and HH:MM:SS(.mmm or :mmm).
    """
    dm = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", time_line)
    if not dm:
        raise ValueError(f"Could not parse date from time line: {time_line!r}")
    year, month, day = map(int, dm.groups())

    tm = re.search(r"(\d{2}):(\d{2}):(\d{2})(?:[.:](\d{1,3}))?", time_line)
    if not tm:
        raise ValueError(f"Could not parse time from time line: {time_line!r}")
    hh, mm, ss = map(int, tm.group(1, 2, 3))
    msec = tm.group(4)
    msec_i = int(msec.ljust(3, "0")) if msec is not None else 0

    dt = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc)
    return dt.timestamp() + (msec_i / 1000.0)


# ----------------------------
# FAST decoding for one block's 3600 hex chars
# ----------------------------
def _decode_data_line_fast(
    data_line_bytes: bytes,
    gain: np.ndarray,
    offset: np.ndarray,
    lux: float,
    volts: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode one block's 3600 hex chars into:
      acc_block: (300,3) float32
      light_block: (300,) float32

    Matches C:
      accel: ((signed12 * 100 - offset[k]) / gain[k])
      light: floor((raw >> 2) * (lux/volts))
    """

    b = data_line_bytes[:DATA_STR_LEN]
    if len(b) < DATA_STR_LEN:
        raise ValueError(f"Data string too short: got {len(b)}, expected >= {DATA_STR_LEN}")

    # ASCII -> nibble (0..15), length 3600
    u8 = np.frombuffer(b, dtype=np.uint8)
    nibbles = _HEX_LUT[u8]

    # 3 hex chars -> 12-bit value; 3600/3 = 1200 values
    trip = nibbles.reshape(-1, 3).astype(np.uint16)          # (1200,3)
    vals = (trip[:, 0] << 8) | (trip[:, 1] << 4) | trip[:, 2]  # (1200,)

    # 4 values per sample: ax, ay, az, light
    vals = vals.reshape(GN_SAMPLES, 4)  # (300,4)

    # Accel: signed 12-bit conversion (v > 2047 -> v-4096)
    raw_acc = vals[:, :3].astype(np.int16)
    raw_acc[raw_acc > 2047] -= 4096

    gain32 = gain.astype(np.float32, copy=False)
    off32 = offset.astype(np.float32, copy=False)

    acc = ((raw_acc.astype(np.float32) * 100.0) - off32) / gain32  # (300,3)

    # Light
    scale = (lux / volts) if volts != 0 else np.nan
    raw_light = vals[:, 3].astype(np.uint16)
    light = np.floor((raw_light >> 2).astype(np.float32) * np.float32(scale)).astype(np.float32)

    return acc, light


# ----------------------------
# Header reader (same logic as your C code)
# ----------------------------
def geneactiv_read_header(fp: BinaryIO) -> GNInfo:
    info = GNInfo()

    # Read first 19 lines
    for _ in range(19):
        line = _readline_text(fp)
        if line is None:
            raise EOFError("Unexpected EOF while reading first 19 header lines")

    # Sampling frequency (key:value)
    _, v = _parseline(fp)
    info.fs = float(int(_parse_first_int(v)))

    # Read until "Calibration Data"
    while True:
        line = _readline_text(fp)
        if line is None:
            raise EOFError("Unexpected EOF while searching for 'Calibration Data'")
        if line.startswith("Calibration Data"):
            break

    # Gain & offset for x,y,z
    for j in range(3):
        _, vg = _parseline(fp)
        info.gain[j] = float(int(_parse_first_int(vg)))
        _, vo = _parseline(fp)
        info.offset[j] = float(int(_parse_first_int(vo)))

    # Volts and lux lines
    line = _readline_text(fp)
    if line is None:
        raise EOFError("Unexpected EOF while reading volts line")
    info.volts = _parse_first_number(line)

    line = _readline_text(fp)
    if line is None:
        raise EOFError("Unexpected EOF while reading lux line")
    info.lux = _parse_first_number(line)

    # Skip 3 lines; the last of these contains number of pages
    last = None
    for _ in range(3):
        last = _readline_text(fp)
        if last is None:
            raise EOFError("Unexpected EOF while skipping header lines (56..58)")

    info.npages = _parse_first_int(last)

    # Final header line
    if _readline_text(fp) is None:
        raise EOFError("Unexpected EOF while reading final header line")

    return info


# ----------------------------
# Block reader (FAST decode inside)
# ----------------------------
def geneactiv_read_block(fp: BinaryIO, info: GNInfo, data: GNData) -> Optional[int]:
    """
    Returns:
      None if EOF
      0 if OK
      1 if (single) fs mismatch warning occurred (like C's GN_READ_E_BLOCK_FS_WARN)
    """

    # Read first line; EOF => done
    line1 = _readline_text(fp)
    if line1 is None:
        return None

    # Re-sync to "Recorded Data" if needed (more robust than the C version)
    if not line1.startswith("Recorded Data"):
        while line1 is not None and not line1.startswith("Recorded Data"):
            line1 = _readline_text(fp)
        if line1 is None:
            return None

    # Skip next 2 lines; third line is sequence number
    if _readline_text(fp) is None:
        raise EOFError("Unexpected EOF while skipping block line 2")
    seq_line = _readline_text(fp)
    if seq_line is None:
        raise EOFError("Unexpected EOF while reading block sequence line")

    N = _parse_first_int(seq_line)
    if N < 0:
        raise ValueError(f"Could not parse sequence number from: {seq_line!r}")

    Nps = N * GN_SAMPLES
    info.max_n = max(info.max_n, N)

    # Timestamp line (raw)
    time_line = _readline_text(fp)
    if time_line is None:
        raise EOFError("Unexpected EOF while reading block timestamp line")

    # Skip one line then read temperature line
    if _readline_text(fp) is None:
        raise EOFError("Unexpected EOF while skipping pre-temperature line")
    temp_line = _readline_text(fp)
    if temp_line is None:
        raise EOFError("Unexpected EOF while reading temperature line")
    temp = float(_parse_first_number(temp_line))
    data.temp[Nps:Nps + GN_SAMPLES] = temp

    # Skip 2 lines then read sampling rate line
    if _readline_text(fp) is None or _readline_text(fp) is None:
        raise EOFError("Unexpected EOF while skipping pre-fs lines")
    fs_line = _readline_text(fp)
    if fs_line is None:
        raise EOFError("Unexpected EOF while reading sampling-rate line")
    fs_block = float(_parse_first_number(fs_line))

    warn_fs = 0
    if not np.isfinite(info.fs):
        info.fs = fs_block
    elif fs_block != info.fs:
        if info.fs_err < 1:
            info.fs_err += 1
            info.fs = fs_block
            warn_fs = 1
        else:
            raise ValueError(f"Sampling rate mismatch again (block={fs_block}, header={info.fs})")

    # Read the data line as BYTES and decode fast
    data_line = fp.readline()
    if not data_line:
        raise EOFError("Unexpected EOF while reading data string")
    data_line = data_line.rstrip(b"\r\n")

    acc_block, light_block = _decode_data_line_fast(
        data_line_bytes=data_line,
        gain=info.gain,
        offset=info.offset,
        lux=info.lux,
        volts=info.volts,
    )

    data.acc[Nps:Nps + GN_SAMPLES, :] = acc_block
    data.light[Nps:Nps + GN_SAMPLES] = light_block

    # Timestamps (fast)
    t0 = _parse_time_line_to_epoch_seconds_utc(time_line)
    data.ts[Nps:Nps + GN_SAMPLES] = t0 + (_GN_ARANGE / info.fs)

    return warn_fs


# ----------------------------
# High-level reader (copy-paste usage)
# ----------------------------
def read_geneactiv_bin(path: str) -> tuple[GNInfo, GNData]:
    """
    Reads header, preallocates arrays using npages, then parses all blocks.

    Returns:
      info, data
    """
    with open(path, "rb") as fp:
        info = geneactiv_read_header(fp)

        if info.npages is None or info.npages <= 0:
            raise ValueError(f"Invalid number of pages parsed from header: {info.npages}")

        n_samples = info.npages * GN_SAMPLES

        # Use float32 for speed/memory; timestamps kept float64
        data = GNData(
            ts=np.full(n_samples, np.nan, dtype=np.float64),
            acc=np.full((n_samples, 3), np.nan, dtype=np.float32),
            temp=np.full(n_samples, np.nan, dtype=np.float32),
            light=np.full(n_samples, np.nan, dtype=np.float32),
        )

        while True:
            res = geneactiv_read_block(fp, info, data)
            if res is None:
                break

        return info, data


def as_pandas_dataframe(info: GNInfo, data: GNData):
    """
    Convenience conversion to pandas DataFrame (UTC time index).
    Note: converting huge arrays to datetime can be slow; do it only if needed.
    """
    import pandas as pd

    dt = pd.to_datetime(data.ts, unit="s", utc=True)
    df = pd.DataFrame(
        {
            "x": data.acc[:, 0],
            "y": data.acc[:, 1],
            "z": data.acc[:, 2],
            "temp": data.temp,
            "light": data.light,
        },
        index=dt,
    )
    df.index.name = "time"
    return df

def as_polars_dataframe(info: GNInfo, data: GNData):
    """
    Convenience conversion to a Polars DataFrame with a UTC Datetime index-like column.

    Notes:
    - Polars doesn't have a true "index" like pandas, so we store time as a column.
    - Converting timestamps to datetime can still be expensive for huge arrays.
    """
    import polars as pl

    # data.ts is seconds since epoch (float). Convert to integer milliseconds for Datetime.
    # (Using ms keeps it fast and matches your msec precision.)
    ts_ms = (data.ts * 1000.0).round().astype(np.int64)

    df = pl.DataFrame(
        {
            "time": ts_ms,
            "x": data.acc[:, 0],
            "y": data.acc[:, 1],
            "z": data.acc[:, 2],
            "temperature": data.temp,
            "light": data.light,
        }
    ).with_columns(
        pl.col("time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    )

    return df

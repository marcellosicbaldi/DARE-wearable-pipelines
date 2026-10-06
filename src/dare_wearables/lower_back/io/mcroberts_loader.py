import struct

import numpy as np
import pandas as pd


PACKET_SIZE = 512
ACC_SENSITIVITY = {4: 1 / (2 * 4096), 8: 1 / 4096, 16: 2 / 4096, 30: 4 / 4096}
GYRO_SENSITIVITY = {500: 1 / 65.5, 1000: 1 / 32.8, 2000: 1 / 16.4, 4000: 1 / 8.2}


def decode_dp7_timestamp_vector(ts_uint32, frac_uint16):
    year = ((ts_uint32 >> 26) & 0x3F) + 2000
    month = (ts_uint32 >> 22) & 0x0F
    day = (ts_uint32 >> 17) & 0x1F
    hour = (ts_uint32 >> 12) & 0x1F
    minute = (ts_uint32 >> 6) & 0x3F
    second = ts_uint32 & 0x3F
    base_ns = np.datetime64(
        f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    )
    return base_ns + np.timedelta64(int(frac_uint16 * 1e9 / 65536), "ns")


def effective_sample_rate(sample_rate, modifier):
    if modifier == 1:
        return sample_rate
    if modifier > 1:
        return sample_rate / modifier
    if modifier < -1:
        return sample_rate * (-modifier)
    if modifier == 0:
        return 1 / sample_rate
    if modifier == -1:
        return 1 / (sample_rate * 60)
    return sample_rate


def packet_sample_times_vector(ts, frac, ts_offset, sample_rate, modifier, sample_count):
    fs = effective_sample_rate(sample_rate, modifier)
    base_time = decode_dp7_timestamp_vector(ts, frac)
    start_time = base_time - np.timedelta64(int(ts_offset * 1e9 / fs), "ns")
    offsets = np.arange(sample_count) * (1e9 / fs)
    return start_time + offsets.astype("timedelta64[ns]")


def read_dp7_as_dataframe_fast(filename, acc_sens=8, gyro_sens=500, verbose=1):
    acc_factor = ACC_SENSITIVITY[acc_sens]
    gyr_factor = GYRO_SENSITIVITY[gyro_sens]

    acc_list, gyr_list = [], []
    acc_times, gyr_times = [], []
    rates = set()

    with open(filename, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        num_packets = filesize // PACKET_SIZE
        f.seek(0)
        if verbose:
            print(f"Packets: {num_packets}")

        for _ in range(num_packets):
            packet = f.read(PACKET_SIZE)
            if not packet:
                break

            sid = packet[1:2]
            if sid not in (b"a", b"g"):
                continue
            sample_count = struct.unpack_from("<H", packet, 22)[0]
            if sample_count == 0 or sample_count > 80:
                continue

            ts_uint = struct.unpack_from("<I", packet, 8)[0]
            frac = struct.unpack_from("<H", packet, 12)[0]
            ts_offset = struct.unpack_from("<h", packet, 14)[0]
            sample_rate = struct.unpack_from("<H", packet, 16)[0]
            modifier = struct.unpack_from("<b", packet, 18)[0]
            if sample_rate == 0:
                raise ValueError("Invalid zero sample rate in sensor packet")

            rates.add(effective_sample_rate(sample_rate, modifier))
            raw = np.frombuffer(packet[24 : 24 + sample_count * 6], dtype="<i2").reshape(-1, 3)
            times = packet_sample_times_vector(
                ts_uint, frac, ts_offset, sample_rate, modifier, sample_count
            )

            if sid == b"a":
                acc_list.append(raw * acc_factor)
                acc_times.append(times)
            elif sid == b"g":
                gyr_list.append(raw * gyr_factor)
                gyr_times.append(times)

    if not acc_list or not gyr_list:
        raise ValueError("Both accelerometer and gyroscope streams are required.")
    if len(rates) != 1:
        raise ValueError("Sensor streams have inconsistent sampling rates.")
    rate = rates.pop()
    acc = pd.DataFrame(np.vstack(acc_list), columns=["ax", "ay", "az"])
    gyr = pd.DataFrame(np.vstack(gyr_list), columns=["gx", "gy", "gz"])
    acc["datetime"] = pd.to_datetime(np.concatenate(acc_times))
    gyr["gyro_datetime"] = pd.to_datetime(np.concatenate(gyr_times))
    for stream, time_col in ((acc, "datetime"), (gyr, "gyro_datetime")):
        if stream[time_col].duplicated().any() or not stream[time_col].is_monotonic_increasing:
            raise ValueError("Sensor stream timestamps must be strictly increasing.")
    # Quarter-period tolerance accommodates timestamp quantization, without
    # attaching a sample from an adjacent sample interval or a missing packet.
    df = pd.merge_asof(acc, gyr, left_on="datetime", right_on="gyro_datetime",
                       direction="nearest", tolerance=pd.Timedelta(seconds=0.25 / rate))
    df = df.dropna(subset=["gyro_datetime"]).copy()
    if df.empty:
        raise ValueError("Sensor streams have no aligned samples.")
    if df.gyro_datetime.duplicated().any():
        raise ValueError("Sensor alignment would reuse a gyroscope sample.")
    df = df.drop(columns="gyro_datetime").reset_index(drop=True)
    df.attrs["sample_rate_hz"] = rate
    df.attrs["unmatched_acc_samples"] = len(acc) - len(df)
    df.attrs["unmatched_gyr_samples"] = len(gyr) - len(df)

    df_rotated = df.copy()
    df_rotated["ay"] = df["ax"]
    df_rotated["ax"] = -df["ay"]
    df_rotated["gy"] = df["gx"]
    df_rotated["gx"] = -df["gy"]

    return df_rotated

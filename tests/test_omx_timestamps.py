"""Synthetic checks for the read-only OMX timestamp investigation tool."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.inspect_omx_timestamps import HEADER, _times, inspect_file
from dare_wearables.lower_back.io.mcroberts_loader import packet_sample_times_vector


def packet(stream, sequence, *, minute=0, second=1, offset=100, count=80):
    p = bytearray(512)
    p[:2] = b"d" + stream
    struct.pack_into("<H", p, 2, 508)
    struct.pack_into("<I", p, 4, sequence)
    stamp = (24 << 26) | (1 << 22) | (1 << 17) | (minute << 6) | second
    struct.pack_into("<I", p, 8, stamp)
    struct.pack_into("<h", p, 14, offset)
    struct.pack_into("<H", p, 16, 100)
    struct.pack_into("<b", p, 18, 1)
    p[21] = 0x32
    struct.pack_into("<H", p, 22, count)
    checksum = (-int(np.frombuffer(p, dtype="<u2").sum())) % 65536
    struct.pack_into("<H", p, 510, checksum)
    return bytes(p)


class OMXTimestampAuditTests(unittest.TestCase):
    def test_vector_decoder_matches_reader_offsets_and_rate_modifiers(self):
        packets = np.frombuffer(packet(b"a", 0) * 5, dtype=HEADER).copy()
        packets["rate"] = [100, 200, 50, 2, 1]
        packets["modifier"] = [1, 2, -2, 0, -1]
        packets["offset"] = [100, -4, 2810, 2, -1]
        packets["fraction"] = [64880, 1310, 400, 65535, 0]
        headers = {name: packets[name] for name in HEADER.names}
        starts, ends, rates = _times(headers)
        for i, row in enumerate(packets):
            actual = packet_sample_times_vector(int(row["timestamp"]), int(row["fraction"]),
                int(row["offset"]), int(row["rate"]), int(row["modifier"]), int(row["count"]))
            np.testing.assert_array_equal(actual[[0,-1]].astype("datetime64[ns]").astype(np.int64), [starts[i], ends[i]])
        np.testing.assert_array_equal(rates, [100, 100, 100, .5, 1/60])

    def test_detects_bad_anchor_without_modifying_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic.OMX"
            payload = b"".join(packet(sid, seq, minute=minute, second=second, offset=offset)
                               for seq,minute,second,offset in ((0,0,1,100),(1,1,2,100),(2,1,2,20),(3,0,4,140))
                               for sid in (b"a",b"g"))
            path.write_bytes(payload)
            result = inspect_file(path)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(len(result["events"]), 2)
            for stream in result["streams"]:
                self.assertEqual(stream["nonincreasing_boundaries"], 1)
                self.assertEqual(stream["sequence_discontinuities"], 0)
            self.assertTrue(all(row["checksum_valid"] for row in result["packet_context"]))
            for event in result["events"]:
                self.assertEqual(event["sequence_delta"], 1)
                self.assertLess(event["timeline_step_seconds"], -59)

    def test_sequence_restart_and_positive_gap_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic.OMX"
            path.write_bytes(b"".join(packet(sid,seq,second=second) for seq,second in ((4,1),(0,11)) for sid in (b"a",b"g")))
            result = inspect_file(path)
            self.assertEqual(result["events"], [])
            for stream in result["streams"]:
                self.assertEqual(stream["sequence_discontinuities"],1)
                self.assertEqual(stream["sequence_nonincreasing"],1)
                self.assertEqual(stream["forward_steps_over_50us"],1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import sys
import re
from datetime import datetime

TIMESTAMP_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
SLOW_DOWN_RE = re.compile(r'/slow_down')
TS_FMT = "%Y-%m-%d %H:%M:%S"


def parse_log(path, output_len, batch_size):
    slow_down_timestamps = []
    last_ts = None

    with open(path) as f:
        for line in f:
            m = TIMESTAMP_RE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), TS_FMT)
            last_ts = ts
            if SLOW_DOWN_RE.search(line):
                slow_down_timestamps.append(ts)

    if len(slow_down_timestamps) < 2:
        print(f"Error: found {len(slow_down_timestamps)} /slow_down request(s), need at least 2.", file=sys.stderr)
        sys.exit(1)

    second_slow_down = slow_down_timestamps[1]
    decode_time = (last_ts - second_slow_down).total_seconds()

    tpot = decode_time / output_len
    output_throughput = batch_size * output_len / decode_time

    print(f"2nd /slow_down  : {second_slow_down}", file=sys.stderr)
    print(f"Last timestamp  : {last_ts}", file=sys.stderr)
    print(f"Decode time     : {decode_time:.2f} seconds", file=sys.stderr)

    # stdout: two values, one per line (for bash capture)
    print(f"{tpot * 1000:.2f}")          # line 1: tpot (ms/token)
    print(f"{output_throughput:.2f}")  # line 2: output_throughput (tokens/s)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <log_path> <output_len> <batch_size>")
        sys.exit(1)
    parse_log(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

#!/usr/bin/env python3
import sys
import re
from datetime import datetime

TIMESTAMP_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
SLOW_DOWN_RE = re.compile(r'/slow_down')
TS_FMT = "%Y-%m-%d %H:%M:%S"


def parse_log(path):
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
        print(f"Error: found {len(slow_down_timestamps)} /slow_down request(s), need at least 2.")
        sys.exit(1)

    second_slow_down = slow_down_timestamps[1]
    diff = (last_ts - second_slow_down).total_seconds()

    print(f"2nd /slow_down : {second_slow_down}", file=sys.stderr)
    print(f"Last timestamp : {last_ts}", file=sys.stderr)
    print(f"Time diff      : {diff:.0f} seconds", file=sys.stderr)

    print(int(diff))  # return value: only the number to stdout


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <log_path>")
        sys.exit(1)
    parse_log(sys.argv[1])

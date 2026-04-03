#!/usr/bin/env python3
import re
import sys
from datetime import datetime

TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SLOW_DOWN_RE = re.compile(r"/slow_down")
DECODE_BATCH_RE = re.compile(r"Decode batch.*#running-req: ([1-9]\d*),")
TS_FMT = "%Y-%m-%d %H:%M:%S"


def fail(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_log(path: str, output_len: int, batch_size: int) -> None:
    slow_down_timestamps = []
    finish_ts = None

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            match = TIMESTAMP_RE.match(line)
            if not match:
                continue

            ts = datetime.strptime(match.group(1), TS_FMT)
            if SLOW_DOWN_RE.search(line):
                slow_down_timestamps.append(ts)
            if DECODE_BATCH_RE.search(line):
                finish_ts = ts

    if not slow_down_timestamps:
        fail("found no /slow_down requests in decode log")
    if finish_ts is None:
        fail("found no decode batch entries with running requests in decode log")

    stop_slow_down_ts = slow_down_timestamps[-1]
    decode_time = (finish_ts - stop_slow_down_ts).total_seconds()
    if decode_time <= 0:
        fail(
            f"invalid decode_time={decode_time:.2f}s; "
            f"stop_slow_down={stop_slow_down_ts}, finish={finish_ts}"
        )

    tpot_ms = decode_time * 1000.0 / output_len
    output_throughput = batch_size * output_len / decode_time

    print(f"Last /slow_down : {stop_slow_down_ts}", file=sys.stderr)
    print(f"Last timestamp  : {finish_ts}", file=sys.stderr)
    print(f"Decode time     : {decode_time:.2f} seconds", file=sys.stderr)
    print(f"{tpot_ms:.2f}")
    print(f"{output_throughput:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        fail(f"Usage: {sys.argv[0]} <log_path> <output_len> <batch_size>")
    parse_log(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

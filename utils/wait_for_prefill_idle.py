#!/usr/bin/env python3
"""
Wait until all dp ranks on a prefill server have no running or waiting requests.
"""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request

DEFAULT_POLL_INTERVAL = 5
DEFAULT_TIMEOUT = 20 * 60
DEFAULT_REQUEST_TIMEOUT = 30


def fetch_loads(url: str, request_timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:
        return json.loads(resp.read().decode())


def all_idle(data: dict) -> tuple[bool, list[dict]]:
    busy = [
        rank
        for rank in data.get("loads", [])
        if rank.get("num_running_reqs", 0) != 0 or rank.get("num_waiting_reqs", 0) != 0
    ]
    return len(busy) == 0, busy


def main():
    parser = argparse.ArgumentParser(description="Wait until prefill server is idle.")
    parser.add_argument(
        "--prefill_url",
        dest="prefill_url",
        required=True,
        help="Base URL of the prefill server, e.g. http://node04:8000",
    )
    parser.add_argument(
        "--poll-interval",
        dest="poll_interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between polling attempts (default: {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Total timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--request-timeout",
        dest="request_timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})",
    )
    args = parser.parse_args()

    loads_url = args.prefill_url.rstrip("/") + "/v1/loads"
    deadline = time.monotonic() + args.timeout

    print(
        f"Polling {loads_url} every {args.poll_interval}s "
        f"(request-timeout {args.request_timeout}s, total-timeout {args.timeout}s) ..."
    )

    while True:
        try:
            data = fetch_loads(loads_url, args.request_timeout)
            idle, busy_ranks = all_idle(data)

            if idle:
                dp_count = data.get("dp_rank_count", len(data.get("loads", [])))
                print(f"All {dp_count} dp ranks are idle. Done.")
                sys.exit(0)

            busy_summary = ", ".join(
                f"dp_rank={rank.get('dp_rank', 'unknown')} "
                f"running={rank.get('num_running_reqs', 0)} "
                f"waiting={rank.get('num_waiting_reqs', 0)}"
                for rank in sorted(
                    busy_ranks,
                    key=lambda item: (item.get("dp_rank") is None, item.get("dp_rank", -1)),
                )
            )
            print(f"[{time.strftime('%H:%M:%S')}] Still busy: {busy_summary}")

        except (urllib.error.URLError, TimeoutError, socket.timeout, KeyError, TypeError) as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Request failed: {exc}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Bad response: {exc}")

        if time.monotonic() >= deadline:
            print(f"Timeout after {args.timeout} seconds. Server still not idle.")
            sys.exit(1)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

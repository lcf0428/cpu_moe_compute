#!/usr/bin/env python3
"""Codex: hold a temporary host CPU idle latency request for one command.

Example: python3 with_cpu_idle_latency.py --latency-us 0 -- bash run_dspark_pr98_0731.sh
The request is host-wide while open and is removed when this process closes it.
"""
import argparse
import os
import signal
import struct
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latency-us', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not 0 <= args.latency_us <= 2147483647:
        parser.error('--latency-us must be between 0 and 2147483647')
    if not command:
        parser.error('a command is required after --')

    child = None
    fd = None
    received = []
    previous = {}

    def forward(signum, frame):
        # Codex: forward shutdown only to the command group created by this wrapper.
        received.append(signum)
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        # Codex: this descriptor owns a temporary PM QoS request, not a persistent setting.
        fd = os.open('/dev/cpu_dma_latency', os.O_RDWR | os.O_CLOEXEC)
        if os.write(fd, struct.pack('=i', args.latency_us)) != 4:
            raise RuntimeError('incomplete CPU latency request write')
        print(f'[Codex] temporary HOST-WIDE CPU idle latency request: {args.latency_us} us', file=sys.stderr, flush=True)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, forward)
        if received:
            return 128 + received[-1]
        # Codex: the child does not inherit the constraint descriptor; the wrapper owns its lifetime.
        child = subprocess.Popen(command, start_new_session=True, close_fds=True)
        if received:
            forward(received[-1], None)
        code = child.wait()
        return code if code >= 0 else 128 - code
    finally:
        # Codex: clean up only this command's remaining descendants before releasing the constraint.
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if fd is not None:
            os.close(fd)
            print('[Codex] CPU idle latency request released', file=sys.stderr, flush=True)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    sys.exit(main())

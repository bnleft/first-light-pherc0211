"""Verify task-local Unix IPC and retained network/write restrictions."""
import argparse
import errno
import json
import os
from pathlib import Path
import socket
import uuid


def denied(action):
    try:
        action()
    except OSError as error:
        assert error.errno in (errno.EPERM, errno.EACCES), repr(error)
        return error.errno
    raise AssertionError("Sandbox unexpectedly permitted the operation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-ipc", choices=("allowed", "blocked"), required=True)
    args = parser.parse_args()
    scratch = Path("/private/tmp/vesuvius-render-preflight-20261018")
    ipc = scratch / "flat-worker-ipc"
    assert Path(os.environ["TMPDIR"]).resolve() == ipc
    ipc.mkdir(exist_ok=True)
    token = uuid.uuid4().hex[:8]
    report = {}
    for label, address in (("loopback_ip_blocked", "127.0.0.1"), ("external_ip_blocked", "203.0.113.1")):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.5)
            report[label] = denied(lambda: connection.connect((address, 9)))
    outside_socket = scratch / f"outside-ipc-{token}.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            report["unix_socket_outside_ipc_blocked"] = denied(lambda: connection.bind(str(outside_socket)))
    finally:
        outside_socket.unlink(missing_ok=True)
    outside_file = Path(f"/private/tmp/flat-write-negative-{token}")
    try:
        report["file_write_outside_scratch_blocked"] = denied(lambda: outside_file.write_bytes(b""))
    finally:
        if outside_file.exists():
            outside_file.unlink()
    local_socket = ipc / f"probe-{token}.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            if args.expect_ipc == "blocked":
                report["task_unix_ipc_blocked"] = denied(lambda: server.bind(str(local_socket)))
            else:
                server.settimeout(1)
                server.bind(str(local_socket))
                server.listen(1)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(1)
                    client.connect(str(local_socket))
                    with server.accept()[0] as accepted:
                        accepted.settimeout(1)
                        client.sendall(b"task-local")
                        assert accepted.recv(10) == b"task-local"
                report["task_unix_ipc_roundtrip"] = True
    finally:
        local_socket.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""
File: port.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Utilities for checking and freeing WAVES application ports.

License: MIT License
"""

from collections.abc import Iterable
from dataclasses import dataclass
import os
import socket
from typing import Any

import psutil


@dataclass(frozen=True, slots=True)
class FreedProcess:
    """Information about a process that was terminated to free a port."""

    pid: int
    name: str
    port: int


def normalize_ports(ports: int | Iterable[int]) -> set[int]:
    """Normalize a single port or an iterable of ports into a set."""

    return {ports} if isinstance(ports, int) else set(ports)


def is_port_in_use(
    host: str,
    port: int,
    timeout: float = 1.0,
) -> bool:
    """Check whether a TCP port is currently accepting connections."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_connection_port(connection: Any) -> int | None:
    """Extract a local port from a psutil connection object."""

    local_address = getattr(connection, "laddr", None)

    if not local_address:
        return None

    port = getattr(local_address, "port", None)

    if port is not None:
        return int(port)

    if isinstance(local_address, tuple) and len(local_address) >= 2:
        return int(local_address[1])

    return None


def find_processes_by_ports(
    ports: int | Iterable[int],
) -> dict[int, list[psutil.Process]]:
    """Find processes that hold the given local TCP ports."""

    ports_to_find = normalize_ports(ports)
    processes_by_port: dict[int, list[psutil.Process]] = {port: [] for port in ports_to_find}
    current_pid = os.getpid()

    for process in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if process.pid == current_pid:
                continue

            connections = process.net_connections(kind="inet")

            for connection in connections:
                port = get_connection_port(connection)

                if port in ports_to_find:
                    processes_by_port[port].append(process)

        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue

    return {port: processes for port, processes in processes_by_port.items() if processes}


def terminate_process(
    process: psutil.Process,
    timeout: float = 3.0,
    force: bool = True,
) -> bool:
    """Terminate a process gracefully and optionally force-kill it."""

    try:
        process.terminate()
        process.wait(timeout=timeout)

        return True

    except psutil.TimeoutExpired:
        if not force:
            return False

        try:
            process.kill()
            process.wait(timeout=timeout)

            return True

        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.TimeoutExpired,
        ):
            return False

    except (
        psutil.AccessDenied,
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
    ):
        return False


def free_ports(
    ports: int | Iterable[int],
    timeout: float = 3.0,
    force: bool = True,
) -> list[FreedProcess]:
    """Free one or more TCP ports by terminating processes that hold them."""

    freed_processes: list[FreedProcess] = []

    for port, processes in find_processes_by_ports(ports).items():
        for process in processes:
            try:
                process_name = process.name()
                process_pid = process.pid

            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
            ):
                continue

            if terminate_process(
                process,
                timeout=timeout,
                force=force,
            ):
                freed_processes.append(
                    FreedProcess(
                        pid=process_pid,
                        name=process_name,
                        port=port,
                    )
                )

    return freed_processes


def ensure_port_available(
    host: str,
    port: int,
    timeout: float = 3.0,
    force: bool = True,
) -> list[FreedProcess]:
    """Ensure that the given port is available before starting WAVES."""

    if not is_port_in_use(host, port):
        return []

    return free_ports(
        port,
        timeout=timeout,
        force=force,
    )

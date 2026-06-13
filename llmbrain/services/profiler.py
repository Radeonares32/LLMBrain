"""Operation profiling layer with wall-time and memory delta tracking."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProfileEntry(BaseModel):
    """Record of a single profiled operation."""

    operation: str
    duration_ms: float
    memory_delta_mb: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfileReport(BaseModel):
    """Aggregated view over all recorded :class:`ProfileEntry` items."""

    entries: list[ProfileEntry]
    total_operations: int
    avg_duration_ms: float
    peak_memory_delta_mb: float
    total_duration_ms: float


# ---------------------------------------------------------------------------
# /proc/self/status helper
# ---------------------------------------------------------------------------


def _vmrss_kb() -> int:
    """Read the current process VmRSS (resident set size) from ``/proc/self/status``.

    Returns ``0`` if the file is unavailable (non-Linux environments).
    """
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return 0
    try:
        with status_path.open(encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class OperationProfiler:
    """Records wall-time durations and memory deltas for named operations.

    Entries are stored in an internal ring buffer capped at ``max_entries``
    items.  Overflow discards the oldest records.

    Usage — context manager::

        profiler = OperationProfiler()

        with profiler.profile("scan_project", metadata={"files": 42}) as entry:
            do_work()
        # entry is now populated with timing/memory data

    Usage — manual::

        entry = profiler.record("my_op", duration_ms=12.5, memory_delta_mb=0.3)
    """

    def __init__(self, max_entries: int = 1000) -> None:
        """Initialise with a fixed-size ring buffer.

        Parameters
        ----------
        max_entries:
            Maximum number of :class:`ProfileEntry` items to retain.
            Oldest entries are dropped when the limit is exceeded.
        """
        self._max_entries = max_entries
        self._entries: deque[ProfileEntry] = deque(maxlen=max_entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        operation: str,
        duration_ms: float,
        memory_delta_mb: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ProfileEntry:
        """Manually record a profiling measurement.

        Parameters
        ----------
        operation:
            Human-readable name for the operation.
        duration_ms:
            Wall-time duration in milliseconds.
        memory_delta_mb:
            Change in resident memory (positive = increase).
        metadata:
            Optional freeform key/value context.

        Returns
        -------
        ProfileEntry
            The newly created and stored entry.
        """
        entry = ProfileEntry(
            operation=operation,
            duration_ms=duration_ms,
            memory_delta_mb=memory_delta_mb,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        return entry

    @contextmanager
    def profile(
        self,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[ProfileEntry, None, None]:
        """Context manager that measures wall-time and memory delta.

        The yielded :class:`ProfileEntry` is a *stub* with placeholder values
        on entry.  On exit, ``duration_ms`` and ``memory_delta_mb`` are set
        and the entry is added to the internal ring buffer.

        Parameters
        ----------
        operation:
            Human-readable name for the operation.
        metadata:
            Optional freeform key/value context attached to the entry.

        Yields
        ------
        ProfileEntry
            The (initially empty) entry that will be completed on exit.

        Example
        -------
        ::

            with profiler.profile("extract_facts") as entry:
                facts = run_extraction()
            print(entry.duration_ms)
        """
        stub = ProfileEntry(
            operation=operation,
            duration_ms=0.0,
            memory_delta_mb=0.0,
            metadata=metadata or {},
        )
        mem_before_kb = _vmrss_kb()
        t_start = time.perf_counter()
        try:
            yield stub
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            mem_after_kb = _vmrss_kb()
            delta_mb = (mem_after_kb - mem_before_kb) / 1024.0

            # Mutate the stub so callers that captured the reference see
            # the final values.
            stub.duration_ms = round(elapsed_ms, 3)
            stub.memory_delta_mb = round(delta_mb, 4)
            self._entries.append(stub)

    def get_report(self) -> ProfileReport:
        """Build and return an aggregate :class:`ProfileReport`.

        Returns a report with zero-valued aggregates when no entries exist.
        """
        entries = list(self._entries)
        n = len(entries)
        if n == 0:
            return ProfileReport(
                entries=[],
                total_operations=0,
                avg_duration_ms=0.0,
                peak_memory_delta_mb=0.0,
                total_duration_ms=0.0,
            )
        total_ms = sum(e.duration_ms for e in entries)
        peak_mem = max(e.memory_delta_mb for e in entries)
        return ProfileReport(
            entries=entries,
            total_operations=n,
            avg_duration_ms=round(total_ms / n, 3),
            peak_memory_delta_mb=round(peak_mem, 4),
            total_duration_ms=round(total_ms, 3),
        )

    def get_slowest(self, n: int = 10) -> list[ProfileEntry]:
        """Return the *n* slowest recorded entries, longest first.

        Parameters
        ----------
        n:
            Number of entries to return.
        """
        return sorted(self._entries, key=lambda e: e.duration_ms, reverse=True)[:n]

    def clear(self) -> None:
        """Discard all stored profile entries."""
        self._entries.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return the full report serialised as a plain :class:`dict`.

        Suitable for JSON serialisation or CLI display.
        """
        return self.get_report().model_dump(mode="json")


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------

#: Global profiler instance shared across the application.
#: Import and use it directly, or pass it explicitly for test isolation.
default_profiler: OperationProfiler = OperationProfiler()

"""CPU and memory resource manager for adaptive worker concurrency."""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ResourceSnapshot(BaseModel):
    """A single point-in-time resource measurement."""

    cpu_percent: float
    memory_percent: float
    memory_mb: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResourcePolicy(BaseModel):
    """Thresholds and limits that govern adaptive concurrency decisions."""

    max_workers: int = 4
    cpu_threshold_high: float = 80.0
    cpu_threshold_low: float = 40.0
    mem_threshold_high: float = 85.0
    mem_threshold_low: float = 50.0
    min_workers: int = 1
    sample_interval_sec: float = 2.0


class ResourceState(StrEnum):
    """Overall system health classification."""

    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ResourceManager:
    """Samples CPU/memory from ``/proc`` and recommends worker concurrency.

    All reads use standard library :mod:`os` / :mod:`pathlib` only — no
    external dependencies (``psutil`` is intentionally not used).
    """

    _HISTORY_SIZE = 60

    def __init__(self, policy: ResourcePolicy | None = None) -> None:
        """Initialise with an optional custom :class:`ResourcePolicy`.

        Parameters
        ----------
        policy:
            Concurrency and threshold configuration.  Defaults to
            :class:`ResourcePolicy` with factory values.
        """
        self._policy = policy or ResourcePolicy()
        self._snapshots: deque[ResourceSnapshot] = deque(maxlen=self._HISTORY_SIZE)
        # Track previous /proc/stat counters for incremental CPU calculation.
        self._prev_cpu_stats: tuple[int, int] | None = None  # (total, idle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def snapshots(self) -> deque[ResourceSnapshot]:
        """Last up-to-60 :class:`ResourceSnapshot` measurements."""
        return self._snapshots

    @property
    def policy(self) -> ResourcePolicy:
        """Active :class:`ResourcePolicy`."""
        return self._policy

    def sample(self) -> ResourceSnapshot:
        """Read current CPU %, memory %, and available memory from ``/proc``.

        Falls back to ``0.0`` for any metric that cannot be read (e.g. non-
        Linux environments).

        Returns
        -------
        ResourceSnapshot
            The captured measurement (also appended to :attr:`snapshots`).
        """
        cpu = self._parse_cpu_percent()
        mem_pct = self._parse_mem_percent()
        mem_mb = self._parse_mem_mb()
        snap = ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=mem_pct,
            memory_mb=mem_mb,
        )
        self._snapshots.append(snap)
        return snap

    def get_state(self) -> ResourceState:
        """Classify current system health based on the latest sample.

        If no snapshot has been taken yet, one is collected first.

        Returns
        -------
        ResourceState
            :attr:`ResourceState.CRITICAL` if either CPU or memory exceeds the
            high threshold; :attr:`ResourceState.DEGRADED` if either is above
            the *low* threshold; otherwise :attr:`ResourceState.NORMAL`.
        """
        snap = self._latest_or_sample()
        p = self._policy
        if snap.cpu_percent >= p.cpu_threshold_high or snap.memory_percent >= p.mem_threshold_high:
            return ResourceState.CRITICAL
        if snap.cpu_percent >= p.cpu_threshold_low or snap.memory_percent >= p.mem_threshold_low:
            return ResourceState.DEGRADED
        return ResourceState.NORMAL

    def recommended_workers(self) -> int:
        """Return the suggested number of concurrent workers.

        The recommendation scales linearly between ``policy.min_workers`` and
        ``policy.max_workers`` based on inverse CPU load.

        * CRITICAL state → ``min_workers``
        * NORMAL state   → ``max_workers``
        * DEGRADED state → interpolated midpoint
        """
        state = self.get_state()
        p = self._policy
        if state == ResourceState.CRITICAL:
            return p.min_workers
        if state == ResourceState.NORMAL:
            return p.max_workers

        # DEGRADED: linear interpolation based on CPU %
        snap = self._latest_or_sample()
        cpu = snap.cpu_percent
        # Map cpu_threshold_low..cpu_threshold_high → max_workers..min_workers
        span_cpu = p.cpu_threshold_high - p.cpu_threshold_low
        span_workers = p.max_workers - p.min_workers
        if span_cpu <= 0 or span_workers <= 0:
            return p.min_workers

        ratio = (cpu - p.cpu_threshold_low) / span_cpu
        ratio = max(0.0, min(1.0, ratio))
        workers = round(p.max_workers - ratio * span_workers)
        return max(p.min_workers, min(p.max_workers, workers))

    def get_stats(self) -> dict[str, object]:
        """Return a summary dict suitable for API/CLI display.

        Returns
        -------
        dict
            Keys: ``avg_cpu``, ``avg_mem``, ``state``, ``recommended_workers``,
            ``snapshot_count``.
        """
        snaps = list(self._snapshots)
        count = len(snaps)
        avg_cpu = round(sum(s.cpu_percent for s in snaps) / count, 2) if count else 0.0
        avg_mem = round(sum(s.memory_percent for s in snaps) / count, 2) if count else 0.0
        return {
            "avg_cpu": avg_cpu,
            "avg_mem": avg_mem,
            "state": self.get_state().value,
            "recommended_workers": self.recommended_workers(),
            "snapshot_count": count,
        }

    # ------------------------------------------------------------------
    # /proc readers
    # ------------------------------------------------------------------

    def _parse_cpu_percent(self) -> float:
        """Compute CPU utilisation using ``/proc/stat`` delta measurement.

        On the very first call (no previous reading) the method sleeps for
        a short interval to collect a meaningful diff.

        Returns ``0.0`` on any read error.
        """
        proc_stat = Path("/proc/stat")
        if not proc_stat.exists():
            return 0.0

        try:
            total1, idle1 = self._read_proc_stat_totals()
        except Exception:
            return 0.0

        if self._prev_cpu_stats is None:
            # First sample: sleep briefly to get a real delta.
            time.sleep(0.1)
            try:
                total2, idle2 = self._read_proc_stat_totals()
            except Exception:
                return 0.0
        else:
            total2, idle2 = self._prev_cpu_stats
            total1_prev = total1
            idle1_prev = idle1
            # swap: total2/idle2 are from the *previous* stored reading
            total1, idle1 = total2, idle2
            try:
                total2, idle2 = self._read_proc_stat_totals()
            except Exception:
                return 0.0
            _ = total1_prev  # silence unused warning
            _ = idle1_prev

        self._prev_cpu_stats = (total2, idle2)

        delta_total = total2 - total1
        delta_idle = idle2 - idle1
        if delta_total <= 0:
            return 0.0
        cpu_used = delta_total - delta_idle
        return round((cpu_used / delta_total) * 100.0, 2)

    @staticmethod
    def _read_proc_stat_totals() -> tuple[int, int]:
        """Read the first ``cpu`` line of ``/proc/stat`` and return
        (total_jiffies, idle_jiffies).

        Raises :class:`OSError` or :class:`ValueError` on parse failure.
        """
        with Path("/proc/stat").open(encoding="ascii") as fh:
            for line in fh:
                if line.startswith("cpu "):
                    parts = line.split()
                    # Fields: user nice system idle iowait irq softirq steal guest guest_nice
                    values = [int(p) for p in parts[1:]]
                    total = sum(values)
                    idle = values[3] if len(values) > 3 else 0
                    # iowait also counts as idle time for our purposes
                    if len(values) > 4:
                        idle += values[4]
                    return total, idle
        raise OSError("/proc/stat: no 'cpu' line found")

    @staticmethod
    def _parse_mem_percent() -> float:
        """Compute memory usage percent from ``/proc/meminfo``.

        Returns ``0.0`` on any read error.
        """
        try:
            info = ResourceManager._read_meminfo()
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", info.get("MemFree", 0))
            if total <= 0:
                return 0.0
            used = total - available
            return round((used / total) * 100.0, 2)
        except Exception:
            return 0.0

    @staticmethod
    def _parse_mem_mb() -> float:
        """Return available memory in megabytes from ``/proc/meminfo``.

        Returns ``0.0`` on any read error.
        """
        try:
            info = ResourceManager._read_meminfo()
            available_kb = info.get("MemAvailable", info.get("MemFree", 0))
            return round(available_kb / 1024.0, 2)
        except Exception:
            return 0.0

    @staticmethod
    def _read_meminfo() -> dict[str, int]:
        """Parse ``/proc/meminfo`` into a mapping of field name → kB value.

        Raises :class:`OSError` if the file is not accessible.
        """
        result: dict[str, int] = {}
        with Path("/proc/meminfo").open(encoding="ascii") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, _, rest = line.partition(":")
                try:
                    result[key.strip()] = int(rest.split()[0])
                except (ValueError, IndexError):
                    pass
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _latest_or_sample(self) -> ResourceSnapshot:
        """Return the most recent snapshot, collecting one if none exists."""
        if self._snapshots:
            return self._snapshots[-1]
        return self.sample()


# ---------------------------------------------------------------------------
# Module-level sentinel to verify importability — no global singleton here
# (callers create their own ResourceManager instances).
# ---------------------------------------------------------------------------

_PROC_STAT_AVAILABLE: bool = os.path.exists("/proc/stat")

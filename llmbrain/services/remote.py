"""Remote API observability layer — health checking and connection state tracking."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────


class ConnectionState(StrEnum):
    """Overall connectivity state of a remote service."""

    CONNECTED = "connected"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


# ── Value objects ─────────────────────────────────────────────────────────────


class ApiVersion(BaseModel):
    """Semantic version triple for a remote API."""

    major: int
    minor: int
    patch: int

    @property
    def version_string(self) -> str:
        """Return dot-separated version string, e.g. '1.2.3'."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, s: str) -> ApiVersion:
        """Parse a version string of the form 'MAJOR.MINOR.PATCH'.

        Raises:
            ValueError: If the string cannot be parsed as a valid semver triple.
        """
        parts = s.strip().lstrip("v").split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid version string {s!r}: expected 'MAJOR.MINOR.PATCH'."
            )
        try:
            return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))
        except ValueError as exc:
            raise ValueError(f"Non-integer component in version string {s!r}: {exc}") from exc


class ServiceEndpoint(BaseModel):
    """Configuration for a single remote service endpoint."""

    name: str = Field(..., description="Unique name identifying this service.")
    base_url: str = Field(
        ..., description="Base URL without trailing slash, e.g. 'http://localhost:8080'."
    )
    api_version: ApiVersion | None = Field(
        default=None, description="Expected API version, if known."
    )
    timeout_sec: float = Field(default=5.0, gt=0, description="Per-request timeout in seconds.")
    enabled: bool = Field(
        default=True, description="Whether health checks are active for this endpoint."
    )


class HealthStatus(BaseModel):
    """Result of a health check against a single service endpoint."""

    service_name: str
    state: ConnectionState
    latency_ms: float | None = None
    last_checked: datetime | None = None
    error: str | None = None
    version: ApiVersion | None = None


# ── Monitor ───────────────────────────────────────────────────────────────────


class RemoteServiceMonitor:
    """Manages health-check polling for a collection of remote service endpoints.

    All network I/O is performed asynchronously via :mod:`httpx`.  No real
    network calls are required for the monitor to function — every probe is
    independently timeout-guarded and degrades gracefully on failure.
    """

    def __init__(self, endpoints: list[ServiceEndpoint] | None = None) -> None:
        self._endpoints: dict[str, ServiceEndpoint] = {}
        self._cache: dict[str, HealthStatus] = {}
        self._lock = asyncio.Lock()

        for ep in endpoints or []:
            self._endpoints[ep.name] = ep

    # ── endpoint management ───────────────────────────────────────────────────

    def add_endpoint(self, endpoint: ServiceEndpoint) -> None:
        """Register a new service endpoint for monitoring."""
        self._endpoints[endpoint.name] = endpoint
        logger.debug("RemoteServiceMonitor: registered endpoint '%s'.", endpoint.name)

    def remove_endpoint(self, name: str) -> None:
        """Remove a service endpoint by name (no-op if not registered)."""
        self._endpoints.pop(name, None)
        self._cache.pop(name, None)
        logger.debug("RemoteServiceMonitor: removed endpoint '%s'.", name)

    # ── health probing ────────────────────────────────────────────────────────

    async def check_endpoint(self, endpoint: ServiceEndpoint) -> HealthStatus:
        """Probe a single endpoint's /health route and return a :class:`HealthStatus`.

        The probe performs a GET to ``<base_url>/health`` with the configured
        timeout.  Any network or HTTP error is caught and mapped to
        :attr:`ConnectionState.OFFLINE` or :attr:`ConnectionState.DEGRADED`.

        Args:
            endpoint: The endpoint configuration to probe.

        Returns:
            A :class:`HealthStatus` reflecting the current connectivity.
        """
        if not endpoint.enabled:
            return HealthStatus(
                service_name=endpoint.name,
                state=ConnectionState.UNKNOWN,
                last_checked=datetime.now(UTC),
                error="Endpoint is disabled.",
            )

        url = endpoint.base_url.rstrip("/") + "/health"
        started = asyncio.get_event_loop().time()

        try:
            async with httpx.AsyncClient(timeout=endpoint.timeout_sec) as client:
                response = await client.get(url)
            elapsed_ms = (asyncio.get_event_loop().time() - started) * 1000.0

            # Parse optional version from response body
            api_version: ApiVersion | None = None
            with _Suppress():
                body: Any = response.json()
                raw_ver = body.get("version") if isinstance(body, dict) else None
                if isinstance(raw_ver, str):
                    api_version = ApiVersion.from_string(raw_ver)

            if response.is_success:
                state = ConnectionState.CONNECTED
                error: str | None = None
            else:
                state = ConnectionState.DEGRADED
                error = f"HTTP {response.status_code}"

            status = HealthStatus(
                service_name=endpoint.name,
                state=state,
                latency_ms=round(elapsed_ms, 2),
                last_checked=datetime.now(UTC),
                error=error,
                version=api_version,
            )

        except httpx.TimeoutException as exc:
            status = HealthStatus(
                service_name=endpoint.name,
                state=ConnectionState.OFFLINE,
                last_checked=datetime.now(UTC),
                error=f"Timeout after {endpoint.timeout_sec}s: {exc}",
            )

        except httpx.ConnectError as exc:
            status = HealthStatus(
                service_name=endpoint.name,
                state=ConnectionState.OFFLINE,
                last_checked=datetime.now(UTC),
                error=f"Connection refused: {exc}",
            )

        except Exception as exc:  # noqa: BLE001
            status = HealthStatus(
                service_name=endpoint.name,
                state=ConnectionState.OFFLINE,
                last_checked=datetime.now(UTC),
                error=f"Unexpected error: {type(exc).__name__}: {exc}",
            )

        logger.debug(
            "RemoteServiceMonitor: %s -> %s (%.1f ms)",
            endpoint.name,
            status.state.value,
            status.latency_ms or 0.0,
        )

        async with self._lock:
            self._cache[endpoint.name] = status

        return status

    async def check_all(self) -> list[HealthStatus]:
        """Probe all registered endpoints concurrently.

        Returns:
            A list of :class:`HealthStatus` objects, one per registered endpoint.
        """
        if not self._endpoints:
            return []

        tasks = [
            asyncio.create_task(self.check_endpoint(ep), name=f"probe-{name}")
            for name, ep in self._endpoints.items()
        ]
        results: list[HealthStatus] = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def check_single(self, name: str) -> HealthStatus | None:
        """Probe a specific endpoint by name.

        Args:
            name: The registered endpoint name.

        Returns:
            A :class:`HealthStatus`, or ``None`` if the name is not registered.
        """
        endpoint = self._endpoints.get(name)
        if endpoint is None:
            logger.warning("RemoteServiceMonitor: unknown endpoint '%s'.", name)
            return None
        return await self.check_endpoint(endpoint)

    # ── cached state accessors ────────────────────────────────────────────────

    def get_cached_status(self) -> list[HealthStatus]:
        """Return the most recent cached health status for all endpoints.

        Returns an empty list if no checks have been performed yet.
        """
        return list(self._cache.values())

    def get_overall_state(self) -> ConnectionState:
        """Aggregate all cached statuses into a single :class:`ConnectionState`.

        Rules:
        - ``UNKNOWN``   — no checks have been performed yet.
        - ``CONNECTED`` — every enabled endpoint is ``CONNECTED``.
        - ``OFFLINE``   — every enabled endpoint is ``OFFLINE``.
        - ``DEGRADED``  — a mix of states.
        """
        statuses = self.get_cached_status()
        if not statuses:
            return ConnectionState.UNKNOWN

        states = {s.state for s in statuses}

        if states == {ConnectionState.CONNECTED}:
            return ConnectionState.CONNECTED
        if states == {ConnectionState.OFFLINE}:
            return ConnectionState.OFFLINE
        return ConnectionState.DEGRADED

    def is_any_connected(self) -> bool:
        """Return ``True`` if at least one endpoint is in the CONNECTED state."""
        return any(s.state is ConnectionState.CONNECTED for s in self.get_cached_status())


# ── context manager helper ────────────────────────────────────────────────────


class _Suppress:
    """Tiny inline suppress() replacement to avoid importing contextlib."""

    def __enter__(self) -> _Suppress:
        return self

    def __exit__(self, *_: object) -> bool:
        return True


# ── Factory ───────────────────────────────────────────────────────────────────


def create_default_monitor() -> RemoteServiceMonitor:
    """Create an empty :class:`RemoteServiceMonitor` with no endpoints pre-registered.

    Callers add endpoints via :meth:`RemoteServiceMonitor.add_endpoint` before
    calling :meth:`RemoteServiceMonitor.check_all`.

    Returns:
        A fresh :class:`RemoteServiceMonitor` instance.
    """
    return RemoteServiceMonitor(endpoints=[])

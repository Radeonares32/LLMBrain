"""Async indexing scheduler — bounded worker pool with resource-adaptive concurrency."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from pydantic import BaseModel

from llmbrain.core.queue import IndexJob, IndexQueue, JobStatus
from llmbrain.core.resource_manager import ResourceManager, ResourcePolicy
from llmbrain.services.profiler import default_profiler


class WorkerResult(BaseModel):
    """Result of a single job execution."""

    job_id: str
    success: bool
    duration_ms: float
    error: str | None = None


class IndexScheduler:
    """Bounded async scheduler that drains the IndexQueue respecting resource limits."""

    def __init__(
        self,
        queue: IndexQueue,
        resource_manager: ResourceManager | None = None,
        max_workers: int = 4,
    ) -> None:
        self.queue = queue
        self.resource_manager = resource_manager or ResourceManager(
            ResourcePolicy(max_workers=max_workers)
        )
        self._max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
        self._running_jobs: dict[str, asyncio.Task] = {}
        self._is_running = False
        self._poll_task: asyncio.Task | None = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._is_running:
            return
        self._is_running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Gracefully stop the scheduler and wait for running jobs."""
        self._is_running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        # Wait for in-flight jobs
        if self._running_jobs:
            await asyncio.gather(*self._running_jobs.values(), return_exceptions=True)

    # ── poll loop ─────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Continuously poll the queue and dispatch jobs up to the worker limit."""
        while self._is_running:
            try:
                # Sample resources and determine allowed concurrency
                self.resource_manager.sample()
                allowed = self.resource_manager.recommended_workers()

                # Dispatch up to allowed new jobs if under limit
                while len(self._running_jobs) < allowed:
                    job = await asyncio.get_event_loop().run_in_executor(
                        None, self.queue.dequeue_next
                    )
                    if job is None:
                        break  # Queue empty
                    task = asyncio.create_task(self._execute_job(job))
                    self._running_jobs[job.id] = task

                # Clean up finished tasks
                done_ids = [jid for jid, t in self._running_jobs.items() if t.done()]
                for jid in done_ids:
                    del self._running_jobs[jid]

            except Exception:
                pass  # Scheduler must never crash

            await asyncio.sleep(0.5)

    # ── job execution ─────────────────────────────────────────────────

    async def _execute_job(self, job: IndexJob) -> WorkerResult:
        """Execute a single job, update its status, and record a profile entry."""
        async with self._semaphore:
            started = time.monotonic()
            now_str = datetime.now(UTC).isoformat()

            # Mark as running
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.queue.update_job(job.id, status=JobStatus.RUNNING, started_at=now_str),
            )

            try:
                with default_profiler.profile(
                    f"scheduler.job.{job.job_type}", metadata={"job_id": job.id}
                ):
                    await self._dispatch_job(job)

                duration_ms = (time.monotonic() - started) * 1000
                completed_str = datetime.now(UTC).isoformat()
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.queue.update_job(
                        job.id,
                        status=JobStatus.COMPLETED,
                        completed_at=completed_str,
                        progress=1.0,
                    ),
                )
                return WorkerResult(job_id=job.id, success=True, duration_ms=duration_ms)

            except Exception as exc:
                duration_ms = (time.monotonic() - started) * 1000
                err_str = str(exc)
                failed_str = datetime.now(UTC).isoformat()

                # Decide: retry or fail
                if job.retry_count < job.max_retries:
                    next_status = JobStatus.RETRYING
                else:
                    next_status = JobStatus.FAILED

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.queue.update_job(
                        job.id,
                        status=next_status,
                        completed_at=failed_str,
                        error=err_str,
                    ),
                )
                return WorkerResult(
                    job_id=job.id, success=False, duration_ms=duration_ms, error=err_str
                )

    async def _dispatch_job(self, job: IndexJob) -> None:
        """Route a job to the appropriate handler by type."""
        # Simulate async work — actual handlers injected by callers or overridden
        handler = _JOB_HANDLERS.get(job.job_type)
        if handler is not None:
            await handler(job)
        else:
            # Default: no-op simulation
            await asyncio.sleep(0.01)

    # ── properties ────────────────────────────────────────────────────

    @property
    def running_jobs(self) -> int:
        return len(self._running_jobs)

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_stats(self) -> dict:
        """Return current scheduler statistics."""
        rm_stats = self.resource_manager.get_stats()
        return {
            "running_jobs": self.running_jobs,
            "is_running": self._is_running,
            "resource_state": rm_stats.get("state", "unknown"),
            "recommended_workers": rm_stats.get("recommended_workers", self._max_workers),
            "avg_cpu_percent": rm_stats.get("avg_cpu", 0.0),
            "avg_mem_percent": rm_stats.get("avg_mem", 0.0),
        }


# ── Default job handlers (no-ops; replaced in tests and production) ──

_JOB_HANDLERS: dict[str, object] = {}


def register_job_handler(job_type: str, handler) -> None:  # type: ignore[type-arg]
    """Register an async handler for a job type."""
    _JOB_HANDLERS[job_type] = handler

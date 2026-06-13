"""Workspace mutation lock to prevent concurrent modifications."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class WorkspaceLock:
    """Controls exclusive write access to a project workspace."""

    def __init__(self, project_id: str, lock_dir: Path) -> None:
        self.project_id = project_id
        self.lock_dir = Path(lock_dir)
        self.lock_file = self.lock_dir / "workspace.lock"

    def acquire(self, task_id: str, session_id: str, timeout_seconds: float = 10.0) -> bool:
        """Acquire exclusive lock, blocking up to timeout_seconds."""
        lock_data = {
            "lock_id": f"lock_{int(time.time())}",
            "project_id": self.project_id,
            "task_id": task_id,
            "session_id": session_id,
            "process_id": os.getpid(),
            "acquisition_time": time.time(),
            "heartbeat": time.time(),
        }

        start_time = time.time()
        while True:
            if not self.lock_file.exists():
                try:
                    # Write to file
                    self.lock_file.write_text(json.dumps(lock_data), encoding="utf-8")
                    return True
                except Exception:
                    pass
            else:
                try:
                    content = self.lock_file.read_text(encoding="utf-8")
                    data = json.loads(content)

                    # Re-acquire if already owned by this process and task
                    if data.get("process_id") == os.getpid() and data.get("task_id") == task_id:
                        data["heartbeat"] = time.time()
                        self.lock_file.write_text(json.dumps(data), encoding="utf-8")
                        return True

                    # Check process viability to detect dead lock holders
                    pid = data.get("process_id")
                    if pid and not self._process_exists(pid):
                        # Owner process has died, safe to take over lock
                        self.lock_file.unlink(missing_ok=True)
                        continue
                except Exception:
                    # Corrupted lock file or race conditions, remove
                    self.lock_file.unlink(missing_ok=True)
                    continue

            if time.time() - start_time >= timeout_seconds:
                return False
            time.sleep(0.1)

    def release(self, task_id: str) -> None:
        """Release the workspace lock if owned by the caller."""
        if self.lock_file.exists():
            try:
                data = json.loads(self.lock_file.read_text(encoding="utf-8"))
                if data.get("task_id") == task_id or data.get("process_id") == os.getpid():
                    self.lock_file.unlink(missing_ok=True)
            except Exception:
                self.lock_file.unlink(missing_ok=True)

    def _process_exists(self, pid: int) -> bool:
        """Verify if the process with the given PID is still active."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

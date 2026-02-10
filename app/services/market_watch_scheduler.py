"""Background scheduler for autonomous market-watch runs."""

from __future__ import annotations

import logging
import threading
from typing import Callable


class MarketWatchScheduler:
    """In-process periodic runner used in local/self-hosted mode."""

    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        interval_hours: int,
        run_job: Callable[[], None],
        logger: logging.Logger,
    ) -> None:
        self.enabled = enabled
        self.mode = mode.strip().lower() if mode else "internal"
        self.interval_seconds = max(1, int(interval_hours)) * 3600
        self.run_job = run_job
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()

    @property
    def is_internal_mode(self) -> bool:
        """Return True when scheduler is responsible for autonomous execution."""

        return self.mode == "internal"

    def start(self) -> bool:
        """Start background loop when enabled and in internal mode."""

        if not self.enabled:
            self.logger.info("market_watch scheduler not started: disabled")
            return False
        if not self.is_internal_mode:
            self.logger.info("market_watch scheduler not started: mode=%s", self.mode)
            return False
        if self._thread and self._thread.is_alive():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="market-watch-scheduler",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("market_watch scheduler started (interval_hours=%s)", self.interval_seconds // 3600)
        return True

    def stop(self) -> None:
        """Request scheduler stop and wait briefly for thread shutdown."""

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        """Run one job on startup, then continue periodically."""

        self._safe_run_once()
        while not self._stop_event.wait(self.interval_seconds):
            self._safe_run_once()

    def _safe_run_once(self) -> None:
        """Execute scheduled job with overlap protection and fault isolation."""

        if not self._run_lock.acquire(blocking=False):
            self.logger.warning("market_watch scheduler skipped overlapping run")
            return

        try:
            self.run_job()
        except Exception:  # pragma: no cover - defensive background job guard
            self.logger.exception("market_watch scheduler run failed")
        finally:
            self._run_lock.release()

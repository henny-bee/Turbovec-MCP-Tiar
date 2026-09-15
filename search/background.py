"""Background relationship-discovery daemon.

Runs :meth:`semantic_radar` on every project at a fixed interval. The worker
waits on an :class:`threading.Event` rather than polling in one-second sleeps,
so a stop request is honoured immediately instead of up to a second later.
"""

from __future__ import annotations

import logging
import threading
from typing import List

logger = logging.getLogger(__name__)

__all__ = ["BackgroundDiscoveryMixin"]

_JOIN_TIMEOUT = 5.0


class BackgroundDiscoveryMixin:
    """Daemon-thread half of :class:`~core.database.VectorDB`."""

    def start_background_discovery(self) -> None:
        """Starts the discovery daemon if it is not already running."""
        if self.bg_thread is not None and self.bg_thread.is_alive():
            logger.info("Background relationship discovery thread is already running.")
            return
        self.bg_stop_event.clear()
        self.bg_thread = threading.Thread(
            target=self._background_discovery_loop,
            name="turbovec-discovery",
            daemon=True,
        )
        self.bg_thread.start()
        logger.info("Background relationship discovery thread started.")

    def stop_background_discovery(self) -> None:
        """Signals the daemon to stop and waits briefly for it to finish."""
        if self.bg_thread is None:
            return
        self.bg_stop_event.set()
        self.bg_thread.join(timeout=_JOIN_TIMEOUT)
        if self.bg_thread.is_alive():
            logger.warning("Background discovery thread did not stop within timeout.")
        self.bg_thread = None
        logger.info("Background relationship discovery thread stopped.")

    # -- worker ------------------------------------------------------------
    def _background_discovery_loop(self) -> None:
        while not self.bg_stop_event.wait(self.bg_interval):
            try:
                self._run_discovery_pass()
            except Exception as exc:
                logger.error(
                    f"Error in background relationship discovery loop: {exc}",
                    exc_info=True,
                )

    def _run_discovery_pass(self) -> None:
        logger.info("Triggering background relationship discovery scan...")
        for project_id in self._discoverable_projects():
            if self.bg_stop_event.is_set():
                return
            logger.info(
                f"Scanning project '{project_id}' for relationship discovery..."
            )
            results = self.semantic_radar(
                similarity_threshold=self.bg_similarity_threshold,
                project_id=project_id,
                auto_create=True,
            )
            created = sum(1 for item in results if item.get("created_edge"))
            if created:
                logger.info(
                    f"Background discovery successfully created {created} edges "
                    f"in project '{project_id}'."
                )

    def _discoverable_projects(self) -> List[str]:
        try:
            projects = [
                row[0]
                for row in self.conn.execute(
                    "SELECT DISTINCT project_id FROM nodes"
                ).fetchall()
            ]
        except Exception as exc:
            logger.error(f"Failed to query projects for background discovery: {exc}")
            projects = []
        return projects or [self.settings.default_project_id]

"""Enterprise event backbone package (Week 10, Phase 3).

Exposes an in-process, deterministic, thread-safe event bus that serves as the
central communication layer for every platform component.
"""

from __future__ import annotations

from .enterprise_event_bus import *  # noqa: F401,F403
from .enterprise_event_bus import __all__  # noqa: F401
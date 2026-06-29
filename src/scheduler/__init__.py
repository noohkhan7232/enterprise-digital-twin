"""Enterprise scheduling & automation package (Week 10, Phase 4).

Exposes a deterministic, tick-driven scheduler that automates time-based,
event-based, and condition-based jobs across the platform and integrates with
the Enterprise Event Bus by composition.
"""

from __future__ import annotations

from .enterprise_scheduler import *  # noqa: F401,F403
from .enterprise_scheduler import __all__  # noqa: F401
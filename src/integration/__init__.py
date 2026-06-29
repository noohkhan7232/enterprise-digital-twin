"""Enterprise integration package (Week 10, Phase 5).

Exposes the Enterprise Integration Layer: a deterministic, thread-safe gateway
that coordinates communication between independent platform modules through
adapters, routing, pipelines, and dispatch - keeping every module loosely
coupled and composed, never modified.
"""

from __future__ import annotations

from .enterprise_integration_layer import *  # noqa: F401,F403
from .enterprise_integration_layer import __all__  # noqa: F401
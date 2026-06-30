"""Enterprise deployment health check.

Validates that a deployed instance of the Enterprise Digital Twin platform is
healthy: that the application code is present, the MLOps and Monitoring
subsystems are available, the HTTP health endpoint responds, and the required
configuration is present and non-empty. Produces a deterministic JSON report
and an exit code suitable for container ``HEALTHCHECK`` and Kubernetes probes.

The HTTP probe is injected (dependency injection), so the checker is fully
deterministic and unit-testable offline; the default probe uses the standard
library with a short timeout and fails closed. Runnable as a CLI::

    python deployment/scripts/health_check.py --root . --endpoint http://localhost:8080/health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["HealthStatus", "CheckResult", "HealthReport", "HealthChecker",
           "default_probe", "main"]

DEFAULT_TIMESTAMP = "2024-01-01T00:00:00+00:00"

# A probe takes a URL and returns (ok, detail).
Probe = Callable[[str], Tuple[bool, str]]


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


_RANK = {HealthStatus.HEALTHY: 0, HealthStatus.DEGRADED: 1, HealthStatus.UNHEALTHY: 2}


def aggregate(statuses: Sequence[HealthStatus]) -> HealthStatus:
    worst = HealthStatus.HEALTHY
    for status in statuses:
        if _RANK[status] > _RANK[worst]:
            worst = status
    return worst


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The immutable outcome of a single health check."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CheckResult.name must be non-empty")
        if not isinstance(self.status, HealthStatus):
            object.__setattr__(self, "status", HealthStatus(self.status))
        object.__setattr__(self, "details",
                           tuple(sorted((str(k), v) for k, v in dict(self.details).items())))

    @property
    def healthy(self) -> bool:
        return self.status is HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "healthy": self.healthy,
            "message": self.message,
            "details": {k: v for k, v in self.details},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckResult":
        return cls(name=data["name"], status=HealthStatus(data["status"]),
                   message=data.get("message", ""),
                   details=tuple(sorted((data.get("details") or {}).items())))


@dataclass(frozen=True, slots=True)
class HealthReport:
    """An immutable, JSON-serialisable health report."""

    results: Tuple[CheckResult, ...] = ()
    generated_at: str = DEFAULT_TIMESTAMP

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def overall_status(self) -> HealthStatus:
        return aggregate([r.status for r in self.results])

    @property
    def healthy(self) -> bool:
        return self.overall_status is HealthStatus.HEALTHY

    @property
    def ready(self) -> bool:
        """Ready to serve traffic if not UNHEALTHY."""
        return self.overall_status is not HealthStatus.UNHEALTHY

    def result(self, name: str) -> Optional[CheckResult]:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        counts = {"healthy": 0, "degraded": 0, "unhealthy": 0}
        for r in self.results:
            counts[r.status.value.lower()] += 1
        return {
            "overall_status": self.overall_status.value,
            "healthy": self.healthy,
            "ready": self.ready,
            "generated_at": self.generated_at,
            "summary": counts,
            "checks": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HealthReport":
        return cls(results=tuple(CheckResult.from_dict(c) for c in data.get("checks", [])),
                   generated_at=data.get("generated_at", DEFAULT_TIMESTAMP))


def default_probe(url: str, *, timeout: float = 2.0) -> Tuple[bool, str]:
    """Probe an HTTP health endpoint with the standard library; fail closed."""
    import urllib.request  # local import keeps the module import-light

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            code = getattr(response, "status", response.getcode())
            if 200 <= int(code) < 300:
                return True, f"HTTP {code}"
            return False, f"HTTP {code}"
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return False, f"unreachable: {type(exc).__name__}"


class HealthChecker:
    """Runs deterministic deployment health checks for a repository root."""

    def __init__(
        self,
        root: str,
        *,
        endpoint: Optional[str] = None,
        probe: Optional[Probe] = None,
        required_packages: Sequence[str] = ("mlops", "monitoring"),
        required_configs: Sequence[str] = ("configs/quality_gate.yaml", "configs/release_policy.yaml"),
        timestamp: str = DEFAULT_TIMESTAMP,
    ) -> None:
        self.root = root
        self.endpoint = endpoint
        self.probe = probe or default_probe
        self.required_packages = tuple(required_packages)
        self.required_configs = tuple(required_configs)
        self.timestamp = timestamp

    # -- checks ------------------------------------------------------------- #
    def check_application(self) -> CheckResult:
        src = os.path.join(self.root, "src")
        if not os.path.isdir(src):
            return CheckResult("application", HealthStatus.UNHEALTHY, "no src/ directory")
        packages = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]
        if not packages:
            return CheckResult("application", HealthStatus.UNHEALTHY, "no application packages")
        return CheckResult("application", HealthStatus.HEALTHY,
                           f"{len(packages)} package(s) present",
                           details={"packages": ",".join(packages)})

    def _package(self, name: str, check_name: str) -> CheckResult:
        path = os.path.join(self.root, "src", name)
        ok = os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py"))
        status = HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY
        return CheckResult(check_name, status,
                           f"{name} subsystem {'available' if ok else 'missing'}")

    def check_mlops(self) -> CheckResult:
        return self._package("mlops", "mlops")

    def check_monitoring(self) -> CheckResult:
        return self._package("monitoring", "monitoring")

    def check_health_endpoint(self) -> CheckResult:
        if not self.endpoint:
            return CheckResult("health_endpoint", HealthStatus.DEGRADED,
                               "no endpoint configured; static checks only")
        ok, detail = self.probe(self.endpoint)
        status = HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY
        return CheckResult("health_endpoint", status, detail,
                           details={"endpoint": self.endpoint})

    def check_configuration(self) -> CheckResult:
        if not self.required_configs:
            return CheckResult("configuration", HealthStatus.HEALTHY, "no configuration required")
        present, empty = [], []
        for rel in self.required_configs:
            path = os.path.join(self.root, rel)
            if not os.path.exists(path):
                continue
            present.append(rel)
            try:
                if os.path.getsize(path) == 0:
                    empty.append(rel)
            except OSError:
                empty.append(rel)
        missing = [c for c in self.required_configs if c not in present]
        if missing:
            return CheckResult("configuration", HealthStatus.UNHEALTHY,
                               f"missing config: {','.join(missing)}",
                               details={"missing": ",".join(missing)})
        if empty:
            return CheckResult("configuration", HealthStatus.DEGRADED,
                               f"empty config: {','.join(empty)}")
        return CheckResult("configuration", HealthStatus.HEALTHY,
                           f"{len(present)} config(s) present")

    # -- aggregation -------------------------------------------------------- #
    def evaluate(self) -> HealthReport:
        checks = [
            self.check_application(),
            self.check_mlops(),
            self.check_monitoring(),
            self.check_health_endpoint(),
            self.check_configuration(),
        ]
        return HealthReport(tuple(checks), self.timestamp)


def _resolve_root(root: Optional[str]) -> str:
    if root:
        return root
    here = os.path.abspath(os.getcwd())
    while here != os.path.dirname(here):
        if os.path.isdir(os.path.join(here, "src")):
            return here
        here = os.path.dirname(here)
    return os.path.abspath(os.getcwd())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deployment health check")
    parser.add_argument("--root", default=None, help="Repository root (auto-detected by default)")
    parser.add_argument("--endpoint", default=os.environ.get("HEALTH_ENDPOINT"),
                        help="HTTP health endpoint URL")
    parser.add_argument("--output", default=None, help="Write the JSON report to this path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = HealthChecker(_resolve_root(args.root), endpoint=args.endpoint).evaluate()
    payload = report.to_json()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    if not args.quiet:
        print(payload)
    return 0 if report.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
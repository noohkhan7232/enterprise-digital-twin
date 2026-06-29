"""Reproducibility engine: capture and verify exact run environments.

The engine produces :class:`ReproducibilitySnapshot` objects that record every
input needed to reproduce a run: interpreter and library versions, dependency
versions, whitelisted environment variables, random seeds, a configuration
snapshot, the dataset version, the git commit, and runtime / hardware info.

Environment capture is performed through an injected
:class:`EnvironmentProvider` (Strategy pattern). A deterministic static
provider is supplied for tests and reproducible pipelines; a system-backed
provider is supplied for production use.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import sys
import threading
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import numpy as _np

from mlops.experiment_models import (
    Clock,
    DeterministicIdGenerator,
    EnvironmentSnapshot,
    IdGenerator,
    LogicalClock,
    MLOpsError,
    ReproducibilitySnapshot,
    ValidationError,
)

__all__ = [
    "ReproducibilityError",
    "EnvironmentProvider",
    "SystemEnvironmentProvider",
    "StaticEnvironmentProvider",
    "ReproducibilityEngine",
    "create_reproducibility_engine",
    "DEFAULT_ENV_ALLOWLIST",
]

DEFAULT_ENV_ALLOWLIST: Tuple[str, ...] = (
    "MLOPS_ENV",
    "MLOPS_SEED",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)


class ReproducibilityError(MLOpsError):
    """Raised on reproducibility capture or verification failures."""


# --------------------------------------------------------------------------- #
# Environment providers (Strategy pattern)
# --------------------------------------------------------------------------- #
class EnvironmentProvider(Protocol):
    """Abstraction over runtime-environment introspection."""

    def python_version(self) -> str: ...
    def numpy_version(self) -> str: ...
    def platform(self) -> str: ...
    def hostname(self) -> str: ...
    def processor(self) -> str: ...
    def cpu_count(self) -> int: ...
    def dependencies(self) -> Mapping[str, str]: ...
    def environment_variables(self, allowlist: Tuple[str, ...]) -> Mapping[str, str]: ...
    def git_commit(self) -> str: ...
    def runtime_info(self) -> Mapping[str, Any]: ...
    def hardware_info(self) -> Mapping[str, Any]: ...


class SystemEnvironmentProvider:
    """Reads the real runtime environment (non-deterministic across hosts)."""

    def __init__(self, dependency_packages: Tuple[str, ...] = ("numpy",), git_commit: str = "") -> None:
        self._packages = dependency_packages
        self._git_commit = git_commit

    def python_version(self) -> str:
        return _platform.python_version()

    def numpy_version(self) -> str:
        return _np.__version__

    def platform(self) -> str:
        return _platform.platform()

    def hostname(self) -> str:
        return _platform.node()

    def processor(self) -> str:
        return _platform.processor()

    def cpu_count(self) -> int:
        return os.cpu_count() or 0

    def dependencies(self) -> Mapping[str, str]:
        from importlib import metadata

        result: Dict[str, str] = {}
        for name in self._packages:
            try:
                result[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                if name == "numpy":
                    result[name] = _np.__version__
        return result

    def environment_variables(self, allowlist: Tuple[str, ...]) -> Mapping[str, str]:
        return {k: os.environ[k] for k in allowlist if k in os.environ}

    def git_commit(self) -> str:
        return self._git_commit

    def runtime_info(self) -> Mapping[str, Any]:
        return {
            "executable": sys.executable,
            "implementation": _platform.python_implementation(),
            "byteorder": sys.byteorder,
            "maxsize": sys.maxsize,
        }

    def hardware_info(self) -> Mapping[str, Any]:
        return {
            "machine": _platform.machine(),
            "architecture": _platform.architecture()[0],
            "cpu_count": os.cpu_count() or 0,
        }


class StaticEnvironmentProvider:
    """A fully deterministic provider used for tests and reproducible runs."""

    def __init__(
        self,
        python_version: str = "3.12.3",
        numpy_version: str = "2.4.4",
        platform_str: str = "Linux-x86_64",
        hostname: str = "build-node",
        processor: str = "x86_64",
        cpu_count: int = 8,
        dependencies: Optional[Mapping[str, str]] = None,
        environment_variables: Optional[Mapping[str, str]] = None,
        git_commit: str = "0" * 40,
        runtime_info: Optional[Mapping[str, Any]] = None,
        hardware_info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._python_version = python_version
        self._numpy_version = numpy_version
        self._platform = platform_str
        self._hostname = hostname
        self._processor = processor
        self._cpu_count = cpu_count
        self._dependencies = dict(dependencies or {"numpy": numpy_version})
        self._environment_variables = dict(environment_variables or {})
        self._git_commit = git_commit
        self._runtime_info = dict(runtime_info or {"implementation": "CPython"})
        self._hardware_info = dict(hardware_info or {"machine": "x86_64", "cpu_count": cpu_count})

    def python_version(self) -> str:
        return self._python_version

    def numpy_version(self) -> str:
        return self._numpy_version

    def platform(self) -> str:
        return self._platform

    def hostname(self) -> str:
        return self._hostname

    def processor(self) -> str:
        return self._processor

    def cpu_count(self) -> int:
        return self._cpu_count

    def dependencies(self) -> Mapping[str, str]:
        return dict(self._dependencies)

    def environment_variables(self, allowlist: Tuple[str, ...]) -> Mapping[str, str]:
        return {k: v for k, v in self._environment_variables.items() if k in allowlist}

    def git_commit(self) -> str:
        return self._git_commit

    def runtime_info(self) -> Mapping[str, Any]:
        return dict(self._runtime_info)

    def hardware_info(self) -> Mapping[str, Any]:
        return dict(self._hardware_info)


# --------------------------------------------------------------------------- #
# Reproducibility engine
# --------------------------------------------------------------------------- #
class ReproducibilityEngine:
    """Captures and verifies reproducibility snapshots."""

    def __init__(
        self,
        provider: Optional[EnvironmentProvider] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        env_allowlist: Tuple[str, ...] = DEFAULT_ENV_ALLOWLIST,
    ) -> None:
        self._provider: EnvironmentProvider = provider or SystemEnvironmentProvider()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="repro")
        self._allowlist = tuple(env_allowlist)
        self._lock = threading.RLock()

    def capture_environment(self) -> EnvironmentSnapshot:
        """Capture a snapshot of the current environment."""
        p = self._provider
        return EnvironmentSnapshot(
            python_version=p.python_version(),
            numpy_version=p.numpy_version(),
            platform=p.platform(),
            hostname=p.hostname(),
            processor=p.processor(),
            cpu_count=p.cpu_count(),
            dependencies=p.dependencies(),
            environment_variables=p.environment_variables(self._allowlist),
            captured_at=self._clock.now(),
        )

    def capture(
        self,
        *,
        random_seed: int,
        numpy_seed: int,
        config: Optional[Mapping[str, Any]] = None,
        dataset_version: str = "",
        snapshot_id: Optional[str] = None,
    ) -> ReproducibilitySnapshot:
        """Capture a complete reproducibility snapshot."""
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise ValidationError("random_seed must be an int")
        if not isinstance(numpy_seed, int) or isinstance(numpy_seed, bool):
            raise ValidationError("numpy_seed must be an int")
        with self._lock:
            sid = snapshot_id or self._ids.generate("repro")
            environment = self.capture_environment()
            return ReproducibilitySnapshot(
                snapshot_id=sid,
                environment=environment,
                random_seed=random_seed,
                numpy_seed=numpy_seed,
                config=dict(config or {}),
                dataset_version=dataset_version,
                git_commit=self._provider.git_commit(),
                runtime_info=self._provider.runtime_info(),
                hardware_info=self._provider.hardware_info(),
                created_at=self._clock.now(),
            )

    def verify(
        self, expected: ReproducibilitySnapshot, actual: ReproducibilitySnapshot
    ) -> bool:
        """Return ``True`` if *actual* reproduces *expected* (ignoring ids/time)."""
        if not isinstance(expected, ReproducibilitySnapshot) or not isinstance(
            actual, ReproducibilitySnapshot
        ):
            raise ValidationError("verify() requires two ReproducibilitySnapshot instances")
        return expected.matches(actual)

    def diff(
        self, expected: ReproducibilitySnapshot, actual: ReproducibilitySnapshot
    ) -> Dict[str, Any]:
        """Return a deterministic diff between two snapshots."""
        return expected.diff(actual)

    @staticmethod
    def to_json(snapshot: ReproducibilitySnapshot, *, indent: int = 2) -> str:
        return json.dumps(snapshot.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def export_json(snapshot: ReproducibilitySnapshot, path: str, *, indent: int = 2) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.to_dict(), indent=indent, sort_keys=True))
        return path


def create_reproducibility_engine(*, deterministic: bool = True) -> ReproducibilityEngine:
    """Factory returning a configured :class:`ReproducibilityEngine`."""
    if deterministic:
        return ReproducibilityEngine(
            provider=StaticEnvironmentProvider(),
            clock=LogicalClock(),
            id_generator=DeterministicIdGenerator(seed="repro"),
        )
    from mlops.experiment_models import SequentialIdGenerator, SystemClock

    return ReproducibilityEngine(
        provider=SystemEnvironmentProvider(),
        clock=SystemClock(),
        id_generator=SequentialIdGenerator(),
    )
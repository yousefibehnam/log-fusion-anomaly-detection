"""The simulated microservice system: services, their dependencies and normal behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Service:
    """Normal behaviour of one service.

    ``latency_ms`` is the service's own processing time (callee time excluded);
    ``cpu_base`` and ``cpu_per_call`` define CPU % as a function of load.
    """

    name: str
    latency_ms: float
    cpu_base: float
    cpu_per_call: float
    memory_mb: float


@dataclass(frozen=True)
class Topology:
    """Services plus caller -> callee edges with the probability that a request makes that call."""

    services: tuple[Service, ...]
    calls: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    entry: str = "frontend"

    def __post_init__(self) -> None:
        names = {s.name for s in self.services}
        if self.entry not in names:
            raise ValueError(f"entry service {self.entry!r} is not defined")
        for caller, edges in self.calls.items():
            for callee, p in edges:
                if caller not in names or callee not in names:
                    raise ValueError(f"edge {caller}->{callee} uses an undefined service")
                if not 0.0 < p <= 1.0:
                    raise ValueError(f"call probability must be in (0, 1]: {caller}->{callee}={p}")
        if self._has_cycle():
            raise ValueError("the call graph must be acyclic")

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.services]

    def service(self, name: str) -> Service:
        for s in self.services:
            if s.name == name:
                return s
        raise KeyError(name)

    def callees(self, name: str) -> list[str]:
        return [callee for callee, _ in self.calls.get(name, ())]

    def callers(self, name: str) -> list[str]:
        return [c for c, edges in self.calls.items() if any(callee == name for callee, _ in edges)]

    def _has_cycle(self) -> bool:
        state: dict[str, int] = {}

        def visit(node: str) -> bool:
            state[node] = 1
            for callee in self.callees(node):
                if state.get(callee) == 1 or (callee not in state and visit(callee)):
                    return True
            state[node] = 2
            return False

        return any(name not in state and visit(name) for name in self.names)


def default_topology() -> Topology:
    """A ten-service online shop: a front end, six business services and two data stores."""
    services = (
        Service("frontend", 5.0, 10.0, 0.20, 512.0),
        Service("auth", 8.0, 6.0, 0.15, 256.0),
        Service("catalog", 12.0, 8.0, 0.20, 384.0),
        Service("cart", 6.0, 5.0, 0.15, 256.0),
        Service("checkout", 15.0, 7.0, 0.30, 384.0),
        Service("payment", 30.0, 5.0, 0.40, 256.0),
        Service("inventory", 10.0, 6.0, 0.20, 320.0),
        Service("shipping", 20.0, 4.0, 0.30, 256.0),
        Service("cache", 1.0, 8.0, 0.05, 1024.0),
        Service("db", 4.0, 15.0, 0.10, 2048.0),
    )
    calls = {
        "frontend": (("auth", 0.9), ("catalog", 0.8), ("cart", 0.5), ("checkout", 0.3)),
        "checkout": (("cart", 1.0), ("payment", 0.9), ("inventory", 0.9), ("shipping", 0.7)),
        "catalog": (("cache", 0.9), ("inventory", 0.5)),
        "cart": (("cache", 1.0),),
        "inventory": (("db", 1.0),),
        "payment": (("db", 1.0),),
        "auth": (("db", 0.6),),
    }
    return Topology(services, calls)

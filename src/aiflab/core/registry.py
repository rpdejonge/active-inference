from collections.abc import Callable
from typing import Any, Generic, TypeVar

from aiflab.core.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Simple named object/factory registry.

    Intended for mapping config names to constructors, classes, or callables.
    """

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._items: dict[str, T] = {}

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(self, name: str, item: T, *, overwrite: bool = False) -> None:
        if not name:
            raise RegistryError(f"{self._namespace}: registry key cannot be empty.")

        if name in self._items and not overwrite:
            raise RegistryError(f"{self._namespace}: item '{name}' already registered.")

        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryError(
                f"{self._namespace}: unknown item '{name}'. Available: {available}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items.keys()))

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple(sorted(self._items.items(), key=lambda kv: kv[0]))

    def clear(self) -> None:
        self._items.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry(namespace={self._namespace!r}, size={len(self._items)})"


Factory = Callable[..., Any]


domain_registry: Registry[Factory] = Registry("domains")
agent_registry: Registry[Factory] = Registry("agents")
model_registry: Registry[Factory] = Registry("models")
state_inference_registry: Registry[Factory] = Registry("state_inference")
policy_inference_registry: Registry[Factory] = Registry("policy_inference")
experiment_runner_registry: Registry[Factory] = Registry("experiment_runners")

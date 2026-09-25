"""A tiny name -> factory registry.

Used for both encoders and algorithms so a config can say ``name: impala`` or
``name: ppo`` and get the right class, without `train.py` importing every
implementation and growing an if-chain.

Deliberately minimal: an unknown name raises immediately, listing what *is*
available. A typo in a YAML file should fail at startup with a readable
message, not silently fall back to a default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            key = name.lower()
            if key in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[key] = factory
            return factory

        return deco

    def get(self, name: str) -> Callable[..., T]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}. Available: {', '.join(self.names())}"
            )
        return self._items[key]

    def build(self, name: str, **kwargs) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __len__(self) -> int:
        return len(self._items)

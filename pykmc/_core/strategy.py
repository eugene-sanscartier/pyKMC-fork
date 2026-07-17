"""Shared plumbing for the pluggable strategy pattern.

This module provides :class:`Strategy`, the domain-free base class that powers
strategy registration and lookup across pyKMC.

Concrete strategies live in their respective modules. See the
[architecture documentation](../dev_strategy_pattern.md) for the overall design and
:func:`pykmc._core.discovery.autodiscover` for how strategy modules are loaded.
"""

from __future__ import annotations

import inspect
from abc import ABC
from typing import ClassVar


class Strategy(ABC):
    """Abstract base for the pluggable strategy mechanism shared by all modules.

    A *strategy* is one interchangeable implementation of a task delegated by a
    facade object (e.g. ``NeighborsList``, ``AtomicEnvironments``, ``RateConstant`). This class provides the registration and lookup plumbing
    reused by every module.

    Each module defines its own base strategy by subclassing ``Strategy`` with
    ``root=True``; that base owns an isolated registry. Concrete strategies then
    subclass the module base, set a unique ``name``, and register themselves
    automatically at class-definition time. Instances are never built directly:
    use :meth:`create` on the module base.

    Attributes
    ----------
    name : str
        Unique identifier of a concrete strategy within its module registry.
        Set as a class attribute by every concrete subclass; validated at
        definition time.
    _registry : dict[str, type[Strategy]]
        Maps strategy names to their classes. Created fresh on each module base
        (``root=True``), so every module has its own, isolated registry.
    _root : type[Strategy]
        The module base strategy that owns the registry. Set on the module base
        and inherited by all its strategies, so any strategy can reach its
        module's registry via ``cls._root._registry``.

    Notes
    -----
    Not instantiable on its own, and ``create`` must be called on a module base
    (which carries ``_registry``), never on ``Strategy`` itself.
    """

    name: ClassVar[str]
    _registry: ClassVar[dict[str, type["Strategy"]]]
    _root: ClassVar[type["Strategy"]]

    def __init_subclass__(cls, root: bool = False, **kwargs) -> None:
        """Register the subclass into its module registry, at definition time.

        Invoked automatically by Python whenever a subclass of ``Strategy`` is
        defined. A module base (``root=True``) gets a fresh isolated registry,
        an intermediate abstract base is skipped, a concrete strategy is
        validated and added to its module's registry.

        Parameters
        ----------
        root : bool, optional
            If ``True``, mark this class as a module base: give it its own empty
            registry instead of registering it. Set only on a module's base
            strategy. Defaults to ``False``.
        **kwargs
            Forwarded to ``super().__init_subclass__``.

        Raises
        ------
        TypeError
            If an abstract subclass declares ``name`` in its own ``__dict__``
            (likely a typo in an abstract-method override), or if a concrete
            strategy does not declare a non-empty string ``name`` directly (not
            inherited).
        RuntimeError
            If ``name`` is already registered in the module (name collision).
        """
        super().__init_subclass__(**kwargs)

        # A module base declares `root=True`: give it its own isolated registry.
        if root:
            cls._registry = {}
            cls._root = cls
            return

        # If the class is still abstract but declared its own name
        if inspect.isabstract(cls):
            if "name" in cls.__dict__:
                unimplemented = sorted(
                    attr
                    for attr in dir(cls)
                    if getattr(getattr(cls, attr, None), "__isabstractmethod__", False)
                )
                raise TypeError(
                    f"{cls.__name__} defines 'name' but is still abstract "
                    f"(unimplemented: {unimplemented})"
                )
            return

        # Each concrete class must declare its own name
        name = cls.__dict__.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError(f"{cls.__name__} must define a non-empty 'name' str")

        # Register into the module's own registry, reached via the inherited root.
        registry = cls._root._registry
        if name in registry:
            raise RuntimeError(
                f"Name collision '{name}': {registry[name].__name__} vs {cls.__name__}"
            )
        registry[name] = cls

    @classmethod
    def create(cls, name: str, **kwargs) -> Strategy:
        """Instantiate the registered strategy identified by ``name``.

        Called on a module base, e.g.
        ``NeighborsListStrategy.create("lammps", config=cfg)``.

        Parameters
        ----------
        name : str
            Name of a strategy registered in this module.
        **kwargs
            Forwarded as-is to the selected strategy's constructor (typically a
            ``config`` object).

        Returns
        -------
        Strategy
            A new instance of the selected strategy.

        Raises
        ------
        TypeError
            If called directly on ``Strategy`` rather than on a module base.
        ValueError
            If ``name`` is not registered in this module's registry.
        """
        if not hasattr(cls, "_registry"):
            raise TypeError(
                "create() must be called on a module base strategy, not on Strategy itself"
            )
        if name not in cls._registry:
            raise ValueError(
                f"Strategy '{name}' unknown. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

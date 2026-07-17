"""Tests for the pluggable strategy pattern (pykmc._core.strategy)."""

import pytest
from abc import abstractmethod
from pykmc._core import Strategy


# Definition of compute strategies :


## Abstract base Strategy
class BaseOperationStrategy(Strategy, root=True):
    @abstractmethod
    def compute(self, a: float, b: float) -> float: ...


## Addition Strategy
class Addition(BaseOperationStrategy):
    name = "addition"

    def compute(self, a, b):
        return a + b


## Multiplication Strategy
class Multiplication(BaseOperationStrategy):
    name = "multiplication"

    def compute(self, a, b):
        return a * b


## Division Strategy
class Division(BaseOperationStrategy):
    name = "division"

    def compute(self, a, b):
        return a / b


## Facade class using defined strategy to compute operation
class ComputeOperation:
    def __init__(self, strategy_name: str) -> None:
        self._strategy = BaseOperationStrategy.create(strategy_name)

    def compute(self, a: float, b: float) -> float:
        return self._strategy.compute(a, b)

    @classmethod
    def create(cls, strategy_name: str) -> "ComputeOperation":
        return cls(strategy_name=strategy_name)


# Test Registration strategies
class TestRegistration:
    def test_all_strategies_are_registered(self):
        assert set(BaseOperationStrategy._registry) == {
            "addition",
            "multiplication",
            "division",
        }

    def test_create_returns_correct_type(self):
        assert isinstance(BaseOperationStrategy.create("addition"), Addition)
        assert isinstance(
            BaseOperationStrategy.create("multiplication"), Multiplication
        )
        assert isinstance(BaseOperationStrategy.create("division"), Division)

    def test_missing_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty 'name'"):

            class NoName(BaseOperationStrategy):
                def compute(self, a, b):
                    return a

    def test_empty_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty 'name'"):

            class EmptyName(BaseOperationStrategy):
                name = ""

                def compute(self, a, b):
                    return a

    def test_name_collision_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Name collision"):

            class Duplicate(BaseOperationStrategy):
                name = "addition"

                def compute(self, a, b):
                    return a + b

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown"):
            BaseOperationStrategy.create("modulo")

    def test_error_message_lists_available_names(self):
        with pytest.raises(ValueError, match="addition"):
            BaseOperationStrategy.create("nonexistent")

    def test_create_on_strategy_base_raises_type_error(self):
        with pytest.raises(TypeError):
            Strategy.create("addition")

    def test_abstract_with_name_raises_type_error(self):
        """A typo in an abstract-method override is caught at class definition."""
        with pytest.raises(TypeError, match="still abstract"):

            class TypoCompute(BaseOperationStrategy):
                name = "typo"

                def computee(self, a, b):  # typo: compute → computee
                    return a + b

    def test_inherited_name_not_accepted(self):
        """A subclass must declare its own name; inheriting a parent's is rejected."""
        with pytest.raises(TypeError, match="non-empty 'name'"):

            class Child(Addition):  # inherits name="addition" from Addition
                def compute(self, a, b):
                    return a - b

    def test_intermediate_abstract_base_not_registered(self):
        class IntermediateOp(BaseOperationStrategy):
            def _helper(self): ...

        class Power(IntermediateOp):
            name = "power"

            def compute(self, a, b):
                return a**b

        assert "power" in BaseOperationStrategy._registry
        assert BaseOperationStrategy._registry["power"] is Power
        assert isinstance(BaseOperationStrategy.create("power"), Power)

    def test_registries_are_isolated_between_roots(self):
        class OtherStrategy(Strategy, root=True):
            @abstractmethod
            def run(self): ...

        class OtherImpl(OtherStrategy):
            name = "addition"

            def run(self):
                return 42

        assert OtherStrategy._registry is not BaseOperationStrategy._registry
        assert OtherStrategy._registry["addition"] is OtherImpl
        assert BaseOperationStrategy._registry["addition"] is Addition


# Test Compute Operations
class TestComputeOperation:
    def test_addition(self):
        assert ComputeOperation.create("addition").compute(3, 2) == 5

    def test_multiplication(self):
        assert ComputeOperation.create("multiplication").compute(3, 2) == 6

    def test_division(self):
        assert ComputeOperation.create("division").compute(6, 2) == 3.0

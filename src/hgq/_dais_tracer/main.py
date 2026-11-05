from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import keras
from da4ml.converter.plugin import DAISTracerPluginBase, _flatten_arr
from da4ml.trace import FixedVariableArray, comb_trace
from keras import KerasTensor, Operation

from .layers import _registry


@dataclass
class OpObj:
    operation: Operation
    args: list
    kwargs: dict
    produces: tuple[KerasTensor, ...]
    requires: tuple[KerasTensor, ...]


class MaybeRename:
    def __init__(self):
        self.counter: dict[str, int] = {}

    def __call__(self, name: str) -> str:
        if name not in self.counter:
            self.counter[name] = 0
            return name
        else:
            self.counter[name] += 1
            return f'{name}#{self.counter[name]}'


def parse_model(model: keras.Model):
    if isinstance(model, keras.Sequential):
        model = model._functional
    operators: dict[int, list[OpObj]] = {}
    for depth, nodes in model._nodes_by_depth.items():
        _oprs = []
        for node in nodes:
            assert isinstance(node.operation, keras.Operation)
            opr = OpObj(
                operation=node.operation,
                args=node.arguments.args,
                kwargs=node.arguments.kwargs,
                produces=node.outputs,
                requires=node.arguments.keras_tensors,
            )
            _oprs.append(opr)
        operators[depth] = _oprs
    return [operators[i] for i in range(max(operators.keys()), -1, -1)]


def replace_tensors(tensor_map: dict[str, FixedVariableArray], obj: Any) -> Any:
    if isinstance(obj, KerasTensor):
        return tensor_map[obj.name]
    if isinstance(obj, list):
        return [replace_tensors(tensor_map, o) for o in obj]
    if isinstance(obj, tuple):
        return tuple(replace_tensors(tensor_map, o) for o in obj)
    if isinstance(obj, dict):
        return {k: replace_tensors(tensor_map, v) for k, v in obj.items()}
    return obj


def _trace_model(
    model: keras.Model,
    inputs: FixedVariableArray | Sequence[FixedVariableArray],
    verbose: bool = False,
    n_nested: int = 0,
) -> dict[str, tuple[FixedVariableArray, ...]]:
    """
    Apply a keras model to a fixed variable array or a sequence of fixed variable arrays.

    Parameters
    ----------
    model : keras.Model
        The keras model to apply.
    inputs : FixedVariableArray or Sequence[FixedVariableArray]
        The input fixed variable array or sequence of fixed variable arrays.
    verbose : bool, optional
        Whether to print the trace, by default False
    """
    if isinstance(inputs, FixedVariableArray):
        inputs = (inputs,)

    assert len(model.inputs) == len(inputs), f'Model has {len(model.inputs)} inputs, got {len(inputs)}'
    tensor_map: dict[str, FixedVariableArray] = {
        keras_tensor.name: da_tensor for keras_tensor, da_tensor in zip(model.inputs, inputs)
    }

    _inputs = _flatten_arr(inputs)

    if verbose and n_nested:
        print(' -> enter:')

    trace: dict[str, tuple[FixedVariableArray, ...]] = {}
    maybe_rename = MaybeRename()

    for ops in parse_model(model):
        for op in ops:
            assert all(t.name in tensor_map for t in op.requires)
            args = replace_tensors(tensor_map, op.args)
            kwargs: dict[str, Any] = replace_tensors(tensor_map, op.kwargs)
            if op.operation.__class__ is keras.layers.InputLayer:
                continue

            if verbose:
                indent = '    ' * n_nested
                print(f'{indent}{op.operation.name} ({op.operation.__class__.__name__})', end='')

            if isinstance(op.operation, keras.Model):
                sub_model = op.operation._functional if isinstance(op.operation, keras.Sequential) else op.operation
                _dump: dict[str, tuple[FixedVariableArray, ...]] = _trace_model(
                    sub_model,
                    args,
                    verbose=verbose,
                    n_nested=n_nested + 1,
                )  # type: ignore
            else:
                mirror_op = _registry[op.operation.__class__](op.operation)
                _dump = mirror_op(*args, **kwargs)
            if verbose:
                comb = comb_trace(_inputs, _flatten_arr(_dump['final']))
                print(f' cumcost: {comb.cost}, latency: {comb.latency[1]}')

            for keras_tensor, da_tensor in zip(op.produces, _dump['final']):
                tensor_map[keras_tensor.name] = da_tensor

            name = maybe_rename(op.operation.name)
            for k, v in _dump.items():
                kk = f'{name}/{k}' if n_nested != 0 else f'/{name}/{k}'
                trace[kk] = v

    if verbose and n_nested:
        indent = '    ' * (n_nested - 1)
        print(f'{indent}<- exit', end='')

    final = tuple(tensor_map[keras_tensor.name] for keras_tensor in model.outputs)
    trace['final'] = final

    return trace


class HGQ2DAISTracer(DAISTracerPluginBase):
    model: keras.Model

    def apply_model(self, verbose: bool, inputs: tuple[FixedVariableArray, ...]):
        dump = {'inputs': inputs}
        _dump = _trace_model(
            self.model,
            inputs,
            verbose=verbose,
        )
        dump.update(_dump)
        return dump, ['final']

    def get_input_shapes(self) -> list[tuple[int, ...]] | None:
        shapes = [inp.shape[1:] for inp in self.model.inputs]
        if any(None in shape or -1 in shape for shape in shapes):
            return None
        return shapes

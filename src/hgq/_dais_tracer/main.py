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


def replace_tensors(tensor_map: dict[KerasTensor, FixedVariableArray], obj: Any) -> Any:
    if isinstance(obj, KerasTensor):
        return tensor_map[obj]
    if isinstance(obj, list):
        return [replace_tensors(tensor_map, o) for o in obj]
    if isinstance(obj, tuple):
        return tuple(replace_tensors(tensor_map, o) for o in obj)
    if isinstance(obj, dict):
        return {k: replace_tensors(tensor_map, v) for k, v in obj.items()}
    return obj


def _apply_nn(
    model: keras.Model,
    inputs: FixedVariableArray | Sequence[FixedVariableArray],
    verbose: bool = False,
    dump: bool = False,
    n_nested: int = 0,
) -> tuple[FixedVariableArray, ...] | dict[str, FixedVariableArray]:
    """
    Apply a keras model to a fixed variable array or a sequence of fixed variable arrays.

    Parameters
    ----------
    model : keras.Model
        The keras model to apply.
    inputs : FixedVariableArray or Sequence[FixedVariableArray]
        The input fixed variable array or sequence of fixed variable arrays.

    Returns
    -------
    tuple of FixedVariableArray
        A tuple containing the output(s) of the model as FixedVariableArray.
    """
    if isinstance(inputs, FixedVariableArray):
        inputs = (inputs,)

    assert len(model.inputs) == len(inputs), f'Model has {len(model.inputs)} inputs, got {len(inputs)}'
    tensor_map = {keras_tensor: da_tensor for keras_tensor, da_tensor in zip(model.inputs, inputs)}

    _inputs = _flatten_arr(inputs)

    if verbose and n_nested:
        print(' -> enter:')

    for ops in parse_model(model):
        for op in ops:
            assert all(t in tensor_map for t in op.requires)
            args = replace_tensors(tensor_map, op.args)
            kwargs: dict[str, Any] = replace_tensors(tensor_map, op.kwargs)
            if op.operation.__class__ is keras.layers.InputLayer:
                continue

            if verbose:
                indent = '    ' * n_nested
                print(f'{indent}{op.operation.name} ({op.operation.__class__.__name__})', end='')

            if isinstance(op.operation, keras.Model):
                sub_model = op.operation._functional if isinstance(op.operation, keras.Sequential) else op.operation
                outputs: tuple[FixedVariableArray, ...] = _apply_nn(
                    sub_model,
                    args,
                    verbose=verbose,
                    dump=False,
                    n_nested=n_nested + 1,
                )  # type: ignore
            else:
                mirror_op = _registry[op.operation.__class__](op.operation)
                outputs = mirror_op(*args, **kwargs)
            if verbose:
                comb = comb_trace(_inputs, _flatten_arr(outputs))
                print(f' cumcost: {comb.cost}, latency: {comb.latency[1]}')

            for keras_tensor, da_tensor in zip(op.produces, outputs):
                tensor_map[keras_tensor] = da_tensor

    if verbose and n_nested:
        indent = '    ' * (n_nested - 1)
        print(f'{indent}<- exit', end='')

    if not dump:
        return tuple(tensor_map[keras_tensor] for keras_tensor in model.outputs)
    else:
        return {k.name: v for k, v in tensor_map.items()}


class HGQ2DAISTracer(DAISTracerPluginBase):
    model: keras.Model

    def apply_model(self, verbose: bool, inputs: tuple[FixedVariableArray, ...]):
        dump: dict[str, FixedVariableArray] = _apply_nn(
            self.model,
            inputs,
            verbose=verbose,
            dump=True,
        )  # type: ignore
        output_names = [out.name for out in self.model.outputs]
        return dump, output_names

    def get_input_shapes(self) -> list[tuple[int, ...]] | None:
        shapes = [inp.shape[1:] for inp in self.model.inputs]
        if any(None in shape or -1 in shape for shape in shapes):
            return None
        return shapes

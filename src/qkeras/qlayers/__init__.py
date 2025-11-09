import inspect

from hgq import layers
from hgq.config import QuantizerConfigScope

from ..initializers import QInitializer as QInitializer  # TODO: use qkeras initializers when called from here
from ..quantizers import get_quantizer

kw_map = {
    'kq_conf': ('kernel_quantizer', 'kq'),
    'bq_conf': ('bias_quantizer', 'bq'),
    'oq_conf': ('output_quantizer', 'oq'),
    'iq_conf': ('input_quantizer', 'iq'),
    'exp_iq_conf': ('exp_input_quantizer', 'exp_iq'),
    'exp_oq_conf': ('exp_output_quantizer', 'exp_oq'),
    'inv_iq_conf': ('inv_input_quantizer', 'inv_iq'),
    'inv_oq_conf': ('inv_output_quantizer', 'inv_oq'),
}

kw_map_inv = {vv: k for k, v in kw_map.items() for vv in v}


def qkeras_layer_wrap(cls: type):
    # base_cls = cls.__bases__[0]
    original_init = cls.__init__
    signature = inspect.signature(original_init)
    params = signature.parameters
    new_params = []
    for v in params.values():
        if v.name in kw_map:
            new_params.append(v.replace(name=kw_map[v.name][0]))
        else:
            new_params.append(v)
    new_signature = signature.replace(parameters=new_params)

    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        for k, v in list(kwargs.items()):
            if k not in kw_map_inv:
                continue
            new_k = kw_map_inv[k]
            assert new_k not in kwargs, f'Duplicate key {new_k}.'
            del kwargs[k]
            v = v if not isinstance(v, str) else get_quantizer(v)
            kwargs[new_k] = v

        if issubclass(cls, layers.QLayerBase):
            # Disable EBOPS; only explicitly enabled quantizers will be used
            kwargs['enable_ebops'] = False
            kwargs['enable_iq'] = kwargs.get('iq_conf') is not None
            kwargs['enable_oq'] = kwargs.get('oq_conf') is not None
        else:
            assert issubclass(cls, layers.Quantizer), f'Unexpected class {cls.__name__}.'
            if 'activation' in kwargs:
                kwargs['config'] = kwargs.pop('activation')
        with QuantizerConfigScope(default_q_type='dummy'):
            return original_init(self, *args, **kwargs)

    __init__.__signature__ = new_signature  # type: ignore
    cls.__init__ = __init__
    return cls


for name, obj in layers.__dict__.items():
    if not isinstance(obj, type):
        continue
    if issubclass(obj, layers.QLayerBase):
        globals()[name] = qkeras_layer_wrap(obj)


QActivation = qkeras_layer_wrap(layers.Quantizer)

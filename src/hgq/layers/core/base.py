import inspect
from abc import ABCMeta
from collections.abc import Callable, Iterable, Sequence
from functools import wraps

import numpy as np
from keras import ops
from keras.initializers import Constant, Initializer
from keras.layers import Layer
from keras.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object

from ...config.layer import global_config
from ...quantizer import Quantizer, QuantizerConfig


def get_method_source(cls, method_name):
    "Get the class that defines the method"
    for parent_cls in inspect.getmro(cls):
        if method_name in parent_cls.__dict__:
            return parent_cls
    assert False, f'Method {method_name} is not defined in {cls}'


def check_save_load_own_variables(cls):
    "Assert that save_own_variables and load_own_variables are not overriden, or they should be defined in the same class"

    if cls.save_own_variables is not Layer.save_own_variables:
        cls_save = get_method_source(cls, 'save_own_variables')
        assert cls_save is cls, f'save_own_variables in {cls} is overriden in {cls_save}, which will likely break save_weights fn'
    if cls.load_own_variables is not Layer.load_own_variables:
        cls_load = get_method_source(cls, 'load_own_variables')
        assert cls_load is cls, f'load_own_variables in {cls} is overriden in {cls_load}, which will likely break load_weights fn'


class QLayerMeta(ABCMeta):
    _wrapped_cls = set()

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        check_save_load_own_variables(cls)
        # ====================================================================
        # =========== Register as Keras serializable if possible =============
        # ====================================================================

        if cls.get_config is not Layer.get_config and cls.__module__.startswith('hgq'):
            original_get_config = cls.get_config

            @wraps(original_get_config)
            def get_config(self):
                config = original_get_config(self)
                config = serialize_keras_object(config)
                return config

            cls.get_config = get_config  # type: ignore
            cls = register_keras_serializable(package='hgq')(cls)

    def __call__(cls: type, *args, **kwargs):
        if cls in QLayerMeta._wrapped_cls:
            return super().__call__(*args, **kwargs)  # type: ignore
        QLayerMeta._wrapped_cls.add(cls)

        # ====================================================================
        # ============ Compute ebops if _compute_ebops presents ==============
        # =============== Apply output quantizer if possible =================
        # ====================================================================

        original_call: Callable = cls.call
        _compute_ebops = cls._compute_ebops

        if original_call is not Layer.call and _compute_ebops is not QLayerBase._compute_ebops:
            VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
            KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
            signature = inspect.signature(original_call)

            # Add kwarg training to signature if not present
            has_training = 'training' in signature.parameters
            has_var_keyword = any(v.kind == VAR_KEYWORD for v in signature.parameters.values())
            new_signature = signature

            if not has_training and not has_var_keyword:
                training_param = inspect.Parameter('training', KEYWORD_ONLY, default=None)
                new_params = signature.parameters.copy()
                new_params['training'] = training_param
                new_signature = signature.replace(parameters=new_params.values())  # type: ignore

            if not getattr(cls, '__no_wrap_call__', False):

                @wraps(original_call)
                def call(self, *args, **kwargs):
                    if has_training:  # remove training flag if the original does not have it
                        training = kwargs.get('training', None)
                    else:
                        training = kwargs.pop('training', None)
                    r = original_call(self, *args, **kwargs)
                    if training or training == 'tracing' and self.enable_ebops:  # noqa: E712, training may be a special wrapper object
                        if isinstance(args[0], (tuple, list)):
                            tensors = args[0]
                        else:
                            tensors = args
                        shapes = ((1,) + shape[1:] for shape in map(ops.shape, tensors))
                        if self.enable_ebops:
                            ebops = self._compute_ebops(*shapes)
                            self._ebops.assign(ops.cast(ebops, self._ebops.dtype))
                            self.add_loss(ebops * self.beta)
                    if not self.enable_oq or self.__output_quantizer_handled__:
                        return r
                    assert not isinstance(r, (tuple, list)), (
                        f'Layer {self.name}({type(self)}) returns multiple outputs, which must be handled in subclasses.'
                    )
                    return self.oq(r, training=training)

                call.__signature__ = new_signature  # type: ignore

                cls.call = call

        return super().__call__(*args, **kwargs)  # type: ignore


class QLayerBase(Layer, metaclass=QLayerMeta):
    """Base class for quantized layers.

    Handles: instantiation of input and output quantizers,
    ebops and beta variable creation.

    call method is wrapped to compute ebops and apply output quantizer if applicable.
    """

    save_own_variables = Layer.save_own_variables
    load_own_variables = Layer.load_own_variables
    __output_quantizer_handled__ = False

    def __init__(
        self,
        enable_ebops: bool | None = None,
        beta0: int | float | np.number | None | Initializer = None,
        enable_oq: bool | None = None,
        enable_iq: bool | None = None,
        oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        beta0 = beta0 if beta0 is not None else global_config['beta0']
        beta0 = Constant(float(beta0)) if not isinstance(beta0, Initializer) else beta0

        self._enable_iq = enable_iq if enable_iq is not None else global_config['enable_iq']
        self._enable_oq = enable_oq if enable_oq is not None else global_config['enable_oq']
        if enable_ebops is None:
            enable_ebops = global_config['enable_ebops'] and self.enable_iq
        self._enable_ebops = enable_ebops
        self._beta0 = beta0

        if self.enable_oq:
            oq_conf = oq_conf or QuantizerConfig('default', 'datalane')
            self._oq = Quantizer(oq_conf, name=f'{self.name}_oq')

    @property
    def enable_iq(self):
        return self._enable_iq

    @property
    def enable_oq(self):
        return self._enable_oq

    @property
    def oq(self):
        if not self.enable_oq:
            raise AttributeError(f'oq has been disabled for {self.name}.')
        return self._oq

    @property
    def beta(self):
        if self._beta is None:
            return ops.cast(0, 'float32')
        return ops.cast(self._beta, ops.dtype(self._beta))

    @property
    def ebops(self):
        if self._ebops is None:
            return ops.cast(0, 'uint32')
        return ops.cast(self._ebops, ops.dtype(self._ebops))

    @property
    def enable_ebops(self):
        return self._enable_ebops

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        if self.enable_ebops:
            self._beta = self.add_weight(
                name='beta',
                shape=(),
                initializer=self._beta0,
                trainable=False,
            )
            self._ebops = self.add_weight(
                name='ebops',
                shape=(),
                initializer=Constant(0.0),
                trainable=False,
                dtype='uint32',
            )
        else:
            self._beta = None
            self._ebops = None

        if not self.enable_oq or self.oq.built:
            return
        self.try_build_output_quantizer(*args, **kwargs)

    def try_build_output_quantizer(self, input_shape=None, *args, **kwargs):
        try:
            output_shape = self.compute_output_shape(input_shape)
        except Exception:
            return
        self.oq.build(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'enable_ebops': self.enable_ebops,
                'beta0': self._beta0,
                'enable_oq': self.enable_oq,
                'enable_iq': self.enable_iq,
                'oq_conf': self.oq.config if self.enable_oq else None,
            }
        )
        return config

    def enable_lora(self, *args, **kwargs):
        raise NotImplementedError('LoRA is not supported in HGQ2.')

    @classmethod
    def from_config(cls, config):
        config = deserialize_keras_object(config)
        return super().from_config(config)

    def _compute_ebops(self, *args, **kwargs):
        raise NotImplementedError('This method is abstract and should be implemented in subclasses.')

    def _post_build(self):
        if self._enable_oq:
            assert hasattr(self, '_oq'), f'Output Quantizer is not defined for {self.name}, but enable_oq is True.'
        if self._enable_iq:
            assert hasattr(self, '_iq'), f'Input Quantizer is not defined for {self.name}, but enable_iq is True.'
        for sublayer in self._flatten_layers():
            assert sublayer.built, f'Sublayer {sublayer.name} is not built for {self.name}'


class QLayerBaseSingleInput(QLayerBase):
    def __init__(
        self,
        iq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.enable_iq:
            iq_conf = iq_conf or QuantizerConfig('default', 'datalane')
            self._iq = Quantizer(iq_conf, name=f'{self.name}_iq')

    @property
    def iq(self):
        if not self.enable_iq:
            raise AttributeError(f'iq has been disabled for {self.name}.')
        return self._iq

    def build(self, input_shape):
        super().build(input_shape)
        if self.enable_iq and not self._iq.built:
            self.iq.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['iq_conf'] = self.iq.config if self.enable_iq else None
        return config


class MultipleQuantizers(Layer):
    # Not subclassing tuple for stupid tensorflow tracing issue
    def __init__(self, configs: Sequence[QuantizerConfig], **kwargs):
        super().__init__(**kwargs)
        self.quantizers = tuple(Quantizer(config) for config in configs)

    def call(self, x, **kwargs):
        assert len(self) == len(x), f'{self.name} ({self.__class__.__name__}) expects {len(self)} inputs, got {len(x)}.'
        return tuple(f(x_, **kwargs) for f, x_ in zip(self, x))

    def build(self, shapes: Sequence[Sequence[int | None]]):  # type: ignore
        for q, shape in zip(self.quantizers, shapes):
            q.build(shape)
        self.built = True

    def __bool__(self):
        return len(self.quantizers) > 0

    def __len__(self):
        return len(self.quantizers)

    def __getitem__(self, item):
        return self.quantizers[item]

    def __iter__(self):
        return iter(self.quantizers)


class QLayerBaseMultiInputs(QLayerBase):
    def __init__(
        self,
        iq_confs: Sequence[QuantizerConfig] | QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._iqs_confs = None
        if self.enable_iq:
            self._iq_confs = iq_confs if iq_confs is not None else QuantizerConfig('default', 'datalane')

    @property
    def iq(self):
        if not self.enable_iq:
            raise AttributeError(f'iq has been disabled for {self.name}.')
        if not self.built:
            raise AttributeError(f'iqs is not available before build for {self.name}.')
        return self._iq

    @property
    def iq_confs(self):
        return self._iq_confs

    def build(self, input_shape):
        super().build(input_shape)
        n_input = len(input_shape)
        for _input_shape in input_shape:
            assert isinstance(_input_shape, Iterable), f'each element of input_shape must be iterable, got {_input_shape}'

        if isinstance(self.iq_confs, QuantizerConfig):
            self._iq_confs = [self.iq_confs] * n_input
        assert len(self.iq_confs) == n_input, (
            f'number of iq_confs must match number of inputs, got {len(self._iq_confs)} != {n_input}'
        )

        self._iq = MultipleQuantizers(self._iq_confs)  # type: ignore
        self._iq.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['iq_confs'] = self.iq_confs if self.enable_iq else None
        return config

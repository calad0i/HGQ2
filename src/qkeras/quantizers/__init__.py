from .quantized_bits import quantized_bits, quantized_relu

__all__ = ['quantized_bits', 'quantized_relu', 'get_quantizer']


def get_quantizer(str_conf: str):
    name = str_conf.split('(', 1)[0]
    if name in globals():
        return globals()[name].from_string(str_conf)
    raise ValueError(f'Unknown quantizer: {name}')

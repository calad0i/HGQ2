import os
import random
import time
from pathlib import Path

import keras
import numpy as np
import pytest


@pytest.fixture(scope='session', autouse=True, params=[42])
def set_random_seed(request):
    """Set random seeds for reproducibility"""

    seed = request.param
    np.random.seed(seed)
    random.seed(seed)
    backend = keras.backend.backend()
    match backend:
        case 'tensorflow':
            import tensorflow as tf

            tf.random.set_seed(seed)
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        case 'torch':
            import torch

            torch.manual_seed(seed)
        case 'jax':
            pass
        case _:
            raise ValueError(f'Unknown backend: {backend}')


@pytest.fixture(scope='session', autouse=True)
def configure_backend():
    backend = keras.backend.backend()

    match backend:
        case 'tensorflow':
            import tensorflow as tf

            tf.config.threading.set_intra_op_parallelism_threads(1)
        case 'torch':
            import torch

            torch.set_float32_matmul_precision('highest')
        case 'jax':
            import jax

            jax.config.update('jax_disable_jit', True)
            jax.config.update('jax_default_matmul_precision', 'float32')
        case _:
            raise ValueError(f'Unknown backend: {backend}')


_last_jax_cache_clear = 0.0


@pytest.fixture(autouse=True)
def _jax_cache_cleanup():
    """After each test, clear JAX compilation cache if 30s+ since last clear."""
    yield
    if keras.backend.backend() != 'jax':
        return
    global _last_jax_cache_clear
    now = time.monotonic()
    if now - _last_jax_cache_clear >= 30:
        import jax

        jax.clear_caches()
        _last_jax_cache_clear = now


@pytest.fixture(scope='session', autouse=True)
def set_hls4ml_configs():
    """Set default hls4ml configuration"""
    os.environ['HLS4ML_BACKEND'] = 'Vivado'


@pytest.fixture(scope='function')
def temp_directory(request: pytest.FixtureRequest):
    root = Path(os.environ.get('HGQ2_TEST_DIR', '/tmp/hgq2_test'))
    root.mkdir(exist_ok=True)

    test_name = request.node.name
    cls_name = request.cls.__name__ if request.cls else None
    if cls_name is None:
        test_dir = root / test_name
    else:
        test_dir = root / f'{cls_name}.{test_name}'
    test_dir.mkdir(exist_ok=True)
    return str(test_dir)


def pytest_sessionfinish(session, exitstatus):
    """whole test run finishes."""
    root = Path(os.environ.get('HGQ2_TEST_DIR', '/tmp/hgq2_test'))
    # Purge empty directories
    for path in root.glob('*'):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def pytest_addoption(parser):
    parser.addoption('--skip-slow', action='store_true', default=False, help='skip slow tests')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: mark test as slow to run')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--skip-slow'):
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason='need --runslow option to run')
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)

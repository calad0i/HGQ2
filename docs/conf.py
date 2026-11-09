# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from typing import Literal

import hgq

sys.path.insert(0, os.path.abspath('../'))

project = 'High Granularity Quantization 2'
copyright = '2025, Chang Sun'
author = 'Chang Sun'
release = str(hgq.__version__)
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

myst_enable_extensions = [
    'amsmath',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]


autosummary_generate = True

extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build']

html_logo = '_static/logo.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}

html_context = {
    'display_github': True,  # Integrate GitHub
    'github_user': 'calad0i',  # Username
    'github_repo': 'HGQ2',  # Repo name
    'github_version': 'master',  # Version
    'conf_py_path': '/docs/',  # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = '_static/icon.svg'


def _skip_triton_ops_internals(app, what, name, obj, skip, options) -> None | Literal[True]:
    # hgq.triton_ops: only enable and enabled are public
    if getattr(obj, '__module__', None) == 'hgq.triton_ops' and name not in ('enable', 'enabled'):
        return True
    return None


def setup(app):
    app.connect('autodoc-skip-member', _skip_triton_ops_internals)

import importlib
import os
import pkgutil
from collections import namedtuple

import torch


def load_ext(name, funcs):
    ext = importlib.import_module(name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext

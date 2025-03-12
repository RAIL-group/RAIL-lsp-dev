import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "lsp_accel",
        sources=["src/main.cpp"],
        language="c++",
    ),
]

setup(
    name='lsp_accel',
    version='0.1.0',
    description='C++ accelerated functions for Subgoal Planning',
    ext_modules=ext_modules,
    include_dirs=[pybind11.get_include(), 'src', '/usr/include/eigen3'],
)

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    name="scope_model",
    version="0.0.1",
    author="ZeZhi Wang",
    author_email="homura@mail.ustc.edu.cn",
    url="https://github.com/bbayukari/scope_example",
    description="A C++ autodiff example for scope algorithm in abess-team/abess",
    long_description="",
    ext_modules=[
        Pybind11Extension("scope_model",
            ["src/model.cpp"],
            include_dirs=["include"],
            cxx_std=17
            ),
    ],
    python_requires=">=3.7",
)

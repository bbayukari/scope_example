# scope_example
A C++ autodiff example for scope algorithm which is implemented in class `ConvexSparseSolver` of Python package `abess`. Specifically, this is a Python package with C++ extension that exports some loss functions which can be used in the autodiff method of scope. Scope is a user-friendly Sparse Convex Optimization algorithm for it only need the objective function written by common python code. If users need the program to become faster, writing the objective function by C++ is a good choice. However, in this way it may be hard to write correct code, so users can be use this repository as reference. As for further advanced usage, users should be familiar with pybind11 (https://github.com/pybind/pybind11), Eigen (https://eigen.tuxfamily.org) and autodiff (https://github.com/autodiff/autodiff).

## Prerequisites
+ A compiler with C++17 support
+ Python 3.7+ and package `pybind11`

## Installation
+ clone this repository
+ pip install ./scope_example

## Test call

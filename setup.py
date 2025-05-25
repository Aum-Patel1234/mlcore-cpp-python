from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "mlcore_cpp",
        ["cpp/bindings/mlcore_bindings.cpp", "cpp/src/linear_regression.cpp"],
        include_dirs=[pybind11.get_include(), "cpp/include"],
        language="c++",
    ),
]

setup(
    name="mlcore",
    version="0.1.0",
    author="Aum",
    description="Machine Learning algorithms with a fast C++ backend via pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

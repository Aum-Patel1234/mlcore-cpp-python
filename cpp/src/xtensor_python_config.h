#ifndef XTENSOR_PYTHON_CONFIG_H
#define XTENSOR_PYTHON_CONFIG_H

#define XTENSOR_PYTHON_USE_BOOST_SIMD
#define XTENSOR_PYTHON_USE_DYNAMIC_API 0

#define FORCE_IMPORT_ARRAY
// The Root of the undefined symbol Problem
//
//     When you link your C++ extension
//     module(like mlcore_cpp.cpython - 312 - x86_64 - linux - gnu.so) to
//     Python, it becomes a shared library.Shared libraries often rely on
//     symbols(functions, variables)
// from other shared
//     libraries.An "undefined symbol" error means that your mlcore_cpp.so
//         file was compiled and linked,
//     but it expects to find a specific function or
//         variable(in this case, xtensor_python_ARRAY_API) at runtime that isn
//         't available in the libraries it' s linked against,
//     or it's not being initialized properly.
#endif // XTENSOR_PYTHON_CONFIG_H

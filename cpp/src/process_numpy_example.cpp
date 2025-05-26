#include <cstddef>
#include <cstdint>
#include <iostream>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void print_array(py::array ndarray) {
  pybind11::buffer_info buffer = ndarray.request();
  auto dtype = py::str(ndarray.dtype()).cast<std::string>();
  std::cout << "ndarray reading via c++!" << std::endl;
  std::cout << "dtype - " << dtype << std::endl;
  std::size_t size = buffer.size;

  if (dtype == "int8") {
    int8_t *ptr = static_cast<int8_t *>(buffer.ptr);
    for (size_t i = 0; i < size; i++)
      std::cout << (int)ptr[i] << " "; // cast to int for readable output
  } else if (dtype == "int32") {
    int32_t *ptr = static_cast<int32_t *>(buffer.ptr);
    for (size_t i = 0; i < size; i++)
      std::cout << ptr[i] << " ";
  } else if (dtype == "float32") {
    float *ptr = static_cast<float *>(buffer.ptr);
    for (size_t i = 0; i < size; i++)
      std::cout << ptr[i] << " ";
  } else {
    double *ptr = static_cast<double *>(buffer.ptr);
    for (size_t i = 0; i < size; i++)
      std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
}

PYBIND11_MODULE(mlcore_cpp, m) {
  m.def("print_array_eg", &print_array, "Print numpy array");
}

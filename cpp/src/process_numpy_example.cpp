#include <cstddef>
#include <iostream>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void print_array(py::array_t<double> ndarray) {
  pybind11::buffer_info buffer = ndarray.request();
  double *ptr = static_cast<double *>(buffer.ptr);
  std::size_t size = buffer.size;

  std::cout << "ndarray reading via c++ : ";
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
}

PYBIND11_MODULE(mlcore_cpp, m) {
  m.def("print_array_eg", &print_array, "Print numpy array");
}

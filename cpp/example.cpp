#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class Example {
public:
  Example() { std::cout << "Hello From Constructor\n" << std::flush; }

  ~Example() { std::cout << "Destructor...\n" << std::flush; }
};

PYBIND11_MODULE(mlcore_cpp, m) {
  py::class_<Example, std::shared_ptr<Example>>(m, "Example").def(py::init<>());
}

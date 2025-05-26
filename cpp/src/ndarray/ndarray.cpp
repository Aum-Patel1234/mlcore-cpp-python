#include "ndarray.h"
#include <cstddef>
#include <cstdint>
#include <cxxabi.h>
#include <iostream>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/typeid.h>
#include <string>
#include <type_traits>
#include <vector>

namespace py = pybind11;

template <typename T> Ndarray<T>::Ndarray(py::array_t<T> numpy_array) {
  py::buffer_info buffer = numpy_array.request();
  this->shape_.assign(buffer.shape.begin(), buffer.shape.end());
  this->arr_size = static_cast<std::size_t>(buffer.size);

  T *ptr = static_cast<T *>(buffer.ptr);
  this->arr = std::vector<T>(ptr, ptr + buffer.size);
  for (std::size_t i = 0; i < arr_size; i++)
    std::cout << this->arr[i] << "\t";
  std::cout << std::endl;
}

template <typename T> std::vector<T> Ndarray<T>::get() const {
  return this->arr;
}

template <typename T> std::size_t Ndarray<T>::size() const {
  return this->arr_size;
}

template <typename T> int Ndarray<T>::ndim() const {
  return static_cast<int>(this->shape_.size());
}

template <typename T> std::vector<int> Ndarray<T>::shape() const {
  return this->shape_;
}

template <typename T> std::string Ndarray<T>::dtype() const {
  if constexpr (std::is_same_v<T, std::int8_t>)
    return "int8_t";
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return "uint8";
  else if constexpr (std::is_same_v<T, std::int16_t>)
    return "int16";
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return "uint16";
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return "int32";
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return "uint32";
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return "int64";
  else if constexpr (std::is_same_v<T, std::uint64_t>)
    return "uint64";
  else if constexpr (std::is_same_v<T, float>)
    return "float32";
  else if constexpr (std::is_same_v<T, double>)
    return "float64";
  else if constexpr (std::is_same_v<T, long double>)
    return "long double";
  else
    return "unknown";
}

// IMPORTANT: as it helps in compile time
template class Ndarray<std::int8_t>;
template class Ndarray<std::uint8_t>;
template class Ndarray<std::int16_t>;
template class Ndarray<std::uint16_t>;
template class Ndarray<std::int32_t>;
template class Ndarray<std::uint32_t>;
template class Ndarray<std::int64_t>;
template class Ndarray<std::uint64_t>;
template class Ndarray<float>;
template class Ndarray<double>;
template class Ndarray<long double>;

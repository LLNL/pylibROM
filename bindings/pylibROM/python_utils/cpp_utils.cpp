#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;

void *
extractSwigPtrAddress(const py::handle &swig_target)
{
    // On python, swig_target.this.__int__() returns the memory address of the wrapped c++ object pointer.
    // We execute the same command here in the c++ side, where the memory address is given as py::object.
    return swig_target.attr("this").attr("__int__")().cast<void *>();
}

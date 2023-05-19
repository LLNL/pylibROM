//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;
PYBIND11_MODULE(pylibROM, m) {
	py::module linalg = m.def_submodule("linalg");

    py::class_<Vector>(linalg, "Vector")

        // Constructor
        .def(py::init<int, bool>())

        // Constructor
        .def(py::init([](py::array_t<double> vec, bool distributed, bool copy_data = true) {
            py::buffer_info buf_info = vec.request();
            int dim = buf_info.shape[0];
            double* data = static_cast<double*>(buf_info.ptr);
            return new Vector(data, dim, distributed, copy_data);
        }))

        .def(py::init<const Vector&>())
        // Bind the assignment operator
        .def("__assign__", [](Vector& self, const Vector& rhs) { self = rhs; return self; })

        // Bind the addition operator
        .def(py::self += py::self)

        // Bind the subtraction operator
        .def(py::self -= py::self)

        //.def(py::self *= py::self)
        .def("fill", [](Vector& self, const double& value) { self = value; })

        // Bind the equal operator (set every element to a scalar)
        //.def("__set_scalar__", [](Vector& self, const double& a) { self = a; })

        // Bind the scaling operator (scale every element by a scalar)
        //.def("__scale__", [](Vector& self, const double& a) { self *= a; })

        .def("get_data", [](const Vector& self) {
            std::vector<double> data(self.dim());
            for (int i = 0; i < self.dim(); ++i) {
                data[i] = self.item(i);
            }
            return data;
        })

        .def("__del__", [](Vector& self) { self.~Vector(); }); // Destructor


}

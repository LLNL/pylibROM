//
// Created by barrow9 on 6/4/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "librom.h"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_NonuniformDMD(pybind11::module_ &m) {

    py::class_<NonuniformDMD, DMD>(m, "NonuniformDMD")

	//constructor, default.
	.def(py::init<std::string>()) //constructor a

    .def(py::init([](int dim,
                     bool alt_output_basis,
                     Vector* state_offset,
                     Vector* derivative_offset) {
        return new NonuniformDMD(dim, alt_output_basis, state_offset, derivative_offset);
    }), 
    py::arg("dim"),
    py::arg("alt_output_basis") = false,
    py::arg("state_offset") = nullptr,
    py::arg("derivative_offset") = nullptr)

    //TODO: needed explicitly?
    .def("__del__", [](NonuniformDMD& self) { self.~NonuniformDMD(); }) // Destructor

    .def("setOffset", &NonuniformDMD::setOffset, py::arg("offset_vector"), py::arg("order"))

    .def("load", &NonuniformDMD::load, py::arg("base_file_name"))
    .def("save", &NonuniformDMD::save, py::arg("base_file_name"));

}

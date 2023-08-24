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

void init_AdaptiveDMD(pybind11::module_ &m) {

    py::class_<AdaptiveDMD, DMD>(m, "AdaptiveDMD")

    .def(py::init([](int dim,
                     double desired_dt,
                     std::string rbf,
                     std::string interp_method,
                     double closest_rbf_val,
                     bool alt_output_basis,
                     Vector* state_offset) {
        return new AdaptiveDMD(dim, desired_dt, rbf, interp_method, closest_rbf_val, alt_output_basis, state_offset);
    }), 
    py::arg("dim"),
    py::arg("desired_dt") = -1.0,
    py::arg("rbf") = "G",
    py::arg("interp_method") = "LS",
    py::arg("closest_rbf_val") = 0.9,
    py::arg("alt_output_basis") = false,
    py::arg("state_offset") = nullptr)

    //TODO: needed explicitly?
    .def("__del__", [](AdaptiveDMD& self) { self.~AdaptiveDMD(); }) // Destructor

    .def("train", py::overload_cast<double, const Matrix*, double>(&AdaptiveDMD::train),
        py::arg("energy_fraction").noconvert(), py::arg("W0") = nullptr, py::arg("linearity_tol") = 0.0)
    .def("train", py::overload_cast<int, const Matrix*, double>(&AdaptiveDMD::train),
        py::arg("k").noconvert(), py::arg("W0") = nullptr, py::arg("linearity_tol") = 0.0)

    .def("getTrueDt", &AdaptiveDMD::getTrueDt)

    .def("getInterpolatedSnapshots", &AdaptiveDMD::getInterpolatedSnapshots);

}

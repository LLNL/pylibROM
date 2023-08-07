/*****************************************************
This file creates the python binder for the NNLSSolver
class as defined in libROM/lib/linalg/NNLS. 

created by Henry Yu 06.21.23
*****************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/NNLS.h"
#include "linalg/Matrix.h"
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;
using namespace pybind11::literals;


void init_NNLSSolver(pybind11::module_ &m) {
    py::class_<NNLSSolver>(m, "NNLSSolver")
        .def(py::init<double, int, int, int, double, double, int, int>(),
            py::arg("const_tol") = 1.0e-14,
            py::arg("min_nnz") = 0,
            py::arg("max_nnz") = 0,
            py::arg("verbosity") = 0,
            py::arg("res_change_termination_tol") = 1.0e-4,
            py::arg("zero_tol") = 1.0e-14,
            py::arg("n_outer") = 100000,
            py::arg("n_inner") = 100000)
        .def("set_verbosity", &NNLSSolver::set_verbosity,
            py::arg("verbosity_in"))
        .def("set_qrresidual_mode", &NNLSSolver::set_qrresidual_mode,
            py::arg("qr_residual_mode"))
        .def("solve_parallel_with_scalapack", &NNLSSolver::solve_parallel_with_scalapack,
            py::arg("matTrans"), py::arg("rhs_lb"), py::arg("rhs_ub"), py::arg("soln"))
        .def("normalize_constraints", &NNLSSolver::normalize_constraints,
            py::arg("matTrans"), py::arg("rhs_lb"), py::arg("rhs_ub"))
        .def("getNumProcs", &NNLSSolver::getNumProcs);

    py::enum_<NNLSSolver::QRresidualMode>(m, "QRresidualMode")
        .value("off", NNLSSolver::QRresidualMode::off)
        .value("on", NNLSSolver::QRresidualMode::on);

}

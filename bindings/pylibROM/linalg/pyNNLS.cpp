/*****************************************************
This file creates the python binder for the NNLSSolver
class as defined in libROM/lib/linalg/NNLS. 

created by Henry Yu 06.21.23
*****************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "NNLS.h"
#include "Matrix.h"
#include "Vector.h"

namespace py = pybind11;
using namespace CAROM;
using namespace pybind11::literals;

NNLSSolver createNNLSSolver(double const_tol, int min_nnz, int max_nnz,
                           int verbosity, double res_change_termination_tol,
                           double zero_tol, int n_outer, int n_inner) {
    return NNLSSolver(const_tol, min_nnz, max_nnz, verbosity,
                      res_change_termination_tol, zero_tol, n_outer, n_inner);
}

PYBIND11_MODULE(nnls, m) {
    py::class_<NNLSSolver>(m, "NNLSSolver")
        .def(py::init(&createNNLSSolver),
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
        .def("getNumProcs", &NNLSSolver::getNumProcs)
        .def_property_readonly("n_outer", &NNLSSolver::n_outer_)
        .def_property_readonly("n_inner", &NNLSSolver::n_inner_)
        .def_property_readonly("zero_tol", &NNLSSolver::zero_tol_)
        .def_property_readonly("const_tol", &NNLSSolver::const_tol_)
        .def_property_readonly("verbosity", &NNLSSolver::verbosity_)
        .def_property_readonly("min_nnz", &NNLSSolver::min_nnz_)
        .def_property_readonly("max_nnz", &NNLSSolver::max_nnz_)
        .def_property_readonly("res_change_termination_tol", &NNLSSolver::res_change_termination_tol_)
        .def_property_readonly("n_proc_max_for_partial_matrix", &NNLSSolver::n_proc_max_for_partial_matrix_)
        .def_property_readonly("normalize_const", &NNLSSolver::normalize_const_)
        .def_property_readonly("QR_reduce_const", &NNLSSolver::QR_reduce_const_)
        .def_property_readonly("NNLS_qrres_on", &NNLSSolver::NNLS_qrres_on_)
        .def_property_readonly("qr_residual_mode", &NNLSSolver::qr_residual_mode_)
        .def_property_readonly("num_procs", &NNLSSolver::getNumProcs);

    py::enum_<NNLSSolver::QRresidualMode>(m, "QRresidualMode")
        .value("off", NNLSSolver::QRresidualMode::off)
        .value("on", NNLSSolver::QRresidualMode::on);

    m.def(...);
}

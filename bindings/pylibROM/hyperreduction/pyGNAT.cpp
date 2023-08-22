#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "hyperreduction/GNAT.h"
#include "linalg/Matrix.h"

namespace py = pybind11;
using namespace CAROM;

void init_GNAT(pybind11::module_ &m) {
   m.def("GNAT", [](const Matrix* f_basis,
                    const int num_f_basis_vectors_used,
                    std::vector<int>& f_sampled_row,
                    std::vector<int>& f_sampled_rows_per_proc,
                    Matrix& f_basis_sampled_inv,
                    const int myid,
                    const int num_procs,
                    const int num_samples_req = -1,
                    std::vector<int>* init_samples = nullptr) {
     
         GNAT(f_basis, num_f_basis_vectors_used, f_sampled_row, f_sampled_rows_per_proc,
             f_basis_sampled_inv, myid, num_procs, num_samples_req, init_samples);
      return std::make_tuple(f_sampled_row, f_sampled_rows_per_proc,f_basis_sampled_inv);
    },py::arg("f_basis"),
    py::arg("num_f_basis_vectors_used"),
    py::arg("f_sampled_row"),
    py::arg("f_sampled_rows_per_proc"),
    py::arg("f_basis_sampled_inv"),
    py::arg("myid"),
    py::arg("num_procs"),
    py::arg("num_samples_req") = -1,
    py::arg("init_samples") = nullptr);
}
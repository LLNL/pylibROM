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
                    Matrix& f_basis_sampled_inv,
                    const int myid,
                    const int num_procs,
                    const int num_samples_req,
                    std::vector<int>& init_samples){  
      
        const int num_basis_vectors = std::min(num_f_basis_vectors_used, f_basis->numColumns());
        const int num_samples = num_samples_req > 0 ? num_samples_req : num_basis_vectors;
        std::vector<int> f_sampled_row(num_samples);
        std::vector<int> f_sampled_rows_per_proc(num_procs);
        GNAT(f_basis, num_f_basis_vectors_used, f_sampled_row, f_sampled_rows_per_proc,
             f_basis_sampled_inv, myid, num_procs, num_samples_req,&init_samples);
     return std::make_tuple(std::move(f_sampled_row), std::move(f_sampled_rows_per_proc));
    },py::arg("f_basis"),
    py::arg("num_f_basis_vectors_used"),
    py::arg("f_basis_sampled_inv"),
    py::arg("myid"),
    py::arg("num_procs"),
    py::arg("num_samples_req") = -1,
    py::arg("init_samples") =  std::vector<int>(0));
}
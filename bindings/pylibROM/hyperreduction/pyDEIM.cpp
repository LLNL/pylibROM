#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "hyperreduction/DEIM.h"
#include <vector>
#include "linalg/Matrix.h"

namespace py = pybind11;
using namespace CAROM;


void init_DEIM(pybind11::module_ &m) {
   m.def("DEIM", [](const Matrix* f_basis,int num_f_basis_vectors_used,
                  Matrix& f_basis_sampled_inv,int myid,int num_procs) {
      int num_basis_vectors = std::min(num_f_basis_vectors_used, f_basis->numColumns());
      std::vector<int> f_sampled_row(num_basis_vectors);
      std::vector<int> f_sampled_rows_per_proc(num_procs);
      DEIM(f_basis, num_f_basis_vectors_used,f_sampled_row, f_sampled_rows_per_proc,f_basis_sampled_inv, myid, num_procs);
      return std::make_tuple(std::move(f_sampled_row), std::move(f_sampled_rows_per_proc));
   });
}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "hyperreduction/QDEIM.h"
#include <vector>
#include "linalg/Matrix.h"

namespace py = pybind11;
using namespace CAROM;

void init_QDEIM(pybind11::module_ &m) {
   m.def("QDEIM", [](const Matrix* f_basis,int num_f_basis_vectors_used,std::vector<int>& f_sampled_row,std::vector<int>& f_sampled_rows_per_proc,
                  Matrix& f_basis_sampled_inv,const int myid,const int num_procs,const int num_samples_req) {
      QDEIM(f_basis, num_f_basis_vectors_used,f_sampled_row, f_sampled_rows_per_proc,f_basis_sampled_inv, myid, num_procs,num_samples_req);
      return std::make_tuple(f_sampled_row, f_sampled_rows_per_proc,f_basis_sampled_inv);
   });
}


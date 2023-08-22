#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <vector>
#include "librom.h"

namespace py = pybind11;
using namespace CAROM;

void init_STSampling(pybind11::module_ &m) {
    m.def("SpaceTimeSampling", [](const Matrix* s_basis,
                  const Matrix* t_basis,
                  const int num_f_basis_vectors_used,
                  std::vector<int>& t_samples,
                  py::array_t<int>  f_sampled_row,
                  py::array_t<int> f_sampled_rows_per_proc,
                  Matrix& s_basis_sampled,
                  const int myid,
                  const int num_procs,
                  const int num_t_samples_req = -1,  
                  const int num_s_samples_req = -1,
                  const bool excludeFinalTime = false) {
      py::buffer_info buf_info_f_sampled_row = f_sampled_row.request();
      int* f_sampled_row_data = static_cast<int*>(buf_info_f_sampled_row.ptr);
      py::buffer_info buf_info_f_sampled_rows_per_proc = f_sampled_rows_per_proc.request();
      int* f_sampled_rows_per_proc_data = static_cast<int*>(buf_info_f_sampled_rows_per_proc.ptr);
      SpaceTimeSampling(s_basis, t_basis,num_f_basis_vectors_used,t_samples,f_sampled_row_data, f_sampled_rows_per_proc_data, s_basis_sampled, myid, num_procs, num_t_samples_req,num_s_samples_req,excludeFinalTime);
      return t_samples;
    }, py::arg("s_basis"), py::arg("t_basis"), py::arg("num_f_basis_vectors_used"),
       py::arg("t_samples"), py::arg("f_sampled_row"), py::arg("f_sampled_rows_per_proc"),
       py::arg("s_basis_sampled"), py::arg("myid"), py::arg("num_procs"),
       py::arg("num_t_samples_req") = -1, py::arg("num_s_samples_req") = -1,
       py::arg("excludeFinalTime") = false);

    m.def("GetSampledSpaceTimeBasis", [](std::vector<int> const& t_samples,
                              const Matrix* t_basis,
                              Matrix const& s_basis_sampled,
                              Matrix& f_basis_sampled_inv) {
      GetSampledSpaceTimeBasis(t_samples,t_basis,s_basis_sampled,f_basis_sampled_inv);
      return t_samples;
    });
}

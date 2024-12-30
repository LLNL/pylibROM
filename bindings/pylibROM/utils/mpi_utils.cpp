//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "utils/mpi_utils.h"

#include "mpicomm.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_mpi_utils(pybind11::module_ &m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API.");
    }
    // m.def("split_dimension", (int ()(const int, const mpi4py_comm&)) &split_dimension);
    m.def("split_dimension", [](const int dim, const mpi4py_comm &comm) {
        return split_dimension(dim, comm.value);
    });
    // m.def("get_global_offsets", (int ()(const int, const MPI_Comm&)) &get_global_offsets);
    m.def("get_global_offsets", [](const int local_dim, const mpi4py_comm &comm) {
        std::vector<int> offsets;
        int global_dim = get_global_offsets(local_dim, offsets, comm.value);
        return py::make_tuple(global_dim, offsets);
    });
}

//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "mpi.h"
#include <mpi4py/mpi4py.h>
#include "utils/mpi_utils.h"

namespace py = pybind11;
using namespace CAROM;

struct mpi4py_comm {
    mpi4py_comm() = default;
    mpi4py_comm(MPI_Comm value) : value(value) {}
    operator MPI_Comm () { return value; }

    MPI_Comm value;
};

namespace pybind11 {
    namespace detail {
        template <> struct type_caster<mpi4py_comm> {
            public:
            PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

            // Python -> C++
            bool load(handle src, bool) {
                PyObject *py_src = src.ptr();

                // Check that we have been passed an mpi4py communicator
                if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
                // Convert to regular MPI communicator
                value.value = *PyMPIComm_Get(py_src);
                } else {
                return false;
                }

                return !PyErr_Occurred();
            }

            // C++ -> Python
            static handle cast(mpi4py_comm src,
                                return_value_policy /* policy */,
                                handle /* parent */)
            {
                // Create an mpi4py handle
                return PyMPIComm_New(src.value);
            }
        };
    }
} // namespace pybind11::detail

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

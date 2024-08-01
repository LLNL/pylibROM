#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mpi.h"
#include <mpi4py/mpi4py.h>

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
                if (src == MPI_COMM_NULL) {
                    return Py_None;
                }

                // Create an mpi4py handle
                return PyMPIComm_New(src.value);
            }
        };
    }
} // namespace pybind11::detail
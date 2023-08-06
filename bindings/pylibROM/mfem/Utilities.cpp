//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include "mfem/Utilities.hpp"

namespace py = pybind11;
using namespace CAROM;

// TODO(kevin): We do not bind mfem-related functions until we figure out how to type-cast SWIG Object.
//              Until then, mfem-related functions need to be re-implemented on python-end, using PyMFEM.

// Some drafts of type_caster for SWIG Object. However, it is not clear how we finish this.

// typedef struct {
//   PyObject_HEAD
//   void *ptr; // This is the pointer to the actual C++ instance
//   void *ty;  // swig_type_info originally, but shouldn't matter
//   int own;
//   PyObject *next;
// } SwigPyObject;

// namespace pybind11 {
//     namespace detail {
//         template <> struct type_caster<SwigPyObject> {
//             public:
//             PYBIND11_TYPE_CASTER(SwigPyObject, _("SwigPyObject"));

//             // Python -> C++
//             bool load(handle src, bool) {
//                 PyObject *py_src = src.ptr();

//                 // // Check that we have been passed an mpi4py communicator
//                 // if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
//                 // // Convert to regular MPI communicator
//                 // value.value = *PyMPIComm_Get(py_src);
//                 // } else {
//                 // return false;
//                 // }

//                 // return !PyErr_Occurred();
//                 return true;
//             }

//             // C++ -> Python
//             static handle cast(SwigPyObject src,
//                                 return_value_policy /* policy */,
//                                 handle /* parent */)
//             {
//                 // // Create an mpi4py handle
//                 // return PyMPIComm_New(src.value);
//                 throw std::runtime_error("SWIG Object casting from c++ to python is not implemented!\n");
//                 return py::cast(SwigPyObject(), return_value_policy::automatic);
//             }
//         };
//     }
// } // namespace pybind11::detail

void init_mfem_Utilities(pybind11::module_ &m) {
    
    m.def("ComputeCtAB", [](const HypreParMatrix& A, const CAROM::Matrix& B, const CAROM::Matrix& C, CAROM::Matrix& CtAB){
        ComputeCtAB(A, B, C, CtAB);
    });
    m.def("ComputeCtAB", [](const Operator& A, const CAROM::Matrix& B, const CAROM::Matrix& C, CAROM::Matrix& CtAB){
        ComputeCtAB(A, B, C, CtAB);
    });
}

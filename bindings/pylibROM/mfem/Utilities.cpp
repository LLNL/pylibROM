//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include "mfem/Utilities.hpp"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace mfem;

void init_mfem_Utilities(pybind11::module_ &m) {
    
    m.def("ComputeCtAB", [](py::object &A,
                            const CAROM::Matrix& B,
                            const CAROM::Matrix& C,
                            CAROM::Matrix& CtAB){
        HypreParMatrix *Aptr = extractSwigPtr<HypreParMatrix>(A);
        CAROM::ComputeCtAB(*Aptr, B, C, CtAB);
    });

    m.def("ComputeCtAB_vec", [](py::object &A,
                                py::object &B,
                                const CAROM::Matrix& C,
                                CAROM::Vector& CtAB_vec){
        HypreParMatrix *Aptr = extractSwigPtr<HypreParMatrix>(A);
        HypreParVector *Bptr = extractSwigPtr<HypreParVector>(B);
        CAROM::ComputeCtAB_vec(*Aptr, *Bptr, C, CtAB_vec);
    });

}

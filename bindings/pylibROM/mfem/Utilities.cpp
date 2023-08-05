//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/operators.h>
// #include <pybind11/stl.h>
#include "mfem/Utilities.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_mfem_Utilities(pybind11::module_ &m) {
    
    m.def("ComputeCtAB", [](const HypreParMatrix& A, const CAROM::Matrix& B, const CAROM::Matrix& C, CAROM::Matrix& CtAB){
        ComputeCtAB(A, B, C, CtAB);
    });
    m.def("ComputeCtAB", [](const Operator& A, const CAROM::Matrix& B, const CAROM::Matrix& C, CAROM::Matrix& CtAB){
        ComputeCtAB(A, B, C, CtAB);
    });
}

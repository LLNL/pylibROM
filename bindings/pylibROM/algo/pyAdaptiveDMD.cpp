#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "algo/AdaptiveDMD.h"
#include "linalg/Vector.h"
#include "linalg/Matrix.h"

namespace py = pybind11;
using namespace CAROM;

void init_AdaptiveDMD(pybind11::module_ &m){

    py::class_<AdaptiveDMD, DMD>(m, "AdaptiveDMD")
        .def(py::init<int, double, std::string, std::string, double, bool, Vector*>(),
            py::arg("dim"),
            py::arg("desired_dt") = -1.0,
            py::arg("rbf") = "G",
            py::arg("interp_method") = "LS",
            py::arg("closest_rbf_val") = 0.9,
            py::arg("alt_output_basis") = false,
            py::arg("state_offset") = nullptr)
        .def("train", (void (AdaptiveDMD::*)(double, const Matrix*, double)) &AdaptiveDMD::train,
            py::arg("energy_fraction").noconvert(),
            py::arg("W0") = nullptr,
            py::arg("linearity_tol") = 0.0)
        .def("train", (void (AdaptiveDMD::*)(int, const Matrix*, double)) &AdaptiveDMD::train,
            py::arg("k").noconvert(),
            py::arg("W0") = nullptr,
            py::arg("linearity_tol") = 0.0)
        .def("getTrueDt", &AdaptiveDMD::getTrueDt)
        .def("getInterpolatedSnapshots", &AdaptiveDMD::getInterpolatedSnapshots);
}


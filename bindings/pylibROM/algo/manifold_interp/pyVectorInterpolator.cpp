#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Matrix.h"
#include "linalg/Vector.h"
#include "algo/manifold_interp/VectorInterpolator.h"

namespace py = pybind11;
using namespace CAROM;

void
init_VectorInterpolator(pybind11::module_ &m)
{
    py::class_<VectorInterpolator>(m, "VectorInterpolator")
        .def(py::init<std::vector<Vector *>, std::vector<Matrix *>, std::vector<Vector *>, int, std::string, std::string, double>(),
             py::arg("parameter_points"),
             py::arg("rotation_matrices"),
             py::arg("reduced_vectors"),
             py::arg("ref_point"),
             py::arg("rbf") = "G",
             py::arg("interp_method") = "LS",
             py::arg("closest_rbf_val") = 0.9)
        .def("interpolate", &VectorInterpolator::interpolate)
        .def("__del__", [](VectorInterpolator& self) { self.~VectorInterpolator(); });

    m.def("obtainInterpolatedVector", &obtainInterpolatedVector);
    m.def("solveLinearSystem", &solveLinearSystem); 
}

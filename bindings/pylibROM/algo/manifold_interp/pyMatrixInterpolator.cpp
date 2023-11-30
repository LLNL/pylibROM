#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Matrix.h"
#include "linalg/Vector.h"
#include "algo/manifold_interp/MatrixInterpolator.h"

namespace py = pybind11;
using namespace CAROM;

void
init_MatrixInterpolator(pybind11::module_ &m)
{
    py::class_<MatrixInterpolator>(m, "MatrixInterpolator")
        .def(py::init<std::vector<Vector *>, std::vector<Matrix *>, std::vector<Matrix *>, int, std::string, std::string, std::string, double>(),
             py::arg("parameter_points"),
             py::arg("rotation_matrices"),
             py::arg("reduced_matrices"),
             py::arg("ref_point"),
             py::arg("matrix_type"),
             py::arg("rbf") = "G",
             py::arg("interp_method") = "LS",
             py::arg("closest_rbf_val") = 0.9)
        .def("interpolate", &MatrixInterpolator::interpolate, py::arg("point"), py::arg("orthogonalize") = false)
        .def("__del__", [](MatrixInterpolator& self) { self.~MatrixInterpolator(); }); 
}

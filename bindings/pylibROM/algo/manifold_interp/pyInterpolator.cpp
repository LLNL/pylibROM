#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Matrix.h"
#include "linalg/Vector.h"
#include "algo/manifold_interp/Interpolator.h"

namespace py = pybind11;
using namespace CAROM;

void
init_Interpolator(pybind11::module_ &m)
{
    m.def("obtainRBFToTrainingPoints", &obtainRBFToTrainingPoints);
    m.def("rbfWeightedSum", &rbfWeightedSum);
    m.def("obtainRBF", &obtainRBF);
    m.def("convertClosestRBFToEpsilon", &convertClosestRBFToEpsilon);
    m.def("obtainRotationMatrices", &obtainRotationMatrices);
}

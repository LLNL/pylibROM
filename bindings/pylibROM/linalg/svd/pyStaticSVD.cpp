#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/svd/StaticSVD.h"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;


class PyStaticSVD : public StaticSVD {
public:
    using StaticSVD::StaticSVD;


    bool takeSample(double* u_in, bool add_without_increase = false) override {
    PYBIND11_OVERRIDE(bool, StaticSVD, takeSample, u_in, add_without_increase);
    }


    const Matrix* getSpatialBasis() override {
    PYBIND11_OVERRIDE(const Matrix*, StaticSVD, getSpatialBasis);
    }


    const Matrix* getTemporalBasis() override {
    PYBIND11_OVERRIDE(const Matrix*, StaticSVD, getTemporalBasis);
    }


    const Vector* getSingularValues() override {
    PYBIND11_OVERRIDE(const Vector*, StaticSVD, getSingularValues);
    }


    const Matrix* getSnapshotMatrix() override {
    PYBIND11_OVERRIDE(const Matrix*, StaticSVD, getSnapshotMatrix);
    }


    PyStaticSVD(Options options) : StaticSVD(options) {}
    static PyStaticSVD* create(Options options) {
    return new PyStaticSVD(options);
    }
   
};


void init_StaticSVD(pybind11::module& m) {
    py::class_<StaticSVD, PyStaticSVD>(m, "StaticSVD")
        .def(py::init(&PyStaticSVD::create), py::arg("options"))
        .def("takeSample", [](StaticSVD& self, py::array_t<double> &u_in, bool add_without_increase = false) {
            bool result = self.takeSample(getVectorPointer(u_in), add_without_increase);
            return result;
        }, py::arg("u_in"), py::arg("add_without_increase") = false)
        .def("getSpatialBasis", (const Matrix* (StaticSVD::*)()) &StaticSVD::getSpatialBasis,py::return_value_policy::reference_internal)
        .def("getTemporalBasis", (const Matrix* (StaticSVD::*)()) &StaticSVD::getTemporalBasis,py::return_value_policy::reference_internal)
        .def("getSingularValues", (const Vector* (StaticSVD::*)()) &StaticSVD::getSingularValues,py::return_value_policy::reference_internal)
        .def("getSnapshotMatrix", (const Matrix* (StaticSVD::*)()) &StaticSVD::getSnapshotMatrix,py::return_value_policy::reference_internal)
        .def("getDim", (int (StaticSVD::*)() const) &StaticSVD::getDim)
        .def("getMaxNumSamples", (int (StaticSVD::*)() const) &StaticSVD::getMaxNumSamples)
        .def("getNumSamples", (int (StaticSVD::*)() const) &StaticSVD::getNumSamples);
        
}



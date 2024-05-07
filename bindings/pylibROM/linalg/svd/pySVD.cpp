#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/svd/SVD.h"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;


class PySVD : public SVD {
        public:
        using SVD::SVD; // Inherit constructors from the base class


        const Matrix* getSpatialBasis() override {
        PYBIND11_OVERRIDE_PURE(const Matrix*, SVD, getSpatialBasis );
        }


        const Matrix* getTemporalBasis() override {
        PYBIND11_OVERRIDE_PURE(const Matrix*, SVD, getTemporalBasis );
        }


        const Vector* getSingularValues() override {
        PYBIND11_OVERRIDE_PURE(const Vector*, SVD, getSingularValues );
        }


        const Matrix* getSnapshotMatrix() override {
        PYBIND11_OVERRIDE_PURE(const Matrix*, SVD, getSnapshotMatrix );
        }


        bool takeSample(double* u_in, bool add_without_increase) override {
        PYBIND11_OVERLOAD_PURE(bool, SVD, takeSample, u_in, add_without_increase);
        }


};


void init_SVD(pybind11::module_ &m) {
    py::class_<SVD, PySVD>(m, "SVD")
        .def(py::init<Options>())
        .def("takeSample", [](SVD& self, py::array_t<double> &u_in, bool add_without_increase = false) {
            return self.takeSample(getVectorPointer(u_in), add_without_increase);
        }, py::arg("u_in"), py::arg("add_without_increase") = false)
        .def("getDim", (int (SVD::*)() const) &SVD::getDim)
        .def("getSpatialBasis", (const Matrix* (SVD::*)()) &SVD::getSpatialBasis)
        .def("getTemporalBasis", (const Matrix* (SVD::*)()) &SVD::getTemporalBasis)
        .def("getSingularValues", (const Vector* (SVD::*)()) &SVD::getSingularValues)
        .def("getSnapshotMatrix", (const Matrix* (SVD::*)()) &SVD::getSnapshotMatrix)
        .def("getMaxNumSamples", (int (SVD::*)() const) &SVD::getMaxNumSamples)
        .def("getNumSamples", (int (SVD::*)() const) &SVD::getNumSamples)
        .def("__del__", [](SVD& self) {
        std::cout << "SVD instance is being destroyed" << std::endl;
        self.~SVD(); });
}

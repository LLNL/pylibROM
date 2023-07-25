#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/svd/SVD.h"


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


        bool takeSample(double* u_in, double time, bool add_without_increase) override {
        PYBIND11_OVERLOAD_PURE(bool, SVD, takeSample, u_in, time, add_without_increase);
        }


};


void init_SVD(pybind11::module_ &m) {
    py::class_<SVD, PySVD>(m, "SVD")
        .def(py::init<Options>())
        .def("takeSample", [](SVD& self, py::array_t<double> u_in, double time,bool add_without_increase = false) {
        py::buffer_info buf_info = u_in.request();
        // if (buf_info.ndim != 1)
        // throw std::runtime_error("Input array must be 1-dimensional");


        double* u_in_data = static_cast<double*>(buf_info.ptr);
        bool result = self.takeSample(u_in_data, time, add_without_increase);
        return result;
        }, py::arg("u_in"), py::arg("time"),py::arg("add_without_increase") = false)
        .def("getDim", (int (SVD::*)() const) &SVD::getDim)
        .def("getSpatialBasis", (const Matrix* (SVD::*)()) &SVD::getSpatialBasis)
        .def("getTemporalBasis", (const Matrix* (SVD::*)()) &SVD::getTemporalBasis)
        .def("getSingularValues", (const Vector* (SVD::*)()) &SVD::getSingularValues)
        .def("getSnapshotMatrix", (const Matrix* (SVD::*)()) &SVD::getSnapshotMatrix)
        .def("getNumBasisTimeIntervals", (int (SVD::*)() const) &SVD::getNumBasisTimeIntervals)
        .def("getBasisIntervalStartTime", (double (SVD::*)(int) const) &SVD::getBasisIntervalStartTime)
        .def("isNewTimeInterval", (bool (SVD::*)() const) &SVD::isNewTimeInterval)
        .def("increaseTimeInterval", (void (SVD::*)()) &SVD::increaseTimeInterval)
        .def("getNumSamples", (int (SVD::*)() const) &SVD::getNumSamples)
        .def("__del__", [](SVD& self) {
        std::cout << "SVD instance is being destroyed" << std::endl;
        self.~SVD(); });
}

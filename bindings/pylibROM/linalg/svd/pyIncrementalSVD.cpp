
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/svd/IncrementalSVD.h"


namespace py = pybind11;
using namespace CAROM;


class PyIncrementalSVD : public IncrementalSVD {


        public:
        using IncrementalSVD::IncrementalSVD;

        const Matrix* getSpatialBasis() override {
        PYBIND11_OVERRIDE(const Matrix*, IncrementalSVD, getSpatialBasis );
        }


        const Matrix* getTemporalBasis() override {
        PYBIND11_OVERRIDE(const Matrix*, IncrementalSVD, getTemporalBasis );
        }


        const Vector* getSingularValues() override {
        PYBIND11_OVERRIDE(const Vector*, IncrementalSVD, getSingularValues );
        }


        const Matrix* getSnapshotMatrix() override {
        PYBIND11_OVERRIDE(const Matrix*, IncrementalSVD, getSnapshotMatrix );
        }


        bool takeSample(double* u_in, double time, bool add_without_increase) override {
        PYBIND11_OVERLOAD(bool, IncrementalSVD, takeSample, u_in, time, add_without_increase);
        }


        ~PyIncrementalSVD() override {
        std::cout << "Destructor of PyIncrementalSVD is called!" << std::endl;
        }


        protected:


        void buildInitialSVD(double* u, double time) override {
        PYBIND11_OVERRIDE_PURE(void, IncrementalSVD, buildInitialSVD, u, time);
        }


        void computeBasis() override {
        PYBIND11_OVERRIDE_PURE(void, IncrementalSVD, computeBasis, );
        }


        void addLinearlyDependentSample(const Matrix* A, const Matrix* W, const Matrix* sigma) override {
        PYBIND11_OVERRIDE_PURE(void, IncrementalSVD, addLinearlyDependentSample, A, W, sigma);
        }


        void addNewSample(const Vector* j, const Matrix* A, const Matrix* W, Matrix* sigma) override {
        PYBIND11_OVERRIDE_PURE(void, IncrementalSVD, addNewSample, j, A, W, sigma);
        }

};

void init_IncrementalSVD(pybind11::module_ &m) {
    py::class_<IncrementalSVD,PyIncrementalSVD>(m, "IncrementalSVD")
        .def(py::init<Options, const std::string&>())
        .def("takeSample", [](IncrementalSVD& self, py::array_t<double> &u_in, double time,bool add_without_increase = false) {
        py::buffer_info buf_info = u_in.request();
        if (buf_info.ndim != 1)
        throw std::runtime_error("Input array must be 1-dimensional");
        double* u_in_data = static_cast<double*>(buf_info.ptr);
        bool result = self.takeSample(u_in_data, time, add_without_increase);
        return result;
        }, py::arg("u_in"), py::arg("time"),py::arg("add_without_increase") = false)
        .def("getSpatialBasis", (const Matrix* (IncrementalSVD::*)()) &IncrementalSVD::getSpatialBasis)
        .def("getTemporalBasis", (const Matrix* (IncrementalSVD::*)()) &IncrementalSVD::getTemporalBasis)
        .def("getSingularValues", (const Vector* (IncrementalSVD::*)()) &IncrementalSVD::getSingularValues)
        .def("getSnapshotMatrix", (const Matrix* (IncrementalSVD::*)()) &IncrementalSVD::getSnapshotMatrix)
        .def("getDim", (int (IncrementalSVD::*)() const) &IncrementalSVD::getDim)
        .def("getNumBasisTimeIntervals", (int (IncrementalSVD::*)() const) &IncrementalSVD::getNumBasisTimeIntervals)
        .def("getBasisIntervalStartTime", (double (IncrementalSVD::*)(int) const) &IncrementalSVD::getBasisIntervalStartTime)
        .def("isNewTimeInterval", (bool (IncrementalSVD::*)() const) &IncrementalSVD::isNewTimeInterval)
        .def("increaseTimeInterval", (void (IncrementalSVD::*)()) &IncrementalSVD::increaseTimeInterval)
        .def("getNumSamples", (int (IncrementalSVD::*)() const) &IncrementalSVD::getNumSamples);
}

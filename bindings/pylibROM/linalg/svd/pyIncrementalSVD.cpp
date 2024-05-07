
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/svd/IncrementalSVD.h"
#include "python_utils/cpp_utils.hpp"

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

    bool takeSample(double* u_in, bool add_without_increase) override {
        PYBIND11_OVERLOAD(bool, IncrementalSVD, takeSample, u_in, add_without_increase);
    }

    ~PyIncrementalSVD() override {
        std::cout << "Destructor of PyIncrementalSVD is called!" << std::endl;
    }

protected:

    void buildInitialSVD(double* u) override {
        PYBIND11_OVERRIDE_PURE(void, IncrementalSVD, buildInitialSVD, u);
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
        .def("takeSample", [](IncrementalSVD& self, py::array_t<double> &u_in, bool add_without_increase = false) {
            bool result = self.takeSample(getVectorPointer(u_in), add_without_increase);
            return result;
        }, py::arg("u_in"), py::arg("add_without_increase") = false)
        .def("getSpatialBasis", (const Matrix* (IncrementalSVD::*)()) &IncrementalSVD::getSpatialBasis)
        .def("getTemporalBasis", (const Matrix* (IncrementalSVD::*)()) &IncrementalSVD::getTemporalBasis)
        .def("getSingularValues", (const Vector* (IncrementalSVD::*)()) &IncrementalSVD::getSingularValues)
        .def("getSnapshotMatrix", (const Matrix* (IncrementalSVD::*)()) &IncrementalSVD::getSnapshotMatrix)
        .def("getDim", (int (IncrementalSVD::*)() const) &IncrementalSVD::getDim)
        .def("getMaxNumSamples", (int (IncrementalSVD::*)() const) &IncrementalSVD::getMaxNumSamples)
        .def("getNumSamples", (int (IncrementalSVD::*)() const) &IncrementalSVD::getNumSamples);
}

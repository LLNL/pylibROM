#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/BasisGenerator.h"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_BasisGenerator(pybind11::module_ &m) {

    py::class_<BasisGenerator>(m, "BasisGenerator")
        // .def(py::init<Options, bool, const std::string&, Database::formats>())
        .def(py::init<const Options&, bool, const std::string&, Database::formats>(),
            py::arg("options"),
            py::arg("incremental"),
            py::arg("basis_file_name") = "",
            py::arg("file_format") = Database::formats::HDF5
        )
        .def("isNextSample", (bool (BasisGenerator::*)(double)) &BasisGenerator::isNextSample)
        .def("updateRightSV", (bool (BasisGenerator::*)()) &BasisGenerator::updateRightSV)
        .def("takeSample", [](BasisGenerator& self, py::array_t<double> &u_in, bool add_without_increase = false) {
            return self.takeSample(getVectorPointer(u_in), add_without_increase);
        }, py::arg("u_in"), py::arg("add_without_increase") = false)
        .def("endSamples", &BasisGenerator::endSamples, py::arg("kind") = "basis")
        .def("writeSnapshot", (void (BasisGenerator::*)()) &BasisGenerator::writeSnapshot)
        .def("loadSamples", (void (BasisGenerator::*)(const std::string&, const std::string&, int, Database::formats)) &BasisGenerator::loadSamples,
            py::arg("base_file_name"),
            py::arg("kind") = "basis",
            py::arg("cut_off") = static_cast<int>(1e9),
            py::arg("db_format") = Database::formats::HDF5
        )
        .def("computeNextSampleTime", [](BasisGenerator& self, py::array_t<double> &u_in, py::array_t<double> &rhs_in, double time) {
             return self.computeNextSampleTime(getVectorPointer(u_in), getVectorPointer(rhs_in), time);
        }, py::arg("u_in"), py::arg("rhs_in"), py::arg("time"))

        .def("getSpatialBasis", (const Matrix* (BasisGenerator::*)()) &BasisGenerator::getSpatialBasis,py::return_value_policy::reference)
        .def("getTemporalBasis", (const Matrix* (BasisGenerator::*)()) &BasisGenerator::getTemporalBasis,py::return_value_policy::reference)
        .def("getSingularValues", (const Vector* (BasisGenerator::*)()) &BasisGenerator::getSingularValues,py::return_value_policy::reference)
        .def("getSnapshotMatrix", (const Matrix* (BasisGenerator::*)()) &BasisGenerator::getSnapshotMatrix,py::return_value_policy::reference)
        .def("getNumSamples",(int (BasisGenerator::*)() const) &BasisGenerator::getNumSamples)
        .def("__del__", [](BasisGenerator& self) { self.~BasisGenerator(); }); // Destructor

}

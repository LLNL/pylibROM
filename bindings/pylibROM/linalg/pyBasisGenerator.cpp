#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/BasisGenerator.h"


namespace py = pybind11;
using namespace CAROM;

void init_BasisGenerator(pybind11::module_ &m) {
    py::enum_<Database::formats>(m, "Formats")
        .value("HDF5", Database::formats::HDF5);

    py::class_<BasisGenerator>(m, "BasisGenerator")
        // .def(py::init<Options, bool, const std::string&, Database::formats>())
        .def(py::init<const Options&, bool, const std::string&, Database::formats>(),py::arg("options"),py::arg("incremental"),py::arg("basis_file_name") = "",py::arg("file_format") = static_cast<int>(Database::formats::HDF5))
        .def("isNextSample", (bool (BasisGenerator::*)(double)) &BasisGenerator::isNextSample)
        .def("updateRightSV", (bool (BasisGenerator::*)()) &BasisGenerator::updateRightSV)
        .def("takeSample", [](BasisGenerator& self, py::array_t<double> u_in, double time, double dt, bool add_without_increase = false) {
            py::buffer_info buf_info = u_in.request();
            if (buf_info.ndim != 1)
                throw std::runtime_error("Input array must be 1-dimensional");

            double* u_in_data = static_cast<double*>(buf_info.ptr);
            return self.takeSample(u_in_data, time, dt, add_without_increase);
        }, py::arg("u_in"), py::arg("time"), py::arg("dt"), py::arg("add_without_increase") = false)
        .def("endSamples", &BasisGenerator::endSamples, py::arg("kind") = "basis")
        .def("writeSnapshot", (void (BasisGenerator::*)()) &BasisGenerator::writeSnapshot)
        .def("loadSamples", &BasisGenerator::loadSamples, py::arg("base_file_name"), py::arg("kind") = "basis", py::arg("cut_off") = 1e9, py::arg("db_format") = static_cast<int>(Database::formats::HDF5))
        .def("computeNextSampleTime", [](BasisGenerator& self, py::array_t<double> u_in, py::array_t<double> rhs_in, double time) {
             py::buffer_info buf_info_u = u_in.request();
             py::buffer_info buf_info_rhs = rhs_in.request();
    
             if (buf_info_u.ndim != 1 || buf_info_rhs.ndim != 1)
               throw std::runtime_error("Input arrays must be 1-dimensional");

             double* u_in_data = static_cast<double*>(buf_info_u.ptr);
             double* rhs_in_data = static_cast<double*>(buf_info_rhs.ptr);

             return self.computeNextSampleTime(u_in_data, rhs_in_data, time);
        }, py::arg("u_in"), py::arg("rhs_in"), py::arg("time"))

        .def("getSpatialBasis", (const Matrix* (BasisGenerator::*)()) &BasisGenerator::getSpatialBasis,py::return_value_policy::reference)
        .def("getTemporalBasis", (const Matrix* (BasisGenerator::*)()) &BasisGenerator::getTemporalBasis,py::return_value_policy::reference)
        .def("getSingularValues", (const Vector* (BasisGenerator::*)()) &BasisGenerator::getSingularValues,py::return_value_policy::reference)
        .def("getSnapshotMatrix", (const Matrix* (BasisGenerator::*)()) &BasisGenerator::getSnapshotMatrix,py::return_value_policy::reference)
        .def("getNumBasisTimeIntervals", (int (BasisGenerator::*)() const) &BasisGenerator::getNumBasisTimeIntervals)
        .def("getBasisIntervalStartTime", (double (BasisGenerator::*)(int) const) &BasisGenerator::getBasisIntervalStartTime, py::arg("which_interval"))
        .def("getNumSamples",(int (BasisGenerator::*)() const) &BasisGenerator::getNumSamples)
        .def("__del__", [](BasisGenerator& self) { self.~BasisGenerator(); }); // Destructor

}

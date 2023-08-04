#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/BasisWriter.h"
#include "linalg/BasisGenerator.h"

namespace py = pybind11;
using namespace CAROM;

void init_BasisWriter(pybind11::module_ &m) {
    py::class_<BasisWriter>(m, "BasisWriter")
        .def(py::init<BasisGenerator*, const std::string&, Database::formats>(), py::arg("basis_generator"), py::arg("base_file_name"), py::arg("db_format") = static_cast<int>(Database::formats::HDF5))
        .def("writeBasis", &BasisWriter::writeBasis, py::arg("kind") = "basis")
        .def("__del__", [](BasisWriter& self) { self.~BasisWriter(); }); 
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Matrix.h"
#include "linalg/Vector.h"
#include "linalg/BasisReader.h"


namespace py = pybind11;
using namespace CAROM;

void init_BasisReader(pybind11::module_ &m) {
     py::class_<BasisReader>(m, "BasisReader")
        .def(py::init<const std::string&, Database::formats, const int>(),
            py::arg("base_file_name"),
            py::arg("file_format") = Database::formats::HDF5,
            py::arg("dim") = -1
        )
        .def("getSpatialBasis",(Matrix* (BasisReader::*)()) &BasisReader::getSpatialBasis)
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(int)) &BasisReader::getSpatialBasis,
            py::arg("n"))
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(int,int)) &BasisReader::getSpatialBasis,
            py::arg("start_col"),
            py::arg("end_col"))
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double)) &BasisReader::getSpatialBasis,
            py::arg("ef").noconvert())
        .def("getTemporalBasis",(Matrix* (BasisReader::*)()) &BasisReader::getTemporalBasis)
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(int)) &BasisReader::getTemporalBasis,
            py::arg("n"))
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(int,int)) &BasisReader::getTemporalBasis,
            py::arg("start_col"),
            py::arg("end_col"))
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double)) &BasisReader::getTemporalBasis,
            py::arg("ef").noconvert())
        .def("getSingularValues",(Vector* (BasisReader::*)()) &BasisReader::getSingularValues)
        .def("getSingularValues",(Vector* (BasisReader::*)(double)) &BasisReader::getSingularValues,
            py::arg("ef"))
        .def("getDim", (int (BasisReader::*)(const std::string)) &BasisReader::getDim,
            py::arg("kind"))
        .def("getNumSamples", (int (BasisReader::*)(const std::string)) &BasisReader::getNumSamples,
            py::arg("kind"))
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)()) &BasisReader::getSnapshotMatrix)
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(int)) &BasisReader::getSnapshotMatrix,
            py::arg("n"))
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(int,int)) &BasisReader::getSnapshotMatrix,
            py::arg("start_col"),
            py::arg("end_col"))
        .def("__del__", [](BasisReader& self) { self.~BasisReader(); }); // Destructor
}

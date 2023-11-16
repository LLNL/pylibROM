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
        .def(py::init<const std::string&, Database::formats>(),
            py::arg("base_file_name"),
            py::arg("file_format") = Database::formats::HDF5
        )
        .def("isNewBasis",(bool (BasisReader::*)(double)) &BasisReader::isNewBasis,
            py::arg("time"))
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double)) &BasisReader::getSpatialBasis,
            py::arg("time"))
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double,int)) &BasisReader::getSpatialBasis,
            py::arg("time"),
            py::arg("n"))
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double,int,int)) &BasisReader::getSpatialBasis,
            py::arg("time"),
            py::arg("start_col"),
            py::arg("end_col"))
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double,double)) &BasisReader::getSpatialBasis,
            py::arg("time"),
            py::arg("ef").noconvert())
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double)) &BasisReader::getTemporalBasis,
            py::arg("time"))
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double,int)) &BasisReader::getTemporalBasis,
            py::arg("time"),
            py::arg("n"))
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double,int,int)) &BasisReader::getTemporalBasis,
            py::arg("time"),
            py::arg("start_col"),
            py::arg("end_col"))
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double,double)) &BasisReader::getTemporalBasis,
            py::arg("time"),
            py::arg("ef").noconvert())
        .def("getSingularValues",(Vector* (BasisReader::*)(double)) &BasisReader::getSingularValues,
            py::arg("time"))
        .def("getSingularValues",(Vector* (BasisReader::*)(double,double)) &BasisReader::getSingularValues,
            py::arg("time"),
            py::arg("ef"))
        .def("getDim", (int (BasisReader::*)(const std::string,double)) &BasisReader::getDim,
            py::arg("kind"),
            py::arg("time"))
        .def("getNumSamples", (int (BasisReader::*)(const std::string,double)) &BasisReader::getNumSamples,
            py::arg("kind"),
            py::arg("time"))
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(double)) &BasisReader::getSnapshotMatrix,
            py::arg("time"))
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(double,int)) &BasisReader::getSnapshotMatrix,
            py::arg("time"),
            py::arg("n"))
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(double,int,int)) &BasisReader::getSnapshotMatrix,
            py::arg("time"),
            py::arg("start_col"),
            py::arg("end_col"))
        .def("__del__", [](BasisReader& self) { self.~BasisReader(); }); // Destructor
}

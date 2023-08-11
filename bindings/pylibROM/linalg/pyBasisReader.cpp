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
        .def("isNewBasis",(bool (BasisReader::*)(double)) &BasisReader::isNewBasis)
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double)) &BasisReader::getSpatialBasis)
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double,int)) &BasisReader::getSpatialBasis)
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double,int,int)) &BasisReader::getSpatialBasis)
        .def("getSpatialBasis",(Matrix* (BasisReader::*)(double,double)) &BasisReader::getSpatialBasis)
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double)) &BasisReader::getTemporalBasis)
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double,int)) &BasisReader::getTemporalBasis)
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double,int,int)) &BasisReader::getTemporalBasis)
        .def("getTemporalBasis",(Matrix* (BasisReader::*)(double,double)) &BasisReader::getTemporalBasis)
        .def("getSingularValues",(Vector* (BasisReader::*)(double)) &BasisReader::getSingularValues)
        .def("getSingularValues",(Vector* (BasisReader::*)(double,double)) &BasisReader::getSingularValues)
        .def("getDim", (int (BasisReader::*)(const std::string,double)) &BasisReader::getDim)
        .def("getNumSamples", (int (BasisReader::*)(const std::string,double)) &BasisReader::getNumSamples) 
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(double)) &BasisReader::getSnapshotMatrix)
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(double,int)) &BasisReader::getSnapshotMatrix)
        .def("getSnapshotMatrix",(Matrix* (BasisReader::*)(double,int,int)) &BasisReader::getSnapshotMatrix)
        .def("__del__", [](BasisReader& self) { self.~BasisReader(); }); // Destructor
}

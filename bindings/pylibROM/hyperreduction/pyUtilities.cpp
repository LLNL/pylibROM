#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "hyperreduction/Utilities.h"
#include "mpi.h"


namespace py = pybind11;
using namespace CAROM;

void init_Utilities(pybind11::module_ &m) {
    py::class_<RowInfo>(m, "RowInfo")
        .def(py::init<>())
        .def_readwrite("row_val", &RowInfo::row_val)
        .def_readwrite("row", &RowInfo::row)
        .def_readwrite("proc", &RowInfo::proc);
}

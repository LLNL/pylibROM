//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/HDFDatabase.h"
#include "utils/pyDatabase.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_HDFDatabase(pybind11::module_ &m) {

    py::class_<HDFDatabase, Database, PyDerivedDatabase<HDFDatabase>> hdfdb(m, "HDFDatabase");

    // Constructor
    hdfdb.def(py::init<>());

    hdfdb.def("create", &HDFDatabase::create);
    hdfdb.def("open", &HDFDatabase::open);
    hdfdb.def("close", &HDFDatabase::close);

    // TODO(kevin): finish binding of member functions.

    // hdfdb.def("__del__", [](HDFDatabase& self) { self.~HDFDatabase(); }); // Destructor

}


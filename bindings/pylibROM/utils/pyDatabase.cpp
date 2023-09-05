//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/Database.h"
#include "utils/pyDatabase.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_Database(pybind11::module_ &m) {

    py::class_<Database, PyDatabase<>> db(m, "Database");

    // Constructor
    db.def(py::init<>());

    py::enum_<Database::formats>(db, "formats")
        .value("HDF5", Database::formats::HDF5)
        .value("CSV", Database::formats::CSV)
        .export_values();

    // TODO(kevin): finish binding of member functions.
    db.def("getIntegerArray", [](
        Database &self, const std::string& key)
    {
        int data;
        self.getInteger(key, data);
        return data;
    });
}

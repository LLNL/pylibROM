//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/CSVDatabase.h"
#include "utils/pyDatabase.hpp"

namespace py = pybind11;
using namespace CAROM;

class PyCSVDatabase : public PyDerivedDatabase<CSVDatabase> {
public:
    using PyDerivedDatabase<CSVDatabase>::PyDerivedDatabase;

    void
    putComplexVector(
        const std::string& file_name,
        const std::vector<std::complex<double>>& data,
        int nelements) override
    {
        PYBIND11_OVERRIDE(
            void,                  /* Return type */
            CSVDatabase,       /* Child class */
            putComplexVector,        /* Name of function in C++ (must match Python name) */
            file_name, data, nelements   /* Argument(s) */
        );
    }

    void
    putStringVector(
        const std::string& file_name,
        const std::vector<std::string>& data,
        int nelements) override
    {
        PYBIND11_OVERRIDE(
            void,                  /* Return type */
            CSVDatabase,       /* Child class */
            putStringVector,        /* Name of function in C++ (must match Python name) */
            file_name, data, nelements   /* Argument(s) */
        );
    }

    // somehow this function is not virtual on c++ side. technically does not need this trampoline?
    void
    getStringVector(
        const std::string& file_name,
        std::vector<std::string> &data,
        bool append = false)
    {
        PYBIND11_OVERRIDE(
            void,                  /* Return type */
            CSVDatabase,       /* Child class */
            getStringVector,        /* Name of function in C++ (must match Python name) */
            file_name, data, append   /* Argument(s) */
        );
    }

    // somehow this function is not virtual on c++ side. technically does not need this trampoline?
    int
    getLineCount(
        const std::string& file_name)
    {
        PYBIND11_OVERRIDE(
            int,                  /* Return type */
            CSVDatabase,       /* Child class */
            getLineCount,        /* Name of function in C++ (must match Python name) */
            file_name           /* Argument(s) */
        );
    }
   
};

void init_CSVDatabase(pybind11::module_ &m) {

    py::class_<CSVDatabase, Database, PyCSVDatabase> csvdb(m, "CSVDatabase");

    // Constructor
    csvdb.def(py::init<>());

    csvdb.def("create", &CSVDatabase::create);
    csvdb.def("open", &CSVDatabase::open);
    csvdb.def("close", &CSVDatabase::close);

    // TODO(kevin): finish binding of member functions.
    csvdb.def("putDoubleArray", &CSVDatabase::putDoubleArray);
    csvdb.def("putDoubleVector", &CSVDatabase::putDoubleVector);
    csvdb.def("putInteger", &CSVDatabase::putInteger);
    csvdb.def("putIntegerArray", &CSVDatabase::putIntegerArray);

}


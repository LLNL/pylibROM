//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/CSVDatabase.h"
#include "utils/pyDatabase.hpp"
#include "python_utils/cpp_utils.hpp"

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
    csvdb.def("putDoubleArray", [](
        CSVDatabase &self, const std::string& key, py::array_t<double> &data, int nelements)
    {
        self.putDoubleArray(key, getVectorPointer(data), nelements);
    });
    csvdb.def("putDoubleVector", &CSVDatabase::putDoubleVector);
    csvdb.def("putInteger", &CSVDatabase::putInteger);
    csvdb.def("putIntegerArray", [](
        CSVDatabase &self, const std::string& key, py::array_t<int> &data, int nelements)
    {
        self.putIntegerArray(key, getVectorPointer(data), nelements);
    });

    csvdb.def("getIntegerArray", [](
        CSVDatabase &self, const std::string& key, int nelements)
    {
        int *dataptr = new int[nelements];
        self.getIntegerArray(key, dataptr, nelements);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    csvdb.def("getIntegerVector", [](
        CSVDatabase &self, const std::string& key, bool append)
    {
        std::vector<int> *datavec = new std::vector<int>;
        self.getIntegerVector(key, *datavec, append);
        return get1DArrayFromPtr(datavec->data(), datavec->size(), true);
    },
    py::arg("key"), py::arg("append") = false);

    csvdb.def("getDoubleArray", [](
        CSVDatabase &self, const std::string& key, int nelements)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    csvdb.def("getDoubleArray", [](
        CSVDatabase &self, const std::string& key, int nelements, const std::vector<int>& idx)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements, idx);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    csvdb.def("getDoubleArray", [](
        CSVDatabase &self, const std::string& key, int nelements,
        int offset, int block_size, int stride)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements, offset, block_size, stride);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    csvdb.def("getDoubleVector", [](
        CSVDatabase &self, const std::string& key, bool append)
    {
        std::vector<double> *datavec = new std::vector<double>();
        self.getDoubleVector(key, *datavec, append);
        return get1DArrayFromPtr(datavec->data(), datavec->size(), true);
    },
    py::arg("key"), py::arg("append") = false);

    csvdb.def("getDoubleArraySize", &CSVDatabase::getDoubleArraySize);

    csvdb.def("getStringVector", [](
        CSVDatabase &self, const std::string& file_name, bool append)
    {
        std::vector<std::string> data;
        self.getStringVector(file_name, data, append);
        return py::cast(data);
    }, py::arg("file_name"), py::arg("append") = false);

    csvdb.def("getLineCount", &CSVDatabase::getLineCount);

}


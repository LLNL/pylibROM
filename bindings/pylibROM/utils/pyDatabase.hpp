//
// Created by sullan2 on 4/20/23.
//
#ifndef PYDATABASE_HPP
#define PYDATABASE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/Database.h"

namespace py = pybind11;
using namespace CAROM;

template <class DatabaseType = Database>
class PyDatabase : public DatabaseType
{
public:
    using DatabaseType::DatabaseType; // Inherit constructors from the base class

    bool
    create(const std::string& file_name) override
    {
        PYBIND11_OVERRIDE_PURE(
            bool,           /* Return type */
            DatabaseType,       /* Parent class */
            create,         /* Name of function in C++ (must match Python name) */
            file_name        /* Argument(s) */
        );
    }

    bool
    open(
        const std::string& file_name,
        const std::string& type) override
    {
        PYBIND11_OVERRIDE_PURE(
            bool,                  /* Return type */
            DatabaseType,              /* Parent class */
            open,                  /* Name of function in C++ (must match Python name) */
            file_name, type         /* Argument(s) */
        );
    }

    bool
    close() override
    {
        PYBIND11_OVERRIDE_PURE(
            bool,                  /* Return type */
            DatabaseType,              /* Parent class */
            close                  /* Name of function in C++ (must match Python name) */
        );
    }

    void
    putIntegerArray(
        const std::string& key,
        const int* const data,
        int nelements) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                  /* Return type */
            DatabaseType,              /* Parent class */
            putIntegerArray,       /* Name of function in C++ (must match Python name) */
            key, data, nelements   /* Argument(s) */
        );
    }

    void
    putDoubleArray(
        const std::string& key,
        const double* const data,
        int nelements) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                  /* Return type */
            DatabaseType,              /* Parent class */
            putDoubleArray,        /* Name of function in C++ (must match Python name) */
            key, data, nelements   /* Argument(s) */
        );
    }

    void
    putDoubleVector(
        const std::string& key,
        const std::vector<double>& data,
        int nelements) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                  /* Return type */
            DatabaseType,              /* Parent class */
            putDoubleVector,        /* Name of function in C++ (must match Python name) */
            key, data, nelements   /* Argument(s) */
        );
    }

    void
    getIntegerArray(
        const std::string& key,
        int* data,
        int nelements) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                  /* Return type */
            DatabaseType,              /* Parent class */
            getIntegerArray,        /* Name of function in C++ (must match Python name) */
            key, data, nelements   /* Argument(s) */
        );
    }

    int
    getDoubleArraySize(const std::string& key) override
    {
        PYBIND11_OVERRIDE_PURE(
            int,                  /* Return type */
            DatabaseType,              /* Parent class */
            getDoubleArraySize,        /* Name of function in C++ (must match Python name) */
            key                     /* Argument(s) */
        );
    }

    void
    getDoubleArray(
        const std::string& key,
        double* data,
        int nelements) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                  /* Return type */
            DatabaseType,              /* Parent class */
            getDoubleArray,        /* Name of function in C++ (must match Python name) */
            key, data, nelements   /* Argument(s) */
        );
    }

    void
    getDoubleArray(
        const std::string& key,
        double* data,
        int nelements,
        const std::vector<int>& idx) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                       /* Return type */
            DatabaseType,                   /* Parent class */
            getDoubleArray,             /* Name of function in C++ (must match Python name) */
            key, data, nelements, idx   /* Argument(s) */
        );
    }

    void
    getDoubleArray(
        const std::string& key,
        double* data,
        int nelements,
        int offset,
        int block_size,
        int stride) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,                       /* Return type */
            DatabaseType,                   /* Parent class */
            getDoubleArray,             /* Name of function in C++ (must match Python name) */
            key, data, nelements,       /* Argument(s) */
            offset, block_size, stride
        );
    }

};

#endif
//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/HDFDatabase.h"
#include "utils/pyDatabase.hpp"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;

void init_HDFDatabase(pybind11::module_ &m) {

    py::class_<HDFDatabase, Database, PyDerivedDatabase<HDFDatabase>> hdfdb(m, "HDFDatabase");

    // Constructor
    hdfdb.def(py::init<>());

    hdfdb.def("create", [](HDFDatabase &self, const std::string& file_name,
                           const mpi4py_comm &comm) -> bool {
        return self.create(file_name, comm.value);
    });
    hdfdb.def("create", [](HDFDatabase &self, const std::string& file_name) -> bool {
        return self.create(file_name, MPI_COMM_NULL);
    });
    hdfdb.def("open", [](HDFDatabase &self, const std::string& file_name,
                         const std::string &type, const mpi4py_comm &comm) -> bool {
        return self.open(file_name, type, comm.value);
    });
    hdfdb.def("open", [](HDFDatabase &self, const std::string& file_name,
                         const std::string &type) -> bool {
        return self.open(file_name, type, MPI_COMM_NULL);
    });
    hdfdb.def("close", &HDFDatabase::close);

    // TODO(kevin): finish binding of member functions.
    hdfdb.def("putDoubleArray", [](
        HDFDatabase &self, const std::string& key, py::array_t<double> &data, int nelements, bool distributed = false)
    {
        self.putDoubleArray(key, getVectorPointer(data), nelements, distributed);
    }, py::arg("key"), py::arg("data"), py::arg("nelements"), py::arg("distributed") = false);
    hdfdb.def("putDoubleVector", &HDFDatabase::putDoubleVector, py::arg("key"), py::arg("data"), py::arg("nelements"), py::arg("distributed") = false);

    hdfdb.def("putInteger", &HDFDatabase::putInteger);
    hdfdb.def("putIntegerArray", [](
        HDFDatabase &self, const std::string& key, py::array_t<int> &data, int nelements, bool distributed = false)
    {
        self.putIntegerArray(key, getVectorPointer(data), nelements);
    }, py::arg("key"), py::arg("data"), py::arg("nelements"), py::arg("distributed") = false);

    hdfdb.def("getIntegerArray", [](
        HDFDatabase &self, const std::string& key, int nelements)
    {
        int *dataptr = new int[nelements];
        self.getIntegerArray(key, dataptr, nelements);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArray", [](
        HDFDatabase &self, const std::string& key, int nelements)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArray", [](
        HDFDatabase &self, const std::string& key, int nelements, const std::vector<int>& idx)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements, idx);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArray", [](
        HDFDatabase &self, const std::string& key, int nelements,
        int offset, int block_size, int stride)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements, offset, block_size, stride);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArraySize", &HDFDatabase::getDoubleArraySize);

    // hdfdb.def("__del__", [](HDFDatabase& self) { self.~HDFDatabase(); }); // Destructor

}


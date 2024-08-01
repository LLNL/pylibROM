#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/HDFDatabaseMPIO.h"
#include "utils/pyDatabase.hpp"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;

class PyHDFDatabaseMPIO : public PyDerivedDatabase<HDFDatabaseMPIO> {
public:
    using PyDerivedDatabase<HDFDatabaseMPIO>::PyDerivedDatabase;

};


void init_HDFDatabaseMPIO(pybind11::module_ &m) {

    py::class_<HDFDatabaseMPIO, HDFDatabase, PyHDFDatabaseMPIO> hdfdb(m, "HDFDatabaseMPIO");

    // Constructor
    hdfdb.def(py::init<>());

    hdfdb.def("create", [](HDFDatabaseMPIO &self, const std::string& file_name,
                           const mpi4py_comm &comm) -> bool {
        return self.create(file_name, comm.value);
    });
    hdfdb.def("open", [](HDFDatabaseMPIO &self, const std::string& file_name,
                         const std::string &type, const mpi4py_comm &comm) -> bool {
        return self.open(file_name, type, comm.value);
    });
    hdfdb.def("close", &HDFDatabase::close);

    // TODO(kevin): finish binding of member functions.
    hdfdb.def("putDoubleArray", [](
        HDFDatabaseMPIO &self, const std::string& key, py::array_t<double> &data, int nelements)
    {
        self.putDoubleArray(key, getVectorPointer(data), nelements);
    });
    hdfdb.def("putDoubleVector", &HDFDatabase::putDoubleVector);

    hdfdb.def("putInteger", &HDFDatabase::putInteger);
    hdfdb.def("putIntegerArray", [](
        HDFDatabaseMPIO &self, const std::string& key, py::array_t<int> &data, int nelements)
    {
        self.putIntegerArray(key, getVectorPointer(data), nelements);
    });

    hdfdb.def("getIntegerArray", [](
        HDFDatabaseMPIO &self, const std::string& key, int nelements)
    {
        int *dataptr = new int[nelements];
        self.getIntegerArray(key, dataptr, nelements);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArray", [](
        HDFDatabaseMPIO &self, const std::string& key, int nelements)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArray", [](
        HDFDatabaseMPIO &self, const std::string& key, int nelements, const std::vector<int>& idx)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements, idx);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("getDoubleArray", [](
        HDFDatabaseMPIO &self, const std::string& key, int nelements,
        int offset, int block_size, int stride)
    {
        double *dataptr = new double[nelements];
        self.getDoubleArray(key, dataptr, nelements, offset, block_size, stride);
        return get1DArrayFromPtr(dataptr, nelements, true);
    });

    hdfdb.def("writeAttribute", &HDFDatabaseMPIO::writeAttribute);
}


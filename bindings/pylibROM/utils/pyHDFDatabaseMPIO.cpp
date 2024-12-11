#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "utils/HDFDatabaseMPIO.h"
#include "utils/pyDatabase.hpp"
#include "python_utils/cpp_utils.hpp"
#include "CAROM_config.h"


namespace py = pybind11;
using namespace CAROM;

class PyHDFDatabaseMPIO : public PyDerivedDatabase<HDFDatabaseMPIO> {
public:
    using PyDerivedDatabase<HDFDatabaseMPIO>::PyDerivedDatabase;

};


void init_HDFDatabaseMPIO(pybind11::module_ &m) {

    py::class_<HDFDatabaseMPIO, HDFDatabase, PyDerivedDatabase<HDFDatabaseMPIO>> hdfdb(m, "HDFDatabaseMPIO");

    // Constructor
    hdfdb.def(py::init<>());

#if HDF5_IS_PARALLEL
    hdfdb.def("create", [](HDFDatabaseMPIO &self, const std::string& file_name,
                           const mpi4py_comm &comm) -> bool {
        return self.create(file_name, comm.value);
    });
    hdfdb.def("create", [](HDFDatabaseMPIO &self, const std::string& file_name) -> bool {
        return self.create(file_name, MPI_COMM_NULL);
    });
    hdfdb.def("open", [](HDFDatabaseMPIO &self, const std::string& file_name,
                         const std::string &type, const mpi4py_comm &comm) -> bool {
        return self.open(file_name, type, comm.value);
    });
    hdfdb.def("close", &HDFDatabaseMPIO::close);

    // TODO(kevin): finish binding of member functions.
    hdfdb.def("putDoubleArray", [](
        HDFDatabaseMPIO &self, const std::string& key, py::array_t<double> &data, int nelements, bool distributed = false)
    {
        self.putDoubleArray(key, getVectorPointer(data), nelements);
    }, py::arg("key"), py::arg("data"), py::arg("nelements"), py::arg("distributed") = false);
    hdfdb.def("putDoubleVector", &HDFDatabaseMPIO::putDoubleVector, py::arg("key"), py::arg("data"), py::arg("nelements"), py::arg("distributed") = false);

    hdfdb.def("putInteger", &HDFDatabaseMPIO::putInteger);
    hdfdb.def("putIntegerArray", [](
        HDFDatabaseMPIO &self, const std::string& key, py::array_t<int> &data, int nelements, bool distributed = false)
    {
        self.putIntegerArray(key, getVectorPointer(data), nelements);
    }, py::arg("key"), py::arg("data"), py::arg("nelements"), py::arg("distributed") = false);

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
#endif
}


//
// Created by barrow9 on 6/4/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/DMD.h"
#include "linalg/Vector.h"
#include "linalg/Matrix.h"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;
using namespace std;

void init_DMD(pybind11::module_ &m) {

    py::class_<DMD>(m, "DMD")
	//constructor, default.
	.def(py::init<string>()) //constructor a
    .def(py::init([](int dim,
                    double dt,
                    bool alt_output_basis = false,
                    Vector *vec = nullptr) {
            return new DMD(dim, dt, alt_output_basis, vec);
        }), py::arg("dim"), py::arg("dt"), py::arg("alt_output_basis") = false, py::arg("vec") = nullptr)

    .def("setOffset", &DMD::setOffset, py::arg("offset_vector"), py::arg("order"))
    .def("takeSample", [](DMD &self, py::array_t<double> &u_in, double t) {
        self.takeSample(getVectorPointer(u_in), t);
    })
    .def("train", py::overload_cast<double, const Matrix*, double>(&DMD::train),
        py::arg("energy_fraction").noconvert(), py::arg("W0") = nullptr, py::arg("linearity_tol") = 0.0)
    .def("train", py::overload_cast<int, const Matrix*, double>(&DMD::train),
        py::arg("k").noconvert(), py::arg("W0") = nullptr, py::arg("linearity_tol") = 0.0)
    .def("projectInitialCondition", &DMD::projectInitialCondition,
            py::arg("init"), py::arg("t_offset") = -1.0)
    .def("predict", &DMD::predict, py::arg("t"), py::arg("deg") = 0)
    .def("getTimeOffset", &DMD::getTimeOffset)
    .def("getNumSamples", &DMD::getNumSamples)
    .def("getDimension", &DMD::getDimension)
    .def("getSnapshotMatrix", &DMD::getSnapshotMatrix)
    .def("load", py::overload_cast<std::string>(&DMD::load), py::arg("base_file_name"))
    .def("load", py::overload_cast<const char*>(&DMD::load), py::arg("base_file_name"))
    .def("save", py::overload_cast<std::string>(&DMD::save), py::arg("base_file_name"))
    .def("save", py::overload_cast<const char*>(&DMD::save), py::arg("base_file_name"))
    .def("summary", &DMD::summary, py::arg("base_file_name"))

    //TODO: needed explicitly?
    .def("__del__", [](DMD& self) { self.~DMD(); }); // Destructor
	/*
        // Constructor
        .def(py::init<>())
        .def(py::init<int, bool>())

        // Constructor
        .def(py::init([](py::array_t<double> vec, bool distributed, bool copy_data = true) {
            py::buffer_info buf_info = vec.request();
            int dim = buf_info.shape[0];
            double* data = static_cast<double*>(buf_info.ptr);
            return new Vector(data, dim, distributed, copy_data);
        }))

        // Bind the copy constructor
        .def(py::init<const Vector&>())

        // Bind the assignment operator
        .def("__assign__", [](Vector& self, const Vector& rhs) { self = rhs; return self; })

        // Bind the addition operator
        .def(py::self += py::self)

        // Bind the subtraction operator
        .def(py::self -= py::self)

        //.def(py::self *= py::self)
        .def("fill", [](Vector& self, const double& value) { self = value; })

        //Bind the equal operator (set every element to a scalar)
        .def("__set_scalar__", [](Vector& self, const double& a) { self = a; })

        //Bind the scaling operator (scale every element by a scalar)
        .def("__scale__", [](Vector& self, const double& a) { self *= a; })

        //Bind transforms
        .def("transform", [](Vector &self, py::function transformer) {
            self.transform([transformer](const int size, double* vector) {
                transformer(size, py::array_t<double>(size, vector));
            });
        })
        .def("transform", [](Vector &self, Vector& result, py::function transformer) {
            self.transform(result, [transformer](const int size, double* vector) {
                transformer(size, py::array_t<double>(size, vector));
            });
        })
        .def("transform", [](Vector &self, Vector* result, py::function transformer) {
            self.transform(result, [transformer](const int size, double* vector) {
                transformer(size, py::array_t<double>(size, vector));
            });
        })
        .def("transform", [](Vector &self, py::function transformer) {
            self.transform([transformer](const int size, double* origVector, double* resultVector) {
                transformer(size, py::array_t<double>(size, origVector), py::array_t<double>(size, resultVector));
            });
        })
        .def("transform", [](Vector &self, Vector& result, py::function transformer) {
            self.transform(result, [transformer](const int size, double* origVector, double* resultVector) {
                transformer(size, py::array_t<double>(size, origVector), py::array_t<double>(size, resultVector));
            });
        })
        .def("transform", [](Vector &self, Vector* result, py::function transformer) {
            self.transform(result, [transformer](const int size, double* origVector, double* resultVector) {
                transformer(size, py::array_t<double>(size, origVector), py::array_t<double>(size, resultVector));
            });
        })

        //Bind set size method
        .def("setSize", &Vector::setSize)

        .def("distributed", &Vector::distributed)

        .def("dim", &Vector::dim)

        .def("inner_product", (double (Vector::*)(const Vector&) const) &Vector::inner_product)
        .def("inner_product", (double (Vector::*)(const Vector*) const) &Vector::inner_product)

        .def("norm", &Vector::norm)
        .def("norm2", &Vector::norm2)
        .def("normalize", &Vector::normalize)

        .def("plus", (Vector* (Vector::*)(const Vector&) const) &Vector::plus)
        .def("plus", (Vector* (Vector::*)(const Vector*) const) &Vector::plus)
//        .def("plus", (void (Vector::*)(const Vector&, Vector*&) const) &Vector::plus)
        .def("plus", (void (Vector::*)(const Vector&, Vector&) const) &Vector::plus)

        .def("plusAx", [](Vector& self, double factor, const Vector& other) {
            Vector* result = self.plusAx(factor, other);
            return result;
        }, py::return_value_policy::automatic)
        .def("plusAx", [](Vector* self, double factor, const Vector* other) {
            Vector* result = self->plusAx(factor, *other);
            return result;
        }, py::return_value_policy::automatic)
//        .def("plusAx", [](Vector& self, double factor, const Vector& other) {
//            Vector* result = new Vector();
//            self.plusAx(factor, other, result);
//            return result;
//        }, py::arg("factor"), py::arg("other"), py::return_value_policy::take_ownership)
        .def("plusAx", [](Vector& self, double factor, const Vector& other) {
            Vector result;
            self.plusAx(factor, other, result);
            return result;
        }, py::return_value_policy::automatic)

        .def("plusEqAx", (void (Vector::*)(double, const Vector&)) &Vector::plusEqAx)
        .def("plusEqAx", [](Vector& self, double factor, const Vector* other) {
            self.plusEqAx(factor, *other);
        }, py::return_value_policy::automatic)

        .def("minus", (Vector* (Vector::*)(const Vector&) const) &Vector::minus)
        .def("minus", (Vector* (Vector::*)(const Vector*) const) &Vector::minus)
        //        .def("minus", (void (Vector::*)(const Vector&, Vector*&) const) &Vector::minus)
        .def("minus", (void (Vector::*)(const Vector&, Vector&) const) &Vector::minus)

//        .def("mult", [](const Vector& self, double factor) {
//            Vector* result = self.mult(factor);
//            return result;
//        }, py::return_value_policy::automatic)
//        .def("mult", [](const Vector& self, double factor, Vector*& result) {
//            self.mult(factor, result);
//        })
//        .def("mult", [](const Vector& self, double factor, Vector& result) {
//            self.mult(factor, result);
//        })
//        .def("__getitem__", (const double& (Vector::*)(int) const) &Vector::item)
//        .def("__setitem__", (double& (Vector::*)(int)) &Vector::item)
//        .def("__call__", (const double& (Vector::*)(int) const) &Vector::operator())
//        .def("__call__", (double& (Vector::*)(int)) &Vector::operator())
        .def("print", &Vector::print)
        .def("write", &Vector::write)
        .def("read", &Vector::read)
        .def("local_read", &Vector::local_read)
        .def("getData", &Vector::getData)
        .def("localMin", &Vector::localMin)

        .def("get_data", [](const Vector& self) {
            std::vector<double> data(self.dim());
            for (int i = 0; i < self.dim(); ++i) {
                data[i] = self.item(i);
            }
            return data;
        })


        .def("__del__", [](Vector& self) { self.~Vector(); }); // Destructor
	*/
}

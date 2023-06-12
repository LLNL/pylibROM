//
// Created by barrow9 on 6/4/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/DMD.h"
#include "linalg/Vector.h"

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
			py::array_t<double> vec)	//state_offset
		{
			py::buffer_info buf_info = vec.request();
			int vdim = buf_info.shape[0];
			double* data = static_cast<double*>(buf_info.ptr);
			
			if(data != NULL)
			{
				Vector* state_offset = new Vector(data, vdim, 
								false, 	//TODO?
								true	//TODO?
									);
				return new DMD(dim, dt, alt_output_basis, state_offset);
			}
			else
				return new DMD(dim, dt, alt_output_basis, NULL);
		}
					)) 
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

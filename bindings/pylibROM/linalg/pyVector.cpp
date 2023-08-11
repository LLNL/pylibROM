//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;

py::buffer_info
bufferInfo(Vector &self)
{
    return py::buffer_info(
        self.getData(),                         /* Pointer to buffer */
        sizeof(double),                          /* Size of one scalar */
        py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
        1,                                      /* Number of dimensions */
        { self.dim() },                         /* Buffer dimensions */
        { sizeof(double) }                       /* Strides (in bytes) for each index */
    );
}

void init_vector(pybind11::module_ &m) {

    py::class_<Vector>(m, "Vector", py::buffer_protocol()) 
        .def_buffer([](Vector &self) -> py::buffer_info {
            return bufferInfo(self);
        })

        // Constructor
        .def(py::init<>())
        .def(py::init<int, bool>())

        // Constructor
        .def(py::init([](py::array_t<double> &vec, bool distributed, bool copy_data = true) {
            py::buffer_info buf_info = vec.request();
            int dim = buf_info.shape[0];
            double* data = static_cast<double*>(buf_info.ptr);
            return new Vector(data, dim, distributed, copy_data);
        }), py::arg("vec"), py::arg("distributed"), py::arg("copy_data") = true) // default value needs to be defined here.

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
        .def("plus", [](const Vector& self,const Vector& other,Vector* result) {
            self.plus(other,*result);
        })
        .def("plus", (void (Vector::*)(const Vector&, Vector&) const) &Vector::plus)


        .def("plusAx", [](Vector& self, double factor, const Vector& other) {
            Vector* result = 0;
            self.plusAx(factor, other,result);
            return result;
        }, py::return_value_policy::automatic)
        .def("plusAx", (Vector* (Vector::*)(double,const Vector*)) &Vector::plusAx, py::return_value_policy::automatic)
        .def("plusAx", [](const Vector& self, double factor, const Vector& other,Vector* result) {
            self.plusAx(factor,other,*result);
        })
        .def("plusAx", (void (Vector::*)(double, const Vector&,Vector&) const) &Vector::plusAx)

        
        .def("plusEqAx", (void (Vector::*)(double, const Vector&)) &Vector::plusEqAx)
        .def("plusEqAx", [](Vector& self, double factor, const Vector* other) {
            self.plusEqAx(factor, *other);
        }, py::return_value_policy::automatic)

        .def("minus", (Vector* (Vector::*)(const Vector&) const) &Vector::minus)
        .def("minus", (Vector* (Vector::*)(const Vector*) const) &Vector::minus)
        .def("minus",[](const Vector& self,const Vector& other,Vector* result){
                self.minus(other,*result); 
        })
        .def("minus", (void (Vector::*)(const Vector&, Vector&) const) &Vector::minus)

        .def("mult", [](const Vector& self, double factor) {
           Vector* result = 0;
           self.mult(factor,result);
           return result;
        }, py::return_value_policy::automatic)
        .def("mult", [](const Vector& self, double factor, Vector* result) {
           self.mult(factor, *result);
        })
        .def("mult", [](const Vector& self, double factor, Vector& result) {
           self.mult(factor, result);
        })

        .def("item", (const double& (Vector::*)(int) const) &Vector::item)
        .def("__getitem__", [](const Vector& self, int i) { 
            return self(i);
        })
        .def("__setitem__", [](Vector& self, int i, double value) { 
            self.item(i) = value; 
        })
        .def("__call__", (const double& (Vector::*)(int) const) &Vector::operator())
        .def("__call__", (double& (Vector::*)(int)) &Vector::operator())
        .def("print", &Vector::print)
        .def("write", &Vector::write)
        .def("read", &Vector::read)
        .def("local_read", &Vector::local_read)
        .def("getData", [](Vector& self) {
            // We provide a view vector, which does not free the memory at its destruction.
            py::capsule buffer_handle([](){});
            // Use this if the C++ memory SHOULD be deallocated
            // once the Python no longer has a reference to it
            // py::capsule buffer_handle(data_buffer, [](void* p){ free(p); });

            return py::array(bufferInfo(self), buffer_handle);
        })
        .def("localMin", &Vector::localMin)

        // TODO: this needs re-naming. confusing with getData.
        .def("get_data", [](const Vector& self) {
            std::vector<double> data(self.dim());
            for (int i = 0; i < self.dim(); ++i) {
                data[i] = self.item(i);
            }
            return data;
        })

        .def("__del__", [](Vector& self) { self.~Vector(); }); // Destructor

    m.def("getCenterPoint", [](std::vector<Vector*>& points, bool use_centroid) {
        return getCenterPoint(points, use_centroid);
    });

    m.def("getCenterPoint", [](std::vector<Vector>& points, bool use_centroid) {
        return getCenterPoint(points, use_centroid);
    });
    
    m.def("getClosestPoint", [](std::vector<Vector*>& points, Vector* test_point) {
        return getClosestPoint(points, test_point);
    });

    m.def("getClosestPoint", [](std::vector<Vector>& points, Vector test_point) {
        return getClosestPoint(points, test_point);
    });

}

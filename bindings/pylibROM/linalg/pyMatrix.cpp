#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Matrix.h"

namespace py = pybind11;
using namespace CAROM;

void init_matrix(pybind11::module_ &m) {

    py::class_<Matrix>(m, "Matrix") 
       
        // Constructor
        .def(py::init<>())
        .def(py::init<int, int, bool, bool>())
        .def(py::init([](py::array_t<double> mat, bool distributed, bool copy_data = true) {
            py::buffer_info buf_info = mat.request();
            int num_rows = buf_info.shape[0];     
            int num_cols = buf_info.shape[1];     
            double* data = static_cast<double*>(buf_info.ptr);
             return new Matrix(data,num_rows, num_cols, distributed, copy_data);
        }))
        
        // Bind the copy constructor
        .def(py::init<const Matrix&>())

        // Bind the assignment operator
        .def("__assign__", [](Matrix& self, const Matrix& rhs) { self = rhs; return self; })

        .def("fill", [](Matrix& self, const double value) { self = value; })

        // Bind the addition operator
        .def(py::self += py::self)

        // Bind the subtraction operator
        .def(py::self -= py::self) 

         //Bind set size method
        .def("setSize", &Matrix::setSize)

         //Bind distributed method
        .def("distributed", &Matrix::distributed)   

        .def("balanced", &Matrix::balanced)

        .def("numRows", &Matrix::numRows)

        .def("numDistributedRows", &Matrix::numDistributedRows) 
        .def("numColumns", &Matrix::numColumns)
         
        .def("getFirstNColumns", (Matrix* (Matrix::*)(int) const) &Matrix::getFirstNColumns)
        // .def("getFirstNColumns", (void (Matrix::*)(int, Matrix*&) const) &Matrix::getFirstNColumns)
        .def("getFirstNColumns", (void (Matrix::*)(int, Matrix&) const) &Matrix::getFirstNColumns)
        
        .def("mult",[](Matrix& self, const Matrix& other){
             Matrix* result = new Matrix();
             self.mult(other,result);
             return result; 
        },py::return_value_policy::take_ownership)
        .def("mult", (Matrix* (Matrix::*)(const Matrix*) const) &Matrix::mult)
        // .def("mult", (void (Matrix::*)(const Matrix&, Matrix*&) const) &Matrix::mult)
        .def("mult", (void (Matrix::*)(const Matrix&, Matrix&) const) &Matrix::mult)
        .def("mult", [](Matrix& self, const Vector& other){
             Vector* result = new Vector();
             self.mult(other,result);
             return result; 
        }, py::return_value_policy::take_ownership)
        .def("mult", (Vector* (Matrix::*)(const Vector*) const) &Matrix::mult, py::return_value_policy::take_ownership)
        // .def("mult", (void (Vector::*)(const Vector&, Vector*&)) &Vector::mult)
        .def("mult", (void (Matrix::*)(const Vector&, Vector&) const) &Matrix::mult)
        .def("pointwise_mult",[](const Matrix& self, int this_row, const Vector& other, Vector& result) {
                self.pointwise_mult(this_row, other, result);
            })
        .def("pointwise_mult",[](const Matrix& self, int this_row, Vector& other) {
                self.pointwise_mult(this_row, other);
            })

        .def("elementwise_mult",[](const Matrix& self, const Matrix& other) {
                Matrix* result = new Matrix();
                self.elementwise_mult(other, result);
                return result;
            }, py::return_value_policy::take_ownership)
        .def("elementwise_mult", (Matrix* (Matrix::*)(const Matrix*) const) &Matrix::elementwise_mult, py::return_value_policy::take_ownership)
        // .def("elementwise_mult",(void (Matrix::*)(const Matrix&,Matrix*&) const) &Matrix::elementwise_mult)
        .def("elementwise_mult",(void (Matrix::*)(const Matrix&,Matrix&) const) &Matrix::elementwise_mult)



   

        .def("get_data", [](const Matrix& self) {
           std::vector<std::vector<double>> data(self.numRows(), std::vector<double>(self.numColumns()));
            for (int i = 0; i < self.numRows(); ++i) {
                for (int j = 0; j < self.numColumns(); ++j) {
                     data[i][j] = self.item(i,j);
                }
            }
            return data;
        }) 
        .def("__del__", [](Matrix& self) { self.~Matrix(); }); // Destructor

};
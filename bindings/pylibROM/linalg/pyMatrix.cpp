#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Matrix.h"

namespace py = pybind11;
using namespace CAROM;

py::buffer_info
bufferInfo(Matrix &self)
{
    return py::buffer_info(
        self.getData(),                          /* Pointer to buffer */
        sizeof(double),                          /* Size of one scalar */
        py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
        2,                                       /* Number of dimensions */
        { self.numRows(), self.numColumns() },   /* Buffer dimensions */
        { sizeof(double) * self.numColumns(),
          sizeof(double) }                       /* Strides (in bytes) for each index */
    );
}

void init_matrix(pybind11::module_ &m) {

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol()) 
        .def_buffer([](Matrix &self) -> py::buffer_info {
            return bufferInfo(self);
        })
       
        // Constructor
        .def(py::init<>())
        .def(py::init<int, int, bool, bool>(),
            py::arg("num_rows"), py::arg("num_cols"), py::arg("distributed"), py::arg("randomized") = false)
        .def(py::init([](py::array_t<double> &mat, bool distributed, bool copy_data = true) {
            py::buffer_info buf_info = mat.request();
            int num_rows = buf_info.shape[0];     
            int num_cols = buf_info.shape[1];     
            double* data = static_cast<double*>(buf_info.ptr);
             return new Matrix(data,num_rows, num_cols, distributed, copy_data);
        }))
        
        // Bind the copy constructor
        .def(py::init<const Matrix&>())

        // NOTE (kevin): Having this destructor cause a fatal python memory error
        //               when defining a derived python class of this Matrix class.
        //               Valgrind does not show any significant memory leak difference
        //               whether or not this destructor is defined. Commenting it for now.
        // .def("__del__", [](Matrix& self) {
        //     self.~Matrix();
        // }) // Destructor

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

        .def("getFirstNColumns", [](Matrix& self,int n, Matrix* result){
              self.getFirstNColumns(n,*result);
        })
        
        .def("getFirstNColumns", (void (Matrix::*)(int, Matrix&) const) &Matrix::getFirstNColumns)
        
        .def("mult",[](const Matrix& self, const Matrix& other){
             Matrix* result = nullptr;
             self.mult(other,result);
             return result; 
        },py::return_value_policy::take_ownership)
        .def("mult", (Matrix* (Matrix::*)(const Matrix*) const) &Matrix::mult)
        .def("mult",[](const Matrix& self,const Matrix& other,Matrix* result){
             self.mult(other,*result);
        })
        .def("mult", (void (Matrix::*)(const Matrix&, Matrix&) const) &Matrix::mult)
        .def("mult", [](Matrix& self, const Vector& other){
             Vector* result = nullptr;
             self.mult(other,result);
             return result; 
        }, py::return_value_policy::take_ownership)
        .def("mult", (Vector* (Matrix::*)(const Vector*) const) &Matrix::mult, py::return_value_policy::take_ownership)
        .def("mult",[](const Matrix& self,const Vector& other,Vector* result){
             self.mult(other,*result);
        })
        .def("mult", (void (Matrix::*)(const Vector&, Vector&) const) &Matrix::mult)

        .def("pointwise_mult",[](const Matrix& self, int this_row, const Vector& other, Vector& result) {
                self.pointwise_mult(this_row, other, result);
            })
        .def("pointwise_mult",[](const Matrix& self, int this_row, Vector& other) {
                self.pointwise_mult(this_row, other);
            })

        .def("elementwise_mult",[](const Matrix& self, const Matrix& other) {
                Matrix* result = nullptr;
                self.elementwise_mult(other, result);
                return result;
            }, py::return_value_policy::take_ownership)
        .def("elementwise_mult", (Matrix* (Matrix::*)(const Matrix*) const) &Matrix::elementwise_mult, py::return_value_policy::take_ownership)
        .def("elementwise_mult",[](const Matrix& self,const Matrix& other,Matrix* result){
             self.elementwise_mult(other,*result);
        })
        .def("elementwise_mult",(void (Matrix::*)(const Matrix&,Matrix&) const) &Matrix::elementwise_mult)
        
        .def("elementwise_square",[](const Matrix& self) {
                Matrix* result = nullptr;
                self.elementwise_square(result);
                return result;
            },py::return_value_policy::take_ownership)
        .def("elementwise_square",[](const Matrix& self,Matrix* result){
             self.elementwise_square(*result);
        })
        .def("elementwise_square",(void (Matrix::*)(Matrix&) const) &Matrix::elementwise_square)

        .def("multPlus", (void (Matrix::*)(Vector&,const Vector&,double) const) &Matrix::multPlus)

        .def("transposeMult",[](const Matrix& self, const Matrix& other) {
                 Matrix* result = nullptr;
                 self.transposeMult(other, result);
                 return result;
             },py::return_value_policy::take_ownership)
        .def("transposeMult", (Matrix* (Matrix::*)(const Matrix*) const) &Matrix::transposeMult)
        .def("transposeMult",[](const Matrix& self,const Matrix& other,Matrix* result){
             self.transposeMult(other,*result);
        })
        .def("transposeMult", (void (Matrix::*)(const Matrix&, Matrix&) const) &Matrix::transposeMult)
        .def("transposeMult",[](const Matrix& self, const Vector& other) {
                 Vector* result = nullptr;
                 self.transposeMult(other, result);
                 return result;
             },py::return_value_policy::take_ownership)
        .def("transposeMult", (Vector* (Matrix::*)(const Vector*) const) &Matrix::transposeMult, py::return_value_policy::take_ownership)
        .def("transposeMult",[](const Matrix& self,const Vector& other,Vector* result){
             self.transposeMult(other,*result);
        })
        .def("transposeMult", (void (Matrix::*)(const Vector&, Vector&) const) &Matrix::transposeMult)

        .def("inverse", (Matrix* (Matrix::*)() const) &Matrix::inverse, py::return_value_policy::take_ownership)
        .def("inverse",[](const Matrix& self,Matrix* result){
             self.inverse(*result);
        })
        .def("inverse", (void (Matrix::*)(Matrix&) const) &Matrix::inverse)
        // NOTE: this cannot be discerned with Matrix* inverse() in python realm. Changing the name is inevitable.
        .def("invert",(void (Matrix::*)()) &Matrix::inverse) 

        .def("getColumn",[](const Matrix& self, int column) {
                 Vector* result = new Vector();
                 self.getColumn(column, result);
                 return result;
             }, py::return_value_policy::take_ownership)
        .def("getColumn",[](const Matrix& self,int column,Vector* result){
             self.getColumn(column,*result);
        })
        .def("getColumn", (void (Matrix::*)(int, Vector&)const) &Matrix::getColumn)
        
        .def("transpose", (void (Matrix::*)()) &Matrix::transpose)

        .def("transposePseudoinverse",(void (Matrix::*)()) &Matrix::transposePseudoinverse)

        .def("qr_factorize", [](const Matrix& self) -> std::vector<std::unique_ptr<Matrix>> {
            std::vector<std::unique_ptr<Matrix>> qr;
            self.qr_factorize(qr);
            return qr;
        })

        // TODO (kevin): due to the difference between python and c++, technically we should not take
        //               row_pivot and row_pivot_owner as input parameters, just returning them in the end as outputs.
        //               Ideally, these two variables should be instantiated within the libROM function itself,
        //               which requires a modification on libROM side as well.
        .def("qrcp_pivots_transpose", [](const Matrix& self, std::vector<int>& row_pivot,
                                          std::vector<int>& row_pivot_owner, int pivots_requested) {
            self.qrcp_pivots_transpose(row_pivot.data(), row_pivot_owner.data(), pivots_requested);
            return std::make_tuple(row_pivot, row_pivot_owner);
        })

        .def("orthogonalize", (void (Matrix::*)(bool, double)) &Matrix::orthogonalize, py::arg("double_pass") = false, py::arg("zero_tol") = 1.0e-15)
        .def("orthogonalize_last", (void (Matrix::*)(int, bool, double)) &Matrix::orthogonalize_last, py::arg("ncols") = -1, py::arg("double_pass") = false, py::arg("zero_tol") = 1.0e-15)

        .def("item", (const double& (Matrix::*)(int, int) const) &Matrix::item)
        .def("__getitem__", [](Matrix& self, int row, int col) { 
            return self(row, col);
        })
        .def("__getitem__", [](Matrix& self, py::tuple indexes) {
            // TODO: raise python-type error
            assert(indexes.size() == 2);
            return self(py::cast<int>(indexes[0]), py::cast<int>(indexes[1]));
        })
        .def("__setitem__", [](Matrix& self, int row, int col, double value) { 
            self(row, col) = value; 
        })
        .def("__setitem__", [](Matrix& self, py::tuple indexes, double value) {
            // TODO: raise python-type error
            assert(indexes.size() == 2);
            self(py::cast<int>(indexes[0]), py::cast<int>(indexes[1])) = value;
        })

        .def("__call__", (const double& (Matrix::*)(int,int) const) &Matrix::operator())
        .def("__call__", (double& (Matrix::*)(int,int)) &Matrix::operator())
        
        .def("print", &Matrix::print)
        .def("write", &Matrix::write)
        .def("read", &Matrix::read)
        .def("local_read", &Matrix::local_read)
        .def("getData", [](Matrix& self) {
            // We provide a view vector, which does not free the memory at its destruction.
            py::capsule buffer_handle([](){});
            // Use this if the C++ memory SHOULD be deallocated
            // once the Python no longer has a reference to it
            // py::capsule buffer_handle(data_buffer, [](void* p){ free(p); });

            return py::array(bufferInfo(self), buffer_handle);
        })

        .def("get_data", [](const Matrix& self) {
           std::vector<std::vector<double>> data(self.numRows(), std::vector<double>(self.numColumns()));
            for (int i = 0; i < self.numRows(); ++i) {
                for (int j = 0; j < self.numColumns(); ++j) {
                     data[i][j] = self.item(i,j);
                }
            }
            return data;
        })

        .def("distribute", &Matrix::distribute)
        .def("gather", &Matrix::gather);
    
    m.def("outerProduct", (Matrix (*)(const Vector&, const Vector&)) &outerProduct);
    m.def("DiagonalMatrixFactory", (Matrix (*)(const Vector&)) &DiagonalMatrixFactory);
    m.def("IdentityMatrixFactory", (Matrix (*)(const Vector&)) &IdentityMatrixFactory);
   
   
    py::class_<EigenPair>(m, "EigenPair")
        .def(py::init<>())
        .def_readwrite("ev", &EigenPair::ev)
        .def_readwrite("eigs", &EigenPair::eigs); 

    py::class_<ComplexEigenPair>(m, "ComplexEigenPair")
        .def(py::init<>())
        .def_readwrite("ev_real", &ComplexEigenPair::ev_real)
        .def_readwrite("ev_imaginary", &ComplexEigenPair::ev_imaginary)
        .def_readwrite("eigs", &ComplexEigenPair::eigs);

    py::class_<SerialSVDDecomposition>(m, "SerialSVDDecomposition")
        .def(py::init<>())
        .def_readwrite("U", &SerialSVDDecomposition::U)
        .def_readwrite("S", &SerialSVDDecomposition::S)
        .def_readwrite("V", &SerialSVDDecomposition::V);
    
    m.def("SerialSVD", (void (*)(Matrix*,Matrix*,Vector*,Matrix*)) &SerialSVD);
    m.def("SerialSVD", [](Matrix* A) {
        Matrix* U = new Matrix();
       Vector* S = new Vector();
        Matrix* V = new Matrix();
        SerialSVD(A,U,S,V);
        SerialSVDDecomposition decomp;
        decomp.U= U;
        decomp.S = S;
        decomp.V = V;
        return decomp;
    });
    m.def("SymmetricRightEigenSolve",(EigenPair (*)(Matrix*) ) &SymmetricRightEigenSolve);
    m.def("NonSymmetricRightEigenSolve",(ComplexEigenPair (*)(Matrix*)) &NonSymmetricRightEigenSolve);
    
    m.def("SpaceTimeProduct", &SpaceTimeProduct, py::return_value_policy::take_ownership,
          py::arg("As"), py::arg("At"), py::arg("Bs"), py::arg("Bt"),
          py::arg("tscale") = static_cast<const std::vector<double>*>(nullptr), py::arg("At0at0") = false,
          py::arg("Bt0at0") = false, py::arg("lagB") = false, py::arg("skip0") = false);
    

    

};
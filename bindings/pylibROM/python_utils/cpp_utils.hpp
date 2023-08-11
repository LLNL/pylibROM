//
// Created by sullan2 on 4/20/23.
//

#ifndef CPP_UTILS_HPP
#define CPP_UTILS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

inline double* getPointer(py::array_t<double> &u_in)
{
    py::buffer_info buf_info = u_in.request();
    // If it is 1D array, we should ensure that the memory is contiguous.
    if ((buf_info.ndim == 1) && (buf_info.strides[0] != sizeof(double)))
    {
        std::string msg = "Input numpy array must have a contiguous memory! - ";
        msg += std::to_string(buf_info.strides[0]) + "\n";
        throw std::runtime_error(msg.c_str());
    }
        
    return static_cast<double*>(buf_info.ptr);
}

inline ssize_t getDim(py::array_t<double> &u_in)
{
    return u_in.request().ndim;
}

inline double* getVectorPointer(py::array_t<double> &u_in)
{
    py::buffer_info buf_info = u_in.request();
    // If it is 1D array, we should ensure that the memory is contiguous.
    if (buf_info.ndim != 1)
    {
        throw std::runtime_error("Input array must be 1-dimensional!\n");
    }
    else if (buf_info.strides[0] != sizeof(double))
    {
        std::string msg = "Input numpy array must have a contiguous memory! - ";
        msg += std::to_string(buf_info.strides[0]) + "\n";
        throw std::runtime_error(msg.c_str());
    }
        
    return static_cast<double*>(buf_info.ptr);
}

#endif

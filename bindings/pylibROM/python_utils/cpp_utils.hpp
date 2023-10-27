//
// Created by sullan2 on 4/20/23.
//

#ifndef CPP_UTILS_HPP
#define CPP_UTILS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename T>
T* getPointer(py::array_t<T> &u_in)
{
    py::buffer_info buf_info = u_in.request();
    // If it is 1D array, we should ensure that the memory is contiguous.
    if ((buf_info.ndim == 1) && (buf_info.strides[0] != sizeof(T)))
    {
        std::string msg = "Input numpy array must have a contiguous memory! - ";
        msg += std::to_string(buf_info.strides[0]) + "\n";
        throw std::runtime_error(msg.c_str());
    }
        
    return static_cast<T*>(buf_info.ptr);
}

template<typename T>
ssize_t getDim(py::array_t<T> &u_in)
{
    return u_in.request().ndim;
}

template<typename T>
T* getVectorPointer(py::array_t<T> &u_in)
{
    py::buffer_info buf_info = u_in.request();
    // If it is 1D array, we should ensure that the memory is contiguous.
    if (buf_info.ndim != 1)
    {
        throw std::runtime_error("Input array must be 1-dimensional!\n");
    }
    else if (buf_info.strides[0] != sizeof(T))
    {
        std::string msg = "Input numpy array must have a contiguous memory! - ";
        msg += std::to_string(buf_info.strides[0]) + "\n";
        throw std::runtime_error(msg.c_str());
    }
        
    return static_cast<T*>(buf_info.ptr);
}

template<typename T>
py::buffer_info
get1DArrayBufferInfo(T *ptr, const int nelem)
{
    return py::buffer_info(
        ptr,                         /* Pointer to buffer */
        sizeof(T),                          /* Size of one scalar */
        py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
        1,                                      /* Number of dimensions */
        { nelem },                         /* Buffer dimensions */
        { sizeof(T) }                       /* Strides (in bytes) for each index */
    );
}

template<typename T>
py::capsule
get1DArrayBufferHandle(T *ptr, const bool free_when_done=false)
{
    if (free_when_done)
        return py::capsule(ptr, [](void *f){
            T *T_ptr = reinterpret_cast<T *>(f);
            delete[] T_ptr;
        });
    else
        return py::capsule([](){});
}

template<typename T>
py::array_t<T>
get1DArrayFromPtr(T *ptr, const int nelem, bool free_when_done=false)
{
    // if empty array, no need to free when done.
    free_when_done = free_when_done && (nelem > 0);
    return py::array(get1DArrayBufferInfo(ptr, nelem),
                     get1DArrayBufferHandle(ptr, free_when_done));
}

template<typename T>
T*
extractSwigPtr(const py::handle &swig_target)
{
    // On python, swig_target.this.__int__() returns the memory address of the wrapped c++ object pointer.
    // We execute the same command here in the c++ side, where the memory address is given as py::object.
    // The returned py::object is simply cast as std::uintptr_t, which is then cast into a c++ object type we want.
    std::uintptr_t temp = swig_target.attr("this").attr("__int__")().cast<std::uintptr_t>();
    return reinterpret_cast<T *>(temp);
}

#endif

//
// Created by barrow9 on 6/4/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/DMD.h"
#include "algo/ParametricDMD.h"
#include "linalg/Vector.h"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace CAROM;
using namespace std;

template <class T>
T* getParametricDMD_Type(
    std::vector<Vector*>& parameter_points,
    std::vector<T*>& dmds,
    Vector* desired_point,
    std::string rbf,
    std::string interp_method,
    double closest_rbf_val,
    bool reorthogonalize_W)
{
    T *parametric_dmd = NULL;
    getParametricDMD(parametric_dmd, parameter_points, dmds, desired_point,
        rbf, interp_method, closest_rbf_val, reorthogonalize_W);
    return parametric_dmd;
}

template <class T>
T* getParametricDMD_Type(
    std::vector<Vector*>& parameter_points,
    std::vector<std::string>& dmd_paths,
    Vector* desired_point,
    std::string rbf = "G",
    std::string interp_method = "LS",
    double closest_rbf_val = 0.9,
    bool reorthogonalize_W = false)
{
    T *parametric_dmd = NULL;
    getParametricDMD(parametric_dmd, parameter_points, dmd_paths, desired_point,
        rbf, interp_method, closest_rbf_val, reorthogonalize_W);
    return parametric_dmd;
}

void init_ParametricDMD(pybind11::module_ &m) {

    // original getParametricDMD pass a template pointer reference T* &parametric_dmd.
    // While it is impossible to bind a template function as itself,
    // this first argument T* &parametric_dmd is mainly for determining the type T.
    // Here we introduce a dummy argument in place of parametric_dmd.
    // This will let python decide which function to use, based on the first argument type.
    // We will need variants of this as we bind more DMD classes,
    // where dmd_type is the corresponding type.
    m.def("getParametricDMD", [](
        py::object &dmd_type,
        std::vector<Vector*>& parameter_points,
        std::vector<DMD*>& dmds,
        Vector* desired_point,
        std::string rbf = "G",
        std::string interp_method = "LS",
        double closest_rbf_val = 0.9,
        bool reorthogonalize_W = false)
    {
        std::string name = dmd_type.attr("__name__").cast<std::string>();
        if (name == "DMD")
            return getParametricDMD_Type<DMD>(parameter_points, dmds, desired_point,
                rbf, interp_method, closest_rbf_val, reorthogonalize_W);
        else
        {
            std::string msg = name + " is not a proper libROM DMD class!\n";
            throw std::runtime_error(msg.c_str());
        }
    },
    py::arg("dmd_type"),
    py::arg("parameter_points"),
    py::arg("dmds"),
    py::arg("desired_point"),
    py::arg("rbf") = "G",
    py::arg("interp_method") = "LS",
    py::arg("closest_rbf_val") = 0.9,
    py::arg("reorthogonalize_W") = false);

    // original getParametricDMD pass a template pointer reference T* &parametric_dmd.
    // While it is impossible to bind a template function as itself,
    // this first argument T* &parametric_dmd is mainly for determining the type T.
    // Here we introduce a dummy argument in place of parametric_dmd.
    // This will let python decide which function to use, based on the first argument type.
    // We will need variants of this as we bind more DMD classes,
    // where dmd_type is the corresponding type.
    m.def("getParametricDMD", [](
        py::object &dmd_type,
        std::vector<Vector*>& parameter_points,
        std::vector<std::string>& dmd_paths,
        Vector* desired_point,
        std::string rbf = "G",
        std::string interp_method = "LS",
        double closest_rbf_val = 0.9,
        bool reorthogonalize_W = false)
    {
        std::string name = dmd_type.attr("__name__").cast<std::string>();
        if (name == "DMD")
            return getParametricDMD_Type<DMD>(parameter_points, dmd_paths, desired_point,
                rbf, interp_method, closest_rbf_val, reorthogonalize_W);
        else
        {
            std::string msg = name + " is not a proper libROM DMD class!\n";
            throw std::runtime_error(msg.c_str());
        }
    },
    py::arg("dmd_type"),
    py::arg("parameter_points"),
    py::arg("dmd_paths"),
    py::arg("desired_point"),
    py::arg("rbf") = "G",
    py::arg("interp_method") = "LS",
    py::arg("closest_rbf_val") = 0.9,
    py::arg("reorthogonalize_W") = false);

}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "linalg/Options.h"

namespace py = pybind11;
using namespace CAROM;
       
void init_Options(pybind11::module_ &m) {

    py::class_<Options>(m, "Options")
        .def(py::init<int, int, bool, bool>(), py::arg("dim_"), py::arg("max_num_samples_"),py::arg("update_right_SV_") = false, py::arg("write_snapshots_") = false)
        .def_readwrite("dim", &Options::dim)
        .def_readwrite("max_num_samples", &Options::max_num_samples)
        .def_readwrite("update_right_SV", &Options::update_right_SV)
        .def_readwrite("write_snapshots", &Options::write_snapshots)
        .def_readwrite("max_basis_dimension", &Options::max_basis_dimension)
        .def_readwrite("singular_value_tol", &Options::singular_value_tol)
        .def_readwrite("debug_algorithm", &Options::debug_algorithm)
        .def_readwrite("randomized", &Options::randomized)
        .def_readwrite("randomized_subspace_dim", &Options::randomized_subspace_dim)
        .def_readwrite("random_seed", &Options::random_seed)
        .def_readwrite("linearity_tol", &Options::linearity_tol)
        .def_readwrite("initial_dt", &Options::initial_dt)
        .def_readwrite("sampling_tol", &Options::sampling_tol)
        .def_readwrite("max_time_between_samples", &Options::max_time_between_samples)
        .def_readwrite("fast_update", &Options::fast_update)
        .def_readwrite("skip_linearly_dependent", &Options::skip_linearly_dependent)
        .def_readwrite("save_state", &Options::save_state)
        .def_readwrite("restore_state", &Options::restore_state)
        .def_readwrite("min_sampling_time_step_scale", &Options::min_sampling_time_step_scale)
        .def_readwrite("sampling_time_step_scale", &Options::sampling_time_step_scale)
        .def_readwrite("max_sampling_time_step_scale", &Options::max_sampling_time_step_scale)
        .def_readwrite("static_svd_preserve_snapshot", &Options::static_svd_preserve_snapshot)
        .def("setMaxBasisDimension", &Options::setMaxBasisDimension, py::arg("max_basis_dimension_"))
        .def("setSingularValueTol", &Options::setSingularValueTol, py::arg("singular_value_tol_"))
        .def("setDebugMode", &Options::setDebugMode,  py::arg("debug_algorithm_"))
        .def("setRandomizedSVD", &Options::setRandomizedSVD, py::arg("randomized_"), py::arg("randomized_subspace_dim_") = -1, py::arg("random_seed_") = 1)
        .def("setIncrementalSVD", &Options::setIncrementalSVD,
                py::arg("linearity_tol_"),
                py::arg("initial_dt_"),
                py::arg("sampling_tol_"),
                py::arg("max_time_between_samples_"),
                py::arg("fast_update_") = false,
                py::arg("fast_update_brand_") = false,
                py::arg("skip_linearly_dependent_") = false
            )
        .def("setStateIO", &Options::setStateIO,py::arg("save_state_"),py::arg("restore_state_"))
        .def("setSamplingTimeStepScale", &Options::setSamplingTimeStepScale,py::arg("min_sampling_time_step_scale_"),py::arg("sampling_time_step_scale_"),py::arg("max_sampling_time_step_scale_"));
}
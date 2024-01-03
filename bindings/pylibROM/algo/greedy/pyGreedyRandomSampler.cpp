#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/greedy/GreedyRandomSampler.h"
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;
using namespace std;

void init_GreedyRandomSampler(py::module &m) {
    py::class_<GreedyRandomSampler, GreedySampler>(m, "GreedyRandomSampler")
        .def(py::init<CAROM::Vector, CAROM::Vector, int, bool, double, double,double, int, int, bool, std::string, std::string, bool, int, bool>(),
            py::arg("param_space_min"),
            py::arg("param_space_max"),
            py::arg("num_parameter_points"),
            py::arg("check_local_rom"),
            py::arg("relative_error_tolerance"),
            py::arg("alpha"),
            py::arg("max_clamp"),
            py::arg("subset_size"),
            py::arg("convergence_subset_size"),
            py::arg("use_latin_hypercube"),
            py::arg("output_log_path") = "",
            py::arg("warm_start_file_name") = "",
            py::arg("use_centroid") = true,
            py::arg("random_seed") = 1,
            py::arg("debug_algorithm") = false
        ) 
        .def(py::init<double, double, int, bool, double, double,double, int, int, bool, std::string, std::string, bool, int, bool>(),
            py::arg("param_space_min"),
            py::arg("param_space_max"),
            py::arg("num_parameter_points"),
            py::arg("check_local_rom"),
            py::arg("relative_error_tolerance"),
            py::arg("alpha"),
            py::arg("max_clamp"),
            py::arg("subset_size"),
            py::arg("convergence_subset_size"),
            py::arg("use_latin_hypercube"),
            py::arg("output_log_path") = "",
            py::arg("warm_start_file_name") = "",
            py::arg("use_centroid") = true,
            py::arg("random_seed") = 1,
            py::arg("debug_algorithm") = false
        )
        .def(py::init<std::string, std::string>(),
            py::arg("base_file_name"),
            py::arg("output_log_path") = ""
        )
        .def("save", &GreedyRandomSampler::save, py::arg("base_file_name"))
        .def("__del__", [](GreedyRandomSampler& self){ self.~GreedyRandomSampler(); });
}

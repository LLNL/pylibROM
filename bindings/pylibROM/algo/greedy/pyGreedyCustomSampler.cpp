#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/greedy/GreedyCustomSampler.h"
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;
using namespace std;

void init_GreedyCustomSampler(pybind11::module_ &m) {
    py::class_<GreedyCustomSampler, GreedySampler>(m, "GreedyCustomSampler")
        .def(py::init<std::vector<CAROM::Vector>, bool, double, double, double, int, int, std::string, std::string, bool, int, bool>(),
            py::arg("parameter_points"),
            py::arg("check_local_rom"),
            py::arg("relative_error_tolerance"),
            py::arg("alpha"),
            py::arg("max_clamp"),
            py::arg("subset_size"),
            py::arg("convergence_subset_size"),
            py::arg("output_log_path") = "",
            py::arg("warm_start_file_name") = "",
            py::arg("use_centroid") = true,
            py::arg("random_seed") = 1,
            py::arg("debug_algorithm") = false)
        .def(py::init<std::vector<double>, bool, double, double, double, int, int, std::string, std::string, bool, int, bool>(),
            py::arg("parameter_points"),
            py::arg("check_local_rom"),
            py::arg("relative_error_tolerance"),
            py::arg("alpha"),
            py::arg("max_clamp"),
            py::arg("subset_size"),
            py::arg("convergence_subset_size"),
            py::arg("output_log_path") = "",
            py::arg("warm_start_file_name") = "",
            py::arg("use_centroid") = true,
            py::arg("random_seed") = 1,
            py::arg("debug_algorithm") = false)
        .def(py::init<std::string, std::string>(),
            py::arg("base_file_name"),
            py::arg("output_log_path") = "")
        .def("__del__", [](GreedyCustomSampler& self){ self.~GreedyCustomSampler(); });
}

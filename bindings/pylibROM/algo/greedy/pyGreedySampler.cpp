#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/greedy/GreedySampler.h"
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;
using namespace std;


class PyGreedySampler : public GreedySampler {
public:
    using GreedySampler::GreedySampler;

    void save(std::string base_file_name) override {
        PYBIND11_OVERRIDE(void,GreedySampler,save,base_file_name );
    } 
protected:
    void constructParameterPoints() override {
        PYBIND11_OVERRIDE_PURE(void, GreedySampler, constructParameterPoints,);
    }
    void getNextParameterPointAfterConvergenceFailure() override {
        PYBIND11_OVERRIDE_PURE(void, GreedySampler, getNextParameterPointAfterConvergenceFailure,);
    }
};

void init_GreedySampler(pybind11::module_ &m) {
    py::class_<GreedyErrorIndicatorPoint>(m, "GreedyErrorIndicatorPoint")
        .def_property_readonly("point", [](GreedyErrorIndicatorPoint &self) {
            return self.point.get();
         })
        .def_property_readonly("localROM", [](GreedyErrorIndicatorPoint &self) {
            return self.localROM.get();
         });
        
    py::class_<GreedySampler,PyGreedySampler>(m, "GreedySampler")
        .def(py::init<std::vector<Vector>, bool, double, double, double, int, int, std::string, std::string, bool, int, bool>(),
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
        .def(py::init<std::vector<double>,bool, double, double, double, int, int, std::string, std::string, bool, int, bool>(), 
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
        .def(py::init<Vector, Vector, int, bool, double, double, double, int, int,std::string, std::string, bool, int, bool>(),
            py::arg("param_space_min"), py::arg("param_space_max"), py::arg("num_parameter_points"),
            py::arg("check_local_rom"), py::arg("relative_error_tolerance"), py::arg("alpha"),
            py::arg("max_clamp"), py::arg("subset_size"), py::arg("convergence_subset_size"),
            py::arg("output_log_path") = "", py::arg("warm_start_file_name") = "",
            py::arg("use_centroid") = true, py::arg("random_seed") = 1,
            py::arg("debug_algorithm") = false
        )
        .def(py::init<double, double, int, bool, double, double, double, int, int,std::string, std::string, bool, int, bool>(),     
            py::arg("param_space_min"), py::arg("param_space_max"), py::arg("num_parameter_points"),
            py::arg("check_local_rom"), py::arg("relative_error_tolerance"), py::arg("alpha"),
            py::arg("max_clamp"), py::arg("subset_size"), py::arg("convergence_subset_size"),
            py::arg("output_log_path") = "", py::arg("warm_start_file_name") = "",
            py::arg("use_centroid") = true, py::arg("random_seed") = 1,
            py::arg("debug_algorithm") = false
        )
        .def(py::init<std::string, std::string>(), py::arg("base_file_name"), py::arg("output_log_path") = "")
        .def("getNextParameterPoint", [](GreedySampler& self) -> std::unique_ptr<Vector> {
            std::shared_ptr<Vector> result = self.getNextParameterPoint();
            return std::make_unique<Vector>(*(result.get()));
        })
        .def("getNextPointRequiringRelativeError", [](GreedySampler& self) -> GreedyErrorIndicatorPoint {
            // Create a deepcopy of the struct, otherwise it will get freed twice
            GreedyErrorIndicatorPoint point = self.getNextPointRequiringRelativeError();
            Vector *t_pt = nullptr;
            Vector *t_lROM = nullptr;

            if (point.point)
            {
                t_pt = new Vector(*(point.point));
            }

            if (point.localROM)
            {
                t_lROM = new Vector(*(point.localROM));
            }

            return createGreedyErrorIndicatorPoint(t_pt, t_lROM);
        }, py::return_value_policy::reference)
        .def("getNextPointRequiringErrorIndicator", [](GreedySampler& self) -> GreedyErrorIndicatorPoint {
            // Create a deepcopy of the struct, otherwise it will get freed twice
            GreedyErrorIndicatorPoint point = self.getNextPointRequiringErrorIndicator();

            Vector *t_pt = nullptr;
            Vector *t_lROM = nullptr;

            if (point.point)
            {
                t_pt = new Vector(*(point.point));
            }

            if (point.localROM)
            {
                t_lROM = new Vector(*(point.localROM));
            }

            return createGreedyErrorIndicatorPoint(t_pt, t_lROM);
        }, py::return_value_policy::reference)
        .def("setPointRelativeError", (void (GreedySampler::*) (double))&GreedySampler::setPointRelativeError)    
        .def("setPointErrorIndicator", (void (GreedySampler::*) (double,int)) &GreedySampler::setPointErrorIndicator)
        .def("getNearestNonSampledPoint", (int (GreedySampler::*) (CAROM::Vector)) &GreedySampler::getNearestNonSampledPoint)
        .def("getNearestROM", [](GreedySampler& self, Vector point) -> std::unique_ptr<Vector> {
            std::shared_ptr<Vector> result = self.getNearestROM(point);
            if (!result)
            {
                return nullptr;
            }
            return std::make_unique<Vector>(*(result.get()));
        })
        .def("getParameterPointDomain", &GreedySampler::getParameterPointDomain)
        .def("getSampledParameterPoints", &GreedySampler::getSampledParameterPoints)
        .def("save", &GreedySampler::save)
        .def("__del__", [](GreedySampler& self){ self.~GreedySampler(); })
        .def("isComplete", &GreedySampler::isComplete);
   
    m.def("createGreedyErrorIndicatorPoint", [](Vector* point, Vector* localROM) {
        return createGreedyErrorIndicatorPoint(point, localROM);
    });
    m.def("createGreedyErrorIndicatorPoint", [](Vector* point, std::shared_ptr<Vector>& localROM) {
        return createGreedyErrorIndicatorPoint(point, localROM);
    });
    m.def("getNearestPoint", [](std::vector<Vector>& paramPoints,Vector point) {
        return getNearestPoint(paramPoints, point);
    });
    m.def("getNearestPoint", [](std::vector<double>& paramPoints, double point) {
        return getNearestPoint(paramPoints, point);
    });
    m.def("getNearestPointIndex", [](std::vector<Vector> paramPoints, Vector point) {
        return getNearestPointIndex(paramPoints, point);
    });
    m.def("getNearestPointIndex", [](std::vector<double> paramPoints, double point) {
        return getNearestPointIndex(paramPoints, point);
    });
}

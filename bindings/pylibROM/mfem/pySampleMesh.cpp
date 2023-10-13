//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include "mfem/SampleMesh.hpp"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace mfem;

void init_mfem_SampleMesh(pybind11::module_ &m) {
    
    m.def("SampleVisualization", [](py::object &pmesh, std::set<int> const& elems,
                                    std::set<int> const& intElems, std::set<int> const& faces,
                                    std::set<int> const& edges, std::set<int> const& vertices,
                                    std::string const& filename, py::object &elemCount,
                                    double elementScaling){
        ParMesh *pmeshPtr = extractSwigPtr<ParMesh>(pmesh);
        Vector *elemCountPtr = nullptr;
        if (!elemCount.is_none())
            elemCountPtr = extractSwigPtr<Vector>(elemCount);

        CAROM::SampleVisualization(pmeshPtr, elems, intElems, faces, edges, vertices,
                                   filename, elemCountPtr, elementScaling);
    },
    py::arg("pmesh"), py::arg("elems"), py::arg("intElems"), py::arg("faces"),
    py::arg("edges"), py::arg("vertices"), py::arg("filename"),
    py::arg("elemCount") = py::none(), py::arg("elementScaling") = 1.0);

    py::class_<CAROM::SampleMeshManager>(m, "SampleMeshManager")

        .def(py::init([](py::list &fespace_, std::string visFilename, double visScale){
            std::vector<ParFiniteElementSpace *> fespacePtr(0);
            for (py::handle obj : fespace_)
                fespacePtr.push_back(extractSwigPtr<ParFiniteElementSpace>(obj));
            return new CAROM::SampleMeshManager(fespacePtr, visFilename, visScale);
        }),
        py::arg("fespace_"), py::arg("visFilename") = "", py::arg("visScale") = 1.0)

        .def("RegisterSampledVariable", &CAROM::SampleMeshManager::RegisterSampledVariable)

        .def("ConstructSampleMesh", &CAROM::SampleMeshManager::ConstructSampleMesh)

        .def("GatherDistributedMatrixRows", &CAROM::SampleMeshManager::GatherDistributedMatrixRows)

        .def("GetSampleFESpace", [](CAROM::SampleMeshManager& self, const int space, py::object &spfespace){
            ParFiniteElementSpace *spfes_target = self.GetSampleFESpace(space);

            // deep copy of the spfes_target.
            void *spfes_address = extractSwigPtr<void>(spfespace);
            ParFiniteElementSpace *spfes = new (spfes_address) ParFiniteElementSpace(*spfes_target);
        })

        .def("GetSampleMesh", [](CAROM::SampleMeshManager& self, py::object &sppmesh){
            ParMesh *sppmesh_target = self.GetSampleMesh();

            // deep copy of the spfes_target.
            void *sppmesh_address = extractSwigPtr<void>(sppmesh);
            ParMesh *sample_pmesh = new (sppmesh_address) ParMesh(*sppmesh_target);
        })

        .def("GetNumVarSamples", &CAROM::SampleMeshManager::GetNumVarSamples)

        .def("GetSampledValues", [](CAROM::SampleMeshManager &self, const std::string variable, py::object &v, CAROM::Vector &s){
            Vector *v_ptr = extractSwigPtr<Vector>(v);
            self.GetSampledValues(variable, *v_ptr, s);
        })

        .def("GetSampleElements", &CAROM::SampleMeshManager::GetSampleElements)

        .def("WriteVariableSampleMap", &CAROM::SampleMeshManager::WriteVariableSampleMap)

        //TODO: needed explicitly?
        .def("__del__", [](CAROM::SampleMeshManager& self){ self.~SampleMeshManager(); }); // Destructor

    py::class_<CAROM::SampleDOFSelector>(m, "SampleDOFSelector")

        .def(py::init<>())

        .def("ReadMapFromFile", &CAROM::SampleDOFSelector::ReadMapFromFile)

        .def("GetSampledValues", [](CAROM::SampleDOFSelector &self, const std::string variable, py::object &v, CAROM::Vector &s){
            Vector *v_ptr = extractSwigPtr<Vector>(v);
            self.GetSampledValues(variable, *v_ptr, s);
        });

}

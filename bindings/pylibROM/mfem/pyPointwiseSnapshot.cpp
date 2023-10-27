//
// Created by sullan2 on 4/20/23.
//
#include <pybind11/pybind11.h>
#include "mfem/PointwiseSnapshot.hpp"
#include "python_utils/cpp_utils.hpp"

namespace py = pybind11;
using namespace mfem;

void init_mfem_PointwiseSnapshot(pybind11::module_ &m) {
    
    py::class_<CAROM::PointwiseSnapshot>(m, "PointwiseSnapshot")

        .def(py::init([](const int sdim, py::array_t<int> &dims){
            return new CAROM::PointwiseSnapshot(sdim, getVectorPointer(dims));
        }))

        .def("SetMesh", [](CAROM::PointwiseSnapshot &self, py::object &pmesh) {
            self.SetMesh(extractSwigPtr<ParMesh>(pmesh));
        })

        .def("GetSnapshot", [](CAROM::PointwiseSnapshot &self, py::object &f, py::object &s) {
            ParGridFunction *fptr = extractSwigPtr<ParGridFunction>(f);
            Vector *sptr = extractSwigPtr<Vector>(s);
            self.GetSnapshot(*fptr, *sptr);
        })

        //TODO: needed explicitly?
        .def("__del__", [](CAROM::PointwiseSnapshot& self) { self.~PointwiseSnapshot(); }); // Destructor

}

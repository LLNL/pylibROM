//
// Created by barrow9 on 6/4/23.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "algo/DMD.h"
#include "linalg/Vector.h"

namespace py = pybind11;
using namespace CAROM;
using namespace std;

void init_DMD(pybind11::module_ &m)
{

     //wrap librom/lib/algo/DMD.h c++ class + its public methods
     py::class_<DMD>(m, "DMD")

         // constructor, take hf5 database file as input.
         .def(py::init<string>())

         .def(py::init<int, double, bool, Vector *>(),
              py::arg("dim"),
              py::arg("dt"),
              py::arg("alt_output_basis") = false,
              py::arg("state_offset") = nullptr)

         .def("setOffset",
              &DMD::setOffset, 
              py::arg("offset_vector"), 
              py::arg("order"))

         //NO
         .def("takeSample",
              &DMD::takeSample, 
              py::arg("u_in"), 
              py::arg("t"))
        //NO
        //.def("takeSample",
        //     [](DMD& self, double* u_in, double t) {
        //         // Convert the 'double *' to the appropriate C++ type, if necessary
        //         // Perform any required operations with the 'double *' argument
        //         // Call the original function passing the converted argument
        //         self.takeSample(u_in, t);
        //     },
        //     py::arg("u_in"),
        //     py::arg("t")
        //)
        //Below is for numpy arrays but that isnt right either.
        //try this? https://people.duke.edu/~ccc14/sta-663-2020/notebooks/S13_pybind11.html
        // Passing in an array of doubles
        /*.def("takeSample",
            [](DMD& self, py::array_t<double> u_in, double t) {
            //void twice(py::array_t<double> xs) {
            py::buffer_info info = u_in.request();
            auto ptr = static_cast<double *>(info.ptr);

            self.takeSample(ptr, t);
            },
            py::arg("u_in"),
             py::arg("t")
        )*/

         .def("train", 
               py::overload_cast<double, const Matrix *, double>(&DMD::train),
               py::arg("energy_fraction"), py::arg("W0") = nullptr, py::arg("linearity_tol") = 0.0)

         .def("train", 
               py::overload_cast<int, const Matrix *, double>(&DMD::train),
               py::arg("k"), 
               py::arg("W0") = nullptr, 
               py::arg("linearity_tol") = 0.0)

         .def("projectInitialCondition", 
               &DMD::projectInitialCondition,
               py::arg("init"), 
               py::arg("t_offset") = -1.0)

         .def("predict", 
               &DMD::predict, 
               py::arg("t"), 
               py::arg("deg") = 0)

         .def("getTimeOffset", 
               &DMD::getTimeOffset)

         .def("getNumSamples", 
               &DMD::getNumSamples)

         .def("getDimension", 
               &DMD::getDimension)

         .def("getSnapshotMatrix", 
               &DMD::getSnapshotMatrix)

         .def("load", 
               py::overload_cast<std::string>(&DMD::load), 
               py::arg("base_file_name"))

         .def("load", 
               py::overload_cast<const char *>(&DMD::load), 
               py::arg("base_file_name"))

         .def("save", 
               py::overload_cast<std::string>(&DMD::save), 
               py::arg("base_file_name"))

         .def("save", 
               py::overload_cast<const char *>(&DMD::save), 
               py::arg("base_file_name"))

         .def("summary", 
               &DMD::summary, 
               py::arg("base_file_name"))

         // TODO: needed explicitly?
         .def("__del__", 
               [](DMD &self)
               { self.~DMD(); }); // Destructor
}

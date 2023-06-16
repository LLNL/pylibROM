#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_vector(pybind11::module_ &);
void init_matrix(pybind11::module_ &);
void init_BasisGenerator(pybind11::module_ &);
void init_Options(pybind11::module_ &m);

PYBIND11_MODULE(pylibROM, m) {
	py::module linalg = m.def_submodule("linalg");
    init_vector(linalg);
    init_matrix(linalg);
    init_BasisGenerator(linalg);
    init_Options(linalg);
}

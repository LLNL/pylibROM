#include <pybind11/pybind11.h>

namespace py = pybind11;

//linalg
void init_vector(pybind11::module_ &);
void init_matrix(pybind11::module_ &);
void init_BasisGenerator(pybind11::module_ &);
void init_Options(pybind11::module_ &m);

//algo
void init_DMD(pybind11::module_ &);

PYBIND11_MODULE(pylibROM, m) {
	py::module linalg = m.def_submodule("linalg");
    init_vector(linalg);
    init_matrix(linalg);
    init_BasisGenerator(linalg);
    init_Options(linalg);
    py::module algo = m.def_submodule("algo");
    init_DMD(algo);
}

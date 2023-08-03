#include <pybind11/pybind11.h>

namespace py = pybind11;

//linalg
void init_vector(pybind11::module_ &);
void init_matrix(pybind11::module_ &);
void init_BasisGenerator(pybind11::module_ &);
void init_BasisWriter(pybind11::module_ &);
void init_BasisReader(pybind11::module_ &);
void init_Options(pybind11::module_ &m);
//linalg/svd
void init_SVD(pybind11::module_ &m); 
void init_StaticSVD(pybind11::module& m);
void init_IncrementalSVD(pybind11::module_ &m); 
void init_mpi_utils(pybind11::module_ &m);

//algo
void init_DMD(pybind11::module_ &);

PYBIND11_MODULE(pylibROM, m) {
	py::module linalg = m.def_submodule("linalg");
    init_vector(linalg);
    init_matrix(linalg);
    init_BasisGenerator(linalg);
    init_BasisWriter(linalg);
    init_BasisReader(linalg);
    init_Options(linalg);
    py::module svd = linalg.def_submodule("svd");
    init_SVD(svd);
    init_StaticSVD(svd);
    init_IncrementalSVD(svd);

    py::module algo = m.def_submodule("algo");
    init_DMD(algo);

    py::module utils = m.def_submodule("utils");
    init_mpi_utils(utils);
}

/*
void init_DMD(pybind11::module_ &);

PYBIND11_MODULE(pylibROM, m) {
	py::module algo = m.def_submodule("algo");
    init_DMD(algo);
}
*/
#include <pybind11/pybind11.h>

#include "CAROM_config.h"
#include "pylibROM_config.h"

// check that libROM has MFEM if pylibROM is using MFEM
#ifdef PYLIBROM_HAS_MFEM
// temporarily disabled until libROM upstream adds this option
// #ifndef CAROM_HAS_MFEM
// #error "libROM was not compiled with MFEM support"
// #endif
#endif

namespace py = pybind11;

//linalg
void init_vector(pybind11::module_ &);
void init_matrix(pybind11::module_ &);
void init_BasisGenerator(pybind11::module_ &);
void init_BasisWriter(pybind11::module_ &);
void init_BasisReader(pybind11::module_ &);
void init_Options(pybind11::module_ &m);
void init_NNLSSolver(pybind11::module_ &m);


//linalg/svd
void init_SVD(pybind11::module_ &m); 
void init_StaticSVD(pybind11::module& m);
void init_IncrementalSVD(pybind11::module_ &m); 

//algo
void init_DMD(pybind11::module_ &);
void init_ParametricDMD(pybind11::module_ &m);
void init_NonuniformDMD(pybind11::module_ &m);
void init_AdaptiveDMD(pybind11::module_ &m);

//algo/manifold_interp
void init_Interpolator(pybind11::module_ &);
void init_VectorInterpolator(pybind11::module_ &);
void init_MatrixInterpolator(pybind11::module_ &);

//algo/greedy
void init_GreedySampler(pybind11::module_ &m);
void init_GreedyCustomSampler(pybind11::module_ &m);
void init_GreedyRandomSampler(pybind11::module_ &m);

//hyperreduction
void init_DEIM(pybind11::module_ &m);
void init_GNAT(pybind11::module_ &m);
void init_QDEIM(pybind11::module_ &m);
void init_S_OPT(pybind11::module_ &m);
void init_STSampling(pybind11::module_ &m);
void init_Utilities(pybind11::module_ &m);

//utils
void init_mpi_utils(pybind11::module_ &m);
void init_Database(pybind11::module_ &m);
void init_HDFDatabase(pybind11::module_ &m);
void init_HDFDatabaseMPIO(pybind11::module_ &m);
void init_CSVDatabase(pybind11::module_ &m);

#ifdef PYLIBROM_HAS_MFEM
//mfem
void init_mfem_Utilities(pybind11::module_ &m);
void init_mfem_PointwiseSnapshot(pybind11::module_ &m);
void init_mfem_SampleMesh(pybind11::module_ &m);
#endif

PYBIND11_MODULE(_pylibROM, m) {
    py::module utils = m.def_submodule("utils");
    init_mpi_utils(utils);
    init_Database(utils);
    init_HDFDatabase(utils);
    init_HDFDatabaseMPIO(utils);
    init_CSVDatabase(utils);
    
	py::module linalg = m.def_submodule("linalg");
    init_vector(linalg);
    init_matrix(linalg);
    init_BasisGenerator(linalg);
    init_BasisWriter(linalg);
    init_BasisReader(linalg);
    init_Options(linalg);
    init_NNLSSolver(linalg);

    py::module svd = linalg.def_submodule("svd");
    init_SVD(svd);
    init_StaticSVD(svd);
    init_IncrementalSVD(svd);

    py::module algo = m.def_submodule("algo");
    init_DMD(algo);
    init_ParametricDMD(algo);
    init_AdaptiveDMD(algo);
    init_NonuniformDMD(algo);

    py::module manifold_interp = algo.def_submodule("manifold_interp");
    init_Interpolator(manifold_interp);
    init_VectorInterpolator(manifold_interp);
    init_MatrixInterpolator(manifold_interp);

    py::module greedy = algo.def_submodule("greedy");
    init_GreedySampler(greedy);
    init_GreedyCustomSampler(greedy);
    init_GreedyRandomSampler(greedy);

    py::module hyperreduction = m.def_submodule("hyperreduction");
    init_DEIM(hyperreduction);
    init_GNAT(hyperreduction);
    init_QDEIM(hyperreduction);
    init_S_OPT(hyperreduction);
    init_STSampling(hyperreduction);
    init_Utilities(hyperreduction);

#ifdef PYLIBROM_HAS_MFEM
    py::module mfem = m.def_submodule("mfem");
    init_mfem_Utilities(mfem);
    init_mfem_PointwiseSnapshot(mfem);
    init_mfem_SampleMesh(mfem);
#endif

    // py::module python_utils = m.def_submodule("python_utils");
}

/*
void init_DMD(pybind11::module_ &);

PYBIND11_MODULE(pylibROM, m) {
	py::module algo = m.def_submodule("algo");
    init_DMD(algo);
}
*/

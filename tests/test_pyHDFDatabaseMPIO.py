import sys
import pytest
import mpi4py
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
    from pylibROM.utils import Database, HDFDatabaseMPIO
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM
    from _pylibROM.utils import Database, HDFDatabaseMPIO
import numpy as np

nrow = 123
ncol = 21
threshold = 1.0e-13


def test_HDFDatabase():
    assert mpi4py.MPI.Is_initialized()
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    dim_rank = 2
    nrow_local = pylibROM.split_dimension(nrow, comm)
    global_dim, offsets = pylibROM.get_global_offsets(nrow_local, comm)
    assert global_dim == nrow

    rng = np.random.default_rng(1234)

    # distribute from a global matrix to keep the same system for different nproc
    snapshots = libROM.Matrix(nrow, ncol, False)
    for i in range(nrow):
        for j in range(ncol):
            snapshots[i, j] = rng.normal(0.0, 1.0)
    snapshots.distribute(nrow_local)

    options = libROM.Options(nrow_local, ncol, True)
    options.setMaxBasisDimension(nrow)
    options.setRandomizedSVD(False)
    options.setDebugMode(True)

    sampler = libROM.BasisGenerator(options, False, "test_basis", Database.formats.HDF5)

    sample = libROM.Vector(nrow_local, True)
    for s in range(ncol):
        for d in range(nrow_local):
            sample[d] = snapshots[d, s]
        sampler.takeSample(sample.getData())

    sampler.writeSnapshot()
    snapshot = sampler.getSnapshotMatrix()
    sampler.endSamples()

    spatial_basis = sampler.getSpatialBasis()

    basis_reader = libROM.BasisReader("test_basis")
    spatial_basis1 = basis_reader.getSpatialBasis()
    assert spatial_basis.numRows() == spatial_basis1.numRows()
    assert spatial_basis.numColumns() == spatial_basis1.numColumns()
    np.testing.assert_array_almost_equal(spatial_basis.getData(), spatial_basis1.getData(), threshold)

    snapshot_reader = libROM.BasisReader("test_basis_snapshot")
    snapshot1 = snapshot_reader.getSnapshotMatrix()
    assert snapshot.numRows() == snapshot1.numRows()
    assert snapshot.numColumns() == snapshot1.numColumns()
    np.testing.assert_array_almost_equal(snapshot.getData(), snapshot1.getData(), threshold)


def test_BaseMPIOCombination():
    assert mpi4py.MPI.Is_initialized()
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    dim_rank = 2
    nrow_local = pylibROM.split_dimension(nrow, comm)
    global_dim, offsets = pylibROM.get_global_offsets(nrow_local, comm)
    assert global_dim == nrow

    base_name = "test_basis"
    mpio_name = "test_mpio"

    options = libROM.Options(nrow_local, ncol, True)
    options.setMaxBasisDimension(nrow)
    options.setRandomizedSVD(False)
    options.setDebugMode(True)

    sampler = libROM.BasisGenerator(options, False, mpio_name, Database.formats.HDF5_MPIO)

    sampler.loadSamples(base_name + "_snapshot", "snapshot", int(1e9), Database.formats.HDF5)
    sampler.writeSnapshot()
    snapshot = sampler.getSnapshotMatrix()

    snapshot_reader = libROM.BasisReader("test_basis_snapshot")
    snapshot1 = snapshot_reader.getSnapshotMatrix()
    assert snapshot.numRows() == snapshot1.numRows()
    assert snapshot.numColumns() == snapshot1.numColumns()
    np.testing.assert_array_almost_equal(snapshot.getData(), snapshot1.getData(), threshold)

    sampler.endSamples()
    spatial_basis = sampler.getSpatialBasis()

    basis_reader = libROM.BasisReader("test_basis")
    spatial_basis1 = basis_reader.getSpatialBasis()
    assert spatial_basis.numRows() == spatial_basis1.numRows()
    assert spatial_basis.numColumns() == spatial_basis1.numColumns()
    np.testing.assert_array_almost_equal(spatial_basis.getData(), spatial_basis1.getData(), threshold)


def test_MPIOBaseCombination():
    assert mpi4py.MPI.Is_initialized()
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    dim_rank = 2
    nrow_local = pylibROM.split_dimension(nrow, comm)
    global_dim, offsets = pylibROM.get_global_offsets(nrow_local, comm)
    assert global_dim == nrow

    base_name = "test_basis2"
    mpio_name = "test_mpio"

    options = libROM.Options(nrow_local, ncol, True)
    options.setMaxBasisDimension(nrow)
    options.setRandomizedSVD(False)
    options.setDebugMode(True)

    sampler = libROM.BasisGenerator(options, False, mpio_name, Database.formats.HDF5)

    sampler.loadSamples(mpio_name + "_snapshot", "snapshot", int(1e9), Database.formats.HDF5_MPIO)
    #sampler.writeSnapshot()
    snapshot = sampler.getSnapshotMatrix()

    snapshot_reader = libROM.BasisReader("test_basis_snapshot")
    snapshot1 = snapshot_reader.getSnapshotMatrix()
    assert snapshot.numRows() == snapshot1.numRows()
    assert snapshot.numColumns() == snapshot1.numColumns()
    np.testing.assert_array_almost_equal(snapshot.getData(), snapshot1.getData(), threshold)

    sampler.endSamples()
    spatial_basis = sampler.getSpatialBasis()

    basis_reader = libROM.BasisReader("test_basis")
    spatial_basis1 = basis_reader.getSpatialBasis()
    assert spatial_basis.numRows() == spatial_basis1.numRows()
    assert spatial_basis.numColumns() == spatial_basis1.numColumns()
    np.testing.assert_array_almost_equal(spatial_basis.getData(), spatial_basis1.getData(), threshold)


def test_partialGetSpatialBasis():
    assert mpi4py.MPI.Is_initialized()
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    nrow_local = pylibROM.split_dimension(nrow, comm)
    global_dim, offsets = pylibROM.get_global_offsets(nrow_local, comm)
    assert global_dim == nrow

    base_name = "test_basis"
    mpio_name = "test_mpio"

    basis_reader = libROM.BasisReader(base_name)
    spatial_basis = basis_reader.getSpatialBasis()

    basis_reader1 = libROM.BasisReader(mpio_name, Database.formats.HDF5_MPIO, nrow_local)
    spatial_basis1 = basis_reader1.getSpatialBasis()

    assert spatial_basis.numRows() == spatial_basis1.numRows()
    assert spatial_basis.numColumns() == spatial_basis1.numColumns()
    np.testing.assert_array_almost_equal(spatial_basis.getData(), spatial_basis1.getData(), threshold)


if __name__ == "__main__":
    pytest.main()

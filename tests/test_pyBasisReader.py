import sys
import pytest
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
    from pylibROM.utils import Database
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM
    from _pylibROM.utils import Database
import numpy as np 

options = libROM.Options(4, 20, True, True)
options.static_svd_preserve_snapshot = True
generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
result = generator.takeSample(np.array([1.0, 2.0, 3.0]))
base_file_name = "test_basisreader_file"
basis_writer = libROM.BasisWriter(generator, base_file_name, Database.formats.HDF5)
basis_writer.writeBasis("basis")
del basis_writer
generator.writeSnapshot()
generator.endSamples()
del generator


def test_getDim():
    basis_reader = libROM.BasisReader("test_basisreader_file", Database.formats.HDF5)
    dim = basis_reader.getDim("basis")
    assert dim == 4


def test_getNumSamples():
    basis_reader = libROM.BasisReader("test_basisreader_file", Database.formats.HDF5)
    num_samples = basis_reader.getNumSamples("basis")
    assert num_samples == 1


def test_getSpatialBasis():
    options = libROM.Options(4, 20, True, True)
    options.static_svd_preserve_snapshot = True
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    result = generator.takeSample(np.array([1.0, 2.0, 3.0]))
    base_file_name = "test_basisreader_file"
    basis_writer = libROM.BasisWriter(generator, base_file_name, Database.formats.HDF5)
    basis_writer.writeBasis("basis")
    del basis_writer
    generator.writeSnapshot()
    generator.endSamples()
    del generator
    basis_reader = libROM.BasisReader("test_basisreader_file", Database.formats.HDF5)

    spatial_basis1 = basis_reader.getSpatialBasis()
    assert(np.allclose(spatial_basis1.getData(),[[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.44659081e-323]]))

    spatial_basis2 = basis_reader.getSpatialBasis(1)
    assert(np.allclose(spatial_basis2.getData(), [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.44659081e-323]]))

    spatial_basis3 = basis_reader.getSpatialBasis(1, 1)
    assert(np.allclose(spatial_basis3.getData(), [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.44659081e-323]]))

    spatial_basis4 = basis_reader.getSpatialBasis(0.7)
    assert(np.allclose(spatial_basis4.getData(), [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731],[-4.44659081e-323]]))


def test_getTemporalBasis():
    basis_reader = libROM.BasisReader("test_basisreader_file", Database.formats.HDF5)

    temporal_basis1 = basis_reader.getTemporalBasis()
    assert(np.array_equal(temporal_basis1.getData(), [[-1.0]]))

    temporal_basis2 = basis_reader.getTemporalBasis(1)
    assert(np.array_equal(temporal_basis2.getData(), [[-1.0]]))

    temporal_basis3 = basis_reader.getTemporalBasis(1, 1)
    assert(np.array_equal(temporal_basis3.getData(), [[-1.0]]))

    temporal_basis4 = basis_reader.getTemporalBasis(0.7)
    assert(np.array_equal(temporal_basis4.getData(), [[-1.0]]))
    

def test_getSingularValues():
    basis_reader = libROM.BasisReader("test_basisreader_file", Database.formats.HDF5)

    singular_values1 = basis_reader.getSingularValues()
    assert(np.array_equal(singular_values1.getData(), [3.7416573867739418]))

    singular_values2 = basis_reader.getSingularValues(0.7)
    assert(np.array_equal(singular_values2.getData(), [3.7416573867739418]))



def test_getSnapshotMatrix():
    basis_reader = libROM.BasisReader("basis.h5_snapshot", Database.formats.HDF5)

    snapshot_matrix1 = basis_reader.getSnapshotMatrix()
    assert(np.allclose(snapshot_matrix1.getData(),[[1.0], [2.0], [3.0], [0.0]]))

    snapshot_matrix2 = basis_reader.getSnapshotMatrix(1)
    assert(np.allclose(snapshot_matrix2.getData(), [[1.0], [2.0], [3.0], [0.0]]))

    snapshot_matrix3 = basis_reader.getSnapshotMatrix(1, 1)
    assert(np.allclose(snapshot_matrix3.getData(), [[1.0], [2.0], [3.0], [0.0]]))


if __name__ == "__main__":
    pytest.main()

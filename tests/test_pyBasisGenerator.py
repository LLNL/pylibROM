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
import h5py


def test_isNextSample():
    options = libROM.Options(4, 20, True, True)
    incremental = False
    basis_file_name = "basis.h5"
    file_format = Database.formats.HDF5
    generator = libROM.BasisGenerator(options, incremental, basis_file_name, file_format)
    time = 1.0
    is_next = generator.isNextSample(time)
    assert is_next

def test_updateRightSV():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    update_success = generator.updateRightSV()
    assert update_success 

def test_takeSample():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5) 
    u_in_data = np.array([1.0, 2.0, 3.0])
    result = generator.takeSample(u_in_data)
    assert result  

def test_writeSnapshot():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    generator.takeSample(np.array([1.0, 2.0, 3.0]))
    generator.writeSnapshot()

def test_computeNextSampleTime():
    options = libROM.Options(4, 20, True, True)
    generator1 = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    generator1.takeSample(np.array([1.0, 2.0, 3.0]))
    base_file_name = "test_basisgenerator_file"
    basis_writer = libROM.BasisWriter(generator1, base_file_name, Database.formats.HDF5)
    basis_writer.writeBasis("basis")
    del basis_writer
    del generator1 
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    kind = "basis"
    cut_off = 10
    db_format = Database.formats.HDF5
    generator.loadSamples(base_file_name, kind, cut_off, db_format)
    u_in = [1.0, 2.0, 3.0]
    rhs_in = [0.1, 0.2, 0.3]
    time = 0.0
    next_sample_time = generator.computeNextSampleTime(u_in, rhs_in, time)
    assert next_sample_time == 0.0
    generator.endSamples()

def test_getSpatialBasis():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    generator.takeSample(np.array([1.0, 2.0, 3.0]))
    spatial_basis = generator.getSpatialBasis()
    expected_spatial_basis = np.array([[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-6.4e-323]])
    assert np.allclose(spatial_basis, expected_spatial_basis)

def test_getTemporalBasis():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    generator.takeSample(np.array([1.0, 2.0, 3.0]))
    temporal_basis = generator.getTemporalBasis()
    expected_temporal_basis = np.array([[-1.0]])
    assert np.allclose(temporal_basis, expected_temporal_basis)
    
def test_getSingularValues():   
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    generator.takeSample(np.array([1.0, 2.0, 3.0]))
    singular_values = generator.getSingularValues()
    assert(np.array_equal(singular_values.getData(),[3.7416573867739418]))

def test_getSnapshotMatrix():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    generator.takeSample(np.array([1.0, 2.0, 3.0]))
    snapshot_matrix = generator.getSnapshotMatrix()
    expected_snapshot_basis = np.array([[1.0], [2.0], [3.0], [5.6e-322]])
    assert np.allclose(snapshot_matrix, expected_snapshot_basis)

def test_getNumSamples():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    num_samples = generator.getNumSamples()
    assert num_samples == 0

if __name__ == "__main__":
    pytest.main()
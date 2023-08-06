# __init__.py is needed only for pure python routines.

def swigdouble2numpyarray(u_swig, u_size):
    from ctypes import c_double
    from numpy import array
    return array((c_double * u_size).from_address(int(u_swig)), copy=False)
    
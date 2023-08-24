# To add pure python routines to this module,
# either define/import the python routine in this file.
# This will combine both c++ bindings/pure python routines into this module.
from .StopWatch import StopWatch

def swigdouble2numpyarray(u_swig, u_size):
    from ctypes import c_double
    from numpy import array
    return array((c_double * u_size).from_address(int(u_swig)), copy=False)
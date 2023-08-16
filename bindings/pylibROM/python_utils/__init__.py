# To add pure python routines to this module,
# either define/import the python routine in this file.
# This will combine both c++ bindings/pure python routines into this module.
import time

def swigdouble2numpyarray(u_swig, u_size):
    from ctypes import c_double
    from numpy import array
    return array((c_double * u_size).from_address(int(u_swig)), copy=False)

# pyMFEM does not provide mfem::StopWatch.
# there is not really a similar stopwatch package in python.. (seriously?)
class StopWatch:
    duration = 0.0
    start_time = 0.0
    stop_time = 0.0
    running = False

    def __init__(self):
        self.Reset()
        return
    
    def Start(self):
        assert(not self.running)
        self.start_time = time.time()
        self.running = True
        return
    
    def Stop(self):
        assert(self.running)
        self.stop_time = time.time()
        self.duration += self.stop_time - self.start_time
        self.running = False
        return
    
    def Reset(self):
        self.duration = 0.0
        self.start_time = 0.0
        self.stop_time = 0.0
        self.running = False
        return
    
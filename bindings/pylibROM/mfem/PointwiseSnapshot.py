import numpy as np
import mfem.par as mfem
from mpi4py import MPI

class PointwiseSnapshot:
    finder = None
    npoints = -1
    dims = [-1] * 3
    spaceDim = -1

    domainMin = mfem.Vector()
    domainMax = mfem.Vector()
    xyz = mfem.Vector()

    def __init__(self, sdim, dims_):
        self.finder = None
        self.spaceDim = sdim
        assert((1 < sdim) and (sdim < 4))

        self.npoints = np.prod(dims_[:self.spaceDim])
        self.dims = np.ones((3,), dtype=int)
        self.dims[:self.spaceDim] = dims_[:self.spaceDim]

        self.xyz.SetSize(self.npoints * self.spaceDim)
        self.xyz.Assign(0.)
        return

    def SetMesh(self, pmesh):
        if (self.finder is not None):
            self.finder.FreeData()  # Free the internal gslib data.
        del self.finder

        assert(pmesh.Dimension() == self.spaceDim)
        assert(pmesh.SpaceDimension() == self.spaceDim)

        self.domainMin, self.domainMax = pmesh.GetBoundingBox(0)

        h = [0.] * 3
        for i in range(self.spaceDim):
            h[i] = (self.domainMax[i] - self.domainMin[i]) / float(self.dims[i] - 1)

        rank = pmesh.GetMyRank()
        if (rank == 0):
            print("PointwiseSnapshot on bounding box from (",
                  self.domainMin[:self.spaceDim],
                  ") to (", self.domainMax[:self.spaceDim], ")")

        # TODO(kevin): we might want to re-write this loop in python manner.
        xyzData = self.xyz.GetDataArray()
        for k in range(self.dims[2]):
            pz = self.domainMin[2] + k * h[2] if (self.spaceDim > 2) else 0.0
            osk = k * self.dims[0] * self.dims[1]

            for j in range(self.dims[1]):
                py = self.domainMin[1] + j * h[1]
                osj = (j * self.dims[0]) + osk

                for i in range(self.dims[0]):
                    px = self.domainMin[0] + i * h[0]
                    xyzData[i + osj] = px
                    if (self.spaceDim > 1): xyzData[self.npoints + i + osj] = py
                    if (self.spaceDim > 2): xyzData[2 * self.npoints + i + osj] = pz

        self.finder = mfem.FindPointsGSLIB(MPI.COMM_WORLD)
        # mfem.FindPointsGSLIB()
        self.finder.Setup(pmesh)
        self.finder.SetL2AvgType(mfem.FindPointsGSLIB.NONE)
        return

    def GetSnapshot(self, f, s):
        vdim = f.FESpace().GetVDim()
        s.SetSize(self.npoints * vdim)

        self.finder.Interpolate(self.xyz, f, s)

        code_out = self.finder.GetCode()
        print(type(code_out))
        print(code_out.__dir__())

        assert(code_out.Size() == self.npoints)

        # Note that Min() and Max() are not defined for Array<unsigned int>
        #MFEM_VERIFY(code_out.Min() >= 0 && code_out.Max() < 2, "");
        cmin = code_out[0]
        cmax = code_out[0]
        # TODO(kevin): does this work for mfem array?
        for c in code_out:
            if (c < cmin):
                cmin = c
            if (c > cmax):
                cmax = c

        assert((cmin >= 0) and (cmax < 2))
        return
# --------------------------------------------------------------------

cdef class DMPlex(DM):

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMPlexCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def clone(self):
        cdef DMPlex dm = <DMPlex>type(self)()
        CHKERR( DMPlexClone(self.dm, &dm.dm) )
        return dm

    def createFromCellList(self, dim, cells, coords, interpolate=True, comm=None):
        cdef DMPlex dm = <DMPlex>type(self)()
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM newdm = NULL
        cdef PetscInt cdim = asInt(dim)
        cdef PetscInt numCells = 0
        cdef PetscInt numCorners = 0
        cdef int      *cellVertices = NULL
        cdef PetscInt numVertices = 0
        cdef PetscInt spaceDim= 0
        cdef double   *vertexCoords = NULL
        cdef int npy_flags = NPY_ARRAY_ALIGNED|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_CARRAY
        cells  = PyArray_FROM_OTF(cells,  NPY_INT,    npy_flags)
        coords = PyArray_FROM_OTF(coords, NPY_DOUBLE, npy_flags)
        if PyArray_NDIM(cells) != 2: raise ValueError(
                ("cell indices must have two dimensions: "
                 "cells.ndim=%d") % (PyArray_NDIM(cells)) )
        if PyArray_NDIM(coords) != 2: raise ValueError(
                ("coords vertices must have two dimensions: "
                 "coords.ndim=%d") % (PyArray_NDIM(coords)) )
        numCells     = <PetscInt> PyArray_DIM(cells,  0)
        numCorners   = <PetscInt> PyArray_DIM(cells,  1)
        numVertices  = <PetscInt> PyArray_DIM(coords, 0)
        spaceDim     = <PetscInt> PyArray_DIM(coords, 1)
        cellVertices = <int*>     PyArray_DATA(cells)
        vertexCoords = <double*>  PyArray_DATA(coords)
        CHKERR( DMPlexCreateFromCellList(ccomm,cdim,numCells,numVertices,numCorners,interp,cellVertices,spaceDim,vertexCoords,&newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

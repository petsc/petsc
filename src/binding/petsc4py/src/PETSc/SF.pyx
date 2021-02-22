# --------------------------------------------------------------------

class SFType(object):
    BASIC      = S_(PETSCSFBASIC)
    NEIGHBOR   = S_(PETSCSFNEIGHBOR)
    ALLGATHERV = S_(PETSCSFALLGATHERV)
    ALLGATHER  = S_(PETSCSFALLGATHER)
    GATHERV    = S_(PETSCSFGATHERV)
    GATHER     = S_(PETSCSFGATHER)
    ALLTOALL   = S_(PETSCSFALLTOALL)
    WINDOW     = S_(PETSCSFWINDOW)

# --------------------------------------------------------------------

cdef class SF(Object):

    Type = SFType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sf
        self.sf  = NULL

    def __dealloc__(self):
        CHKERR( PetscSFDestroy(&self.sf) )
        self.sf = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSFView(self.sf, vwr) )

    def destroy(self):
        CHKERR( PetscSFDestroy(&self.sf) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSF newsf = NULL
        CHKERR( PetscSFCreate(ccomm, &newsf) )
        PetscCLEAR(self.obj); self.sf = newsf
        return self

    def setType(self, sf_type):
        cdef PetscSFType cval = NULL
        sf_type = str2bytes(sf_type, &cval)
        CHKERR( PetscSFSetType(self.sf, cval) )

    def getType(self):
        cdef PetscSFType cval = NULL
        CHKERR( PetscSFGetType(self.sf, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscSFSetFromOptions(self.sf) )

    def setUp(self):
        CHKERR( PetscSFSetUp(self.sf) )

    def reset(self):
        CHKERR( PetscSFReset(self.sf) )

    #

    def getGraph(self):
        """nleaves can be determined from the size of local"""
        cdef PetscInt nroots = 0, nleaves = 0
        cdef const PetscInt *ilocal = NULL
        cdef const PetscSFNode *iremote = NULL
        CHKERR( PetscSFGetGraph(self.sf, &nroots, &nleaves, &ilocal, &iremote) )
        if ilocal == NULL:
            local = arange(0, nleaves, 1)
        else:
            local = array_i(nleaves, ilocal)
        remote = array_i(nleaves*2, <const PetscInt*>iremote)
        remote = remote.reshape(nleaves, 2)
        return toInt(nroots), local, remote

    def setGraph(self, nroots, local, remote):
        """
        The nleaves argument is determined from the size of local and/or remote.
        local may be None, meaning contiguous storage.
        remote should be 2*nleaves long as (rank, index) pairs.
        """
        cdef PetscInt cnroots = asInt(nroots)
        cdef PetscInt nleaves = 0
        cdef PetscInt nremote = 0
        cdef PetscInt *ilocal = NULL
        cdef PetscSFNode* iremote = NULL
        remote = iarray_i(remote, &nremote, <PetscInt**>&iremote)
        if local is not None:
            local = iarray_i(local, &nleaves, &ilocal)
            assert 2*nleaves == nremote
        else:
            assert nremote % 2 == 0
            nleaves = nremote // 2
        CHKERR( PetscSFSetGraph(self.sf, cnroots, nleaves, ilocal, PETSC_COPY_VALUES, iremote, PETSC_COPY_VALUES) )

    def setRankOrder(self, flag):
        cdef PetscBool bval = asBool(flag)
        CHKERR( PetscSFSetRankOrder(self.sf, bval) )

    #

    def getMulti(self):
        cdef SF sf = SF()
        CHKERR( PetscSFGetMultiSF(self.sf, &sf.sf) )
        PetscINCREF(sf.obj)
        return sf

    def createInverse(self):
        cdef SF sf = SF()
        CHKERR( PetscSFCreateInverseSF(self.sf, &sf.sf) )
        return sf

    def computeDegree(self):
        cdef const PetscInt *cdegree = NULL
        cdef PetscInt nroots
        CHKERR( PetscSFComputeDegreeBegin(self.sf, &cdegree) )
        CHKERR( PetscSFComputeDegreeEnd(self.sf, &cdegree) )
        CHKERR( PetscSFGetGraph(self.sf, &nroots, NULL, NULL, NULL) )
        degree = array_i(nroots, cdegree)
        return degree

    def createEmbeddedRootSF(self, selected):
        cdef PetscInt nroots = asInt(len(selected))
        cdef PetscInt *cselected = NULL
        selected = iarray_i(selected, &nroots, &cselected)
        cdef SF sf = SF()
        CHKERR( PetscSFCreateEmbeddedRootSF(self.sf, nroots, cselected, &sf.sf) )
        return sf

    def createEmbeddedLeafSF(self, selected):
        cdef PetscInt nleaves = asInt(len(selected))
        cdef PetscInt *cselected = NULL
        selected = iarray_i(selected, &nleaves, &cselected)
        cdef SF sf = SF()
        CHKERR( PetscSFCreateEmbeddedLeafSF(self.sf, nleaves, cselected, &sf.sf) )
        return sf

    def bcastBegin(self, unit, ndarray rootdata, ndarray leafdata, op):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        CHKERR( PetscSFBcastBegin(self.sf, dtype, <const void*>PyArray_DATA(rootdata),
                                  <void*>PyArray_DATA(leafdata), cop) )

    def bcastEnd(self, unit, ndarray rootdata, ndarray leafdata, op):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        CHKERR( PetscSFBcastEnd(self.sf, dtype, <const void*>PyArray_DATA(rootdata),
                                <void*>PyArray_DATA(leafdata), cop) )

    def reduceBegin(self, unit, ndarray leafdata, ndarray rootdata, op):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        CHKERR( PetscSFReduceBegin(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                   <void*>PyArray_DATA(rootdata), cop) )

    def reduceEnd(self, unit, ndarray leafdata, ndarray rootdata, op):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        CHKERR( PetscSFReduceEnd(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                 <void*>PyArray_DATA(rootdata), cop) )

    def scatterBegin(self, unit, ndarray multirootdata, ndarray leafdata):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        CHKERR( PetscSFScatterBegin(self.sf, dtype, <const void*>PyArray_DATA(multirootdata),
                                    <void*>PyArray_DATA(leafdata)) )

    def scatterEnd(self, unit, ndarray multirootdata, ndarray leafdata):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        CHKERR( PetscSFScatterEnd(self.sf, dtype, <const void*>PyArray_DATA(multirootdata),
                                  <void*>PyArray_DATA(leafdata)) )

    def gatherBegin(self, unit, ndarray leafdata, ndarray multirootdata):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        CHKERR( PetscSFGatherBegin(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                   <void*>PyArray_DATA(multirootdata)) )

    def gatherEnd(self, unit, ndarray leafdata, ndarray multirootdata):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        CHKERR( PetscSFGatherEnd(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                 <void*>PyArray_DATA(multirootdata)) )

    def fetchAndOpBegin(self, unit, rootdata, leafdata, leafupdate, op):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        CHKERR( PetscSFFetchAndOpBegin(self.sf, dtype, <void*>PyArray_DATA(rootdata),
                                       <const void*>PyArray_DATA(leafdata),
                                       <void*>PyArray_DATA(leafupdate), cop) )

    def fetchAndOpEnd(self, unit, rootdata, leafdata, leafupdate, op):
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        CHKERR( PetscSFFetchAndOpEnd(self.sf, dtype, <void*>PyArray_DATA(rootdata),
                                     <const void*>PyArray_DATA(leafdata),
                                     <void*>PyArray_DATA(leafupdate), cop) )

# --------------------------------------------------------------------

del SFType

# --------------------------------------------------------------------

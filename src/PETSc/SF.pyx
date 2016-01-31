# --------------------------------------------------------------------

class SFType(object):
    BASIC  = S_(PETSCSFBASIC)
    WINDOW = S_(PETSCSFWINDOW)

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
        CHKERR( PetscObjectGetType(<PetscObject>self.sf, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscSFSetFromOptions(self.sf) )

    def setUp(self):
        CHKERR( PetscSFSetUp(self.sf) )

    def reset(self):
        CHKERR( PetscSFReset(self.sf) )

    #

    def getGraph(self):
        cdef PetscInt nroots = 0, nleaves = 0
        cdef const_PetscInt *ilocal = NULL
        cdef const_PetscSFNode *iremote = NULL
        CHKERR( PetscSFGetGraph(self.sf, &nroots, &nleaves, &ilocal, &iremote) )
        local = array_i(nleaves, ilocal)
        remote = []
        for i in range(toInt(nleaves)):
            sfnode = (toInt(iremote[asInt(i)].rank), toInt(iremote[asInt(i)].index))
            remote.append( sfnode )
        return toInt(nroots), toInt(nleaves), local, remote

    def setGraph(self, nroots, nleaves, local, remote):
        cdef PetscInt cnroots = asInt(nroots)
        cdef PetscInt cnleaves = asInt(nleaves)
        cdef PetscInt nlocal = 0
        cdef PetscInt *ilocal = NULL
        local = iarray_i(local, &nlocal, &ilocal)
        cdef PetscSFNode* iremote = NULL
        CHKERR( PetscMalloc(nleaves*sizeof(PetscSFNode), &iremote) )
        cdef int i = 0
        for rank, index in remote:
            iremote[i].rank  = asInt(rank)
            iremote[i].index = asInt(index)
            i += 1
        CHKERR( PetscSFSetGraph(self.sf, cnroots, cnleaves, ilocal, PETSC_COPY_VALUES, iremote, PETSC_OWN_POINTER) )

    def setRankOrder(self, flag):
        cdef PetscBool bval = flag
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

# --------------------------------------------------------------------

del SFType

# --------------------------------------------------------------------

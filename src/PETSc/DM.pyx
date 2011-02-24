# --------------------------------------------------------------------

cdef class DM(Object):

    def ___cinit__(self):
        self.dm = <PetscDM*> self.obj

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( DMView(self.dm[0], vwr) )

    def destroy(self):
        CHKERR( DMDestroy(self.dm[0]) )
        self.dm[0] = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm[0] = newdm
        return self

    def setType(self, dm_type):
        cdef const_char *cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        CHKERR( DMSetType(self.dm[0], cval) )

    def getType(self):
        cdef PetscDMType cval = NULL
        CHKERR( DMGetType(self.dm[0], &cval) )
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DMSetOptionsPrefix(self.dm[0], cval) )

    def setFromOptions(self):
        CHKERR( DMSetFromOptions(self.dm[0]) )

    def setUp(self):
        CHKERR( DMSetUp(self.dm[0]) )
        return self

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')
    #

    def createGlobalVec(self):
        cdef Vec vg = Vec()
        CHKERR( DMCreateGlobalVector(self.dm[0], &vg.vec) )
        return vg

    def createLocalVec(self):
        cdef Vec vl = Vec()
        CHKERR( DMCreateLocalVector(self.dm[0], &vl.vec) )
        return vl

    def globalToLocal(self, Vec vg not None, Vec vl not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMGlobalToLocalBegin(self.dm[0], vg.vec, im, vl.vec) )
        CHKERR( DMGlobalToLocalEnd  (self.dm[0], vg.vec, im, vl.vec) )

    def localToGlobal(self, Vec vl not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMLocalToGlobalBegin(self.dm[0], vl.vec, im, vg.vec) )
        CHKERR( DMLocalToGlobalEnd(self.dm[0], vl.vec, im, vg.vec) )

    #

    def createMat(self, mat_type=None):
        cdef PetscMatType mtype = MATAIJ
        mat_type = str2bytes(mat_type, &mtype)
        if mtype == NULL: mtype = MATAIJ
        cdef Mat mat = Mat()
        CHKERR( DMGetMatrix(self.dm[0], mtype, &mat.mat) )
        return mat

    def getInterpolation(self, DM dm not None):
        cdef Mat A = Mat()
        cdef Vec scale = Vec()
        CHKERR( DMGetInterpolation(self.dm[0], dm.dm[0], 
                                   &A.mat, &scale.vec))
        return(A, scale)

    def getInjection(self, DM dm not None):
        cdef Scatter sct = Scatter()
        CHKERR( DMGetInjection(self.dm[0], dm.dm[0], &sct.sct) )
        return sct

    def refine(self, comm=None):
        cdef MPI_Comm dmcomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm[0], &dmcomm) )
        dmcomm = def_Comm(comm, dmcomm)
        cdef DM dm = DM()
        CHKERR( DMRefine(self.dm[0], dmcomm, &dm.dm[0]) )
        return dm

    def coarsen(self, comm=None):
        cdef MPI_Comm dmcomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm[0], &dmcomm) )
        dmcomm = def_Comm(comm, dmcomm)
        cdef DM dm = DM()
        CHKERR( DMCoarsen(self.dm[0], dmcomm, &dm.dm[0]) )
        return dm

    #

    def getLGMap(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DMGetLocalToGlobalMapping(self.dm[0], &lgm.lgm) )
        PetscIncref(<PetscObject>lgm.lgm)
        return lgm

    def getLGMapBlock(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DMGetLocalToGlobalMappingBlock(self.dm[0], &lgm.lgm) )
        PetscIncref(<PetscObject>lgm.lgm)
        return lgm

# --------------------------------------------------------------------

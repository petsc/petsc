# --------------------------------------------------------------------

cdef class DM(Object):

    def __cinit__(self):
        self.obj = <PetscObject*> &self.dm
        self.dm  = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( DMView(self.dm, vwr) )

    def destroy(self):
        CHKERR( DMDestroy(&self.dm) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def setType(self, dm_type):
        cdef const_char *cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        CHKERR( DMSetType(self.dm, cval) )

    def getType(self):
        cdef PetscDMType cval = NULL
        CHKERR( DMGetType(self.dm, &cval) )
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DMSetOptionsPrefix(self.dm, cval) )

    def setFromOptions(self):
        CHKERR( DMSetFromOptions(self.dm) )

    def setUp(self):
        CHKERR( DMSetUp(self.dm) )
        return self

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')

    #

    def getBlockSize(self):
        cdef PetscInt bs = 1
        CHKERR( DMGetBlockSize(self.dm, &bs) )
        return toInt(bs)

    def createGlobalVec(self):
        cdef Vec vg = Vec()
        CHKERR( DMCreateGlobalVector(self.dm, &vg.vec) )
        return vg

    def createLocalVec(self):
        cdef Vec vl = Vec()
        CHKERR( DMCreateLocalVector(self.dm, &vl.vec) )
        return vl

    def globalToLocal(self, Vec vg not None, Vec vl not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMGlobalToLocalBegin(self.dm, vg.vec, im, vl.vec) )
        CHKERR( DMGlobalToLocalEnd  (self.dm, vg.vec, im, vl.vec) )

    def localToGlobal(self, Vec vl not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMLocalToGlobalBegin(self.dm, vl.vec, im, vg.vec) )
        CHKERR( DMLocalToGlobalEnd(self.dm, vl.vec, im, vg.vec) )

    def getLGMap(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DMGetLocalToGlobalMapping(self.dm, &lgm.lgm) )
        PetscINCREF(lgm.obj)
        return lgm

    def getLGMapBlock(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DMGetLocalToGlobalMappingBlock(self.dm, &lgm.lgm) )
        PetscINCREF(lgm.obj)
        return lgm

    #

    def createMat(self, mat_type=None):
        cdef PetscMatType mtype = MATAIJ
        mat_type = str2bytes(mat_type, &mtype)
        if mtype == NULL: mtype = MATAIJ
        cdef Mat mat = Mat()
        CHKERR( DMGetMatrix(self.dm, mtype, &mat.mat) )
        return mat

    def getInterpolation(self, DM dm not None):
        cdef Mat A = Mat()
        cdef Vec scale = Vec()
        CHKERR( DMGetInterpolation(self.dm, dm.dm, 
                                   &A.mat, &scale.vec))
        return(A, scale)

    def getInjection(self, DM dm not None):
        cdef Scatter sct = Scatter()
        CHKERR( DMGetInjection(self.dm, dm.dm, &sct.sct) )
        return sct

    def getAggregates(self, DM dm not None):
        cdef Mat mat = Mat()
        CHKERR( DMGetAggregates(self.dm, dm.dm, &mat.mat) )
        return mat

    def refine(self, comm=None):
        cdef MPI_Comm dmcomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm, &dmcomm) )
        dmcomm = def_Comm(comm, dmcomm)
        cdef PetscDM newdm = NULL
        CHKERR( DMRefine(self.dm, dmcomm, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        return dm

    def coarsen(self, comm=None):
        cdef MPI_Comm dmcomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm, &dmcomm) )
        dmcomm = def_Comm(comm, dmcomm)
        cdef PetscDM newdm = NULL
        CHKERR( DMCoarsen(self.dm, dmcomm, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        return dm

    def refineHierarchy(self, nlevels):
        cdef PetscInt i, n = asInt(nlevels)
        cdef PetscDM *newdmf = NULL
        cdef object tmp = oarray_p(empty_p(n), NULL, <void**>&newdmf)
        CHKERR( DMRefineHierarchy(self.dm, n, newdmf) )
        cdef DM dmf = None
        cdef list hierarchy = []
        for i from 0 <= i < n:
            dmf = subtype_DM(newdmf[i])()
            dmf.dm = newdmf[i]
            hierarchy.append(dmf)
        return hierarchy

    def coarsenHierarchy(self, nlevels):
        cdef PetscInt i, n = asInt(nlevels)
        cdef PetscDM *newdmc = NULL
        cdef object tmp = oarray_p(empty_p(n),NULL, <void**>&newdmc)
        CHKERR( DMCoarsenHierarchy(self.dm, n, newdmc) )
        cdef DM dmc = None
        cdef list hierarchy = []
        for i from 0 <= i < n:
            dmc = subtype_DM(newdmc[i])()
            dmc.dm = newdmc[i]
            hierarchy.append(dmc)
        return hierarchy

# --------------------------------------------------------------------

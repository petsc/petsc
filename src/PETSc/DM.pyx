# --------------------------------------------------------------------

class DMType(object):
    DA        = S_(DMDA_type)
    COMPOSITE = S_(DMCOMPOSITE)
    SLICED    = S_(DMSLICED)
    SHELL     = S_(DMSHELL)
    PLEX      = S_(DMPLEX)
    CARTESIAN = S_(DMCARTESIAN)
    REDUNDANT = S_(DMREDUNDANT)
    PATCH     = S_(DMPATCH)
    MOAB      = S_(DMMOAB)
    NETWORK   = S_(DMNETWORK)

class DMBoundaryType(object):
    NONE     = DM_BOUNDARY_NONE
    GHOSTED  = DM_BOUNDARY_GHOSTED
    MIRROR   = DM_BOUNDARY_MIRROR
    PERIODIC = DM_BOUNDARY_PERIODIC
    TWIST    = DM_BOUNDARY_TWIST

# --------------------------------------------------------------------

cdef class DM(Object):

    Type         = DMType
    BoundaryType = DMBoundaryType

    #

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

    def clone(self):
        cdef DM dm = type(self)()
        CHKERR( DMClone(self.dm, &dm.dm) )
        return dm

    def setType(self, dm_type):
        cdef const_char *cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        CHKERR( DMSetType(self.dm, cval) )

    def getType(self):
        cdef PetscDMType cval = NULL
        CHKERR( DMGetType(self.dm, &cval) )
        return bytes2str(cval)

    def getDimension(self):
        cdef PetscInt dim = 0
        CHKERR( DMGetDimension(self.dm, &dim) )
        return toInt(dim)

    def setDimension(self, dim):
        cdef PetscInt cdim = asInt(dim)
        CHKERR( DMSetDimension(self.dm, cdim) )


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

    def setVecType(self, vec_type):
        cdef PetscVecType vtype = NULL
        vec_type = str2bytes(vec_type, &vtype)
        CHKERR( DMSetVecType(self.dm, vtype) )

    def createGlobalVec(self):
        cdef Vec vg = Vec()
        CHKERR( DMCreateGlobalVector(self.dm, &vg.vec) )
        return vg

    def createLocalVec(self):
        cdef Vec vl = Vec()
        CHKERR( DMCreateLocalVector(self.dm, &vl.vec) )
        return vl

    def getGlobalVec(self):
        cdef Vec vg = Vec()
        CHKERR( DMGetGlobalVector(self.dm, &vg.vec) )
        PetscINCREF(vg.obj)
        return vg

    def restoreGlobalVec(self, Vec vg):
        CHKERR( PetscObjectDereference(<PetscObject>vg.vec) ) 
        CHKERR( DMRestoreGlobalVector(self.dm, &vg.vec) )

    def getLocalVec(self):
        cdef Vec vl = Vec()
        CHKERR( DMGetLocalVector(self.dm, &vl.vec) )
        PetscINCREF(vl.obj)
        return vl

    def restoreLocalVec(self, Vec vl):
        CHKERR( PetscObjectDereference(<PetscObject>vl.vec) ) 
        CHKERR( DMRestoreLocalVector(self.dm, &vl.vec) )

    def globalToLocal(self, Vec vg, Vec vl, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMGlobalToLocalBegin(self.dm, vg.vec, im, vl.vec) )
        CHKERR( DMGlobalToLocalEnd  (self.dm, vg.vec, im, vl.vec) )

    def localToGlobal(self, Vec vl, Vec vg, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMLocalToGlobalBegin(self.dm, vl.vec, im, vg.vec) )
        CHKERR( DMLocalToGlobalEnd(self.dm, vl.vec, im, vg.vec) )

    def localToLocal(self, Vec vl, Vec vlg, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMLocalToLocalBegin(self.dm, vl.vec, im, vlg.vec) )
        CHKERR( DMLocalToLocalEnd  (self.dm, vl.vec, im, vlg.vec) )

    def getLGMap(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DMGetLocalToGlobalMapping(self.dm, &lgm.lgm) )
        PetscINCREF(lgm.obj)
        return lgm

    #

    def getCoordinateDM(self):
        cdef DM cdm = type(self)()
        CHKERR( DMGetCoordinateDM(self.dm, &cdm.dm) )
        PetscINCREF(cdm.obj)
        return cdm

    def getCoordinateSection(self):
        cdef Section sec = Section()
        CHKERR( DMGetCoordinateSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    def setCoordinates(self, Vec c):
        CHKERR( DMSetCoordinates(self.dm, c.vec) )

    def getCoordinates(self):
        cdef Vec c = Vec()
        CHKERR( DMGetCoordinates(self.dm, &c.vec) )
        PetscINCREF(c.obj)
        return c

    def setCoordinatesLocal(self, Vec c):
        CHKERR( DMSetCoordinatesLocal(self.dm, c.vec) )

    def getCoordinatesLocal(self):
        cdef Vec c = Vec()
        CHKERR( DMGetCoordinatesLocal(self.dm, &c.vec) )
        PetscINCREF(c.obj)
        return c

    #

    def setMatType(self, mat_type):
        """Set matrix type to be used by DM.createMat"""
        cdef PetscMatType mtype = NULL
        mat_type = str2bytes(mat_type, &mtype)
        CHKERR( DMSetMatType(self.dm, mtype) )

    def createMat(self):
        cdef Mat mat = Mat()
        CHKERR( DMCreateMatrix(self.dm, &mat.mat) )
        return mat

    def createInterpolation(self, DM dm):
        cdef Mat A = Mat()
        cdef Vec scale = Vec()
        CHKERR( DMCreateInterpolation(self.dm, dm.dm,
                                   &A.mat, &scale.vec))
        return(A, scale)

    def createInjection(self, DM dm):
        cdef Mat inject = Mat()
        CHKERR( DMCreateInjection(self.dm, dm.dm, &inject.mat) )
        return inject

    def createAggregates(self, DM dm):
        cdef Mat mat = Mat()
        CHKERR( DMCreateAggregates(self.dm, dm.dm, &mat.mat) )
        return mat

    def convert(self, dm_type):
        cdef const_char *cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        cdef PetscDM newdm = NULL
        CHKERR( DMConvert(self.dm, cval, &newdm) )
        cdef DM dm = <DM>subtype_DM(newdm)()
        dm.dm = newdm
        return dm

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

    def getRefineLevel(self):
        cdef PetscInt n = 0
        CHKERR( DMGetRefineLevel(self.dm, &n) )
        return toInt(n)

    def setRefineLevel(self, level):
        cdef PetscInt clevel = asInt(level)
        CHKERR( DMSetRefineLevel(self.dm, clevel) )

    def getCoarsenLevel(self):
        cdef PetscInt n = 0
        CHKERR( DMGetCoarsenLevel(self.dm, &n) )
        return toInt(n)

    #

    def adaptLabel(self, label):
        cdef const_char *cval = NULL
        cdef PetscDMLabel clbl = NULL
        label = str2bytes(label, &cval)
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        cdef DM newdm = DMPlex()
        CHKERR( DMAdaptLabel(self.dm, clbl, &newdm.dm) )
        return newdm

    def adaptMetric(self, Vec metric, label=None):
        cdef const_char *cval = NULL
        cdef PetscDMLabel clbl = NULL
        label = str2bytes(label, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        cdef DM newdm = DMPlex()
        CHKERR( DMAdaptMetric(self.dm, metric.vec, clbl, &newdm.dm) )
        return newdm

    #

    def setDefaultSection(self, Section sec):
        CHKERR( DMSetDefaultSection(self.dm, sec.sec) )

    def getDefaultSection(self):
        cdef Section sec = Section()
        CHKERR( DMGetDefaultSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    def setDefaultGlobalSection(self, Section sec):
        CHKERR( DMSetDefaultGlobalSection(self.dm, sec.sec) )

    def getDefaultGlobalSection(self):
        cdef Section sec = Section()
        CHKERR( DMGetDefaultGlobalSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    def createDefaultSF(self, Section localsec, Section globalsec):
        CHKERR( DMCreateDefaultSF(self.dm, localsec.sec, globalsec.sec) )

    def getDefaultSF(self):
        cdef SF sf = SF()
        CHKERR( DMGetDefaultSF(self.dm, &sf.sf) )
        PetscINCREF(sf.obj)
        return sf

    def getPointSF(self):
        cdef SF sf = SF()
        CHKERR( DMGetPointSF(self.dm, &sf.sf) )
        PetscINCREF(sf.obj)
        return sf

    def setPointSF(self, SF sf):
        CHKERR( DMSetPointSF(self.dm, sf.sf) )

    def getNumLabels(self):
        cdef PetscInt nLabels = 0
        CHKERR( DMGetNumLabels(self.dm, &nLabels) )
        return toInt(nLabels)

    def getLabelName(self, index):
        cdef PetscInt cindex = asInt(index)
        cdef const_char *cname = NULL
        CHKERR( DMGetLabelName(self.dm, cindex, &cname) )
        return bytes2str(cname)

    def hasLabel(self, name):
        cdef PetscBool flag = PETSC_FALSE
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMHasLabel(self.dm, cname, &flag) )
        return toBool(flag)

    def createLabel(self, name):
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMCreateLabel(self.dm, cname) )

    def removeLabel(self, name):
        cdef const_char *cname = NULL
        cdef PetscDMLabel clbl = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMRemoveLabel(self.dm, cname, &clbl) )
        # CHKERR( DMLabelDestroy(&clbl) )

    def getLabelValue(self, name, point):
        cdef PetscInt cpoint = asInt(point), value = 0
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabelValue(self.dm, cname, cpoint, &value) )
        return toInt(value)

    def setLabelValue(self, name, point, value):
        cdef PetscInt cpoint = asInt(point), cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMSetLabelValue(self.dm, cname, cpoint, cvalue) )

    def clearLabelValue(self, name, point, value):
        cdef PetscInt cpoint = asInt(point), cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMClearLabelValue(self.dm, cname, cpoint, cvalue) )

    def getLabelSize(self, name):
        cdef PetscInt size = 0
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabelSize(self.dm, cname, &size) )
        return toInt(size)

    def getLabelIdIS(self, name):
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS lis = IS()
        CHKERR( DMGetLabelIdIS(self.dm, cname, &lis.iset) )
        return lis

    def getStratumSize(self, name, value):
        cdef PetscInt size = 0
        cdef PetscInt cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetStratumSize(self.dm, cname, cvalue, &size) )
        return toInt(size)

    def getStratumIS(self, name, value):
        cdef PetscInt cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS sis = IS()
        CHKERR( DMGetStratumIS(self.dm, cname, cvalue, &sis.iset) )
        return sis

    def clearLabelStratum(self, name, value):
        cdef PetscInt cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMClearLabelStratum(self.dm, cname, cvalue) )

    def setLabelOutput(self, name, output):
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscBool coutput = output
        CHKERR( DMSetLabelOutput(self.dm, cname, coutput) )

    def getLabelOutput(self, name):
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscBool coutput = PETSC_FALSE
        CHKERR( DMGetLabelOutput(self.dm, cname, &coutput) )
        return coutput

    # backward compatibility
    createGlobalVector = createGlobalVec
    createLocalVector = createLocalVec
    getMatrix = createMatrix = createMat

    def setKSPComputeOperators(self, operators, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operators, args, kargs)
        self.set_attr('__operators__', context)
        CHKERR( DMKSPSetComputeOperators(self.dm, KSP_ComputeOps, <void*>context) )

    def createFieldDecomposition(self):
        cdef PetscInt clen = 0
        cdef PetscIS *cis = NULL
        cdef PetscDM *cdm = NULL
        cdef char** cnamelist = NULL

        CHKERR( DMCreateFieldDecomposition(self.dm, &clen, &cnamelist, &cis, &cdm) )

        cdef list isets = [ref_IS(cis[i]) for i from 0 <= i < clen]
        cdef list dms   = []
        cdef list names = []
        cdef DM dm = None

        for i from 0 <= i < clen:
            if cdm != NULL:
                dm = subtype_DM(cdm[i])()
                dm.dm = cdm[i]
                PetscINCREF(dm.obj)
                dms.append(dm)
            else:
                dms.append(None)

            name = bytes2str(cnamelist[i])
            names.append(name)
            CHKERR( PetscFree(cnamelist[i]) )

            CHKERR( ISDestroy(&cis[i]) )
            CHKERR( DMDestroy(&cdm[i]) )

        CHKERR( PetscFree(cis) )
        CHKERR( PetscFree(cdm) )
        CHKERR( PetscFree(cnamelist) )

        return (names, isets, dms)

    def setSNESFunction(self, function, args=None, kargs=None):
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr('__function__', context)
            CHKERR( DMSNESSetFunction(self.dm, SNES_Function, <void*>context) )
        else:
            CHKERR( DMSNESSetFunction(self.dm, NULL, NULL) )

    def setSNESJacobian(self, jacobian, args=None, kargs=None):
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__jacobian__', context)
            CHKERR( DMSNESSetJacobian(self.dm, SNES_Jacobian, <void*>context) )
        else:
            CHKERR( DMSNESSetJacobian(self.dm, NULL, NULL) )

# --------------------------------------------------------------------

del DMType
del DMBoundaryType

# --------------------------------------------------------------------

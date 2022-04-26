# --------------------------------------------------------------------

class DMType(object):
    DA        = S_(DMDA_type)
    COMPOSITE = S_(DMCOMPOSITE)
    SLICED    = S_(DMSLICED)
    SHELL     = S_(DMSHELL)
    PLEX      = S_(DMPLEX)
    REDUNDANT = S_(DMREDUNDANT)
    PATCH     = S_(DMPATCH)
    MOAB      = S_(DMMOAB)
    NETWORK   = S_(DMNETWORK)
    FOREST    = S_(DMFOREST)
    P4EST     = S_(DMP4EST)
    P8EST     = S_(DMP8EST)
    SWARM     = S_(DMSWARM)
    PRODUCT   = S_(DMPRODUCT)
    STAG      = S_(DMSTAG)

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

    def load(self, Viewer viewer):
        CHKERR( DMLoad(self.dm, viewer.vwr) )
        return self

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
        cdef PetscDMType cval = NULL
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

    def getCoordinateDim(self):
        cdef PetscInt dim = 0
        CHKERR( DMGetCoordinateDim(self.dm, &dim) )
        return toInt(dim)

    def setCoordinateDim(self, dim):
        cdef PetscInt cdim = asInt(dim)
        CHKERR( DMSetCoordinateDim(self.dm, cdim) )

    def setOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DMSetOptionsPrefix(self.dm, cval) )

    def getOptionsPrefix(self):
        cdef const char *cval = NULL
        CHKERR( DMGetOptionsPrefix(self.dm, &cval) )
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DMAppendOptionsPrefix(self.dm, cval) )

    def setFromOptions(self):
        CHKERR( DMSetFromOptions(self.dm) )

    def viewFromOptions(self, name, Object obj=None):
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( DMViewFromOptions(self.dm, cobj, cname) )

    def setUp(self):
        CHKERR( DMSetUp(self.dm) )
        return self

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')

    #

    def setBasicAdjacency(self, useCone, useClosure):
        cdef PetscBool uC  = useCone
        cdef PetscBool uCl = useClosure
        CHKERR( DMSetBasicAdjacency(self.dm, uC, uCl) )

    def getBasicAdjacency(self):
        cdef PetscBool uC  = PETSC_FALSE
        cdef PetscBool uCl = PETSC_FALSE
        CHKERR( DMGetBasicAdjacency(self.dm, &uC, &uCl) )
        return toBool(uC), toBool(uCl)

    def setFieldAdjacency(self, field, useCone, useClosure):
        cdef PetscInt  f   = asInt(field)
        cdef PetscBool uC  = useCone
        cdef PetscBool uCl = useClosure
        CHKERR( DMSetAdjacency(self.dm, f, uC, uCl) )

    def getFieldAdjacency(self, field):
        cdef PetscInt  f   = asInt(field)
        cdef PetscBool uC  = PETSC_FALSE
        cdef PetscBool uCl = PETSC_FALSE
        CHKERR( DMGetAdjacency(self.dm, f, &uC, &uCl) )
        return toBool(uC), toBool(uCl)

    #

    def createSubDM(self, fields):
        cdef IS iset = IS()
        cdef DM subdm = DM()
        cdef PetscInt *ifields = NULL
        cdef PetscInt numFields = 0
        fields = iarray_i(fields, &numFields, &ifields)
        CHKERR( DMCreateSubDM( self.dm, numFields, ifields, &iset.iset, &subdm.dm) )
        return iset, subdm

    #

    def setNumFields(self, numFields):
        cdef PetscInt cnum = asInt(numFields)
        CHKERR( DMSetNumFields(self.dm, cnum) )

    def getNumFields(self):
        cdef PetscInt cnum = 0
        CHKERR( DMGetNumFields(self.dm, &cnum) )
        return toInt(cnum)

    def setField(self, index, Object field, label=None):
        cdef PetscInt     cidx = asInt(index)
        cdef PetscObject  cobj = field.obj[0]
        cdef PetscDMLabel clbl = NULL
        assert label is None
        CHKERR( DMSetField(self.dm, cidx, clbl, cobj) )

    def getField(self, index):
        cdef PetscInt     cidx = asInt(index)
        cdef PetscObject  cobj = NULL
        cdef PetscDMLabel clbl = NULL
        CHKERR( DMGetField(self.dm, cidx, &clbl, &cobj) )
        assert clbl == NULL
        cdef Object field = subtype_Object(cobj)()
        field.obj[0] = cobj
        PetscINCREF(field.obj)
        return (field, None)

    def addField(self, Object field, label=None):
        cdef PetscObject  cobj = field.obj[0]
        cdef PetscDMLabel clbl = NULL
        assert label is None
        CHKERR( DMAddField(self.dm, clbl, cobj) )

    def clearFields(self):
        CHKERR( DMClearFields(self.dm) )

    def copyFields(self, DM dm):
        CHKERR( DMCopyFields(self.dm, dm.dm) )

    def createDS(self):
        CHKERR( DMCreateDS(self.dm) )

    def clearDS(self):
        CHKERR( DMClearDS(self.dm) )

    def getDS(self):
        cdef DS ds = DS()
        CHKERR( DMGetDS(self.dm, &ds.ds) )
        PetscINCREF(ds.obj)
        return ds

    def copyDS(self, DM dm):
        CHKERR( DMCopyDS(self.dm, dm.dm) )

    def copyDisc(self, DM dm):
        CHKERR( DMCopyDisc(self.dm, dm.dm) )

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

    def projectCoordinates(self, FE disc):
        CHKERR( DMProjectCoordinates(self.dm, disc.fe))
        return self

    def getBoundingBox(self):
        cdef PetscInt i,dim=0
        CHKERR( DMGetCoordinateDim(self.dm, &dim) )
        cdef PetscReal gmin[3], gmax[3]
        CHKERR( DMGetBoundingBox(self.dm, gmin, gmax) )
        return tuple([(toReal(gmin[i]), toReal(gmax[i]))
                      for i from 0 <= i < dim])

    def getLocalBoundingBox(self):
        cdef PetscInt i,dim=0
        CHKERR( DMGetCoordinateDim(self.dm, &dim) )
        cdef PetscReal lmin[3], lmax[3]
        CHKERR( DMGetLocalBoundingBox(self.dm, lmin, lmax) )
        return tuple([(toReal(lmin[i]), toReal(lmax[i]))
                      for i from 0 <= i < dim])

    def localizeCoordinates(self):
        CHKERR( DMLocalizeCoordinates(self.dm) )
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

    def createRestriction(self, DM dm):
        cdef Mat mat = Mat()
        CHKERR( DMCreateRestriction(self.dm, dm.dm, &mat.mat) )
        return mat

    def convert(self, dm_type):
        cdef PetscDMType cval = NULL
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
        cdef const char *cval = NULL
        cdef PetscDMLabel clbl = NULL
        label = str2bytes(label, &cval)
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        cdef DM newdm = DMPlex()
        CHKERR( DMAdaptLabel(self.dm, clbl, &newdm.dm) )
        return newdm

    def adaptMetric(self, Vec metric, bdLabel=None, rgLabel=None):
        cdef const char *cval = NULL
        cdef PetscDMLabel cbdlbl = NULL
        cdef PetscDMLabel crglbl = NULL
        bdLabel = str2bytes(bdLabel, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &cbdlbl) )
        rgLabel = str2bytes(rgLabel, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &crglbl) )
        cdef DM newdm = DMPlex()
        CHKERR( DMAdaptMetric(self.dm, metric.vec, cbdlbl, crglbl, &newdm.dm) )
        return newdm

    def getLabel(self, name):
        cdef const char *cname = NULL
        cdef DMLabel dmlabel = DMLabel()
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabel(self.dm, cname, &dmlabel.dmlabel) )
        PetscINCREF(dmlabel.obj)
        return dmlabel

    #

    def setSection(self, Section sec):
        CHKERR( DMSetSection(self.dm, sec.sec) )

    def getSection(self):
        cdef Section sec = Section()
        CHKERR( DMGetSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    def setGlobalSection(self, Section sec):
        CHKERR( DMSetGlobalSection(self.dm, sec.sec) )

    def getGlobalSection(self):
        cdef Section sec = Section()
        CHKERR( DMGetGlobalSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    setDefaultSection = setSection
    getDefaultSection = getSection
    setDefaultGlobalSection = setGlobalSection
    getDefaultGlobalSection = getGlobalSection

    def createSectionSF(self, Section localsec, Section globalsec):
        CHKERR( DMCreateSectionSF(self.dm, localsec.sec, globalsec.sec) )

    def getSectionSF(self):
        cdef SF sf = SF()
        CHKERR( DMGetSectionSF(self.dm, &sf.sf) )
        PetscINCREF(sf.obj)
        return sf

    def setSectionSF(self, SF sf):
        CHKERR( DMSetSectionSF(self.dm, sf.sf) )

    createDefaultSF = createSectionSF
    getDefaultSF = getSectionSF
    setDefaultSF = setSectionSF

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
        cdef const char *cname = NULL
        CHKERR( DMGetLabelName(self.dm, cindex, &cname) )
        return bytes2str(cname)

    def hasLabel(self, name):
        cdef PetscBool flag = PETSC_FALSE
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMHasLabel(self.dm, cname, &flag) )
        return toBool(flag)

    def createLabel(self, name):
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMCreateLabel(self.dm, cname) )

    def removeLabel(self, name):
        cdef const char *cname = NULL
        cdef PetscDMLabel clbl = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMRemoveLabel(self.dm, cname, &clbl) )
        # TODO: Once DMLabel is wrapped, this should return the label, like the C function.
        CHKERR( DMLabelDestroy(&clbl) )

    def getLabelValue(self, name, point):
        cdef PetscInt cpoint = asInt(point), value = 0
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabelValue(self.dm, cname, cpoint, &value) )
        return toInt(value)

    def setLabelValue(self, name, point, value):
        cdef PetscInt cpoint = asInt(point), cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMSetLabelValue(self.dm, cname, cpoint, cvalue) )

    def clearLabelValue(self, name, point, value):
        cdef PetscInt cpoint = asInt(point), cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMClearLabelValue(self.dm, cname, cpoint, cvalue) )

    def getLabelSize(self, name):
        cdef PetscInt size = 0
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabelSize(self.dm, cname, &size) )
        return toInt(size)

    def getLabelIdIS(self, name):
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS lis = IS()
        CHKERR( DMGetLabelIdIS(self.dm, cname, &lis.iset) )
        return lis

    def getStratumSize(self, name, value):
        cdef PetscInt size = 0
        cdef PetscInt cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetStratumSize(self.dm, cname, cvalue, &size) )
        return toInt(size)

    def getStratumIS(self, name, value):
        cdef PetscInt cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS sis = IS()
        CHKERR( DMGetStratumIS(self.dm, cname, cvalue, &sis.iset) )
        return sis

    def clearLabelStratum(self, name, value):
        cdef PetscInt cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMClearLabelStratum(self.dm, cname, cvalue) )

    def setLabelOutput(self, name, output):
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscBool coutput = output
        CHKERR( DMSetLabelOutput(self.dm, cname, coutput) )

    def getLabelOutput(self, name):
        cdef const char *cname = NULL
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

    def addCoarsenHook(self, coarsenhook, restricthook, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}

        if coarsenhook is not None:
            coarsencontext = (coarsenhook, args, kargs)

            coarsenhooks = self.get_attr('__coarsenhooks__')
            if coarsenhooks is None:
                coarsenhooks = [coarsencontext]
                CHKERR( DMCoarsenHookAdd(self.dm, DM_PyCoarsenHook, NULL, <void*>NULL) )
            else:
                coarsenhooks.append(coarsencontext)
            self.set_attr('__coarsenhooks__', coarsenhooks)

        if restricthook is not None:
            restrictcontext = (restricthook, args, kargs)

            restricthooks = self.get_attr('__restricthooks__')
            if restricthooks is None:
                restricthooks = [restrictcontext]
                CHKERR( DMCoarsenHookAdd(self.dm, NULL, DM_PyRestrictHook, <void*>NULL) )
            else:
                restricthooks.append(restrictcontext)
            self.set_attr('__restricthooks__', restricthooks)

    # --- application context ---

    property appctx:
        def __get__(self):
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

    # --- discretization space ---

    property ds:
        def __get__(self):
            return self.getDS()
        def __set__(self, value):
            self.setDS(value)

# --------------------------------------------------------------------

del DMType
del DMBoundaryType

# --------------------------------------------------------------------

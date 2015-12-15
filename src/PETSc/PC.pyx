# --------------------------------------------------------------------

class PCType(object):
    # native
    NONE         = S_(PCNONE)
    JACOBI       = S_(PCJACOBI)
    SOR          = S_(PCSOR)
    LU           = S_(PCLU)
    SHELL        = S_(PCSHELL)
    BJACOBI      = S_(PCBJACOBI)
    MG           = S_(PCMG)
    EISENSTAT    = S_(PCEISENSTAT)
    ILU          = S_(PCILU)
    ICC          = S_(PCICC)
    ASM          = S_(PCASM)
    GASM         = S_(PCGASM)
    KSP          = S_(PCKSP)
    COMPOSITE    = S_(PCCOMPOSITE)
    REDUNDANT    = S_(PCREDUNDANT)
    SPAI         = S_(PCSPAI)
    NN           = S_(PCNN)
    CHOLESKY     = S_(PCCHOLESKY)
    PBJACOBI     = S_(PCPBJACOBI)
    MAT          = S_(PCMAT)
    HYPRE        = S_(PCHYPRE)
    PARMS        = S_(PCPARMS)
    FIELDSPLIT   = S_(PCFIELDSPLIT)
    TFS          = S_(PCTFS)
    ML           = S_(PCML)
    GALERKIN     = S_(PCGALERKIN)
    EXOTIC       = S_(PCEXOTIC)
    CP           = S_(PCCP)
    BFBT         = S_(PCBFBT)
    LSC          = S_(PCLSC)
    PYTHON       = S_(PCPYTHON)
    PFMG         = S_(PCPFMG)
    SYSPFMG      = S_(PCSYSPFMG)
    REDISTRIBUTE = S_(PCREDISTRIBUTE)
    SVD          = S_(PCSVD)
    GAMG         = S_(PCGAMG)
    SACUSP       = S_(PCSACUSP)
    SACUSPPOLY   = S_(PCSACUSPPOLY)
    BICGSTABCUSP = S_(PCBICGSTABCUSP)
    AINVCUSP     = S_(PCAINVCUSP)
    BDDC         = S_(PCBDDC)
    KACZMARZ     = S_(PCKACZMARZ)

class PCSide(object):
    # native
    LEFT      = PC_LEFT
    RIGHT     = PC_RIGHT
    SYMMETRIC = PC_SYMMETRIC
    # aliases
    L = LEFT
    R = RIGHT
    S = SYMMETRIC

class PCASMType(object):
    NONE        = PC_ASM_NONE
    BASIC       = PC_ASM_BASIC
    RESTRICT    = PC_ASM_RESTRICT
    INTERPOLATE = PC_ASM_INTERPOLATE

class PCGASMType(object):
    NONE        = PC_GASM_NONE
    BASIC       = PC_GASM_BASIC
    RESTRICT    = PC_GASM_RESTRICT
    INTERPOLATE = PC_GASM_INTERPOLATE

class PCGAMGType(object):
    AGG       = S_(PCGAMGAGG)
    GEO       = S_(PCGAMGGEO)
    CLASSICAL = S_(PCGAMGCLASSICAL)

class PCCompositeType(object):
    ADDITIVE                 = PC_COMPOSITE_ADDITIVE
    MULTIPLICATIVE           = PC_COMPOSITE_MULTIPLICATIVE
    SYMMETRIC_MULTIPLICATIVE = PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE
    SPECIAL                  = PC_COMPOSITE_SPECIAL
    SCHUR                    = PC_COMPOSITE_SCHUR

class PCFieldSplitSchurPreType(object):
    SELF                     = PC_FIELDSPLIT_SCHUR_PRE_SELF
    SELFP                    = PC_FIELDSPLIT_SCHUR_PRE_SELFP
    A11                      = PC_FIELDSPLIT_SCHUR_PRE_A11
    USER                     = PC_FIELDSPLIT_SCHUR_PRE_USER
    FULL                     = PC_FIELDSPLIT_SCHUR_PRE_FULL

class PCFieldSplitSchurFactType(object):
    DIAG                     = PC_FIELDSPLIT_SCHUR_FACT_DIAG
    LOWER                    = PC_FIELDSPLIT_SCHUR_FACT_LOWER
    UPPER                    = PC_FIELDSPLIT_SCHUR_FACT_UPPER
    FULL                     = PC_FIELDSPLIT_SCHUR_FACT_FULL

# --------------------------------------------------------------------

cdef class PC(Object):

    Type = PCType
    Side = PCSide

    ASMType       = PCASMType
    GASMType      = PCGASMType
    GAMGType      = PCGAMGType
    CompositeType = PCCompositeType
    SchurFactType = PCFieldSplitSchurFactType
    SchurPreType  = PCFieldSplitSchurPreType

    # --- xxx ---

    def __cinit__(self):
        self.obj = <PetscObject*> &self.pc
        self.pc = NULL

    def __call__(self, x, y=None):
        if y is None: # XXX do this better
            y = self.getOperators()[0].createVecLeft()
        self.apply(x, y)
        return y

    # --- xxx ---

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PCView(self.pc, vwr) )

    def destroy(self):
        CHKERR( PCDestroy(&self.pc) )
        self.pc = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscPC newpc = NULL
        CHKERR( PCCreate(ccomm, &newpc) )
        PetscCLEAR(self.obj); self.pc = newpc
        return self

    def setType(self, pc_type):
        cdef PetscPCType cval = NULL
        pc_type = str2bytes(pc_type, &cval)
        CHKERR( PCSetType(self.pc, cval) )

    def getType(self):
        cdef PetscPCType cval = NULL
        CHKERR( PCGetType(self.pc, &cval) )
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( PCSetOptionsPrefix(self.pc, cval) )

    def getOptionsPrefix(self):
        cdef const_char *cval = NULL
        CHKERR( PCGetOptionsPrefix(self.pc, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PCSetFromOptions(self.pc) )

    def setOperators(self, Mat A=None, Mat P=None):
        cdef PetscMat amat=NULL
        if A is not None: amat = A.mat
        cdef PetscMat pmat=amat
        if P is not None: pmat = P.mat
        CHKERR( PCSetOperators(self.pc, amat, pmat) )

    def getOperators(self):
        cdef Mat A = Mat(), P = Mat()
        CHKERR( PCGetOperators(self.pc, &A.mat, &P.mat) )
        PetscINCREF(A.obj)
        PetscINCREF(P.obj)
        return (A, P)

    def setUseAmat(self, flag):
        cdef PetscBool cflag = PETSC_FALSE
        if flag:
            cflag = PETSC_TRUE
        CHKERR( PCSetUseAmat(self.pc, cflag) )

    def setUp(self):
        CHKERR( PCSetUp(self.pc) )

    def reset(self):
        CHKERR( PCReset(self.pc) )

    def setUpOnBlocks(self):
        CHKERR( PCSetUpOnBlocks(self.pc) )

    def apply(self, Vec x not None, Vec y not None):
        CHKERR( PCApply(self.pc, x.vec, y.vec) )

    def applyTranspose(self, Vec x not None, Vec y not None):
        CHKERR( PCApplyTranspose(self.pc, x.vec, y.vec) )

    def applySymmetricLeft(self, Vec x not None, Vec y not None):
        CHKERR( PCApplySymmetricLeft(self.pc, x.vec, y.vec) )

    def applySymmetricRight(self, Vec x not None, Vec y not None):
        CHKERR( PCApplySymmetricRight(self.pc, x.vec, y.vec) )

    # --- discretization space ---

    def getDM(self):
        cdef PetscDM newdm = NULL
        CHKERR( PCGetDM(self.pc, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setDM(self, DM dm not None):
        CHKERR( PCSetDM(self.pc, dm.dm) )

    def setCoordinates(self, coordinates):
        cdef ndarray xyz = iarray(coordinates, NPY_PETSC_REAL)
        if PyArray_ISFORTRAN(xyz): xyz = PyArray_Copy(xyz)
        if PyArray_NDIM(xyz) != 2: raise ValueError(
            ("coordinates must have two dimensions: "
             "coordinates.ndim=%d") % (PyArray_NDIM(xyz)) )
        cdef PetscInt nvtx = <PetscInt> PyArray_DIM(xyz, 0)
        cdef PetscInt ndim = <PetscInt> PyArray_DIM(xyz, 1)
        cdef PetscReal *coords = <PetscReal*> PyArray_DATA(xyz)
        CHKERR( PCSetCoordinates(self.pc, ndim, nvtx, coords) )

    # --- Python ---

    def createPython(self, context=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscPC newpc = NULL
        CHKERR( PCCreate(ccomm, &newpc) )
        PetscCLEAR(self.obj); self.pc = newpc
        CHKERR( PCSetType(self.pc, PCPYTHON) )
        CHKERR( PCPythonSetContext(self.pc, <void*>context) )
        return self

    def setPythonContext(self, context):
        CHKERR( PCPythonSetContext(self.pc, <void*>context) )

    def getPythonContext(self):
        cdef void *context = NULL
        CHKERR( PCPythonGetContext(self.pc, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type):
        cdef const_char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( PCPythonSetType(self.pc, cval) )

    # --- ASM ---

    def setASMType(self, asmtype):
        cdef PetscPCASMType cval = asmtype
        CHKERR( PCASMSetType(self.pc, cval) )

    def setASMOverlap(self, overlap):
        cdef PetscInt ival = asInt(overlap)
        CHKERR( PCASMSetOverlap(self.pc, ival) )

    def setASMLocalSubdomains(self, nsd):
        cdef PetscInt n = asInt(nsd)
        CHKERR( PCASMSetLocalSubdomains(self.pc, n, NULL, NULL) )

    def setASMTotalSubdomains(self, nsd):
        cdef PetscInt N = asInt(nsd)
        CHKERR( PCASMSetTotalSubdomains(self.pc, N, NULL, NULL) )

    def getASMSubKSP(self):
        cdef PetscInt i = 0, n = 0
        cdef PetscKSP *p = NULL
        CHKERR( PCASMGetSubKSP(self.pc, &n, NULL, &p) )
        return [ref_KSP(p[i]) for i from 0 <= i <n]

    # --- GASM ---

    def setGASMType(self, gasmtype):
        cdef PetscPCGASMType cval = gasmtype
        CHKERR( PCGASMSetType(self.pc, cval) )

    def setGASMOverlap(self, overlap):
        cdef PetscInt ival = asInt(overlap)
        CHKERR( PCGASMSetOverlap(self.pc, ival) )

    # --- GAMG ---

    def setGAMGType(self, gamgtype):
        cdef PetscPCGAMGType cval = NULL
        gamgtype = str2bytes(gamgtype, &cval)
        CHKERR( PCGAMGSetType(self.pc, cval) )

    def setGAMGLevels(self, levels):
        cdef PetscInt ival = asInt(levels)
        CHKERR( PCGAMGSetNlevels(self.pc, ival) )

    def setGAMGSmooths(self, smooths):
        cdef PetscInt ival = asInt(smooths)
        CHKERR( PCGAMGSetNSmooths(self.pc, ival) )

    # --- Hypre ---

    def getHYPREType(self):
        cdef PetscPCHYPREType cval = NULL
        CHKERR( PCHYPREGetType(self.pc, &cval) )
        return bytes2str(cval)

    def setHYPREType(self, hypretype):
        cdef PetscPCHYPREType cval = NULL
        hypretype = str2bytes(hypretype, &cval)
        CHKERR( PCHYPRESetType(self.pc, cval) )

    def setHYPREDiscreteCurl(self, Mat mat not None):
        CHKERR( PCHYPRESetDiscreteCurl(self.pc, mat.mat) )

    def setHYPREDiscreteGradient(self, Mat mat not None):
        CHKERR( PCHYPRESetDiscreteGradient(self.pc, mat.mat) )

    def setHYPRESetAlphaPoissonMatrix(self, Mat mat not None):
        CHKERR( PCHYPRESetAlphaPoissonMatrix(self.pc, mat.mat) )

    def setHYPRESetBetaPoissonMatrix(self, Mat mat=None):
        cdef PetscMat pmat = NULL
        if mat is not None: pmat = mat.mat
        CHKERR( PCHYPRESetBetaPoissonMatrix(self.pc, pmat) )

    def setHYPRESetEdgeConstantVectors(self, Vec ozz not None,
                                       Vec zoz not None, Vec zzo=None):
        cdef PetscVec zzo_vec = NULL
        if zzo is not None: zzo_vec = zzo.vec
        CHKERR( PCHYPRESetEdgeConstantVectors(self.pc, ozz.vec, zoz.vec,
                                              zzo_vec) )

    # --- Factor ---

    def setFactorSolverPackage(self, solver):
        cdef PetscMatSolverPackage cval = NULL
        solver = str2bytes(solver, &cval)
        CHKERR( PCFactorSetMatSolverPackage(self.pc, cval) )

    def getFactorSolverPackage(self):
        cdef PetscMatSolverPackage cval = NULL
        CHKERR( PCFactorGetMatSolverPackage(self.pc, &cval) )
        return bytes2str(cval)

    def setFactorOrdering(self, ord_type=None, nzdiag=None, reuse=None):
        cdef PetscMatOrderingType cval = NULL
        if ord_type is not None:
            ord_type = str2bytes(ord_type, &cval)
            CHKERR( PCFactorSetMatOrderingType(self.pc, cval) )
        cdef PetscReal rval = 0
        if nzdiag is not None:
            rval = asReal(nzdiag)
            CHKERR( PCFactorReorderForNonzeroDiagonal(self.pc, rval) )
        cdef PetscBool bval = PETSC_FALSE
        if reuse is not None:
            bval = PETSC_TRUE if reuse else PETSC_FALSE
            CHKERR( PCFactorSetReuseOrdering(self.pc, bval) )

    def setFactorPivot(self, zeropivot=None, inblocks=None):
        cdef PetscReal rval = 0
        if zeropivot is not None:
            rval = asReal(zeropivot)
            CHKERR( PCFactorSetZeroPivot(self.pc, rval) )
        cdef PetscBool bval = PETSC_FALSE
        if inblocks is not None:
            bval2 = PETSC_TRUE if inblocks else PETSC_FALSE
            CHKERR( PCFactorSetPivotInBlocks(self.pc, bval) )

    def setFactorShift(self, shift_type=None, amount=None):
        cdef PetscMatFactorShiftType cval = MAT_SHIFT_NONE
        if shift_type is not None:
            cval = matfactorshifttype(shift_type)
            CHKERR( PCFactorSetShiftType(self.pc, cval) )
        cdef PetscReal rval = 0
        if amount is not None:
            rval = asReal(amount)
            CHKERR( PCFactorSetShiftAmount(self.pc, rval) )

    def setFactorLevels(self, levels):
        cdef PetscInt ival = asInt(levels)
        CHKERR( PCFactorSetLevels(self.pc, ival) )

    def getFactorMatrix(self):
        cdef Mat mat = Mat()
        CHKERR( PCFactorGetMatrix(self.pc, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

   # --- FieldSplit ---

    def setFieldSplitType(self, ctype):
        cdef PetscPCCompositeType cval = ctype
        CHKERR( PCFieldSplitSetType(self.pc, cval) )

    def setFieldSplitIS(self, *fields):
        cdef object name = None
        cdef IS field = None
        cdef const_char *cname = NULL
        for name, field in fields:
            name = str2bytes(name, &cname)
            CHKERR( PCFieldSplitSetIS(self.pc, cname, field.iset) )

    def setFieldSplitFields(self, bsize, *fields):
        cdef PetscInt bs = asInt(bsize)
        CHKERR( PCFieldSplitSetBlockSize(self.pc, bs) )
        cdef object name = None
        cdef object field = None
        cdef const_char *cname = NULL
        cdef PetscInt nfields = 0, *ifields = NULL
        for name, field in fields:
            name = str2bytes(name, &cname)
            field = iarray_i(field, &nfields, &ifields)
            CHKERR( PCFieldSplitSetFields(self.pc, cname,
                                          nfields, ifields, ifields) )

    def getFieldSplitSubKSP(self):
        cdef PetscInt i = 0, n = 0
        cdef PetscKSP *p = NULL
        cdef object subksp = None
        try:
            CHKERR( PCFieldSplitGetSubKSP(self.pc, &n, &p) )
            subksp = [ref_KSP(p[i]) for i from 0 <= i <n]
        finally:
            CHKERR( PetscFree(p) )
        return subksp

    def setFieldSplitSchurFactType(self, ctype):
        cdef PetscPCFieldSplitSchurFactType cval = ctype
        CHKERR( PCFieldSplitSetSchurFactType(self.pc, cval) )

    def setFieldSplitSchurPreType(self, ptype, Mat pre=None):
        cdef PetscPCFieldSplitSchurPreType pval = ptype
        cdef PetscMat pmat = NULL
        if pre is not None: pmat = pre.mat
        CHKERR( PCFieldSplitSetSchurPre(self.pc, pval, pmat) )

    def setReusePreconditioner(self, flag):
        cdef PetscBool cflag = PETSC_FALSE
        if flag:
            cflag = PETSC_TRUE
        CHKERR( PCSetReusePreconditioner(self.pc, cflag) )

    # --- COMPOSITE ---

    def setCompositeType(self, ctype):
        cdef PetscPCCompositeType cval = ctype
        CHKERR( PCCompositeSetType(self.pc, cval) )

    def getCompositePC(self, n):
        cdef PC pc = PC()
        cdef cn = asInt(n)
        CHKERR( PCCompositeGetPC(self.pc, cn, &pc.pc) )
        PetscINCREF(pc.obj)
        return pc

    def addCompositePC(self, pc_type):
        cdef PetscPCType cval = NULL
        pc_type = str2bytes(pc_type, &cval)
        CHKERR( PCCompositeAddPC(self.pc, cval) )

    # --- KSP ---

    def getKSP(self):
        cdef KSP ksp = KSP()
        CHKERR( PCKSPGetKSP(self.pc, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    # --- MG ---
    def getMGLevels(self):
        cdef PetscInt levels
        CHKERR( PCMGGetLevels(self.pc, &levels) )
        return toInt(levels)

    def getMGCoarseSolve(self):
        cdef KSP ksp = KSP()
        CHKERR( PCMGGetCoarseSolve(self.pc, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def setMGInterpolation(self, level, Mat mat not None):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PCMGSetInterpolation(self.pc, clevel, mat.mat) )

    def getMGInterpolation(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef Mat interpolation = Mat()
        CHKERR( PCMGGetInterpolation(self.pc, clevel, &interpolation.mat) )
        PetscINCREF(interpolation.obj)
        return interpolation

    def setMGRestriction(self, level, Mat mat not None):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PCMGSetRestriction(self.pc, clevel, mat.mat) )

    def getMGRestriction(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef Mat restriction = Mat()
        CHKERR( PCMGGetRestriction(self.pc, clevel, &restriction.mat) )
        PetscINCREF(restriction.obj)
        return restriction

    def setMGRScale(self, level, Vec rscale not None):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PCMGSetRScale(self.pc, clevel, rscale.vec) )

    def getMGRScale(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef Vec rscale = Vec()
        CHKERR( PCMGGetRScale(self.pc, clevel, &rscale.vec) )
        PetscINCREF(rscale.obj)
        return rscale

    def getMGSmoother(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef KSP ksp = KSP()
        CHKERR( PCMGGetSmoother(self.pc, clevel, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def getMGSmootherDown(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef KSP ksp = KSP()
        CHKERR( PCMGGetSmootherDown(self.pc, clevel, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def getMGSmootherUp(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef KSP ksp = KSP()
        CHKERR( PCMGGetSmootherUp(self.pc, clevel, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def setMGCyclesOnLevel(self, level, ncycle):
        cdef PetscInt clevel = asInt(level)
        cdef PetscInt c = asInt(ncycle)
        CHKERR( PCMGSetCyclesOnLevel(self.pc, clevel, c) )

    def setMGRhs(self, level, Vec rhs not None):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PCMGSetRhs(self.pc, clevel, rhs.vec) )

    def setMGX(self, level, Vec x not None):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PCMGSetX(self.pc, clevel, x.vec) )

    def setMGR(self, level, Vec r not None):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PCMGSetR(self.pc, clevel, r.vec) )

# --------------------------------------------------------------------

del PCType
del PCSide
del PCASMType
del PCGASMType
del PCGAMGType
del PCCompositeType
del PCFieldSplitSchurPreType
del PCFieldSplitSchurFactType

# --------------------------------------------------------------------

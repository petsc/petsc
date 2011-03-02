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
    KSP          = S_(PCKSP)
    COMPOSITE    = S_(PCCOMPOSITE)
    REDUNDANT    = S_(PCREDUNDANT)
    SPAI         = S_(PCSPAI)
    NN           = S_(PCNN)
    CHOLESKY     = S_(PCCHOLESKY)
    PBJACOBI     = S_(PCPBJACOBI)
    MAT          = S_(PCMAT)
    HYPRE        = S_(PCHYPRE)
    FIELDSPLIT   = S_(PCFIELDSPLIT)
    TFS          = S_(PCTFS)
    ML           = S_(PCML)
    PROMETHEUS   = S_(PCPROMETHEUS)
    GALERKIN     = S_(PCGALERKIN)
    EXOTIC       = S_(PCEXOTIC)
    OPENMP       = S_(PCOPENMP)
    SUPPORTGRAPH = S_(PCSUPPORTGRAPH)
    ASA          = S_(PCASA)
    CP           = S_(PCCP)
    BFBT         = S_(PCBFBT)
    LSC          = S_(PCLSC)
    PYTHON       = S_(PCPYTHON)
    PFMG         = S_(PCPFMG)
    SYSPFMG      = S_(PCSYSPFMG)
    REDISTRIBUTE = S_(PCREDISTRIBUTE)
    SACUSP       = S_(PCSACUSP)

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

class PCCompositeType(object):
    ADDITIVE                 = PC_COMPOSITE_ADDITIVE
    MULTIPLICATIVE           = PC_COMPOSITE_MULTIPLICATIVE
    SYMMETRIC_MULTIPLICATIVE = PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE
    SPECIAL                  = PC_COMPOSITE_SPECIAL
    SCHUR                    = PC_COMPOSITE_SCHUR

# --------------------------------------------------------------------

cdef class PC(Object):

    Type = PCType
    Side = PCSide

    ASMType       = PCASMType
    CompositeType = PCCompositeType

    # --- xxx ---

    def __cinit__(self):
        self.obj = <PetscObject*> &self.pc
        self.pc = NULL

    def __call__(self, x, y=None):
        if y is None: # XXX do this better
            y = self.getOperators()[0].getVecLeft()
        self.apply(x, y)
        return y

    # --- xxx ---

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PCView(self.pc, vwr) )

    def destroy(self):
        CHKERR( PCDestroy(self.pc) )
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

    def setOperators(self, Mat A=None, Mat P=None, structure=None):
        cdef PetscMat amat=NULL
        if A is not None: amat = A.mat
        cdef PetscMat pmat=amat
        if P is not None: pmat = P.mat
        cdef PetscMatStructure flag = matstructure(structure)
        CHKERR( PCSetOperators(self.pc, amat, pmat, flag) )

    def getOperators(self):
        cdef Mat A = Mat(), P = Mat()
        cdef PetscMatStructure flag = MAT_DIFFERENT_NONZERO_PATTERN
        CHKERR( PCGetOperators(self.pc, &A.mat, &P.mat, &flag) )
        PetscIncref(<PetscObject>A.mat)
        PetscIncref(<PetscObject>P.mat)
        return (A, P, flag)

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
        cdef PetscPCASMType  cval = asmtype
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
                                          nfields, ifields) )

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

# --------------------------------------------------------------------

del PCType
del PCSide
del PCASMType
del PCCompositeType

# --------------------------------------------------------------------

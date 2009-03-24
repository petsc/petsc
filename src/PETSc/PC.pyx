# --------------------------------------------------------------------

class PCType(object):
    # native
    NONE       = PCNONE
    JACOBI     = PCJACOBI
    SOR        = PCSOR
    LU         = PCLU
    SHELL      = PCSHELL
    BJACOBI    = PCBJACOBI
    MG         = PCMG
    EISENSTAT  = PCEISENSTAT
    ILU        = PCILU
    ICC        = PCICC
    ASM        = PCASM
    KSP        = PCKSP
    COMPOSITE  = PCCOMPOSITE
    REDUNDANT  = PCREDUNDANT
    SPAI       = PCSPAI
    NN         = PCNN
    CHOLESKY   = PCCHOLESKY
    SAMG       = PCSAMG
    PBJACOBI   = PCPBJACOBI
    MAT        = PCMAT
    HYPRE      = PCHYPRE
    FIELDSPLIT = PCFIELDSPLIT
    TFS        = PCTFS
    ML         = PCML
    PROMETHEUS = PCPROMETHEUS
    GALERKIN   = PCGALERKIN
    #
    PYTHON = PCPYTHON

class PCSide(object):
    # native
    LEFT      = PC_LEFT
    RIGHT     = PC_RIGHT
    SYMMETRIC = PC_SYMMETRIC
    # aliases
    L = LEFT
    R = RIGHT
    S = SYMMETRIC

# --------------------------------------------------------------------

cdef class PC(Object):

    Type = PCType
    Side = PCSide

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.pc
        self.pc = NULL

    def __call__(self, x, y=None):
        if y is None: # XXX do this better
            y = self.getOperators()[0].getVecLeft()
        self.apply(x, y)
        return y

    #

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
        CHKERR( PCSetType(self.pc, str2cp(pc_type)) )

    def getType(self):
        cdef PetscPCType pc_type = NULL
        CHKERR( PCGetType(self.pc, &pc_type) )
        return cp2str(pc_type)

    def setOptionsPrefix(self, prefix):
        CHKERR( PCSetOptionsPrefix(self.pc, str2cp(prefix)) )

    def getOptionsPrefix(self):
        cdef const_char_p prefix = NULL
        CHKERR( PCGetOptionsPrefix(self.pc, &prefix) )
        return cp2str(prefix)

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


    #

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
        CHKERR( PCPythonSetType(self.pc, str2cp(py_type)) )

    #

# --------------------------------------------------------------------

del PCType
del PCSide

# --------------------------------------------------------------------

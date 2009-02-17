# --------------------------------------------------------------------

cdef extern from *:
    ctypedef char* char_p       "char*"
    ctypedef char* const_char_p "const char*"

cdef inline object cp2str(const_char_p p):
    if p == NULL: return None
    else:         return p

cdef inline char_p str2cp(object s) except ? NULL:
    if s is None: return NULL
    else:         return s

include "allocate.pxi"

# --------------------------------------------------------------------

# Vile hack for raising a exception and not contaminating traceback

cdef extern from *:
    enum: PETSC_ERR_PYTHON "(-1)"

cdef extern from *:
    void pyx_raise"__Pyx_Raise"(object, object, void*)

cdef inline int SETERR(int ierr):
    pyx_raise(Error, ierr, NULL)
    return ierr

cdef inline int CHKERR(int ierr) except -1:
    if ierr == 0: return 0                 # no error
    if ierr == PETSC_ERR_PYTHON: return -1 # error in Python call
    pyx_raise(Error, ierr, NULL)           # error in PETSc call
    return -1

# --------------------------------------------------------------------

# PETSc support
# -------------

cdef extern from "petsc.h":
    ctypedef long   PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar
    ctypedef PetscInt    const_PetscInt    "const PetscInt"
    ctypedef PetscReal   const_PetscReal   "const PetscReal"
    ctypedef PetscScalar const_PetscScalar "const PetscScalar"

cdef extern from "compat.h":
    pass

cdef extern from "custom.h":
    pass

cdef extern from "scalar.h":
    object      toScalar "PyPetscScalar_FromPetscScalar" (PetscScalar)
    PetscScalar asScalar "PyPetscScalar_AsPetscScalar"   (object) except*

# --------------------------------------------------------------------

# NumPy support
# -------------

include "arraynpy.pxi"

import_numpy()

IntType     = numpy_typeobj(NPY_PETSC_INT)
RealType    = numpy_typeobj(NPY_PETSC_REAL)
ScalarType  = numpy_typeobj(NPY_PETSC_SCALAR)
ComplexType = numpy_typeobj(NPY_PETSC_COMPLEX)

# --------------------------------------------------------------------

include "petscdef.pxi"
include "petscmem.pxi"
include "petscopt.pxi"
include "petscsys.pxi"
include "petsclog.pxi"
include "petscmpi.pxi"
include "petscobj.pxi"
include "petscvwr.pxi"
include "petscrand.pxi"
include "petscis.pxi"
include "petscvec.pxi"
include "petscmat.pxi"
include "petscpc.pxi"
include "petscksp.pxi"
include "petscsnes.pxi"
include "petscts.pxi"
include "petscao.pxi"
include "petscda.pxi"

# --------------------------------------------------------------------

__doc__ = u"""
Portable, Extensible Toolkit for Scientific Computation
"""

include "Const.pyx"
include "Error.pyx"
include "Options.pyx"
include "Sys.pyx"
include "Log.pyx"
include "Comm.pyx"
include "Object.pyx"
include "Viewer.pyx"
include "Random.pyx"
include "IS.pyx"
include "Vec.pyx"
include "Mat.pyx"
include "PC.pyx"
include "KSP.pyx"
include "SNES.pyx"
include "TS.pyx"
include "AO.pyx"
include "DA.pyx"

# --------------------------------------------------------------------

include "CAPI.pyx"

# --------------------------------------------------------------------

cdef extern from "Python.h":
    int Py_IsInitialized()

cdef object tracebacklist = []

cdef int traceback(int line,
                   const_char_p cfun,
                   const_char_p cfile,
                   const_char_p cdir,
                   int n, int p,
                   const_char_p mess,
                   void *ctx) with gil:
    cdef PetscLogDouble mem=0
    cdef PetscLogDouble rss=0
    cdef const_char_p text=NULL
    #
    if not Py_IsInitialized():
        return PetscTBEH(line, cfun, cfile, cdir, n, p, mess, ctx)
    if  (<void*>tracebacklist) == NULL:
        return PetscTBEH(line, cfun, cfile, cdir, n, p, mess, ctx)
    #
    cdef object tbl = tracebacklist
    fun = cp2str(cfun)
    fnm = cp2str(cfile)
    dnm = cp2str(cdir)
    m = "%s() line %d in %s%s" % (fun, line, dnm, fnm)
    tbl.insert(0, m)
    if p != 1: return n
    #
    del tbl[1:] # clear any previous stuff
    if n == PETSC_ERR_MEM: # special case
        PetscMallocGetCurrentUsage(&mem)
        PetscMemoryGetCurrentUsage(&rss)
        m = "Out of memory. " \
            "Allocated: %d, " \
            "Used by process: %d" % (mem, rss)
        tbl.append(m)
    else:
        PetscErrorMessage(n, &text, NULL)
    if text != NULL: tbl.append(cp2str(text))
    if mess != NULL: tbl.append(cp2str(mess))
    return n

# --------------------------------------------------------------------

cdef extern from "Python.h":
    int Py_AtExit(void (*)())
    void PySys_WriteStderr(char*,...)

cdef extern from "stdio.h" nogil:
    ctypedef struct FILE
    FILE *stderr
    int fprintf(FILE *, char *, ...)

cdef extern from "initpkg.h":
    int PetscInitializeAllPackages(char[])

cdef extern from "libpetsc4py.h":
    int PetscPythonRegisterAll(char[])

cdef int    PyPetsc_Argc = 0
cdef char** PyPetsc_Argv = NULL

cdef int getinitargs(object args, int *argc, char **argv[]) except -1:
    # allocate command line arguments
    cdef int i, c = 0
    cdef char **v = NULL
    args = [str(a) for a in args]
    args = [a for a in args if a]
    c = <int>    len(args)
    v = <char**> malloc((c+1)*sizeof(char*))
    if v == NULL: raise MemoryError
    else: memset(v, 0, (c+1)*sizeof(char*))
    try:
        for 0 <= i < c:
            v[i] = strdup(str2cp(args[i]))
            if v[i] == NULL: raise MemoryError
    except:
        delinitargs(&c, &v); raise
    argc[0] = c; argv[0] = v
    return 0

cdef void delinitargs(int *argc, char **argv[]) nogil:
    # dallocate command line arguments
    cdef int i, c = argc[0]
    cdef char** v = argv[0]
    argc[0] = 0; argv[0] = NULL;
    if c >= 0 and v != NULL:
        for 0 <= i < c:
            if  v[i] != NULL: free(v[i])
        free(v)

cdef void finalize() nogil:
    cdef int ierr = 0
    # deallocate command line arguments
    global PyPetsc_Argc; global PyPetsc_Argv;
    delinitargs(&PyPetsc_Argc, &PyPetsc_Argv)
    # manage PETSc finalization
    if not (<int>PetscInitializeCalled): return
    if (<int>PetscFinalizeCalled): return
    # deinstall custom error handler
    ierr = PetscPopErrorHandler()
    if ierr != 0:
        fprintf(stderr, "PetscPopErrorHandler() failed "
                "[error code: %d]\n", ierr)
    # finalize PETSc
    ierr = PetscFinalize()
    if ierr != 0:
        fprintf(stderr, "PetscFinalize() failed "
                "[error code: %d]\n", ierr)
    # and we are done, see you later !!

cdef int initialize(object args) except -1:
    if (<int>PetscInitializeCalled):
        return 1
    if (<int>PetscFinalizeCalled):
        return 0
    # allocate command line arguments
    global PyPetsc_Argc; global PyPetsc_Argv;
    getinitargs(args, &PyPetsc_Argc, &PyPetsc_Argv)
    # initialize PETSc
    CHKERR( PetscInitialize(&PyPetsc_Argc, &PyPetsc_Argv, NULL, NULL) )
    # install custom error handler
    CHKERR( PetscPushErrorHandler(traceback, <void*>tracebacklist) )
    # register finalization function
    if Py_AtExit(finalize) < 0:
        PySys_WriteStderr("warning: could not register"
                          "PetscFinalize() with Py_AtExit()")
    return 1 # and we are done, enjoy !!

cdef extern from *:
    PetscCookie PETSC_OBJECT_COOKIE    "PETSC_OBJECT_COOKIE"
    PetscCookie PETSC_VIEWER_COOKIE    "PETSC_VIEWER_COOKIE"
    PetscCookie PETSC_RANDOM_COOKIE    "PETSC_RANDOM_COOKIE"
    PetscCookie PETSC_IS_COOKIE        "IS_COOKIE"
    PetscCookie PETSC_LGMAP_COOKIE     "IS_LTOGM_COOKIE"
    PetscCookie PETSC_VEC_COOKIE       "VEC_COOKIE"
    PetscCookie PETSC_SCATTER_COOKIE   "VEC_SCATTER_COOKIE"
    PetscCookie PETSC_MAT_COOKIE       "MAT_COOKIE"
    PetscCookie PETSC_NULLSPACE_COOKIE "MAT_NULLSPACE_COOKIE"
    PetscCookie PETSC_PC_COOKIE        "PC_COOKIE"
    PetscCookie PETSC_KSP_COOKIE       "KSP_COOKIE"
    PetscCookie PETSC_SNES_COOKIE      "SNES_COOKIE"
    PetscCookie PETSC_TS_COOKIE        "TS_COOKIE"
    PetscCookie PETSC_AO_COOKIE        "AO_COOKIE"
    PetscCookie PETSC_DA_COOKIE        "DM_COOKIE"

cdef int register(char path[]) except -1:
    # make sure all PETSc packages are initialized
    CHKERR( PetscInitializeAllPackages(NULL) )
    # register custom implementations
    CHKERR( PetscPythonRegisterAll(path) )
    # register Python types
    TypeRegistryAdd(PETSC_OBJECT_COOKIE,    Object)
    TypeRegistryAdd(PETSC_VIEWER_COOKIE,    Viewer)
    TypeRegistryAdd(PETSC_RANDOM_COOKIE,    Random)
    TypeRegistryAdd(PETSC_IS_COOKIE,        IS)
    TypeRegistryAdd(PETSC_LGMAP_COOKIE,     LGMap)
    TypeRegistryAdd(PETSC_VEC_COOKIE,       Vec)
    TypeRegistryAdd(PETSC_SCATTER_COOKIE,   Scatter)
    TypeRegistryAdd(PETSC_MAT_COOKIE,       Mat)
    TypeRegistryAdd(PETSC_NULLSPACE_COOKIE, NullSpace)
    TypeRegistryAdd(PETSC_PC_COOKIE,        PC)
    TypeRegistryAdd(PETSC_KSP_COOKIE,       KSP)
    TypeRegistryAdd(PETSC_SNES_COOKIE,      SNES)
    TypeRegistryAdd(PETSC_TS_COOKIE,        TS)
    TypeRegistryAdd(PETSC_AO_COOKIE,        AO)
    TypeRegistryAdd(PETSC_DA_COOKIE,        DA)
    return 0 # and we are done, enjoy !!

# --------------------------------------------------------------------

def _initialize(args=None):
    if args is None: args = ()
    global tracebacklist
    Error._traceback_ = tracebacklist
    #
    global __file__
    cdef char* path = str2cp(__file__)
    cdef int ready = initialize(args)
    if ready: register(path)
    #
    global COMM_NULL, COMM_SELF, COMM_WORLD
    (<Comm?>COMM_NULL).comm  = MPI_COMM_NULL
    (<Comm?>COMM_SELF).comm  = PETSC_COMM_SELF
    (<Comm?>COMM_WORLD).comm = PETSC_COMM_WORLD
    #
    global PETSC_COMM_DEFAULT
    PETSC_COMM_DEFAULT = PETSC_COMM_WORLD

def _finalize():
    finalize()
    #
    global petsc2type
    petsc2type.clear()
    #
    global COMM_NULL, COMM_SELF, COMM_WORLD
    (<Comm?>COMM_NULL).comm  = MPI_COMM_NULL
    (<Comm?>COMM_SELF).comm  = MPI_COMM_NULL
    (<Comm?>COMM_WORLD).comm = MPI_COMM_NULL
    #
    global PETSC_COMM_DEFAULT
    PETSC_COMM_DEFAULT = MPI_COMM_NULL

# --------------------------------------------------------------------

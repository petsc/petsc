# --------------------------------------------------------------------

cdef extern from *:
    ctypedef ssize_t Py_intptr_t
    ctypedef size_t  Py_uintptr_t

# --------------------------------------------------------------------

cdef inline object bytes2str(const char p[]):
     if p == NULL:
         return None
     cdef bytes s = <char*>p
     if isinstance(s, str):
         return s
     else:
         return s.decode()

cdef inline object str2bytes(object s, const char *p[]):
    if s is None:
        p[0] = NULL
        return None
    if not isinstance(s, bytes):
        s = s.encode()
    p[0] = <const char*>(<char*>s)
    return s

cdef inline object S_(const char p[]):
     if p == NULL: return None
     cdef object s = <char*>p
     return s if isinstance(s, str) else s.decode()


# --------------------------------------------------------------------

# Vile hack for raising a exception and not contaminating traceback

cdef extern from *:
    enum: PETSC_ERR_PYTHON

cdef extern from *:
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(int ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return ierr

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == 0:
        return 0 # no error
    if ierr == PETSC_ERR_PYTHON:
        return -1 # error in Python call
    <void>SETERR(ierr)
    return -1

# --------------------------------------------------------------------

# PETSc support
# -------------

cdef extern from "compat.h": pass
cdef extern from "custom.h": pass

cdef extern from *:
    ctypedef long   PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar

cdef extern from "scalar.h":
    object      PyPetscScalar_FromPetscScalar(PetscScalar)
    PetscScalar PyPetscScalar_AsPetscScalar(object) except? <PetscScalar>-1.0

cdef inline object toBool(PetscBool value):
    return True if value else False
cdef inline PetscBool asBool(object value) except? <PetscBool>0:
    return PETSC_TRUE if value else PETSC_FALSE

cdef inline object toInt(PetscInt value):
    return value
cdef inline PetscInt asInt(object value) except? -1:
    return value

cdef inline object toReal(PetscReal value):
    return value
cdef inline PetscReal asReal(object value) except? -1:
    return value

cdef inline object toScalar(PetscScalar value):
    return PyPetscScalar_FromPetscScalar(value)
cdef inline PetscScalar asScalar(object value) except? <PetscScalar>-1.0:
    return PyPetscScalar_AsPetscScalar(value)

# --------------------------------------------------------------------

# NumPy support
# -------------

include "arraynpy.pxi"

import_array()

IntType     = PyArray_TypeObjectFromType(NPY_PETSC_INT)
RealType    = PyArray_TypeObjectFromType(NPY_PETSC_REAL)
ScalarType  = PyArray_TypeObjectFromType(NPY_PETSC_SCALAR)
ComplexType = PyArray_TypeObjectFromType(NPY_PETSC_COMPLEX)

# --------------------------------------------------------------------

include "petscdef.pxi"
include "petscmem.pxi"
include "petscopt.pxi"
include "petscmpi.pxi"
include "petscsys.pxi"
include "petsclog.pxi"
include "petscobj.pxi"
include "petscvwr.pxi"
include "petscrand.pxi"
include "petscis.pxi"
include "petscsf.pxi"
include "petscvec.pxi"
include "petscdt.pxi"
include "petscfe.pxi"
include "petscsct.pxi"
include "petscsec.pxi"
include "petscmat.pxi"
include "petscpc.pxi"
include "petscksp.pxi"
include "petscsnes.pxi"
include "petscts.pxi"
include "petsctao.pxi"
include "petscao.pxi"
include "petscdm.pxi"
include "petscds.pxi"
include "petscdmda.pxi"
include "petscdmplex.pxi"
include "petscdmstag.pxi"
include "petscdmcomposite.pxi"
include "petscdmshell.pxi"
include "petscdmlabel.pxi"
include "petscdmswarm.pxi"
include "petscpartitioner.pxi"

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
include "SF.pyx"
include "Vec.pyx"
include "DT.pyx"
include "FE.pyx"
include "Scatter.pyx"
include "Section.pyx"
include "Mat.pyx"
include "PC.pyx"
include "KSP.pyx"
include "SNES.pyx"
include "TS.pyx"
include "TAO.pyx"
include "AO.pyx"
include "DM.pyx"
include "DS.pyx"
include "DMDA.pyx"
include "DMPlex.pyx"
include "DMStag.pyx"
include "DMComposite.pyx"
include "DMShell.pyx"
include "DMLabel.pyx"
include "DMSwarm.pyx"
include "Partitioner.pyx"

# --------------------------------------------------------------------

include "CAPI.pyx"

# --------------------------------------------------------------------

cdef extern from "Python.h":
    int Py_IsInitialized() nogil

cdef extern from * nogil:
    int PetscTBEH(MPI_Comm,int,char*,char*,
                  int,PetscErrorType,char*,void*)

cdef object tracebacklist = []

cdef int traceback(MPI_Comm       comm,
                   int            line,
                   const char    *cfun,
                   const char    *cfile,
                   int            n,
                   PetscErrorType p,
                   const char    *mess,
                   void          *ctx) with gil:
    cdef PetscLogDouble mem=0
    cdef PetscLogDouble rss=0
    cdef const char    *text=NULL
    global tracebacklist
    cdef object tbl = tracebacklist
    fun = bytes2str(cfun)
    fnm = bytes2str(cfile)
    m = "%s() line %d in %s" % (fun, line, fnm)
    tbl.insert(0, m)
    if p != PETSC_ERROR_INITIAL:
        return n
    #
    del tbl[1:] # clear any previous stuff
    if n == PETSC_ERR_MEM: # special case
        PetscMallocGetCurrentUsage(&mem)
        PetscMemoryGetCurrentUsage(&rss)
        m = ("Out of memory. "
             "Allocated: %d, "
             "Used by process: %d") % (mem, rss)
        tbl.append(m)
    else:
        PetscErrorMessage(n, &text, NULL)
    if text != NULL: tbl.append(bytes2str(text))
    if mess != NULL: tbl.append(bytes2str(mess))
    <void>comm; <void>ctx; # unused
    return n

cdef int PetscPythonErrorHandler(
    MPI_Comm       comm,
    int            line,
    const char    *cfun,
    const char    *cfile,
    int            n,
    PetscErrorType p,
    const char    *mess,
    void          *ctx) nogil:
    global tracebacklist
    if Py_IsInitialized() and (<void*>tracebacklist) != NULL:
        return traceback(comm, line, cfun, cfile, n, p, mess, ctx)
    else:
        return PetscTBEH(comm, line, cfun, cfile, n, p, mess, ctx)

# --------------------------------------------------------------------

cdef extern from "stdlib.h" nogil:
    void* malloc(size_t)
    void* realloc (void*,size_t)
    void free(void*)

cdef extern from "stdarg.h" nogil:
    ctypedef struct va_list:
        pass

cdef extern from "string.h"  nogil:
    void* memset(void*,int,size_t)
    void* memcpy(void*,void*,size_t)
    char* strdup(char*)

cdef extern from "Python.h":
    int Py_AtExit(void (*)())
    void PySys_WriteStderr(char*,...)

cdef extern from "stdio.h" nogil:
    ctypedef struct FILE
    FILE *stderr
    int fprintf(FILE *, char *, ...)

cdef extern from "initpkg.h":
    int PetscInitializePackageAll()

cdef extern from "libpetsc4py.h":
    int import_libpetsc4py() except -1
    int PetscPythonRegisterAll()

cdef int    PyPetsc_Argc = 0
cdef char** PyPetsc_Argv = NULL

cdef int getinitargs(object args, int *argc, char **argv[]) except -1:
    # allocate command line arguments
    cdef int i, c = 0
    cdef char **v = NULL
    if args is None: args = []
    args = [str(a).encode() for a in args]
    args = [a for a in args if a]
    c = <int> len(args)
    v = <char**> malloc(<size_t>(c+1)*sizeof(char*))
    if v == NULL: raise MemoryError
    memset(v, 0, <size_t>(c+1)*sizeof(char*))
    try:
        for 0 <= i < c:
            v[i] = strdup(args[i])
            if v[i] == NULL:
                raise MemoryError
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
    global PetscVFPrintf; global prevfprintf;
    delinitargs(&PyPetsc_Argc, &PyPetsc_Argv)
    # manage PETSc finalization
    if not (<int>PetscInitializeCalled): return
    if (<int>PetscFinalizeCalled): return
    # stop stdout/stderr redirect
    if (prevfprintf != NULL):
        PetscVFPrintf = prevfprintf
        prevfprintf = NULL
    # deinstall Python error handler
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

cdef int PetscVFPrintf_PythonStd(FILE *fd, const char formt[], va_list ap):
    import sys
    cdef char cstring[8192]
    cdef size_t stringlen = sizeof(cstring)
    cdef size_t final_pos
    if (fd == PETSC_STDOUT) and not (sys.stdout == sys.__stdout__):
        CHKERR( PetscVSNPrintf(&cstring[0],stringlen,formt,&final_pos,ap))
        if final_pos > 0 and cstring[final_pos-1] == '\x00':
            final_pos -= 1
        ustring = cstring[:final_pos].decode('UTF-8')
        sys.stdout.write(ustring)
    elif (fd == PETSC_STDERR) and not (sys.stderr == sys.__stderr__):
        CHKERR( PetscVSNPrintf(&cstring[0],stringlen,formt,&final_pos,ap))
        if final_pos > 0 and cstring[final_pos-1] == '\x00':
            final_pos -= 1
        ustring = cstring[:final_pos].decode('UTF-8')
        sys.stderr.write(ustring)
    else:
        PetscVFPrintfDefault(fd, formt, ap)
    return 0

cdef int(*prevfprintf)(FILE*, const char*, va_list)
prevfprintf = NULL

cdef int _push_vfprintf(int (*vfprintf)(FILE *, const char*, va_list)) except -1:
    global PetscVFPrintf, prevfprintf
    assert prevfprintf == NULL
    prevfprintf = PetscVFPrintf
    PetscVFPrintf = vfprintf

cdef int _pop_vfprintf() except -1:
    global PetscVFPrintf, prevfprintf
    assert prevfprintf != NULL
    PetscVFPrintf = prevfprintf
    prevfprintf == NULL

cdef int initialize(object args, object comm) except -1:
    if (<int>PetscInitializeCalled): return 1
    if (<int>PetscFinalizeCalled):   return 0
    # allocate command line arguments
    global PyPetsc_Argc; global PyPetsc_Argv;
    getinitargs(args, &PyPetsc_Argc, &PyPetsc_Argv)
    # communicator
    global PETSC_COMM_WORLD
    PETSC_COMM_WORLD = def_Comm(comm, PETSC_COMM_WORLD)
    # initialize PETSc
    CHKERR( PetscInitialize(&PyPetsc_Argc, &PyPetsc_Argv, NULL, NULL) )
    # install Python error handler
    cdef PetscErrorHandlerFunction handler = NULL
    handler = <PetscErrorHandlerFunction>PetscPythonErrorHandler
    CHKERR( PetscPushErrorHandler(handler, NULL) )
    import sys
    if (sys.stdout != sys.__stdout__) or (sys.stderr != sys.__stderr__):
        _push_vfprintf(&PetscVFPrintf_PythonStd)
    # register finalization function
    if Py_AtExit(finalize) < 0:
        PySys_WriteStderr(b"warning: could not register %s with Py_AtExit()",
                          b"PetscFinalize()")
    return 1 # and we are done, enjoy !!

cdef extern from *:
    PetscClassId PETSC_OBJECT_CLASSID      "PETSC_OBJECT_CLASSID"
    PetscClassId PETSC_VIEWER_CLASSID      "PETSC_VIEWER_CLASSID"
    PetscClassId PETSC_RANDOM_CLASSID      "PETSC_RANDOM_CLASSID"
    PetscClassId PETSC_IS_CLASSID          "IS_CLASSID"
    PetscClassId PETSC_LGMAP_CLASSID       "IS_LTOGM_CLASSID"
    PetscClassId PETSC_SF_CLASSID          "PETSCSF_CLASSID"
    PetscClassId PETSC_VEC_CLASSID         "VEC_CLASSID"
    PetscClassId PETSC_SECTION_CLASSID     "PETSC_SECTION_CLASSID"
    PetscClassId PETSC_MAT_CLASSID         "MAT_CLASSID"
    PetscClassId PETSC_NULLSPACE_CLASSID   "MAT_NULLSPACE_CLASSID"
    PetscClassId PETSC_PC_CLASSID          "PC_CLASSID"
    PetscClassId PETSC_KSP_CLASSID         "KSP_CLASSID"
    PetscClassId PETSC_SNES_CLASSID        "SNES_CLASSID"
    PetscClassId PETSC_TS_CLASSID          "TS_CLASSID"
    PetscClassId PETSC_TAO_CLASSID         "TAO_CLASSID"
    PetscClassId PETSC_AO_CLASSID          "AO_CLASSID"
    PetscClassId PETSC_DM_CLASSID          "DM_CLASSID"
    PetscClassId PETSC_DS_CLASSID          "PETSCDS_CLASSID"
    PetscClassId PETSC_PARTITIONER_CLASSID "PETSCPARTITIONER_CLASSID"
    PetscClassId PETSC_FE_CLASSID          "PETSCFE_CLASSID"
    PetscClassId PETSC_DMLABEL_CLASSID     "DMLABEL_CLASSID"

cdef bint registercalled = 0

cdef const char *citation = b"""\
@Article{Dalcin2011,
  Author = {Lisandro D. Dalcin and Rodrigo R. Paz and Pablo A. Kler and Alejandro Cosimo},
  Title = {Parallel distributed computing using {P}ython},
  Journal = {Advances in Water Resources},
  Note = {New Computational Methods and Software Tools},
  Volume = {34},
  Number = {9},
  Pages = {1124--1139},
  Year = {2011},
  DOI = {http://dx.doi.org/10.1016/j.advwatres.2011.04.013}
}
"""

cdef int register() except -1:
    global registercalled
    if registercalled: return 0
    registercalled = True
    # register citation
    CHKERR( PetscCitationsRegister(citation, NULL) )
    # make sure all PETSc packages are initialized
    CHKERR( PetscInitializePackageAll() )
    # register custom implementations
    import_libpetsc4py()
    CHKERR( PetscPythonRegisterAll() )
    # register Python types
    PyPetscType_Register(PETSC_OBJECT_CLASSID,      Object)
    PyPetscType_Register(PETSC_VIEWER_CLASSID,      Viewer)
    PyPetscType_Register(PETSC_RANDOM_CLASSID,      Random)
    PyPetscType_Register(PETSC_IS_CLASSID,          IS)
    PyPetscType_Register(PETSC_LGMAP_CLASSID,       LGMap)
    PyPetscType_Register(PETSC_SF_CLASSID,          SF)
    PyPetscType_Register(PETSC_VEC_CLASSID,         Vec)
    PyPetscType_Register(PETSC_SECTION_CLASSID,     Section)
    PyPetscType_Register(PETSC_MAT_CLASSID,         Mat)
    PyPetscType_Register(PETSC_NULLSPACE_CLASSID,   NullSpace)
    PyPetscType_Register(PETSC_PC_CLASSID,          PC)
    PyPetscType_Register(PETSC_KSP_CLASSID,         KSP)
    PyPetscType_Register(PETSC_SNES_CLASSID,        SNES)
    PyPetscType_Register(PETSC_TS_CLASSID,          TS)
    PyPetscType_Register(PETSC_TAO_CLASSID,         TAO)
    PyPetscType_Register(PETSC_PARTITIONER_CLASSID, Partitioner)
    PyPetscType_Register(PETSC_AO_CLASSID,          AO)
    PyPetscType_Register(PETSC_DM_CLASSID,          DM)
    PyPetscType_Register(PETSC_DS_CLASSID,          DS)
    PyPetscType_Register(PETSC_FE_CLASSID,          FE)
    PyPetscType_Register(PETSC_DMLABEL_CLASSID,     DMLabel)
    return 0 # and we are done, enjoy !!

# --------------------------------------------------------------------

def _initialize(args=None, comm=None):
    global tracebacklist
    Error._traceback_ = tracebacklist
    global PetscError
    PetscError = Error
    #
    cdef int ready = initialize(args, comm)
    if ready: register()
    #
    global __COMM_SELF__, __COMM_WORLD__
    __COMM_SELF__.comm  = PETSC_COMM_SELF
    __COMM_WORLD__.comm = PETSC_COMM_WORLD
    #
    global PETSC_COMM_DEFAULT
    PETSC_COMM_DEFAULT = PETSC_COMM_WORLD

def _finalize():
    finalize()
    #
    global __COMM_SELF__
    __COMM_SELF__.comm  = MPI_COMM_NULL
    global __COMM_WORLD__
    __COMM_WORLD__.comm = MPI_COMM_NULL
    #
    global PETSC_COMM_DEFAULT
    PETSC_COMM_DEFAULT = MPI_COMM_NULL
    #
    global type_registry
    type_registry.clear()
    global stage_registry
    stage_registry.clear()
    global class_registry
    class_registry.clear()
    global event_registry
    event_registry.clear()
    global citations_registry
    citations_registry.clear()

def _push_python_vfprintf():
    _push_vfprintf(&PetscVFPrintf_PythonStd)

def _pop_python_vfprintf():
    _pop_vfprintf()

def _stdout_is_stderr():
    global PETSC_STDOUT, PETSC_STDERR;
    return PETSC_STDOUT == PETSC_STDERR
# --------------------------------------------------------------------

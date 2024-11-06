# --------------------------------------------------------------------

cdef extern from * nogil:
    """
    #include "lib-petsc/compat.h"
    #include "lib-petsc/custom.h"

    /* Silence Clang warnings in Cython-generated C code */
    #if defined(__clang__)
      #pragma clang diagnostic ignored "-Wextra-semi-stmt"
      #pragma clang diagnostic ignored "-Wparentheses-equality"
      #pragma clang diagnostic ignored "-Wunreachable-code-fallthrough"
      #pragma clang diagnostic ignored "-Woverlength-strings"
      #pragma clang diagnostic ignored "-Wunreachable-code"
      #pragma clang diagnostic ignored "-Wundef"
    #elif defined(__GNUC__) || defined(__GNUG__)
      #pragma GCC diagnostic ignored "-Wstrict-aliasing"
      #pragma GCC diagnostic ignored "-Wtype-limits"
    #endif
    """

# --------------------------------------------------------------------

cdef extern from * nogil:
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

# SETERR Support
# --------------

cdef extern from *:
    """
#if PY_VERSION_HEX < 0X30C0000
static PyObject *PyErr_GetRaisedException()
{
    PyObject *t, *v, *tb;
    PyErr_Fetch(&t, &v, &tb);
    PyErr_NormalizeException(&t, &v, &tb);
    if (tb != NULL) PyException_SetTraceback(v, tb);
    Py_XDECREF(t);
    Py_XDECREF(tb);
    return v;
}
static void PyErr_SetRaisedException(PyObject *v)
{
    PyObject *t = (PyObject *)Py_TYPE(v);
    PyObject *tb = PyException_GetTraceback(v);
    Py_XINCREF(t);
    Py_XINCREF(tb);
    PyErr_Restore(t, v, tb);
}
#endif
    """
    void PyErr_SetObject(object, object)
    PyObject *PyExc_RuntimeError
    PyObject *PyErr_GetRaisedException()
    void PyErr_SetRaisedException(PyObject*)
    void PyException_SetCause(PyObject*, PyObject*)

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(PetscErrorCode ierr) noexcept nogil:
    cdef PyObject *exception = NULL, *cause = NULL
    with gil:
        cause = PyErr_GetRaisedException()
        if (<void*>PetscError) != NULL:
            PyErr_SetObject(PetscError, <long>ierr)
        else:
            PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
        if cause != NULL:
            exception = PyErr_GetRaisedException()
            PyException_SetCause(exception, cause)
            PyErr_SetRaisedException(exception)
    return 0

cdef inline PetscErrorCode CHKERR(PetscErrorCode ierr) except PETSC_ERR_PYTHON nogil:
    if ierr == PETSC_SUCCESS:
        return PETSC_SUCCESS # no error
    <void>SETERR(ierr)
    return PETSC_ERR_PYTHON

# SETERRMPI Support
# -----------------

cdef extern from * nogil:
    enum: MPI_SUCCESS
    enum: MPI_MAX_ERROR_STRING
    int MPI_Error_string(int, char[], int*)
    PetscErrorCode PetscSNPrintf(char[], size_t, const char[], ...)
    PetscErrorCode PetscERROR(MPI_Comm, char[], PetscErrorCode, int, char[], char[])

cdef inline int SETERRMPI(int ierr) noexcept nogil:
    cdef char mpi_err_str[MPI_MAX_ERROR_STRING]
    cdef int  result_len = <int>sizeof(mpi_err_str)
    <void>memset(mpi_err_str, 0, <size_t>result_len)
    <void>MPI_Error_string(ierr, mpi_err_str, &result_len)
    <void>result_len  # unused-but-set-variable
    cdef char error_str[MPI_MAX_ERROR_STRING+64]
    <void>PetscSNPrintf(error_str, sizeof(error_str), b"MPI Error %s %d", mpi_err_str, ierr)
    <void>PetscERROR(PETSC_COMM_SELF, "Unknown Python Function", PETSC_ERR_MPI, PETSC_ERROR_INITIAL, "%s", error_str)
    <void>SETERR(PETSC_ERR_MPI)
    return 0

cdef inline PetscErrorCode CHKERRMPI(int ierr) except PETSC_ERR_PYTHON nogil:
    if ierr == MPI_SUCCESS:
        return PETSC_SUCCESS
    <void>SETERRMPI(ierr)
    return PETSC_ERR_PYTHON

# --------------------------------------------------------------------

# PETSc support
# -------------

cdef extern from * nogil:
    ctypedef long   PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar

cdef extern from "<petsc4py/pyscalar.h>":
    object      PyPetscScalar_FromPetscScalar(PetscScalar)
    PetscScalar PyPetscScalar_AsPetscScalar(object) except? <PetscScalar>-1.0

cdef extern from "<petsc4py/pybuffer.h>":
    int  PyPetscBuffer_FillInfo(Py_buffer*, void*, PetscInt, char, int, int) except -1
    void PyPetscBuffer_Release(Py_buffer*)

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

include "dlpack.pxi"

# --------------------------------------------------------------------

include "typing.pxi"
include "petscdef.pxi"
include "petscmem.pxi"
include "petscopt.pxi"
include "petscmpi.pxi"
include "petscsys.pxi"
include "petsclog.pxi"
include "petscobj.pxi"
include "petscvwr.pxi"
include "petscrand.pxi"
include "petscdevice.pxi"
include "petsclayout.pxi"
include "petscis.pxi"
include "petscsf.pxi"
include "petscvec.pxi"
include "petscdt.pxi"
include "petscfe.pxi"
include "petscsct.pxi"
include "petscsec.pxi"
include "petscmat.pxi"
include "petscmatpartitioning.pxi"
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
include "petscspace.pxi"
include "petscdmutils.pxi"
include "petscpyappctx.pxi"

# --------------------------------------------------------------------

__doc__ = """
Portable, Extensible Toolkit for Scientific Computation.
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
include "Device.pyx"
include "IS.pyx"
include "SF.pyx"
include "Vec.pyx"
include "DT.pyx"
include "FE.pyx"
include "Scatter.pyx"
include "Section.pyx"
include "Mat.pyx"
include "MatPartitioning.pyx"
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
include "Space.pyx"
include "DMUtils.pyx"

# --------------------------------------------------------------------

include "CAPI.pyx"
include "libpetsc4py.pyx"

# --------------------------------------------------------------------

cdef extern from "Python.h":
    int Py_IsInitialized() nogil
    int PyList_Insert(object, Py_ssize_t, object) except -1
    int PyList_Append(object, object) except -1

cdef extern from * nogil:
    PetscErrorCode PetscTBEH(MPI_Comm, int, char*, char*, int, PetscErrorType, char*, void*)

cdef object tracebacklist = []

cdef PetscErrorCode traceback(
    MPI_Comm       comm,
    int            line,
    const char    *cfunc,
    const char    *cfile,
    PetscErrorCode n,
    PetscErrorType p,
    const char    *mess,
    void          *ctx,
) except (<PetscErrorCode>-1) with gil:
    cdef PetscLogDouble mem=0
    cdef PetscLogDouble rss=0
    cdef const char    *text=NULL
    global tracebacklist
    cdef object tbl = tracebacklist
    cdef object fun = bytes2str(cfunc)
    cdef object fnm = bytes2str(cfile)
    cdef object m = "%s() at %s:%d" % (fun, fnm, line)
    PyList_Insert(tbl, 0, m)
    if p != PETSC_ERROR_INITIAL:
        return n
    #
    del tbl[1:] # clear any previous stuff
    if n == PETSC_ERR_MEM: # special case
        PetscMallocGetCurrentUsage(&mem)
        PetscMemoryGetCurrentUsage(&rss)
        m = (
            "Out of memory. "
            "Allocated: %d, "
            "Used by process: %d"
        ) % (mem, rss)
        PyList_Append(tbl, m)
    else:
        PetscErrorMessage(n, &text, NULL)
    if text != NULL: PyList_Append(tbl, bytes2str(text))
    if mess != NULL: PyList_Append(tbl, bytes2str(mess))
    <void>comm # unused
    <void>ctx  # unused
    return n

cdef PetscErrorCode PetscPythonErrorHandler(
    MPI_Comm       comm,
    int            line,
    const char    *cfunc,
    const char    *cfile,
    PetscErrorCode n,
    PetscErrorType p,
    const char    *mess,
    void          *ctx,
) except (<PetscErrorCode>-1) nogil:
    global tracebacklist
    if (<void*>tracebacklist) != NULL and Py_IsInitialized():
        return traceback(comm, line, cfunc, cfile, n, p, mess, ctx)
    else:
        return PetscTBEH(comm, line, cfunc, cfile, n, p, mess, ctx)

# --------------------------------------------------------------------

cdef extern from "<stdlib.h>" nogil:
    void* malloc(size_t)
    void* realloc (void*, size_t)
    void free(void*)

cdef extern from "<stdarg.h>" nogil:
    ctypedef struct va_list:
        pass

cdef extern from "<string.h>" nogil:
    void* memset(void*, int, size_t)
    void* memcpy(void*, void*, size_t)
    char* strdup(char*)

cdef extern from "<stdio.h>" nogil:
    ctypedef struct FILE
    FILE *stderr
    int fprintf(FILE *, char *, ...)

cdef extern from "Python.h":
    int Py_AtExit(void (*)() noexcept nogil)
    void PySys_WriteStderr(char*, ...)

cdef extern from * nogil:
    """
    #include "lib-petsc/initpkg.h"
    """
    PetscErrorCode PetscInitializePackageAll()

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
    except Exception:
        delinitargs(&c, &v); raise
    argc[0] = c; argv[0] = v
    return 0

cdef void delinitargs(int *argc, char **argv[]) noexcept nogil:
    # dallocate command line arguments
    cdef int i, c = argc[0]
    cdef char** v = argv[0]
    argc[0] = 0; argv[0] = NULL
    if c >= 0 and v != NULL:
        for 0 <= i < c:
            if  v[i] != NULL: free(v[i])
        free(v)

cdef void finalize() noexcept nogil:
    cdef int ierr = 0
    # deallocate command line arguments
    global PyPetsc_Argc, PyPetsc_Argv
    global PetscVFPrintf, prevfprintf
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
        fprintf(stderr,
                "PetscPopErrorHandler() failed "
                "[error code: %d]\n", ierr)
    # finalize PETSc
    ierr = PetscFinalize()
    if ierr != 0:
        fprintf(stderr,
                "PetscFinalize() failed "
                "[error code: %d]\n", ierr)
    # and we are done, see you later !!

# --------------------------------------------------------------------

cdef extern from *:
    PetscErrorCode (*PetscVFPrintf)(FILE*, const char*, va_list) except PETSC_ERR_PYTHON nogil

cdef PetscErrorCode (*prevfprintf)(FILE*, const char*, va_list) except PETSC_ERR_PYTHON nogil
prevfprintf = NULL

cdef PetscErrorCode PetscVFPrintf_PythonStdStream(
    FILE *fd, const char fmt[], va_list ap,
) except PETSC_ERR_PYTHON with gil:
    import sys
    cdef char cstring[8192]
    cdef size_t stringlen = sizeof(cstring)
    cdef size_t final_pos = 0
    if (fd == PETSC_STDOUT) and not (sys.stdout == sys.__stdout__):
        CHKERR(PetscVSNPrintf(&cstring[0], stringlen, fmt, &final_pos, ap))
        if final_pos > 0 and cstring[final_pos-1] == '\x00':
            final_pos -= 1
        ustring = cstring[:final_pos].decode('UTF-8')
        sys.stdout.write(ustring)
    elif (fd == PETSC_STDERR) and not (sys.stderr == sys.__stderr__):
        CHKERR(PetscVSNPrintf(&cstring[0], stringlen, fmt, &final_pos, ap))
        if final_pos > 0 and cstring[final_pos-1] == '\x00':
            final_pos -= 1
        ustring = cstring[:final_pos].decode('UTF-8')
        sys.stderr.write(ustring)
    else:
        CHKERR(PetscVFPrintfDefault(fd, fmt, ap))
    return PETSC_SUCCESS

cdef int _push_vfprintf(
    PetscErrorCode (*vfprintf)(FILE*, const char*, va_list) except PETSC_ERR_PYTHON nogil,
) except -1:
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
    global PyPetsc_Argc, PyPetsc_Argv
    getinitargs(args, &PyPetsc_Argc, &PyPetsc_Argv)
    # communicator
    global PETSC_COMM_WORLD
    PETSC_COMM_WORLD = def_Comm(comm, PETSC_COMM_WORLD)
    # initialize PETSc
    CHKERR(PetscInitialize(&PyPetsc_Argc, &PyPetsc_Argv, NULL, NULL))
    # install Python error handler
    cdef PetscErrorHandlerFunction handler = NULL
    handler = <PetscErrorHandlerFunction>PetscPythonErrorHandler
    CHKERR(PetscPushErrorHandler(handler, NULL))
    # redirect PETSc std streams
    import sys
    if (sys.stdout != sys.__stdout__) or (sys.stderr != sys.__stderr__):
        _push_vfprintf(&PetscVFPrintf_PythonStdStream)
    # register finalization function
    if Py_AtExit(finalize) < 0:
        PySys_WriteStderr(b"warning: could not register %s with Py_AtExit()",
                          b"PetscFinalize()")
    return 1 # and we are done, enjoy !!

cdef extern from * nogil:
    PetscClassId PETSC_OBJECT_CLASSID           "PETSC_OBJECT_CLASSID"
    PetscClassId PETSC_VIEWER_CLASSID           "PETSC_VIEWER_CLASSID"
    PetscClassId PETSC_RANDOM_CLASSID           "PETSC_RANDOM_CLASSID"
    PetscClassId PETSC_IS_CLASSID               "IS_CLASSID"
    PetscClassId PETSC_LGMAP_CLASSID            "IS_LTOGM_CLASSID"
    PetscClassId PETSC_SF_CLASSID               "PETSCSF_CLASSID"
    PetscClassId PETSC_VEC_CLASSID              "VEC_CLASSID"
    PetscClassId PETSC_SECTION_CLASSID          "PETSC_SECTION_CLASSID"
    PetscClassId PETSC_MAT_CLASSID              "MAT_CLASSID"
    PetscClassId PETSC_MAT_PARTITIONING_CLASSID "MAT_PARTITIONING_CLASSID"
    PetscClassId PETSC_NULLSPACE_CLASSID        "MAT_NULLSPACE_CLASSID"
    PetscClassId PETSC_PC_CLASSID               "PC_CLASSID"
    PetscClassId PETSC_KSP_CLASSID              "KSP_CLASSID"
    PetscClassId PETSC_SNES_CLASSID             "SNES_CLASSID"
    PetscClassId PETSC_TS_CLASSID               "TS_CLASSID"
    PetscClassId PETSC_TAO_CLASSID              "TAO_CLASSID"
    PetscClassId PETSC_AO_CLASSID               "AO_CLASSID"
    PetscClassId PETSC_DM_CLASSID               "DM_CLASSID"
    PetscClassId PETSC_DS_CLASSID               "PETSCDS_CLASSID"
    PetscClassId PETSC_PARTITIONER_CLASSID      "PETSCPARTITIONER_CLASSID"
    PetscClassId PETSC_FE_CLASSID               "PETSCFE_CLASSID"
    PetscClassId PETSC_DMLABEL_CLASSID          "DMLABEL_CLASSID"
    PetscClassId PETSC_SPACE_CLASSID            "PETSCSPACE_CLASSID"
    PetscClassId PETSC_DUALSPACE_CLASSID        "PETSCDUALSPACE_CLASSID"
    PetscClassId PETSC_DEVICE_CLASSID           "PETSC_DEVICE_CLASSID"
    PetscClassId PETSC_DEVICE_CONTEXT_CLASSID   "PETSC_DEVICE_CONTEXT_CLASSID"

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
  DOI = {https://doi.org/10.1016/j.advwatres.2011.04.013}
}
"""

cdef int register() except -1:
    global registercalled
    if registercalled: return 0
    registercalled = True
    # register citation
    CHKERR(PetscCitationsRegister(citation, NULL))
    # make sure all PETSc packages are initialized
    CHKERR(PetscInitializePackageAll())
    # register custom implementations
    CHKERR(PetscPythonRegisterAll())
    # register Python types
    PyPetscType_Register(PETSC_OBJECT_CLASSID,           Object)
    PyPetscType_Register(PETSC_VIEWER_CLASSID,           Viewer)
    PyPetscType_Register(PETSC_RANDOM_CLASSID,           Random)
    PyPetscType_Register(PETSC_DEVICE_CLASSID,           Device)
    PyPetscType_Register(PETSC_DEVICE_CONTEXT_CLASSID,   DeviceContext)
    PyPetscType_Register(PETSC_IS_CLASSID,               IS)
    PyPetscType_Register(PETSC_LGMAP_CLASSID,            LGMap)
    PyPetscType_Register(PETSC_SF_CLASSID,               SF)
    PyPetscType_Register(PETSC_VEC_CLASSID,              Vec)
    PyPetscType_Register(PETSC_SECTION_CLASSID,          Section)
    PyPetscType_Register(PETSC_MAT_CLASSID,              Mat)
    PyPetscType_Register(PETSC_MAT_PARTITIONING_CLASSID, MatPartitioning)
    PyPetscType_Register(PETSC_NULLSPACE_CLASSID,        NullSpace)
    PyPetscType_Register(PETSC_PC_CLASSID,               PC)
    PyPetscType_Register(PETSC_KSP_CLASSID,              KSP)
    PyPetscType_Register(PETSC_SNES_CLASSID,             SNES)
    PyPetscType_Register(PETSC_TS_CLASSID,               TS)
    PyPetscType_Register(PETSC_TAO_CLASSID,              TAO)
    PyPetscType_Register(PETSC_PARTITIONER_CLASSID,      Partitioner)
    PyPetscType_Register(PETSC_AO_CLASSID,               AO)
    PyPetscType_Register(PETSC_DM_CLASSID,               DM)
    PyPetscType_Register(PETSC_DS_CLASSID,               DS)
    PyPetscType_Register(PETSC_FE_CLASSID,               FE)
    PyPetscType_Register(PETSC_DMLABEL_CLASSID,          DMLabel)
    PyPetscType_Register(PETSC_SPACE_CLASSID,            Space)
    PyPetscType_Register(PETSC_DUALSPACE_CLASSID,        DualSpace)
    return 0 # and we are done, enjoy !!

# --------------------------------------------------------------------


def _initialize(args=None, comm=None):
    import atexit
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
    # Register finalizer
    atexit.register(_pre_finalize)


def _pre_finalize():
    # Called while the Python interpreter is still running
    garbage_cleanup()


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
    _push_vfprintf(&PetscVFPrintf_PythonStdStream)


def _pop_python_vfprintf():
    _pop_vfprintf()


def _stdout_is_stderr():
    global PETSC_STDOUT, PETSC_STDERR
    return PETSC_STDOUT == PETSC_STDERR

# --------------------------------------------------------------------

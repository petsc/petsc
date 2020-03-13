#cython: cdivision=True
#cython: binding=False
#cython: auto_pickle=False
#cython: autotestdict=False
#cython: warn.multiple_declarators=False
cimport cython

# ----------

from petsc4py.PETSc cimport *
cdef extern from "custom.h": pass

# ----------

cdef extern from "Python.h":
    bint Py_IsInitialized() nogil
    ctypedef struct PyObject
    void Py_INCREF(PyObject*)
    void Py_DECREF(PyObject*)
    void Py_CLEAR(PyObject*)
    object PyModule_New(char*)
    bint PyModule_Check(object)
    object PyImport_Import(object)

# --------------------------------------------------------------------

cdef extern from "custom.h":
    void initlibpetsc4py() nogil except *

cdef public int import_libpetsc4py() nogil except -1:
    initlibpetsc4py()
    return 0

# --------------------------------------------------------------------

cdef extern from * nogil:
    MPI_Comm MPI_COMM_NULL
    MPI_Comm PETSC_COMM_SELF
    MPI_Comm PETSC_COMM_WORLD
    int MPI_Comm_size(MPI_Comm,int*)
    int MPI_Comm_rank(MPI_Comm,int*)

    ctypedef int PetscErrorCode
    enum: PETSC_ERR_SUP
    enum: PETSC_ERR_USER
    enum: PETSC_ERROR_INITIAL
    enum: PETSC_ERROR_REPEAT
    PetscErrorCode PetscERROR(MPI_Comm,char[],PetscErrorCode,int,char[],char[])

    ctypedef enum PetscBool:
        PETSC_TRUE
        PETSC_FALSE
    ctypedef long   PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar
    ctypedef PetscInt    const_PetscInt    "const PetscInt"
    ctypedef PetscReal   const_PetscReal   "const PetscReal"
    ctypedef PetscScalar const_PetscScalar "const PetscScalar"
    PetscReal PetscMin(PetscReal,PetscReal)

    ctypedef struct _p_PetscObject:
        MPI_Comm comm
        char     *prefix
        PetscInt refct
    ctypedef int PetscClassId
    PetscErrorCode PetscObjectGetClassId(PetscObject,PetscClassId*)
    PetscErrorCode PetscObjectGetComm(PetscObject,MPI_Comm*)
    PetscErrorCode PetscObjectCompose(PetscObject,char[],PetscObject)
    PetscErrorCode PetscObjectQuery(PetscObject,char[],PetscObject*)
    PetscErrorCode PetscObjectReference(PetscObject)
    ctypedef void (*PetscVoidFunction)()
    PetscErrorCode PetscObjectComposeFunction(PetscObject,char[],void (*ptr)())
    PetscErrorCode PetscObjectTypeCompare(PetscObject,char[],PetscBool*)
    PetscErrorCode PetscObjectChangeTypeName(PetscObject, char[])

    ctypedef struct _p_PetscOptionItems
    ctypedef _p_PetscOptionItems* PetscOptionItems
    ctypedef struct _n_PetscOptions
    ctypedef _n_PetscOptions* PetscOptions
    PetscErrorCode PetscOptionsString(char[],char[],char[],char[],char[],size_t,PetscBool*)
    PetscErrorCode PetscOptionsGetString(PetscOptions,char[],char[],char[],size_t,PetscBool*)

    char *PETSCVIEWERASCII
    char *PETSCVIEWERSTRING
    PetscErrorCode PetscViewerASCIIPrintf(PetscViewer,char[],...)
    PetscErrorCode PetscViewerStringSPrintf(PetscViewer,char[],...)

cdef inline object      toInt(PetscInt value):          return value
cdef inline PetscInt    asInt(object value)  except?-1: return value
cdef inline object      toReal(PetscReal value):        return value
cdef inline PetscReal   asReal(object value) except?-1: return value
cdef extern from "scalar.h":
    object      toScalar"PyPetscScalar_FromPetscScalar"(PetscScalar)
    PetscScalar asScalar"PyPetscScalar_AsPetscScalar"(object) except*

# --------------------------------------------------------------------

# NumPy support
# -------------

cdef extern from "string.h"  nogil:
    void* memset(void*,int,size_t)
    void* memcpy(void*,void*,size_t)
    char* strdup(char*)

include "arraynpy.pxi"

import_array()

# --------------------------------------------------------------------

cdef extern from * nogil:
    enum: PETSC_ERR_PYTHON

cdef extern from * nogil:
    enum: IERR "PETSC_ERR_PYTHON"

cdef char *FUNCT = NULL
cdef char *fstack[1024]
cdef int   istack = 0

cdef inline void FunctionBegin(char name[]) nogil:
    global FUNCT
    FUNCT = name
    global fstack, istack
    fstack[istack] = FUNCT
    istack += 1
    if istack >= 1024:
        istack = 0
    return

cdef inline PetscErrorCode FunctionEnd() nogil:
    global FUNCT
    FUNCT = NULL
    global fstack, istack
    istack -= 1
    if istack < 0:
        istack = 1024
    FUNCT = fstack[istack]
    return 0

cdef PetscErrorCode PetscSETERR(PetscErrorCode ierr,char msg[]) nogil:
    global fstack, istack
    istack = 0; fstack[istack] = NULL;
    global FUNCT
    return PetscERROR(PETSC_COMM_SELF,FUNCT,ierr,
                      PETSC_ERROR_INITIAL, msg, NULL)

cdef PetscErrorCode PetscCHKERR(PetscErrorCode ierr) nogil:
    global fstack, istack
    istack = 0; fstack[istack] = NULL;
    global FUNCT
    return PetscERROR(PETSC_COMM_SELF,FUNCT,ierr,
                      PETSC_ERROR_REPEAT, b"", NULL)


cdef extern from *:
    void PyErr_SetObject(object, object)
    PyObject *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError
from petsc4py.PETSc import Error as PetscError

cdef inline void PythonSETERR(PetscErrorCode ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)

cdef inline int CHKERR(PetscErrorCode ierr) nogil except -1:
    if ierr == 0:
        return 0
    if ierr == PETSC_ERR_PYTHON:
        #PetscCHKERR(ierr)
        return -1
    if Py_IsInitialized():
        PythonSETERR(ierr)
    PetscCHKERR(ierr)
    return -1

cdef PetscErrorCode UNSUPPORTED(char msg[]) nogil:
    global FUNCT
    return PetscERROR(PETSC_COMM_SELF,FUNCT,PETSC_ERR_USER,
                      PETSC_ERROR_INITIAL,b"method %s()",msg)

# --------------------------------------------------------------------

cdef inline PetscInt getRef(void *pobj) nogil:
    cdef PetscObject obj = <PetscObject>pobj
    if obj == NULL: return 0
    else: return obj.refct

cdef inline void addRef(void *pobj) nogil:
    cdef PetscObject obj = <PetscObject>pobj
    if obj != NULL: obj.refct += 1

cdef inline void delRef(void *pobj) nogil:
    cdef PetscObject obj = <PetscObject>pobj
    if obj != NULL: obj.refct -= 1

cdef inline PetscObject newRef(void *pobj) nogil:
    cdef PetscObject obj = <PetscObject>pobj
    cdef int ierr = 0
    if obj != NULL:
        ierr = PetscObjectReference(obj)
        if ierr: return NULL # XXX warning!
    return obj

cdef inline char* getPrefix(void *pobj) nogil:
    cdef PetscObject obj = <PetscObject>pobj
    if obj == NULL: return NULL
    return obj.prefix

cdef inline int getCommSize(void *pobj) nogil:
    cdef PetscObject obj = <PetscObject>pobj
    if obj == NULL: return 0
    cdef int size = 0
    MPI_Comm_size(obj.comm,&size)
    return size

cdef inline Viewer Viewer_(PetscViewer p):
    cdef Viewer ob = Viewer.__new__(Viewer)
    ob.obj[0] = newRef(p)
    return ob

cdef inline IS IS_(PetscIS p):
    cdef IS ob = IS.__new__(IS)
    ob.obj[0] = newRef(p)
    return ob

cdef inline Vec Vec_(PetscVec p):
    cdef Vec ob = Vec.__new__(Vec)
    ob.obj[0] = newRef(p)
    return ob

cdef inline Mat Mat_(PetscMat p):
    cdef Mat ob = Mat.__new__(Mat)
    ob.obj[0] = newRef(p)
    return ob

cdef inline PC PC_(PetscPC p):
    cdef PC ob = PC.__new__(PC)
    ob.obj[0] = newRef(p)
    return ob

cdef inline KSP KSP_(PetscKSP p):
    cdef KSP ob = KSP.__new__(KSP)
    ob.obj[0] = newRef(p)
    return ob

cdef inline SNES SNES_(PetscSNES p):
    cdef SNES ob = SNES.__new__(SNES)
    ob.obj[0] = newRef(p)
    return ob

cdef inline TS TS_(PetscTS p):
    cdef TS ob = TS.__new__(TS)
    ob.obj[0] = newRef(p)
    return ob

# --------------------------------------------------------------------

cdef extern from *:
    ctypedef char const_char "const char"

cdef inline object bytes2str(const_char p[]):
    if p == NULL: return None
    cdef bytes s = <char*>p
    if not isinstance(s, str):
        return s.decode()
    else:
        return s

cdef object parse_url(object url):
    path, name = url.rsplit(":", 1)
    return (path, name)

cdef dict module_cache = {}

cdef object load_module(object path):
    if path in module_cache:
        return module_cache[path]
    module = PyModule_New("__petsc__")
    module.__file__ = path
    module.__package__ = None
    module_cache[path] = module
    try:
        source = open(path, 'rU')
        try:
            code = compile(source.read(), path, 'exec')
        finally:
            source.close()
        exec(code, module.__dict__)
    except:
        del module_cache[path]
        raise
    return module

# -----------------------------------------------------------------------------

@cython.internal
cdef class _PyObj:

    cdef object self
    cdef bytes  name

    def __getattr__(self, attr):
        return getattr(self.self, attr, None)

    cdef int setcontext(self, void *ctx, Object base) except -1:
        #
        if ctx == <void*>self.self:
            return 0
        #
        cdef object destroy = self.destroy
        if destroy is not None:
            destroy(base)
            destroy = None
        #
        if ctx == NULL:
            self.self = None
            self.name = None
            return 0
        #
        self.self = <object>ctx
        self.name = None
        cdef object create = self.create
        if create is not None:
            create(base)
            create = None
        return 0

    cdef int getcontext(self, void **ctx) except -1:
        if ctx == NULL: return 0
        if self.self is not None:
            ctx[0] = <void*> self.self
        else:
            ctx[0] = NULL
        return 0

    cdef int setname(self, char name[]) except -1:
        if name != NULL and name[0] != 0:
            self.name = name
        else:
            self.name = None
        return 0

    cdef char* getname(self) except? NULL:
        if self.self is None:
            return NULL
        if self.name is not None:
            return self.name
        cdef ctx = self.self
        cdef name = None
        if PyModule_Check(ctx):
            name = getattr(ctx, '__name__', None)
        else:
            modname = getattr(ctx, '__module__', None)
            clsname = None
            cls = getattr(ctx, '__class__', None)
            if cls:
                clsname = getattr(cls, '__name__', None)
                if not modname:
                    modname = getattr(cls, '__module__', None)
            if modname and clsname:
                name = modname + '.' + clsname
            elif clsname:
                name = clsname
            elif modname:
                name = modname
        if name is not None:
            self.name = name.encode()
        if self.name is not None:
            return self.name
        return NULL

cdef createcontext(char name_p[]):
    if name_p == NULL: return None
    cdef name = bytes2str(name_p)
    cdef mod, path, modname=None
    cdef cls, attr, clsname=None
    # path/to/filename.py:{function|class}
    if ':' in name:
        path, attr = parse_url(name)
        mod = load_module(path)
        if attr:
            cls = getattr(mod, attr)
            return cls()
        else:
            return mod
    # package.module[.{function|class}]
    if '.' in name:
        modname, clsname = name.rsplit('.', 1)
        mod = PyImport_Import(modname)
        if hasattr(mod, clsname):
            cls = getattr(mod, clsname)
            if not PyModule_Check(cls):
                return cls()
    # package[.module]
    mod = PyImport_Import(name)
    return mod

cdef int viewcontext(_PyObj ctx, PetscViewer viewer) except -1:
    cdef PetscBool isascii = PETSC_FALSE, isstring = PETSC_FALSE
    CHKERR( PetscObjectTypeCompare(<PetscObject>viewer, PETSCVIEWERASCII,  &isascii)  )
    CHKERR( PetscObjectTypeCompare(<PetscObject>viewer, PETSCVIEWERSTRING, &isstring) )
    cdef char *name = ctx.getname()
    if isascii:
        if name == NULL: name = b"unknown/no yet set"
        CHKERR( PetscViewerASCIIPrintf(viewer, b"  Python: %s\n", name) )
    if isstring:
        if name == NULL: name = b"<unknown>"
        CHKERR( PetscViewerStringSPrintf(viewer,"%s", name) )
    return 0

# --------------------------------------------------------------------

cdef extern from * nogil:
    struct _n_PetscLayout
    ctypedef _n_PetscLayout* PetscLayout
    PetscErrorCode PetscLayoutSetLocalSize(PetscLayout,PetscInt)
    PetscErrorCode PetscLayoutSetSize(PetscLayout,PetscInt)
    PetscErrorCode PetscLayoutGetBlockSize(PetscLayout,PetscInt*)
    PetscErrorCode PetscLayoutSetBlockSize(PetscLayout,PetscInt)
    PetscErrorCode PetscLayoutSetUp(PetscLayout)

cdef extern from * nogil:
    ctypedef enum NormType:
        NORM_1
        NORM_2
    ctypedef enum InsertMode:
        INSERT_VALUES
        ADD_VALUES
    PetscErrorCode VecDestroy(PetscVec*)
    PetscErrorCode VecDuplicate(PetscVec,PetscVec*)
    PetscErrorCode VecCopy(PetscVec,PetscVec)
    PetscErrorCode VecSet(PetscVec,PetscScalar)
    PetscErrorCode VecScale(PetscVec,PetscScalar)
    PetscErrorCode VecShift(PetscVec,PetscScalar)
    PetscErrorCode VecAXPY(PetscVec,PetscScalar,PetscVec)
    PetscErrorCode VecAXPBY(PetscVec,PetscScalar,PetscScalar,PetscVec)
    PetscErrorCode VecNorm(PetscVec,NormType,PetscReal*)

# --------------------------------------------------------------------

cdef extern from * nogil:
    ctypedef enum MatDuplicateOption:
        MAT_DO_NOT_COPY_VALUES
        MAT_COPY_VALUES
        MAT_SHARE_NONZERO_PATTERN
    ctypedef enum MatAssemblyType:
        MAT_FLUSH_ASSEMBLY
        MAT_FINAL_ASSEMBLY
    ctypedef enum MatOption:
        pass
    ctypedef enum MatStructure:
        SAME_NONZERO_PATTERN
        DIFFERENT_NONZERO_PATTERN
        SUBSET_NONZERO_PATTERN
    ctypedef enum MatReuse:
        MAT_IGNORE_MATRIX
        MAT_INITIAL_MATRIX
        MAT_REUSE_MATRIX

    ctypedef enum MatSORType:
        SOR_FORWARD_SWEEP
        SOR_BACKWARD_SWEEP
        SOR_SYMMETRIC_SWEEP
        SOR_LOCAL_FORWARD_SWEEP
        SOR_LOCAL_BACKWARD_SWEEP
        SOR_LOCAL_SYMMETRIC_SWEEP
        SOR_ZERO_INITIAL_GUESS
        SOR_EISENSTAT
        SOR_APPLY_UPPER
        SOR_APPLY_LOWER

cdef extern from * nogil:
    struct _MatOps:
        PetscErrorCode (*destroy)(PetscMat) except IERR
        PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscMat) except IERR
        PetscErrorCode (*view)(PetscMat,PetscViewer) except IERR
        PetscErrorCode (*duplicate)(PetscMat,MatDuplicateOption,PetscMat*) except IERR
        PetscErrorCode (*copy)(PetscMat,PetscMat,MatStructure) except IERR
        PetscErrorCode (*createsubmatrix)(PetscMat,PetscIS,PetscIS,MatReuse,PetscMat*) except IERR
        PetscErrorCode (*setoption)(PetscMat,MatOption,PetscBool) except IERR
        PetscErrorCode (*setup)(PetscMat) except IERR
        PetscErrorCode (*assemblybegin)(PetscMat,MatAssemblyType) except IERR
        PetscErrorCode (*assemblyend)(PetscMat,MatAssemblyType) except IERR
        PetscErrorCode (*zeroentries)(PetscMat) except IERR
        PetscErrorCode (*zerorowscolumns)(PetscMat,PetscInt,PetscInt*,PetscScalar,PetscVec,PetscVec) except IERR
        PetscErrorCode (*scale)(PetscMat,PetscScalar) except IERR
        PetscErrorCode (*shift)(PetscMat,PetscScalar) except IERR
        PetscErrorCode (*sor)(PetscMat,PetscVec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,PetscVec) except IERR
        PetscErrorCode (*getvecs)(PetscMat,PetscVec*,PetscVec*) except IERR
        PetscErrorCode (*mult)(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*multtranspose)(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*multhermitian"multhermitiantranspose")(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*multadd)(PetscMat,PetscVec,PetscVec,PetscVec) except IERR
        PetscErrorCode (*multtransposeadd)(PetscMat,PetscVec,PetscVec,PetscVec) except IERR
        PetscErrorCode (*multhermitianadd"multhermitiantransposeadd")(PetscMat,PetscVec,PetscVec,PetscVec) except IERR
        PetscErrorCode (*multdiagonalblock)(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*solve)(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*solvetranspose)(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*solveadd)(PetscMat,PetscVec,PetscVec,PetscVec) except IERR
        PetscErrorCode (*solvetransposeadd)(PetscMat,PetscVec,PetscVec,PetscVec) except IERR
        PetscErrorCode (*getdiagonal)(PetscMat,PetscVec) except IERR
        PetscErrorCode (*setdiagonal"diagonalset")(PetscMat,PetscVec,InsertMode) except IERR
        PetscErrorCode (*diagonalscale)(PetscMat,PetscVec,PetscVec) except IERR
        PetscErrorCode (*norm)(PetscMat,NormType,PetscReal*) except IERR
        PetscErrorCode (*realpart)(PetscMat) except IERR
        PetscErrorCode (*imagpart"imaginarypart")(PetscMat) except IERR
        PetscErrorCode (*conjugate)(PetscMat) except IERR
    ctypedef _MatOps *MatOps
    struct _p_Mat:
        void *data
        MatOps ops
        PetscBool assembled
        PetscBool preallocated
        PetscLayout rmap, cmap
cdef extern from * nogil:
    PetscErrorCode MatCreateVecs(PetscMat,PetscVec*,PetscVec*)
    PetscErrorCode MatIsSymmetricKnown(PetscMat,PetscBool*,PetscBool*)
    PetscErrorCode MatIsHermitianKnown(PetscMat,PetscBool*,PetscBool*)
    PetscErrorCode MatMult(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatMultTranspose(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatMultHermitian"MatMultHermitianTranspose"(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatSolve(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatSolveTranspose(PetscMat,PetscVec,PetscVec)

@cython.internal
cdef class _PyMat(_PyObj): pass
cdef inline _PyMat PyMat(PetscMat mat):
    if mat != NULL and mat.data != NULL:
        return <_PyMat>mat.data
    else:
        return _PyMat.__new__(_PyMat)

cdef public PetscErrorCode MatPythonGetContext(PetscMat mat, void **ctx) \
    except IERR:
    FunctionBegin(b"MatPythonGetContext")
    PyMat(mat).getcontext(ctx)
    return FunctionEnd()

cdef public PetscErrorCode MatPythonSetContext(PetscMat mat, void *ctx) \
    except IERR:
    FunctionBegin(b"MatPythonSetContext")
    PyMat(mat).setcontext(ctx, Mat_(mat))
    return FunctionEnd()

cdef PetscErrorCode MatPythonSetType_PYTHON(PetscMat mat, char name[]) \
    except IERR with gil:
    FunctionBegin(b"MatPythonSetType_PYTHON")
    if name == NULL: return FunctionEnd() # XXX
    cdef object ctx = createcontext(name)
    MatPythonSetContext(mat, <void*>ctx)
    PyMat(mat).setname(name)
    return FunctionEnd()

cdef PetscErrorCode MatCreate_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatCreate_Python")
    #
    cdef MatOps ops       = mat.ops
    ops.destroy           = MatDestroy_Python
    ops.setfromoptions    = MatSetFromOptions_Python
    ops.view              = MatView_Python
    ops.duplicate         = MatDuplicate_Python
    ops.copy              = MatCopy_Python
    ops.createsubmatrix   = MatCreateSubMatrix_Python
    ops.setoption         = MatSetOption_Python
    ops.setup             = MatSetUp_Python
    ops.assemblybegin     = MatAssemblyBegin_Python
    ops.assemblyend       = MatAssemblyEnd_Python
    ops.zeroentries       = MatZeroEntries_Python
    ops.zerorowscolumns   = MatZeroRowsColumns_Python
    ops.scale             = MatScale_Python
    ops.shift             = MatShift_Python
    ops.getvecs           = MatCreateVecs_Python
    ops.mult              = MatMult_Python
    ops.sor               = MatSOR_Python
    ops.multtranspose     = MatMultTranspose_Python
    ops.multhermitian     = MatMultHermitian_Python
    ops.multadd           = MatMultAdd_Python
    ops.multtransposeadd  = MatMultTransposeAdd_Python
    ops.multhermitianadd  = MatMultHermitianAdd_Python
    ops.multdiagonalblock = MatMultDiagonalBlock_Python
    ops.solve             = MatSolve_Python
    ops.solvetranspose    = MatSolveTranspose_Python
    ops.solveadd          = MatSolveAdd_Python
    ops.solvetransposeadd = MatSolveTransposeAdd_Python
    ops.getdiagonal       = MatGetDiagonal_Python
    ops.setdiagonal       = MatSetDiagonal_Python
    ops.diagonalscale     = MatDiagonalScale_Python
    ops.norm              = MatNorm_Python
    ops.realpart          = MatRealPart_Python
    ops.imagpart          = MatImagPart_Python
    ops.conjugate         = MatConjugate_Python
    #
    mat.assembled    = PETSC_TRUE  # XXX
    mat.preallocated = PETSC_FALSE # XXX
    #
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>mat,b"MatGetDiagonalBlock_C",
            <PetscVoidFunction>MatGetDiagonalBlock_Python) )
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>mat,b"MatPythonSetType_C",
            <PetscVoidFunction>MatPythonSetType_PYTHON) )
    CHKERR( PetscObjectChangeTypeName(
            <PetscObject>mat,MATPYTHON) )
    #
    cdef ctx = PyMat(NULL)
    mat.data = <void*> ctx
    Py_INCREF(<PyObject*>mat.data)
    return FunctionEnd()

cdef PetscErrorCode MatDestroy_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatDestroy_Python")
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>mat,b"MatGetDiagonalBlock_C",
            <PetscVoidFunction>NULL) )
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>mat,b"MatPythonSetType_C",
            <PetscVoidFunction>NULL) )
    CHKERR( PetscObjectChangeTypeName(
            <PetscObject>mat,NULL) )
    #
    if not Py_IsInitialized(): return FunctionEnd()
    try:
        addRef(mat)
        MatPythonSetContext(mat, NULL)
    finally:
        delRef(mat)
        Py_DECREF(<PyObject*>mat.data)
        mat.data = NULL
    return FunctionEnd()

cdef PetscErrorCode MatSetFromOptions_Python(
    PetscOptionItems *PetscOptionsObject,
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSetFromOptions_Python")
    #
    cdef char name[2048], *defval = PyMat(mat).getname()
    cdef PetscBool found = PETSC_FALSE
    cdef PetscOptionItems *opts "PetscOptionsObject" = PetscOptionsObject
    CHKERR( PetscOptionsString(
            b"-mat_python_type",b"Python [package.]module[.{class|function}]",
            b"MatPythonSetType",defval,name,sizeof(name),&found) ); <void>opts;
    if found and name[0]:
        CHKERR( MatPythonSetType_PYTHON(mat,name) )
    #
    cdef setFromOptions = PyMat(mat).setFromOptions
    if setFromOptions is not None:
        setFromOptions(Mat_(mat))
    return FunctionEnd()

cdef PetscErrorCode MatView_Python(
    PetscMat mat,
    PetscViewer vwr,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatView_Python")
    viewcontext(PyMat(mat), vwr)
    cdef view = PyMat(mat).view
    if view is not None:
        view(Mat_(mat), Viewer_(vwr))
    return FunctionEnd()

cdef PetscErrorCode MatDuplicate_Python(
    PetscMat mat,
    MatDuplicateOption op,
    PetscMat* out,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatDuplicate_Python")
    cdef duplicate = PyMat(mat).duplicate
    if duplicate is None: return UNSUPPORTED(b"duplicate")
    cdef Mat m = duplicate(Mat_(mat), <long>op)
    out[0] = m.mat; m.mat = NULL
    return FunctionEnd()

cdef PetscErrorCode MatCopy_Python(
    PetscMat mat,
    PetscMat out,
    MatStructure op,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatCopy_Python")
    cdef copy = PyMat(mat).copy
    if copy is None: return UNSUPPORTED(b"copy")
    copy(Mat_(mat), Mat_(out), <long>op)
    return FunctionEnd()

cdef PetscErrorCode MatGetDiagonalBlock_Python(
    PetscMat  mat,
    PetscMat  *out
    ) \
    except IERR with gil:
    FunctionBegin(b"MatGetDiagonalBlock_Python")
    cdef getDiagonalBlock = PyMat(mat).getDiagonalBlock
    if getDiagonalBlock is None:
        if getCommSize(mat) == 1:
            out[0] = mat
            return FunctionEnd()
    if getDiagonalBlock is None: return UNSUPPORTED(b"getDiagonalBlock")
    cdef Mat sub = getDiagonalBlock(Mat_(mat))
    if sub is not None: out[0] = sub.mat
    return FunctionEnd()

cdef PetscErrorCode MatCreateSubMatrix_Python(
    PetscMat mat,
    PetscIS  row,
    PetscIS  col,
    MatReuse op,
    PetscMat *out,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatCreateSubMatrix_Python")
    cdef createSubMatrix = PyMat(mat).createSubMatrix
    if createSubMatrix is None: return UNSUPPORTED(b"createSubMatrix")
    cdef Mat sub = None
    if op == MAT_IGNORE_MATRIX:
        sub = None
    elif op == MAT_INITIAL_MATRIX:
        sub = createSubMatrix(Mat_(mat), IS_(row), IS_(col), None)
        if sub is not None:
            addRef(sub.mat)
    elif op == MAT_REUSE_MATRIX:
        sub = createSubMatrix(Mat_(mat), IS_(row), IS_(col), Mat_(out[0]))
    if sub is not None:
        out[0] = sub.mat
    return FunctionEnd()

cdef PetscErrorCode MatSetOption_Python(
    PetscMat mat,
    MatOption op,
    PetscBool flag,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSetOption_Python")
    cdef setOption = PyMat(mat).setOption
    if setOption is not None:
        setOption(Mat_(mat), <long>op, <bint>(<int>flag))
    return FunctionEnd()

cdef PetscErrorCode MatSetUp_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSetUp_Python")
    cdef PetscInt rbs = -1, cbs = -1
    CHKERR( PetscLayoutGetBlockSize(mat.rmap,&rbs) )
    CHKERR( PetscLayoutGetBlockSize(mat.cmap,&cbs) )
    if rbs == -1: rbs = 1
    if cbs == -1: cbs = rbs
    CHKERR( PetscLayoutSetBlockSize(mat.rmap,rbs)  )
    CHKERR( PetscLayoutSetBlockSize(mat.cmap,cbs)  )
    CHKERR( PetscLayoutSetUp(mat.rmap)             )
    CHKERR( PetscLayoutSetUp(mat.cmap)             )
    mat.preallocated = PETSC_TRUE
    #
    cdef char name[2048]
    cdef PetscBool found = PETSC_FALSE
    if PyMat(mat).self is None:
        CHKERR( PetscOptionsGetString(NULL,
                getPrefix(mat), b"-mat_python_type",
                name,sizeof(name),&found) )
        if found and name[0]:
            CHKERR( MatPythonSetType_PYTHON(mat,name) )
    if PyMat(mat).self is None:
        return PetscSETERR(PETSC_ERR_USER,
            "Python context not set, call one of \n"
            " * MatPythonSetType(mat,\"[package.]module.class\")\n"
            " * MatSetFromOptions(mat) and pass option "
            "-mat_python_type [package.]module.class")
    #
    cdef setUp = PyMat(mat).setUp
    if setUp is not None:
        setUp(Mat_(mat))
    return FunctionEnd()

cdef PetscErrorCode MatAssemblyBegin_Python(
    PetscMat mat,
    MatAssemblyType at,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatAssemblyBegin_Python")
    cdef assembly = PyMat(mat).assemblyBegin
    if assembly is not None:
        assembly(Mat_(mat), <long>at)
    return FunctionEnd()

cdef PetscErrorCode MatAssemblyEnd_Python(
    PetscMat mat,
    MatAssemblyType at,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatAssemblyEnd_Python")
    cdef assembly = PyMat(mat).assemblyEnd
    if assembly is None:
        assembly = PyMat(mat).assembly
    if assembly is not None:
        assembly(Mat_(mat), <int>at)
    return FunctionEnd()

cdef PetscErrorCode MatZeroEntries_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatZeroEntries_Python")
    cdef zeroEntries = PyMat(mat).zeroEntries
    if zeroEntries is None: return UNSUPPORTED(b"zeroEntries")
    zeroEntries(Mat_(mat))
    return FunctionEnd()

cdef PetscErrorCode MatZeroRowsColumns_Python(
    PetscMat mat,
    PetscInt numRows,
    const PetscInt* rows,
    PetscScalar diag,
    PetscVec x,
    PetscVec b,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatZeroRowsColumns_Python")
    cdef zeroRowsColumns = PyMat(mat).zeroRowsColumns
    if zeroRowsColumns is None: return UNSUPPORTED(b"zeroRowsColumns")
    cdef ndarray pyrows = array_i(numRows, rows)
    zeroRowsColumns(Mat_(mat), pyrows, toScalar(diag), Vec_(x), Vec_(b))
    return FunctionEnd()

cdef PetscErrorCode MatScale_Python(
    PetscMat mat,
    PetscScalar s,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatScale_Python")
    cdef scale = PyMat(mat).scale
    if scale is None: return UNSUPPORTED(b"scale")
    scale(Mat_(mat), toScalar(s))
    return FunctionEnd()

cdef PetscErrorCode MatShift_Python(
    PetscMat mat,
    PetscScalar s,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatShift_Python")
    cdef shift = PyMat(mat).shift
    if shift is None: return UNSUPPORTED(b"shift")
    shift(Mat_(mat), toScalar(s))
    return FunctionEnd()

cdef PetscErrorCode MatCreateVecs_Python(
    PetscMat mat,
    PetscVec *x,
    PetscVec *y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatCreateVecs_Python")
    cdef createVecs = PyMat(mat).createVecs
    if createVecs is None:
        try:
            mat.ops.getvecs = NULL
            CHKERR( MatCreateVecs(mat,x,y) )
        finally:
            mat.ops.getvecs = MatCreateVecs_Python
        return FunctionEnd()
    if createVecs is None: return UNSUPPORTED(b"createVecs")
    cdef Vec u, v
    u, v = createVecs(Mat_(mat))
    if x != NULL:
        x[0] = u.vec
        u.vec = NULL
    if y != NULL:
        y[0] = v.vec
        v.vec = NULL
    return FunctionEnd()

cdef PetscErrorCode MatMult_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMult_Python")
    cdef mult = PyMat(mat).mult
    if mult is None: return UNSUPPORTED(b"mult")
    mult(Mat_(mat), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatMultTranspose_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMultTranspose_Python")
    cdef multTranspose = PyMat(mat).multTranspose
    cdef PetscBool symmset, symmknown
    if multTranspose is None:
        symmset = symmknown = PETSC_FALSE
        CHKERR( MatIsSymmetricKnown(mat,&symmset,&symmknown) )
        if symmset and symmknown:
            CHKERR( MatMult(mat,x,y) )
            return FunctionEnd()
    if multTranspose is None: return UNSUPPORTED(b"multTranspose")
    multTranspose(Mat_(mat), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatMultHermitian_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMultHermitian_Python")
    cdef multHermitian = PyMat(mat).multHermitian
    cdef PetscBool hermset, hermknown
    if multHermitian is None:
        hermset = hermknown = PETSC_FALSE
        CHKERR( MatIsHermitianKnown(mat,&hermset,&hermknown) )
        if hermset and hermknown:
            CHKERR( MatMult(mat,x,y) )
            return FunctionEnd()
    if multHermitian is None: return UNSUPPORTED(b"multHermitian")
    multHermitian(Mat_(mat), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatMultAdd_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec v,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMultAdd_Python")
    cdef multAdd = PyMat(mat).multAdd
    cdef PetscVec t = NULL

    if multAdd is None:
        if v == y:
            CHKERR( VecDuplicate(y, &t) )
            CHKERR( MatMult(mat,x,t) )
            CHKERR( VecAXPY(y,1.0,t) )
            CHKERR( VecDestroy(&t) )
        else:
            CHKERR( MatMult(mat,x,y) )
            CHKERR( VecAXPY(y,1.0,v) )
        return FunctionEnd()
    if multAdd is None: return UNSUPPORTED(b"multAdd")
    multAdd(Mat_(mat), Vec_(x), Vec_(v), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatMultTransposeAdd_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec v,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMultTransposeAdd_Python")
    cdef multTransposeAdd = PyMat(mat).multTransposeAdd
    if multTransposeAdd is None:
        CHKERR( MatMultTranspose(mat,x,y) )
        CHKERR( VecAXPY(y,1.0,v)          )
        return FunctionEnd()
    if multTransposeAdd is None: return UNSUPPORTED(b"multTransposeAdd")
    multTransposeAdd(Mat_(mat), Vec_(x), Vec_(v), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatMultHermitianAdd_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec v,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMultHermitianAdd_Python")
    cdef multHermitianAdd = PyMat(mat).multHermitianAdd
    if multHermitianAdd is None:
        CHKERR( MatMultHermitian(mat,x,y) )
        CHKERR( VecAXPY(y,1.0,v)          )
        return FunctionEnd()
    if multHermitianAdd is None: return UNSUPPORTED(b"multHermitianAdd")
    multHermitianAdd(Mat_(mat), Vec_(x), Vec_(v), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatMultDiagonalBlock_Python(
    PetscMat mat,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatMultDiagonalBlock_Python")
    cdef multDiagonalBlock = PyMat(mat).multDiagonalBlock
    if multDiagonalBlock is None: return UNSUPPORTED(b"multDiagonalBlock")
    multDiagonalBlock(Mat_(mat), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode MatSolve_Python(
    PetscMat mat,
    PetscVec b,
    PetscVec x,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSolve_Python")
    cdef solve = PyMat(mat).solve
    if solve is None: return UNSUPPORTED(b"solve")
    solve(Mat_(mat), Vec_(b), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode MatSolveTranspose_Python(
    PetscMat mat,
    PetscVec b,
    PetscVec x,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSolveTranspose_Python")
    cdef solveTranspose = PyMat(mat).solveTranspose
    cdef PetscBool symmset, symmknown
    if solveTranspose is None:
        symmset = symmknown = PETSC_FALSE
        CHKERR( MatIsSymmetricKnown(mat,&symmset,&symmknown) )
        if symmset and symmknown:
            CHKERR( MatSolve(mat,b,x) )
            return FunctionEnd()
    if solveTranspose is None: return UNSUPPORTED(b"solveTranspose")
    solveTranspose(Mat_(mat), Vec_(b), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode MatSolveAdd_Python(
    PetscMat mat,
    PetscVec b,
    PetscVec y,
    PetscVec x,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSolveAdd_Python")
    cdef solveAdd = PyMat(mat).solveAdd
    if solveAdd is None:
        CHKERR( MatSolve(mat,b,x) )
        CHKERR( VecAXPY(x,1.0,y)  )
        return FunctionEnd()
    if solveAdd is None: return UNSUPPORTED(b"solveAdd")
    solveAdd(Mat_(mat), Vec_(b), Vec_(y), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode MatSolveTransposeAdd_Python(
    PetscMat mat,
    PetscVec b,
    PetscVec y,
    PetscVec x,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSolveTransposeAdd_Python")
    cdef solveTransposeAdd = PyMat(mat).solveTransposeAdd
    if solveTransposeAdd is None:
        CHKERR( MatSolveTranspose(mat,b,x) )
        CHKERR( VecAXPY(x,1.0,y)           )
        return FunctionEnd()
    if solveTransposeAdd is None: return UNSUPPORTED(b"solveTransposeAdd")
    solveTransposeAdd(Mat_(mat), Vec_(b), Vec_(y), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode MatSOR_Python(
    PetscMat mat,
    PetscVec b,
    PetscReal omega,
    MatSORType sortype,
    PetscReal shift,
    PetscInt its,
    PetscInt lits,
    PetscVec x
    )\
    except IERR with gil:
    FunctionBegin(b"MatSOR_Python")
    cdef SOR = PyMat(mat).SOR
    if SOR is None: return UNSUPPORTED(b"SOR")
    SOR(Mat_(mat), Vec_(b), asReal(omega), asInt(sortype), asReal(shift), asInt(its), asInt(lits), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode MatGetDiagonal_Python(
    PetscMat mat,
    PetscVec v,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatGetDiagonal_Python")
    cdef getDiagonal = PyMat(mat).getDiagonal
    if getDiagonal is None: return UNSUPPORTED(b"getDiagonal")
    getDiagonal(Mat_(mat), Vec_(v))
    return FunctionEnd()

cdef PetscErrorCode MatSetDiagonal_Python(
    PetscMat mat,
    PetscVec v,
    InsertMode im,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatSetDiagonal_Python")
    cdef setDiagonal = PyMat(mat).setDiagonal
    cdef bint addv = True if im == ADD_VALUES else False
    if setDiagonal is None: return UNSUPPORTED(b"setDiagonal")
    setDiagonal(Mat_(mat), Vec_(v), addv)
    return FunctionEnd()

cdef PetscErrorCode MatDiagonalScale_Python(
    PetscMat mat,
    PetscVec l,
    PetscVec r,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatDiagonalScale_Python")
    cdef diagonalScale = PyMat(mat).diagonalScale
    if diagonalScale is None: return UNSUPPORTED(b"diagonalScale")
    diagonalScale(Mat_(mat), Vec_(l), Vec_(r))
    return FunctionEnd()

cdef PetscErrorCode MatNorm_Python(
    PetscMat mat,
    NormType ntype,
    PetscReal *nrm,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatNorm_Python")
    cdef norm = PyMat(mat).norm
    if norm is None: return UNSUPPORTED(b"norm")
    retval = norm(Mat_(mat), ntype)
    nrm[0] = <PetscReal>retval
    return FunctionEnd()

cdef PetscErrorCode MatRealPart_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatRealPart_Python")
    cdef realPart = PyMat(mat).realPart
    if realPart is None: return UNSUPPORTED(b"realPart")
    realPart(Mat_(mat))
    return FunctionEnd()

cdef PetscErrorCode MatImagPart_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatImagPart_Python")
    cdef imagPart = PyMat(mat).imagPart
    if imagPart is None: return UNSUPPORTED(b"imagPart")
    imagPart(Mat_(mat))
    return FunctionEnd()

cdef PetscErrorCode MatConjugate_Python(
    PetscMat mat,
    ) \
    except IERR with gil:
    FunctionBegin(b"MatConjugate_Python")
    cdef conjugate = PyMat(mat).conjugate
    if conjugate is None: return UNSUPPORTED(b"conjugate")
    conjugate(Mat_(mat))
    return FunctionEnd()

# --------------------------------------------------------------------

cdef extern from * nogil:
    struct _PCOps:
      PetscErrorCode (*destroy)(PetscPC) except IERR
      PetscErrorCode (*setup)(PetscPC) except IERR
      PetscErrorCode (*reset)(PetscPC) except IERR
      PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscPC) except IERR
      PetscErrorCode (*view)(PetscPC,PetscViewer) except IERR
      PetscErrorCode (*presolve)(PetscPC,PetscKSP,PetscVec,PetscVec) except IERR
      PetscErrorCode (*postsolve)(PetscPC,PetscKSP,PetscVec,PetscVec) except IERR
      PetscErrorCode (*apply)(PetscPC,PetscVec,PetscVec) except IERR
      PetscErrorCode (*applytranspose)(PetscPC,PetscVec,PetscVec) except IERR
      PetscErrorCode (*applysymmetricleft)(PetscPC,PetscVec,PetscVec) except IERR
      PetscErrorCode (*applysymmetricright)(PetscPC,PetscVec,PetscVec) except IERR
    ctypedef _PCOps *PCOps
    struct _p_PC:
        void *data
        PCOps ops

@cython.internal
cdef class _PyPC(_PyObj): pass
cdef inline _PyPC PyPC(PetscPC pc):
    if pc != NULL and pc.data != NULL:
        return <_PyPC>pc.data
    else:
        return _PyPC.__new__(_PyPC)

cdef public PetscErrorCode PCPythonGetContext(PetscPC pc, void **ctx) \
    except IERR:
    FunctionBegin(b"PCPythonGetContext")
    PyPC(pc).getcontext(ctx)
    return FunctionEnd()

cdef public PetscErrorCode PCPythonSetContext(PetscPC pc, void *ctx) \
    except IERR:
    FunctionBegin(b"PCPythonSetContext")
    PyPC(pc).setcontext(ctx, PC_(pc))
    return FunctionEnd()

cdef PetscErrorCode PCPythonSetType_PYTHON(PetscPC pc, char name[]) \
    except IERR with gil:
    FunctionBegin(b"PCPythonSetType_PYTHON")
    if name == NULL: return FunctionEnd() # XXX
    cdef object ctx = createcontext(name)
    PCPythonSetContext(pc, <void*>ctx)
    PyPC(pc).setname(name)
    return FunctionEnd()

cdef PetscErrorCode PCCreate_Python(
    PetscPC pc,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCCreate_Python")
    #
    cdef PCOps ops          = pc.ops
    ops.reset               = PCReset_Python
    ops.destroy             = PCDestroy_Python
    ops.setup               = PCSetUp_Python
    ops.setfromoptions      = PCSetFromOptions_Python
    ops.view                = PCView_Python
    ops.presolve            = PCPreSolve_Python
    ops.postsolve           = PCPostSolve_Python
    ops.apply               = PCApply_Python
    ops.applytranspose      = PCApplyTranspose_Python
    ops.applysymmetricleft  = PCApplySymmetricLeft_Python
    ops.applysymmetricright = PCApplySymmetricRight_Python
    #
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>pc, b"PCPythonSetType_C",
            <PetscVoidFunction>PCPythonSetType_PYTHON) )
    #
    cdef ctx = PyPC(NULL)
    pc.data = <void*> ctx
    Py_INCREF(<PyObject*>pc.data)
    return FunctionEnd()

cdef PetscErrorCode PCDestroy_Python(
    PetscPC pc,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCDestroy_Python")
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>pc, b"PCPythonSetType_C",
            <PetscVoidFunction>NULL) )
    #
    if not Py_IsInitialized(): return FunctionEnd()
    try:
        addRef(pc)
        PCPythonSetContext(pc, NULL)
    finally:
        delRef(pc)
        Py_DECREF(<PyObject*>pc.data)
        pc.data = NULL
    return FunctionEnd()

cdef PetscErrorCode PCSetUp_Python(
    PetscPC pc,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCSetUp_Python")
    #
    cdef char name[2048]
    cdef PetscBool found = PETSC_FALSE
    if PyPC(pc).self is None:
        CHKERR( PetscOptionsGetString(NULL,
                getPrefix(pc), b"-pc_python_type",
                name,sizeof(name),&found) )
        if found and name[0]:
            CHKERR( PCPythonSetType_PYTHON(pc,name) )
    if PyPC(pc).self is None:
        return PetscSETERR(PETSC_ERR_USER,
            "Python context not set, call one of \n"
            " * PCPythonSetType(pc,\"[package.]module.class\")\n"
            " * PCSetFromOptions(pc) and pass option "
            "-pc_python_type [package.]module.class")
    #
    cdef setUp = PyPC(pc).setUp
    if setUp is not None:
        setUp(PC_(pc))
    #
    cdef o = PyPC(pc)
    cdef PCOps ops = pc.ops
    if o.applyTranspose is None:
        ops.applytranspose = NULL
    if o.applySymmetricLeft is None:
        ops.applysymmetricleft = NULL
    if o.applySymmetricRight is None:
        ops.applysymmetricright = NULL
    #
    return FunctionEnd()

cdef PetscErrorCode PCReset_Python(
    PetscPC pc,
    ) \
    except IERR with gil:
    if getRef(pc) == 0: return 0
    FunctionBegin(b"PCReset_Python")
    cdef reset = PyPC(pc).reset
    if reset is not None:
        reset(PC_(pc))
    return FunctionEnd()

cdef PetscErrorCode PCSetFromOptions_Python(
    PetscOptionItems *PetscOptionsObject,
    PetscPC pc,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCSetFromOptions_Python")
    #
    cdef char name[2048], *defval = PyPC(pc).getname()
    cdef PetscBool found = PETSC_FALSE
    cdef PetscOptionItems *opts "PetscOptionsObject" = PetscOptionsObject
    CHKERR( PetscOptionsString(
            b"-pc_python_type",b"Python [package.]module[.{class|function}]",
            b"PCPythonSetType",defval,name,sizeof(name),&found) ); <void>opts;
    if found and name[0]:
        CHKERR( PCPythonSetType_PYTHON(pc,name) )
    #
    cdef setFromOptions = PyPC(pc).setFromOptions
    if setFromOptions is not None:
        setFromOptions(PC_(pc))
    return FunctionEnd()

cdef PetscErrorCode PCView_Python(
    PetscPC     pc,
    PetscViewer vwr,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCView_Python")
    viewcontext(PyPC(pc), vwr)
    cdef view = PyPC(pc).view
    if view is not None:
        view(PC_(pc), Viewer_(vwr))
    return FunctionEnd()

cdef PetscErrorCode PCPreSolve_Python(
    PetscPC  pc,
    PetscKSP ksp,
    PetscVec b,
    PetscVec x,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCPreSolve_Python")
    cdef preSolve = PyPC(pc).preSolve
    if preSolve is not None:
        preSolve(PC_(pc), KSP_(ksp), Vec_(b), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode PCPostSolve_Python(
    PetscPC  pc,
    PetscKSP ksp,
    PetscVec b,
    PetscVec x,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCPostSolve_Python")
    cdef postSolve = PyPC(pc).postSolve
    if postSolve is not None:
        postSolve(PC_(pc), KSP_(ksp), Vec_(b), Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode PCApply_Python(
    PetscPC  pc,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCApply_Python")
    cdef apply = PyPC(pc).apply
    apply(PC_(pc), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode PCApplyTranspose_Python(
    PetscPC  pc,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCApplyTranspose_Python")
    cdef applyTranspose = PyPC(pc).applyTranspose
    applyTranspose(PC_(pc), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode PCApplySymmetricLeft_Python(
    PetscPC  pc,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCApplySymmetricLeft_Python")
    cdef applySymmetricLeft = PyPC(pc).applySymmetricLeft
    applySymmetricLeft(PC_(pc), Vec_(x), Vec_(y))
    return FunctionEnd()

cdef PetscErrorCode PCApplySymmetricRight_Python(
    PetscPC  pc,
    PetscVec x,
    PetscVec y,
    ) \
    except IERR with gil:
    FunctionBegin(b"PCApplySymmetricRight_Python")
    cdef applySymmetricRight = PyPC(pc).applySymmetricRight
    applySymmetricRight(PC_(pc), Vec_(x), Vec_(y))
    return FunctionEnd()

# --------------------------------------------------------------------

cdef extern from * nogil:
    ctypedef enum KSPNormType:
        KSP_NORM_NONE
        KSP_NORM_PRECONDITIONED
        KSP_NORM_UNPRECONDITIONED
        KSP_NORM_NATURAL
    ctypedef enum PCSide:
        PC_LEFT
        PC_RIGHT
        PC_SYMMETRIC
    ctypedef enum KSPConvergedReason:
        KSP_CONVERGED_ITERATING
        KSP_DIVERGED_ITS
cdef extern from * nogil:
    struct _KSPOps:
      PetscErrorCode (*destroy)(PetscKSP) except IERR
      PetscErrorCode (*setup)(PetscKSP) except IERR
      PetscErrorCode (*reset)(PetscKSP) except IERR
      PetscErrorCode (*setfromoptions)(PetscOptionItems*, PetscKSP) except IERR
      PetscErrorCode (*view)(PetscKSP,PetscViewer) except IERR
      PetscErrorCode (*solve)(PetscKSP) except IERR
      PetscErrorCode (*buildsolution)(PetscKSP,PetscVec,PetscVec*) except IERR
      PetscErrorCode (*buildresidual)(PetscKSP,PetscVec,PetscVec,PetscVec*) except IERR
    ctypedef _KSPOps *KSPOps
    struct _p_KSP:
        void *data
        KSPOps ops
        PetscBool transpose_solve
        PetscInt iter"its",max_its"max_it"
        PetscReal norm"rnorm"
        KSPConvergedReason reason
    PetscErrorCode KSPCreate(MPI_Comm,PetscKSP*)
    PetscErrorCode KSPSetSupportedNorm(PetscKSP,KSPNormType,PCSide,PetscInt)
    PetscErrorCode KSPSolve(PetscKSP,PetscVec,PetscVec)
    PetscErrorCode KSPSetOperators(PetscKSP,PetscMat,PetscMat)
cdef extern from * nogil:
    PetscErrorCode KSPGetRhs(PetscKSP,PetscVec*)
    PetscErrorCode KSPGetSolution(PetscKSP,PetscVec*)
    PetscErrorCode KSPBuildSolutionDefault(PetscKSP,PetscVec,PetscVec*)
    PetscErrorCode KSPBuildResidualDefault(PetscKSP,PetscVec,PetscVec,PetscVec*)
    PetscErrorCode KSPGetIterationNumber(PetscKSP,PetscInt*)
    PetscErrorCode KSPBuildSolution(PetscKSP,PetscVec,PetscVec*)
    PetscErrorCode KSPBuildResidual(PetscKSP,PetscVec,PetscVec,PetscVec*)
    PetscErrorCode KSPConverged(PetscKSP,PetscInt,PetscReal,KSPConvergedReason*)
    PetscErrorCode KSPLogHistory(PetscKSP,PetscReal)
    PetscErrorCode KSPMonitor(PetscKSP,PetscInt,PetscReal)


@cython.internal
cdef class _PyKSP(_PyObj): pass
cdef inline _PyKSP PyKSP(PetscKSP ksp):
    if ksp != NULL and ksp.data != NULL:
        return <_PyKSP>ksp.data
    else:
        return _PyKSP.__new__(_PyKSP)

cdef public PetscErrorCode KSPPythonGetContext(PetscKSP ksp, void **ctx) \
    except IERR:
    FunctionBegin(b"KSPPythonGetContext")
    PyKSP(ksp).getcontext(ctx)
    return FunctionEnd()

cdef public PetscErrorCode KSPPythonSetContext(PetscKSP ksp, void *ctx) \
    except IERR:
    FunctionBegin(b"KSPPythonSetContext")
    PyKSP(ksp).setcontext(ctx, KSP_(ksp))
    return FunctionEnd()

cdef PetscErrorCode KSPPythonSetType_PYTHON(PetscKSP ksp, char name[]) \
    except IERR with gil:
    FunctionBegin(b"KSPPythonSetType_PYTHON")
    if name == NULL: return FunctionEnd() # XXX
    cdef object ctx = createcontext(name)
    KSPPythonSetContext(ksp, <void*>ctx)
    PyKSP(ksp).setname(name)
    return FunctionEnd()

cdef PetscErrorCode KSPCreate_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPCreate_Python")
    #
    cdef KSPOps ops    = ksp.ops
    ops.reset          = KSPReset_Python
    ops.destroy        = KSPDestroy_Python
    ops.setup          = KSPSetUp_Python
    ops.setfromoptions = KSPSetFromOptions_Python
    ops.view           = KSPView_Python
    ops.solve          = KSPSolve_Python
    ops.buildsolution  = KSPBuildSolution_Python
    ops.buildresidual  = KSPBuildResidual_Python
    #
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>ksp, b"KSPPythonSetType_C",
            <PetscVoidFunction>KSPPythonSetType_PYTHON) )
    #
    cdef ctx = PyKSP(NULL)
    ksp.data = <void*> ctx
    Py_INCREF(<PyObject*>ksp.data)
    #
    CHKERR( KSPSetSupportedNorm(
            ksp, KSP_NORM_PRECONDITIONED,   PC_LEFT,      3) )
    CHKERR( KSPSetSupportedNorm(
            ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT,     3) )
    CHKERR( KSPSetSupportedNorm(
            ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT,      2) )
    CHKERR( KSPSetSupportedNorm(
            ksp, KSP_NORM_PRECONDITIONED,   PC_RIGHT,     2) )
    CHKERR( KSPSetSupportedNorm(
            ksp, KSP_NORM_PRECONDITIONED,   PC_SYMMETRIC, 1) )
    CHKERR( KSPSetSupportedNorm(
            ksp, KSP_NORM_UNPRECONDITIONED, PC_SYMMETRIC, 1) )
    return FunctionEnd()

cdef PetscErrorCode KSPDestroy_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPDestroy_Python")
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>ksp, b"KSPPythonSetType_C",
            <PetscVoidFunction>NULL))
    #
    if not Py_IsInitialized(): return FunctionEnd()
    try:
        addRef(ksp)
        KSPPythonSetContext(ksp, NULL)
    finally:
        delRef(ksp)
        Py_DECREF(<PyObject*>ksp.data)
        ksp.data = NULL
    return FunctionEnd()

cdef PetscErrorCode KSPSetUp_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPSetUp_Python")
    #
    cdef char name[2048]
    cdef PetscBool found = PETSC_FALSE
    if PyKSP(ksp).self is None:
        CHKERR( PetscOptionsGetString(NULL,
                getPrefix(ksp), b"-ksp_python_type",
                name,sizeof(name),&found) )
        if found and name[0]:
            CHKERR( KSPPythonSetType_PYTHON(ksp,name) )
    if PyKSP(ksp).self is None:
        return PetscSETERR(PETSC_ERR_USER,
            "Python context not set, call one of \n"
            " * KSPPythonSetType(ksp,\"[package.]module.class\")\n"
            " * KSPSetFromOptions(ksp) and pass option "
            "-ksp_python_type [package.]module.class")
    #
    cdef setUp = PyKSP(ksp).setUp
    if setUp is not None:
        setUp(KSP_(ksp))
    return FunctionEnd()

cdef PetscErrorCode KSPReset_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    if getRef(ksp) == 0: return 0
    FunctionBegin(b"KSPReset_Python")
    CHKERR( PetscObjectCompose(<PetscObject>ksp,b"@ksp.vec_work_sol",NULL) )
    CHKERR( PetscObjectCompose(<PetscObject>ksp,b"@ksp.vec_work_res",NULL) )
    cdef reset = PyKSP(ksp).reset
    if reset is not None:
        reset(KSP_(ksp))
    return FunctionEnd()

cdef PetscErrorCode KSPSetFromOptions_Python(
    PetscOptionItems *PetscOptionsObject,
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPSetFromOptions_Python")
    #
    cdef char name[2048], *defval = PyKSP(ksp).getname()
    cdef PetscBool found = PETSC_FALSE
    cdef PetscOptionItems *opts "PetscOptionsObject" = PetscOptionsObject
    CHKERR( PetscOptionsString(
            b"-ksp_python_type",b"Python [package.]module[.{class|function}]",
            b"KSPPythonSetType",defval,name,sizeof(name),&found) ); <void>opts;
    if found and name[0]:
        CHKERR( KSPPythonSetType_PYTHON(ksp,name) )
    #
    cdef setFromOptions = PyKSP(ksp).setFromOptions
    if setFromOptions is not None:
        setFromOptions(KSP_(ksp))
    return FunctionEnd()

cdef PetscErrorCode KSPView_Python(
    PetscKSP    ksp,
    PetscViewer vwr,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPView_Python")
    viewcontext(PyKSP(ksp), vwr)
    cdef view = PyKSP(ksp).view
    if view is not None:
        view(KSP_(ksp), Viewer_(vwr))
    return FunctionEnd()

cdef PetscErrorCode KSPBuildSolution_Python(
    PetscKSP ksp,
    PetscVec v,
    PetscVec *V,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPBuildSolution_Python")
    cdef PetscVec x = v
    cdef buildSolution = PyKSP(ksp).buildSolution
    if buildSolution is not None:
        if x == NULL: pass # XXX
        buildSolution(KSP_(ksp), Vec_(x))
        if V != NULL: V[0] = x
    else:
        CHKERR( KSPBuildSolutionDefault(ksp, v, V) )
    return FunctionEnd()

cdef PetscErrorCode KSPBuildResidual_Python(
    PetscKSP ksp,
    PetscVec t,
    PetscVec v,
    PetscVec *V,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPBuildResidual_Python")
    cdef buildResidual = PyKSP(ksp).buildResidual
    if buildResidual is not None:
        buildResidual(KSP_(ksp), Vec_(t), Vec_(v))
        if V != NULL: V[0] = v
    else:
        CHKERR( KSPBuildResidualDefault(ksp, t, v, V) )
    return FunctionEnd()

cdef PetscErrorCode KSPSolve_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPSolve_Python")
    cdef PetscVec B = NULL, X = NULL
    CHKERR( KSPGetRhs(ksp,&B)      )
    CHKERR( KSPGetSolution(ksp,&X) )
    #
    ksp.iter = 0
    ksp.reason = KSP_CONVERGED_ITERATING
    #
    cdef solve = None
    if ksp.transpose_solve:
        solve = PyKSP(ksp).solveTranspose
    else:
        solve = PyKSP(ksp).solve
    if solve is not None:
        solve(KSP_(ksp),Vec_(B),Vec_(X))
    else:
        KSPSolve_Python_default(ksp,B,X)
    return FunctionEnd()

cdef PetscErrorCode KSPSolve_Python_default(
    PetscKSP ksp,
    PetscVec B,
    PetscVec X,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPSolve_Python_default")
    #
    cdef PetscVec t = NULL
    CHKERR( PetscObjectQuery(
            <PetscObject>ksp,
             b"@ksp.vec_work_sol",
             <PetscObject*>&t) )
    if t == NULL:
        CHKERR( VecDuplicate(X,&t) )
        CHKERR( PetscObjectCompose(
                <PetscObject>ksp,
                 b"@ksp.vec_work_sol",
                 <PetscObject>t) )
    cdef PetscVec v = NULL
    CHKERR( PetscObjectQuery(
            <PetscObject>ksp,
             b"@ksp.vec_work_res",
             <PetscObject*>&v) )
    if v == NULL:
        CHKERR( VecDuplicate(B,&v) )
        CHKERR( PetscObjectCompose(
                <PetscObject>ksp,
                 b"@ksp.vec_work_res",
                 <PetscObject>v) )
    #
    cdef PetscInt its = 0
    cdef PetscVec R = NULL
    cdef PetscReal rnorm = 0
    #
    CHKERR( KSPBuildResidual(ksp,t,v,&R) )
    CHKERR( VecNorm(R,NORM_2,&rnorm)     )
    #
    CHKERR( KSPConverged(ksp,ksp.iter,rnorm,&ksp.reason) )
    CHKERR( KSPLogHistory(ksp,ksp.norm) )
    CHKERR( KSPMonitor(ksp,ksp.iter,ksp.norm) )
    for its from 0 <= its < ksp.max_its:
        if ksp.reason: break
        KSPPreStep_Python(ksp)
        #
        KSPStep_Python(ksp,B,X)
        CHKERR( KSPBuildResidual(ksp,t,v,&R) )
        CHKERR( VecNorm(R,NORM_2,&rnorm)     )
        ksp.iter += 1
        #
        KSPPostStep_Python(ksp)
        CHKERR( KSPConverged(ksp,ksp.iter,rnorm,&ksp.reason) )
        CHKERR( KSPLogHistory(ksp,ksp.norm) )
        CHKERR( KSPMonitor(ksp,ksp.iter,ksp.norm) )
    if ksp.iter == ksp.max_its:
        if ksp.reason == KSP_CONVERGED_ITERATING:
            ksp.reason = KSP_DIVERGED_ITS
    #
    return FunctionEnd()

cdef PetscErrorCode KSPPreStep_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPPreStep_Python")
    cdef preStep = PyKSP(ksp).preStep
    if preStep is not None:
        preStep(KSP_(ksp))
    return FunctionEnd()

cdef PetscErrorCode KSPPostStep_Python(
    PetscKSP ksp,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPPostStep_Python")
    cdef postStep = PyKSP(ksp).postStep
    if postStep is not None:
        postStep(KSP_(ksp))
    return FunctionEnd()

cdef PetscErrorCode KSPStep_Python(
    PetscKSP ksp,
    PetscVec B,
    PetscVec X,
    ) \
    except IERR with gil:
    FunctionBegin(b"KSPStep_Python")
    cdef step = None
    if ksp.transpose_solve:
        step = PyKSP(ksp).stepTranspose
        if step is None: return UNSUPPORTED(b"stepTranspose")
    else:
        step = PyKSP(ksp).step
        if step is None: return UNSUPPORTED(b"step")
    step(KSP_(ksp),Vec_(B),Vec_(X))
    return FunctionEnd()

# --------------------------------------------------------------------

cdef extern from * nogil:
    ctypedef enum SNESConvergedReason:
      SNES_CONVERGED_ITERATING
      SNES_DIVERGED_MAX_IT
cdef extern from * nogil:
    struct _SNESOps:
      PetscErrorCode (*destroy)(PetscSNES) except IERR
      PetscErrorCode (*setup)(PetscSNES) except IERR
      PetscErrorCode (*reset)(PetscSNES) except IERR
      PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscSNES) except IERR
      PetscErrorCode (*view)(PetscSNES,PetscViewer) except IERR
      PetscErrorCode (*solve)(PetscSNES) except IERR
    ctypedef _SNESOps *SNESOps
    struct _p_SNES:
        void *data
        SNESOps ops
        PetscInt  iter,max_its,linear_its
        PetscReal norm,rtol,ttol
        SNESConvergedReason reason
        PetscVec vec_sol,vec_sol_update,vec_func
        PetscMat jacobian,jacobian_pre
        PetscKSP ksp
    PetscErrorCode SNESCreate(MPI_Comm,PetscSNES*)
    PetscErrorCode SNESSolve(PetscSNES,PetscVec,PetscVec)
    ctypedef PetscErrorCode (*SNESFunction)(PetscSNES,PetscVec,PetscVec,void*)
    PetscErrorCode SNESGetFunction(PetscSNES,PetscVec*,SNESFunction*,void*)
    PetscErrorCode SNESSetFunction(PetscSNES,PetscVec,SNESFunction,void*)
    int SNESComputeFunction(PetscSNES,PetscVec,PetscVec)
    ctypedef PetscErrorCode (*SNESJacobian)(PetscSNES,PetscVec,PetscMat,PetscMat,void*)
    PetscErrorCode SNESGetJacobian(PetscSNES,PetscMat*,PetscMat*,SNESJacobian*,void*)
    PetscErrorCode SNESSetJacobian(PetscSNES,PetscMat,PetscMat,SNESJacobian,void*)
    int SNESComputeJacobian(PetscSNES,PetscVec,PetscMat,PetscMat)
    SNESJacobian MatMFFDComputeJacobian
    PetscErrorCode SNESGetKSP(PetscSNES,PetscKSP*)
    PetscErrorCode SNESGetLineSearch(PetscSNES,PetscSNESLineSearch*)
    PetscErrorCode SNESLineSearchApply(PetscSNESLineSearch,PetscVec,PetscVec,PetscReal*,PetscVec)
cdef extern from * nogil:
    PetscErrorCode SNESGetRhs(PetscSNES,PetscVec*)
    PetscErrorCode SNESGetSolution(PetscSNES,PetscVec*)
    PetscErrorCode SNESGetSolutionUpdate(PetscSNES,PetscVec*)
    PetscErrorCode SNESGetIterationNumber(PetscSNES,PetscInt*)
    PetscErrorCode SNESGetLinearSolveIterations(PetscSNES,PetscInt*)
    PetscErrorCode SNESGetConvergedReason(PetscSNES,SNESConvergedReason*)
    PetscErrorCode SNESConverged(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*)
    PetscErrorCode SNESLogHistory(PetscSNES,PetscReal,PetscInt)
    PetscErrorCode SNESMonitor(PetscSNES,PetscInt,PetscReal)
    PetscErrorCode SNESSetFromOptions(PetscSNES)


@cython.internal
cdef class _PySNES(_PyObj): pass
cdef inline _PySNES PySNES(PetscSNES snes):
    if snes != NULL and snes.data != NULL:
        return <_PySNES>snes.data
    else:
        return _PySNES.__new__(_PySNES)

cdef public PetscErrorCode SNESPythonGetContext(PetscSNES snes, void **ctx) \
    except IERR:
    FunctionBegin(b"SNESPythonGetContext ")
    PySNES(snes).getcontext(ctx)
    return FunctionEnd()

cdef public PetscErrorCode SNESPythonSetContext(PetscSNES snes, void *ctx) \
    except IERR:
    FunctionBegin(b"SNESPythonSetContext ")
    PySNES(snes).setcontext(ctx, SNES_(snes))
    return FunctionEnd()

cdef PetscErrorCode SNESPythonSetType_PYTHON(PetscSNES snes, char name[]) \
    except IERR with gil:
    FunctionBegin(b"SNESPythonSetType_PYTHON")
    if name == NULL: return FunctionEnd() # XXX
    cdef object ctx = createcontext(name)
    SNESPythonSetContext(snes, <void*>ctx)
    PySNES(snes).setname(name)
    return FunctionEnd()

cdef PetscErrorCode SNESCreate_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESCreate_Python")
    #
    cdef SNESOps ops   = snes.ops
    cdef PetscSNESLineSearch ls = NULL
    ops.reset          = SNESReset_Python
    ops.destroy        = SNESDestroy_Python
    ops.setup          = SNESSetUp_Python
    ops.setfromoptions = SNESSetFromOptions_Python
    ops.view           = SNESView_Python
    ops.solve          = SNESSolve_Python
    #
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>snes, b"SNESPythonSetType_C",
            <PetscVoidFunction>SNESPythonSetType_PYTHON) )
    #
    cdef ctx = PySNES(NULL)
    snes.data = <void*> ctx

    # Ensure that the SNES has a linesearch object early enough that
    # it gets setFromOptions.
    CHKERR( SNESGetLineSearch(snes, &ls) )
    Py_INCREF(<PyObject*>snes.data)
    return FunctionEnd()

cdef PetscErrorCode SNESDestroy_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESDestroy_Python")
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>snes, b"SNESPythonSetType_C",
            <PetscVoidFunction>NULL) )
    #
    if not Py_IsInitialized(): return FunctionEnd()
    try:
        addRef(snes)
        SNESPythonSetContext(snes, NULL)
    finally:
        delRef(snes)
        Py_DECREF(<PyObject*>snes.data)
        snes.data = NULL
    return FunctionEnd()

cdef PetscErrorCode SNESSetUp_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESSetUp_Python")
    #
    #SNESGetKSP(snes,&snes.ksp)
    #
    cdef char name[2048]
    cdef PetscBool found = PETSC_FALSE
    if PySNES(snes).self is None:
        CHKERR( PetscOptionsGetString(NULL,
                getPrefix(snes),b"-snes_python_type",
                name,sizeof(name),&found) )
        if found and name[0]:
            CHKERR( SNESPythonSetType_PYTHON(snes,name) )
    if PySNES(snes).self is None:
        return PetscSETERR(PETSC_ERR_USER,
            "Python context not set, call one of \n"
            " * SNESPythonSetType(snes,\"[package.]module.class\")\n"
            " * SNESSetFromOptions(snes) and pass option "
            "-snes_python_type [package.]module.class")
    #
    cdef setUp = PySNES(snes).setUp
    if setUp is not None:
        setUp(SNES_(snes))
    return FunctionEnd()

cdef PetscErrorCode SNESReset_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    if getRef(snes) == 0: return 0
    FunctionBegin(b"SNESReset_Python")
    cdef reset = PySNES(snes).reset
    if reset is not None:
        reset(SNES_(snes))
    return FunctionEnd()

cdef PetscErrorCode SNESSetFromOptions_Python(
    PetscOptionItems *PetscOptionsObject,
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESSetFromOptions_Python")
    #
    cdef char name[2048], *defval = PySNES(snes).getname()
    cdef PetscBool found = PETSC_FALSE
    cdef PetscOptionItems *opts "PetscOptionsObject" = PetscOptionsObject
    CHKERR( PetscOptionsString(
            b"-snes_python_type",b"Python [package.]module[.{class|function}]",
            b"SNESPythonSetType",defval,name,sizeof(name),&found) ); <void>opts;
    if found and name[0]:
        CHKERR( SNESPythonSetType_PYTHON(snes,name) )
    #
    cdef setFromOptions = PySNES(snes).setFromOptions
    if setFromOptions is not None:
        setFromOptions(SNES_(snes))
    return FunctionEnd()

cdef PetscErrorCode SNESView_Python(
    PetscSNES   snes,
    PetscViewer vwr,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESView_Python")
    viewcontext(PySNES(snes), vwr)
    cdef view = PySNES(snes).view
    if view is not None:
        view(SNES_(snes), Viewer_(vwr))
    return FunctionEnd()

cdef PetscErrorCode SNESSolve_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESSolve_Python")
    cdef PetscVec b = NULL, x = NULL
    CHKERR( SNESGetRhs(snes,&b)      )
    CHKERR( SNESGetSolution(snes,&x) )
    #
    snes.iter = 0
    snes.reason = SNES_CONVERGED_ITERATING
    #
    cdef solve = PySNES(snes).solve
    if solve is not None:
        solve(SNES_(snes), Vec_(b) if b != NULL else None, Vec_(x))
    else:
        SNESSolve_Python_default(snes)
    #
    return FunctionEnd()

cdef PetscErrorCode SNESSolve_Python_default(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESSolve_Python_default")
    #
    cdef PetscVec X=NULL, F=NULL, Y=NULL
    cdef PetscSNESLineSearch ls
    CHKERR( SNESGetSolution(snes,&X)           )
    CHKERR( SNESGetFunction(snes,&F,NULL,NULL) )
    CHKERR( SNESGetSolutionUpdate(snes,&Y)     )
    CHKERR( SNESGetLineSearch(snes, &ls)       )
    cdef PetscInt  its=0, lits=0
    cdef PetscReal xnorm = 0.0
    cdef PetscReal fnorm = 0.0
    cdef PetscReal ynorm = 0.0
    #
    CHKERR( VecSet(Y,0.0)                 )
    CHKERR( SNESComputeFunction(snes,X,F) )
    CHKERR( VecNorm(X,NORM_2,&xnorm)      )
    CHKERR( VecNorm(F,NORM_2,&fnorm)      )
    #
    CHKERR( SNESConverged(snes,snes.iter,xnorm,ynorm,fnorm,&snes.reason) )
    CHKERR( SNESLogHistory(snes,snes.norm,lits) )
    CHKERR( SNESMonitor(snes,snes.iter,snes.norm) )
    for its from 0 <= its < snes.max_its:
        if snes.reason: break
        SNESPreStep_Python(snes)
        #
        lits = -snes.linear_its
        SNESStep_Python(snes,X,F,Y)
        lits += snes.linear_its
        #
        CHKERR( SNESLineSearchApply(ls,X,F,&fnorm,Y) )
        CHKERR( VecNorm(X,NORM_2,&xnorm)      )
        CHKERR( VecNorm(Y,NORM_2,&ynorm)      )
        snes.iter += 1
        #
        SNESPostStep_Python(snes)
        CHKERR( SNESConverged(snes,snes.iter,xnorm,ynorm,fnorm,&snes.reason) )
        CHKERR( SNESLogHistory(snes,snes.norm,lits) )
        CHKERR( SNESMonitor(snes,snes.iter,snes.norm) )
    if snes.iter == snes.max_its:
        if snes.reason == SNES_CONVERGED_ITERATING:
            snes.reason = SNES_DIVERGED_MAX_IT
    #
    return FunctionEnd()

cdef PetscErrorCode SNESPreStep_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESPreStep_Python")
    cdef preStep = PySNES(snes).preStep
    if preStep is not None:
        preStep(SNES_(snes))
    return FunctionEnd()

cdef PetscErrorCode SNESPostStep_Python(
    PetscSNES snes,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESPostStep_Python")
    cdef postStep = PySNES(snes).postStep
    if postStep is not None:
        postStep(SNES_(snes))
    return FunctionEnd()

cdef PetscErrorCode SNESStep_Python(
    PetscSNES snes,
    PetscVec  X,
    PetscVec  F,
    PetscVec  Y,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESStep_Python")
    cdef step = PySNES(snes).step
    if step is not None:
        step(SNES_(snes),Vec_(X),Vec_(F),Vec_(Y))
    else:
        SNESStep_Python_default(snes,X,F,Y)
    return FunctionEnd()

cdef PetscErrorCode SNESStep_Python_default(
    PetscSNES snes,
    PetscVec  X,
    PetscVec  F,
    PetscVec  Y,
    ) \
    except IERR with gil:
    FunctionBegin(b"SNESStep_Python_default")
    cdef PetscMat J = NULL, P = NULL
    cdef PetscInt lits = 0
    CHKERR( SNESGetJacobian(snes,&J,&P,NULL,NULL)   )
    CHKERR( SNESComputeJacobian(snes,X,J,P)         )
    CHKERR( KSPSetOperators(snes.ksp,J,P)           )
    CHKERR( KSPSolve(snes.ksp,F,Y)                  )
    CHKERR( KSPGetIterationNumber(snes.ksp,&lits)   )
    snes.linear_its += lits
    return FunctionEnd()

# --------------------------------------------------------------------


cdef extern from * nogil:
    ctypedef enum TSProblemType:
        TS_LINEAR
        TS_NONLINEAR
    ctypedef enum TSConvergedReason:
      TS_CONVERGED_ITERATING
      TS_CONVERGED_TIME
      TS_CONVERGED_ITS
      TS_DIVERGED_NONLINEAR_SOLVE
      TS_DIVERGED_STEP_REJECTED

cdef extern from * nogil:
    struct _TSOps:
      PetscErrorCode (*destroy)(PetscTS) except IERR
      PetscErrorCode (*setup)(PetscTS) except IERR
      PetscErrorCode (*reset)(PetscTS) except IERR
      PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscTS) except IERR
      PetscErrorCode (*view)(PetscTS,PetscViewer) except IERR
      PetscErrorCode (*step)(PetscTS) except IERR
      PetscErrorCode (*rollback)(PetscTS) except IERR
      PetscErrorCode (*interpolate)(PetscTS,PetscReal,PetscVec) except IERR
      PetscErrorCode (*evaluatestep)(PetscTS,PetscInt,PetscVec,PetscBool*) except IERR
      PetscErrorCode (*solve)(PetscTS) except IERR
      PetscErrorCode (*snesfunction)(PetscSNES,PetscVec,PetscVec,PetscTS) except IERR
      PetscErrorCode (*snesjacobian)(PetscSNES,PetscVec,PetscMat,PetscMat,PetscTS) except IERR
    ctypedef _TSOps *TSOps
    struct _TSUserOps:
      PetscErrorCode (*prestep)(PetscTS) except IERR
      PetscErrorCode (*prestage)(PetscTS,PetscReal) except IERR
      PetscErrorCode (*poststage)(PetscTS,PetscReal,PetscInt,PetscVec*) except IERR
      PetscErrorCode (*poststep)(PetscTS) except IERR
      PetscErrorCode (*rhsfunction)(PetscTS,PetscReal,PetscVec,PetscVec,void*) except IERR
      PetscErrorCode (*ifunction)  (PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,void*) except IERR
      PetscErrorCode (*rhsjacobian)(PetscTS,PetscReal,PetscVec,PetscMat,PetscMat,void*) except IERR
      PetscErrorCode (*ijacobian)  (PetscTS,PetscReal,PetscVec,PetscVec,PetscReal,PetscMat,PetscMat,void*) except IERR
    ctypedef _TSUserOps *TSUserOps
    struct _p_TSAdapt
    ctypedef _p_TSAdapt *PetscTSAdapt "TSAdapt"
    struct _p_TS:
        void *data
        PetscDM dm
        PetscTSAdapt adapt
        TSOps ops
        TSUserOps userops
        TSProblemType problem_type
        PetscInt  snes_its
        PetscInt  ksp_its
        PetscInt  reject
        PetscInt  max_reject
        PetscInt  steps
        PetscReal ptime
        PetscVec  vec_sol
        PetscReal time_step
        PetscInt  max_steps
        PetscReal max_time
        TSConvergedReason reason
        PetscSNES snes
        PetscBool usessnes
cdef extern from * nogil:
    PetscErrorCode TSGetKSP(PetscTS,PetscKSP*)
    PetscErrorCode TSGetSNES(PetscTS,PetscSNES*)
    PetscErrorCode TSPreStep(PetscTS)
    PetscErrorCode TSPreStage(PetscTS,PetscReal)
    PetscErrorCode TSPostStage(PetscTS,PetscReal,PetscInt,PetscVec*)
    PetscErrorCode TSPostStep(PetscTS)
    PetscErrorCode TSAdaptCheckStage(PetscTSAdapt,PetscTS,PetscReal,PetscVec,PetscBool*)
    PetscErrorCode TSMonitor(PetscTS,PetscInt,PetscReal,PetscVec)
    PetscErrorCode TSComputeIFunction(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscBool)
    PetscErrorCode TSComputeIJacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscReal,PetscMat,PetscMat,PetscBool)
    PetscErrorCode TSComputeI2Function(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscVec)
    PetscErrorCode TSComputeI2Jacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscReal,PetscReal,PetscMat,PetscMat)
    PetscErrorCode SNESTSFormFunction(PetscSNES,PetscVec,PetscVec,void*)
    PetscErrorCode SNESTSFormJacobian(PetscSNES,PetscVec,PetscMat,PetscMat,void*)

@cython.internal
cdef class _PyTS(_PyObj): pass
cdef inline _PyTS PyTS(PetscTS ts):
    if ts != NULL and ts.data != NULL:
        return <_PyTS>ts.data
    else:
        return _PyTS.__new__(_PyTS)

cdef public PetscErrorCode TSPythonGetContext(PetscTS ts, void **ctx) \
    except IERR:
    FunctionBegin(b"TSPythonGetContext")
    PyTS(ts).getcontext(ctx)
    return FunctionEnd()

cdef public PetscErrorCode TSPythonSetContext(PetscTS ts, void *ctx) \
    except IERR:
    FunctionBegin(b"TSPythonSetContext")
    PyTS(ts).setcontext(ctx, TS_(ts))
    return FunctionEnd()

cdef PetscErrorCode TSPythonSetType_PYTHON(PetscTS ts, char name[]) \
    except IERR with gil:
    FunctionBegin(b"TSPythonSetType_PYTHON")
    if name == NULL: return FunctionEnd() # XXX
    cdef object ctx = createcontext(name)
    TSPythonSetContext(ts, <void*>ctx)
    PyTS(ts).setname(name)
    return  0

cdef PetscErrorCode TSCreate_Python(
    PetscTS ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSCreate_Python")
    #
    cdef TSOps ops     = ts.ops
    ops.reset          = TSReset_Python
    ops.destroy        = TSDestroy_Python
    ops.setup          = TSSetUp_Python
    ops.setfromoptions = TSSetFromOptions_Python
    ops.view           = TSView_Python
    ops.step           = TSStep_Python
    ops.rollback       = TSRollBack_Python
    ops.interpolate    = TSInterpolate_Python
    ops.evaluatestep   = TSEvaluateStep_Python
    ops.snesfunction   = SNESTSFormFunction_Python
    ops.snesjacobian   = SNESTSFormJacobian_Python
    #
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>ts, b"TSPythonSetType_C",
            <PetscVoidFunction>TSPythonSetType_PYTHON) )
    #
    ts.usessnes = PETSC_TRUE
    #
    cdef ctx = PyTS(NULL)
    ts.data = <void*> ctx
    Py_INCREF(<PyObject*>ts.data)
    return FunctionEnd()

cdef PetscErrorCode TSDestroy_Python(
    PetscTS ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSDestroy_Python")
    CHKERR( PetscObjectComposeFunction(
            <PetscObject>ts, b"TSPythonSetType_C",
            <PetscVoidFunction>NULL) )
    #
    if not Py_IsInitialized(): return FunctionEnd()
    try:
        addRef(ts)
        TSPythonSetContext(ts, NULL)
    finally:
        delRef(ts)
        Py_DECREF(<PyObject*>ts.data)
        ts.data = NULL
    return FunctionEnd()

cdef PetscErrorCode TSSetUp_Python(
    PetscTS ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSSetUp_Python")
    #
    cdef PetscVec vec_update = NULL
    CHKERR( VecDuplicate(ts.vec_sol,&vec_update) )
    CHKERR( PetscObjectCompose(<PetscObject>ts,
                                b"@ts.vec_update",
                                <PetscObject>vec_update) )
    CHKERR( VecDestroy(&vec_update) )
    cdef PetscVec vec_dot = NULL
    CHKERR( VecDuplicate(ts.vec_sol,&vec_dot) )
    CHKERR( PetscObjectCompose(<PetscObject>ts,
                                b"@ts.vec_dot",
                                <PetscObject>vec_dot) )
    CHKERR( VecDestroy(&vec_dot) )
    #
    cdef char name[2048]
    cdef PetscBool found = PETSC_FALSE
    if PyTS(ts).self is None:
        CHKERR( PetscOptionsGetString(NULL,
                getPrefix(ts),b"-ts_python_type",
                name,sizeof(name),&found) )
        if found and name[0]:
            CHKERR( TSPythonSetType_PYTHON(ts,name) )
    if PyTS(ts).self is None:
        return PetscSETERR(PETSC_ERR_USER,
            "Python context not set, call one of \n"
            " * TSPythonSetType(ts,\"[package.]module.class\")\n"
            " * TSSetFromOptions(ts) and pass option "
            "-ts_python_type [package.]module.class")
    #
    cdef setUp = PyTS(ts).setUp
    if setUp is not None:
        setUp(TS_(ts))
    return FunctionEnd()

cdef PetscErrorCode TSReset_Python(
    PetscTS ts,
    ) \
    except IERR with gil:
    if getRef(ts) == 0: return 0
    FunctionBegin(b"TSReset_Python")
    #
    CHKERR( PetscObjectCompose(<PetscObject>ts, b"@ts.vec_update", NULL) )
    CHKERR( PetscObjectCompose(<PetscObject>ts, b"@ts.vec_dot",    NULL) )
    #
    cdef reset = PyTS(ts).reset
    if reset is not None:
        reset(TS_(ts))
    return FunctionEnd()

cdef PetscErrorCode TSSetFromOptions_Python(
    PetscOptionItems *PetscOptionsObject,
    PetscTS ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSSetFromOptions_Python")
    cdef char name[2048], *defval = PyTS(ts).getname()
    cdef PetscBool found = PETSC_FALSE
    cdef PetscOptionItems *opts "PetscOptionsObject" = PetscOptionsObject
    CHKERR( PetscOptionsString(
            b"-ts_python_type",b"Python [package.]module[.{class|function}]",
            b"TSPythonSetType",defval,name,sizeof(name),&found) ); <void>opts;
    if found and name[0]:
        CHKERR( TSPythonSetType_PYTHON(ts,name) )
    #
    cdef setFromOptions = PyTS(ts).setFromOptions
    if setFromOptions is not None:
        setFromOptions(TS_(ts))
    CHKERR( SNESSetFromOptions(ts.snes) )
    return FunctionEnd()

cdef PetscErrorCode TSView_Python(
    PetscTS ts,
    PetscViewer vwr,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSView_Python")
    viewcontext(PyTS(ts), vwr)
    cdef view = PyTS(ts).view
    if view is not None:
        view(TS_(ts), Viewer_(vwr))
    return FunctionEnd()

cdef PetscErrorCode TSStep_Python(
    PetscTS   ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSStep_Python")
    cdef step = PyTS(ts).step
    if step is not None:
        step(TS_(ts))
    else:
        TSStep_Python_default(ts)
    return FunctionEnd()

cdef PetscErrorCode TSRollBack_Python(
    PetscTS   ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSRollBack_Python")
    cdef rollback = PyTS(ts).rollback
    if rollback is None: return UNSUPPORTED(b"rollback")
    rollback(TS_(ts))
    return FunctionEnd()

cdef PetscErrorCode TSInterpolate_Python(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSInterpolate _Python")
    cdef interpolate = PyTS(ts).interpolate
    if interpolate is None: return UNSUPPORTED(b"interpolate")
    interpolate(TS_(ts),toReal(t),Vec_(x))
    return FunctionEnd()

cdef PetscErrorCode TSEvaluateStep_Python(
    PetscTS   ts,
    PetscInt  o,
    PetscVec  x,
    PetscBool *flag,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSEvaluateStep _Python")
    cdef evaluatestep = PyTS(ts).evaluatestep
    if evaluatestep is None: return UNSUPPORTED(b"evaluatestep")
    cdef done = evaluatestep(TS_(ts),toInt(o),Vec_(x))
    if flag != NULL:
        flag[0] = PETSC_TRUE if done else PETSC_FALSE
    elif not done:
        return PetscSETERR(PETSC_ERR_USER, "Cannot evaluate step")
    return FunctionEnd()

cdef PetscErrorCode SNESTSFormFunction_Python(
    PetscSNES snes,
    PetscVec  x,
    PetscVec  f,
    PetscTS   ts,
    ) \
    except IERR with gil:
    #
    cdef formSNESFunction = PyTS(ts).formSNESFunction
    if formSNESFunction is not None:
        args = (SNES_(snes),Vec_(x),Vec_(f),TS_(ts))
        formSNESFunction(args)
        return FunctionEnd()
    #
    cdef PetscVec dx = NULL
    CHKERR( PetscObjectQuery(
            <PetscObject>ts,
             b"@ts.vec_dot",
             <PetscObject*>&dx) )
    #
    cdef PetscReal t = ts.ptime + ts.time_step
    cdef PetscReal a = 1.0/ts.time_step
    CHKERR( VecCopy(ts.vec_sol,dx)          )
    CHKERR( VecAXPBY(dx,+a,-a,x)            )
    CHKERR( TSComputeIFunction(ts,t,x,dx,f,PETSC_FALSE) )
    return FunctionEnd()

cdef PetscErrorCode SNESTSFormJacobian_Python(
    PetscSNES snes,
    PetscVec  x,
    PetscMat  A,
    PetscMat  B,
    PetscTS   ts,
    ) \
    except IERR with gil:
    #
    cdef formSNESJacobian = PyTS(ts).formSNESJacobian
    if formSNESJacobian is not None:
        args = (SNES_(snes),Vec_(x),Mat_(A),Mat_(B),TS_(ts))
        formSNESJacobian(*args)
        return FunctionEnd()
    #
    cdef PetscVec dx = NULL
    CHKERR( PetscObjectQuery(
            <PetscObject>ts,
             b"@ts.vec_dot",
             <PetscObject*>&dx) )
    #
    cdef PetscReal t = ts.ptime + ts.time_step
    cdef PetscReal a = 1.0/ts.time_step
    CHKERR( VecCopy(ts.vec_sol,dx)                )
    CHKERR( VecAXPBY(dx,+a,-a,x)                  )
    CHKERR( TSComputeIJacobian(ts,t,x,dx,a,A,B,PETSC_FALSE) )
    return FunctionEnd()

cdef PetscErrorCode TSSolveStep_Python(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSSolveStep_Python")
    #
    cdef solveStep = PyTS(ts).solveStep
    if solveStep is not None:
        solveStep(TS_(ts), <double>t, Vec_(x))
        return FunctionEnd()
    #
    cdef PetscInt nits = 0, lits = 0
    CHKERR( SNESSolve(ts.snes, NULL, x) )
    CHKERR( SNESGetIterationNumber(ts.snes,&nits) )
    CHKERR( SNESGetLinearSolveIterations(ts.snes,&lits) )
    ts.snes_its += nits
    ts.ksp_its  += lits
    return FunctionEnd()

cdef PetscErrorCode TSAdaptStep_Python(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscReal *nextdt,
    PetscBool *stepok,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSAdaptStep_Python")
    nextdt[0] = ts.time_step
    stepok[0] = PETSC_TRUE
    #
    cdef adaptStep = PyTS(ts).adaptStep
    if adaptStep is None: return FunctionEnd()
    cdef object retval
    cdef double dt
    cdef bint   ok
    retval = adaptStep(TS_(ts), <double>t, Vec_(x))
    if retval is None: pass
    elif isinstance(retval, float):
        dt = retval
        nextdt[0] = <PetscReal>dt
        stepok[0] = PETSC_TRUE
    elif isinstance(retval, bool):
        ok = retval
        nextdt[0] = ts.time_step
        stepok[0] = PETSC_TRUE if ok else PETSC_FALSE
    else:
        dt, ok = retval
        nextdt[0] = <PetscReal>dt
        stepok[0] = PETSC_TRUE if ok else PETSC_FALSE
    return FunctionEnd()

cdef PetscErrorCode TSStep_Python_default(
    PetscTS ts,
    ) \
    except IERR with gil:
    FunctionBegin(b"TSStep_Python_default")
    cdef PetscVec vec_update = NULL
    CHKERR( PetscObjectQuery(
            <PetscObject>ts,
             b"@ts.vec_update",
             <PetscObject*>&vec_update) )
    #
    cdef PetscInt  r = 0
    cdef PetscReal tt = ts.ptime
    cdef PetscReal dt = ts.time_step
    cdef PetscBool accept  = PETSC_TRUE
    cdef PetscBool stageok = PETSC_TRUE
    for r from 0 <= r < ts.max_reject:
        tt = ts.ptime + ts.time_step
        CHKERR( VecCopy(ts.vec_sol,vec_update) )
        CHKERR( TSPreStage(ts,tt+dt) )
        TSSolveStep_Python(ts,tt,vec_update)
        CHKERR( TSPostStage(ts,tt+dt,0,&vec_update) );
        CHKERR( TSAdaptCheckStage(ts.adapt,ts,tt+dt,vec_update,&stageok) );
        if not stageok:
            ts.reject += 1
            continue
        TSAdaptStep_Python(ts,tt,vec_update,&dt,&accept)
        if not accept:
            ts.time_step = dt
            ts.reject += 1
            continue
        CHKERR( VecCopy(vec_update,ts.vec_sol) )
        ts.ptime += ts.time_step
        ts.time_step = dt
        break
    if (not stageok or not accept) and ts.reason == 0:
        ts.reason = TS_DIVERGED_STEP_REJECTED
    return FunctionEnd()

# --------------------------------------------------------------------

cdef PetscErrorCode PetscPythonMonitorSet_Python(
    PetscObject obj_p,
    const_char *url_p,
    ) \
    except IERR with gil:
    FunctionBegin(b"PetscPythonMonitorSet_Python")
    assert obj_p != NULL
    assert url_p != NULL
    assert url_p[0] != 0
    #
    cdef PetscClassId classid = 0
    CHKERR( PetscObjectGetClassId(obj_p,&classid) )
    cdef type klass = PyPetscType_Lookup(classid)
    cdef Object ob = klass()
    ob.obj[0] = newRef(obj_p)
    #
    cdef url = bytes2str(url_p)
    if ':' in url:
        path, names = parse_url(url)
    else:
        path, names = url, 'monitor'
    module = load_module(path)
    for attr in names.split(','):
        monitor = getattr(module, attr)
        if isinstance(monitor, type):
            monitor = monitor(ob)
        ob.setMonitor(monitor)
    #
    return FunctionEnd()

# --------------------------------------------------------------------

cdef extern from * nogil:

  char* MATPYTHON  '"python"'
  char* KSPPYTHON  '"python"'
  char* PCPYTHON   '"python"'
  char* SNESPYTHON '"python"'
  char* TSPYTHON   '"python"'

  ctypedef PetscErrorCode MatCreateFunction  (PetscMat)  except IERR
  ctypedef PetscErrorCode PCCreateFunction   (PetscPC)   except IERR
  ctypedef PetscErrorCode KSPCreateFunction  (PetscKSP)  except IERR
  ctypedef PetscErrorCode SNESCreateFunction (PetscSNES) except IERR
  ctypedef PetscErrorCode TSCreateFunction   (PetscTS)   except IERR

  PetscErrorCode MatRegister  (char[],MatCreateFunction* )
  PetscErrorCode PCRegister   (char[],PCCreateFunction*  )
  PetscErrorCode KSPRegister  (char[],KSPCreateFunction* )
  PetscErrorCode SNESRegister (char[],SNESCreateFunction*)
  PetscErrorCode TSRegister   (char[],TSCreateFunction*  )

  PetscErrorCode (*PetscPythonMonitorSet_C) \
      (PetscObject, const_char[]) except IERR


cdef public PetscErrorCode PetscPythonRegisterAll() except IERR:
    FunctionBegin(b"PetscPythonRegisterAll")

    # Python subtypes
    CHKERR( MatRegister ( MATPYTHON,  MatCreate_Python  ) )
    CHKERR( PCRegister  ( PCPYTHON,   PCCreate_Python   ) )
    CHKERR( KSPRegister ( KSPPYTHON,  KSPCreate_Python  ) )
    CHKERR( SNESRegister( SNESPYTHON, SNESCreate_Python ) )
    CHKERR( TSRegister  ( TSPYTHON,   TSCreate_Python   ) )

    # Python monitors
    global PetscPythonMonitorSet_C
    PetscPythonMonitorSet_C = PetscPythonMonitorSet_Python

    return FunctionEnd()

# --------------------------------------------------------------------

# Cython 0.16 - Function prototypes for unused entries are not emitted
# despite them being marked as 'public'. The lack of explicit extern
# "C" binding causes trouble with C++ builds of PETSc. The bogus calls
# below trick Cython to treat all these entries as used.

if PETSC_FALSE:
    <void>import_libpetsc4py()
    <void>MatPythonGetContext(NULL,NULL)
    <void>MatPythonSetContext(NULL,NULL)
    <void>PCPythonGetContext(NULL,NULL)
    <void>PCPythonSetContext(NULL,NULL)
    <void>KSPPythonGetContext(NULL,NULL)
    <void>KSPPythonSetContext(NULL,NULL)
    <void>SNESPythonGetContext(NULL,NULL)
    <void>SNESPythonSetContext(NULL,NULL)
    <void>TSPythonGetContext(NULL,NULL)
    <void>TSPythonSetContext(NULL,NULL)
    <void>PetscPythonRegisterAll()

# --------------------------------------------------------------------

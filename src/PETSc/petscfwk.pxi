# -----------------------------------------------------------------------------

cdef extern from "petscsys.h" nogil:

    struct _p_PetscFwk
    ctypedef _p_PetscFwk *PetscFwk
    int PetscFwkCall(PetscFwk, char[])
    int PetscFwkGetURL(PetscFwk, char**)
    int PetscFwkSetURL(PetscFwk, char[])
    #
    int PetscFwkCreate(MPI_Comm,PetscFwk*)
    int PetscFwkView(PetscFwk,PetscViewer)
    int PetscFwkRegisterComponent(PetscFwk,char[])
    int PetscFwkRegisterComponentURL(PetscFwk,char[],char[])
    int PetscFwkRegisterDependence(PetscFwk,char[],char[])
    int PetscFwkGetComponent(PetscFwk,char[],PetscFwk*,PetscTruth*)
    int PetscFwkVisit(PetscFwk, char[])
    int PetscFwkDestroy(PetscFwk)
    PetscFwk PETSC_FWK_DEFAULT_(MPI_Comm)


# -----------------------------------------------------------------------------


cdef dict fwk_cache = {}
__fwk_cache__ = fwk_cache

cdef extern from "Python.h":
    object PyModule_New(char *)
    void Py_XINCREF(object o)
    void Py_XDECREF(object o)

cdef int Fwk_Call(
    PetscFwk         pcomponent,
    const_char       *pmessage,
    void             *pvtable
    ) except PETSC_ERR_PYTHON with gil:
    #
    assert pcomponent != NULL
    assert pmessage   != NULL
    #
    cdef Fwk component = <Fwk> Fwk()
    PetscIncref(<PetscObject>pcomponent)
    component.fwk     = pcomponent
    #
    cdef message = bytes2str(pmessage)
    #
    cdef object klass = <object> pvtable
    #
    cdef func = None
    try:
        func = getattr(klass, message)
        func(component)
    except AttributeError:
        try:
            func = getattr(klass, "call")
            func(component,message)
        except AttributeError:
            raise AttributeError("Fwk '%s' has no suitable func in vtable '%s' to respond to message '%s'" % (component.getName(), str(klass), message))
    return 0


cdef int Fwk_SetVTable(
    PetscFwk          component_p,
    const_char        *path_p, 
    const_char        *name_p,
    void              **vtable_p
    ) except PETSC_ERR_PYTHON with gil:
    #
    assert path_p != NULL
    assert name_p != NULL
    #
    cdef str path = bytes2str(path_p)
    cdef str name = bytes2str(name_p)
    #
    cdef module = fwk_cache.get(path)
    if module is None:
        module = PyModule_New("__petsc__")
        module.__file__    = path
        module.__package__ = None
        fwk_cache[path] = module
        try:
            source = open(path, 'rU').read()
            code = compile(source, path, 'exec')
            namespace = module.__dict__
            exec code in namespace
        except:
            del fwk_cache[path]
            raise
    #
    cdef klass = None
    try:
        klass = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            "Cannot load class %s() from file '%s'"
            % (name, path))
    #
    Py_XINCREF(klass)
    vtable_p[0] = <void*>klass
    return 0

cdef int Fwk_ClearVTable(
     PetscFwk component_p, 
     void     **vtable_p
     ) except PETSC_ERR_PYTHON with gil:
    #
    cdef object klass = <object> vtable_p[0]
    Py_XDECREF(klass)
    vtable_p[0] = NULL
    return 0


cdef extern from "Python.h":
    ctypedef struct PyObject
    PyObject *Py_None
    PyObject *PyErr_Occurred()
    void PyErr_Fetch(PyObject**,PyObject**,PyObject**)
    void PyErr_NormalizeException(PyObject**,PyObject**,PyObject**)
    void PyErr_Display(PyObject*,PyObject*,PyObject*)
    void PyErr_Restore(PyObject*,PyObject*,PyObject*)

cdef int Fwk_PrintError() with gil:
    if PyErr_Occurred() == NULL: return 0
    cdef PyObject *exc=NULL,*val=NULL,*tb=NULL
    PyErr_Fetch(&exc,&val,&tb)
    PyErr_NormalizeException(&exc,&val,&tb)
    PyErr_Display(exc if exc != NULL else Py_None,
                  val if val != NULL else Py_None,
                  tb  if tb  != NULL else Py_None)
    PyErr_Restore(exc,val,tb)
    return 0

# -----------------------------------------------------------------------------

cdef extern from *:

    ctypedef int (*PetscFwkPythonCallFunction)(
        PetscFwk, const_char[], void *
        ) except PETSC_ERR_PYTHON with gil

    ctypedef int (*PetscFwkPythonSetVTableFunction)(
        PetscFwk, const_char[], const_char[], void**
        ) except PETSC_ERR_PYTHON with gil

    ctypedef int (*PetscFwkPythonClearVTableFunction)(
        PetscFwk, void**
        )  except PETSC_ERR_PYTHON with gil

    ctypedef int (*PetscFwkPythonPrintErrorFunction)(
        ) with gil

    cdef PetscFwkPythonCallFunction \
        PetscFwkPythonCall

    cdef PetscFwkPythonSetVTableFunction \
        PetscFwkPythonSetVTable

    cdef PetscFwkPythonClearVTableFunction \
        PetscFwkPythonClearVTable

    cdef PetscFwkPythonPrintErrorFunction \
        PetscFwkPythonPrintError

PetscFwkPythonCall          = Fwk_Call
PetscFwkPythonClearVTable   = Fwk_ClearVTable
PetscFwkPythonSetVTable     = Fwk_SetVTable
PetscFwkPythonPrintError    = Fwk_PrintError

# -----------------------------------------------------------------------------

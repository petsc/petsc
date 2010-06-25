# -----------------------------------------------------------------------------

cdef extern from "petscsys.h" nogil:

    struct _p_PetscFwk
    ctypedef _p_PetscFwk *PetscFwk
    int PetscFwkCreate(MPI_Comm,PetscFwk*)
    int PetscFwkDestroy(PetscFwk)
    int PetscFwkRegisterComponent(PetscFwk,char[])
    int PetscFwkRegisterDependence(PetscFwk,char[],char[])
    int PetscFwkGetComponent(PetscFwk,char[],PetscObject*,PetscTruth*)
    int PetscFwkConfigure(PetscFwk,PetscInt)
    int PetscFwkViewConfigurationOrder(PetscFwk,PetscViewer)
    PetscFwk PETSC_FWK_DEFAULT_(MPI_Comm)

# -----------------------------------------------------------------------------

cdef dict fwk_cache = {}
__fwk_cache__ = fwk_cache

cdef extern from "Python.h":
    object PyModule_New(char *)

cdef int Fwk_ImportConfigure(
    const_char_p url_p,
    const_char_p path_p, 
    const_char_p name_p,
    void         **configure_p,
    ) except PETSC_ERR_PYTHON with gil:
    #
    assert url_p != NULL
    assert path_p != NULL
    assert name_p != NULL
    assert configure_p != NULL
    #
    cdef str url  = cp2str(url_p)
    cdef str path = cp2str(path_p) + '.py'
    cdef str name = 'PetscFwkConfigure'+cp2str(name_p)
    #
    cdef module = fwk_cache.get(path)
    if module is None:
        module = PyModule_New("__petsc__")
        module.__file__ = path
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
    cdef configure = None
    try:
        configure = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            "Cannot load configuration function %s() from file '%s'"
            % (name, path))
    #
    configure_p[0] = <void*>configure
    return 0

cdef int Fwk_ComponentConfigure(
    void        *pconfigure,
    PetscFwk    pfwk,
    PetscInt    pstate, 
    PetscObject *pcomponent,
    ) except PETSC_ERR_PYTHON with gil:
    #
    assert pconfigure != NULL
    assert pfwk != NULL
    #
    cdef configure = <object> pconfigure
    cdef PetscInt state = asInt(pstate)
    cdef Fwk fwk = <Fwk> Fwk()
    PetscIncref(<PetscObject>pfwk)
    fwk.fwk = pfwk
    #
    cdef PetscClassId classid = 0
    cdef Object component = None
    cdef type klass = None
    if pcomponent != NULL:
        if pcomponent[0] != NULL:
            CHKERR( PetscObjectGetClassId(pcomponent[0], &classid) )
            klass = TypeRegistryGet(classid)
            component = klass()
            PetscIncref(pcomponent[0])
            component.obj[0] = pcomponent[0]

    cdef object result = configure(fwk, state, component)
    if result is not None:
        component = result
    
    if pcomponent != NULL:
        if component is not None:
            PetscIncref(component.obj[0])
            PetscDecref(pcomponent[0])
            pcomponent[0] = component.obj[0]
    #
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

    ctypedef int (*PetscFwkPythonImportConfigureFunction)(
        const_char_p, const_char_p, const_char_p, void **,
        ) nogil except PETSC_ERR_PYTHON

    ctypedef int (*PetscFwkPythonConfigureComponentFunction)(
        void*, PetscFwk, PetscInt, PetscObject*,
        ) nogil except PETSC_ERR_PYTHON

    ctypedef int (*PetscFwkPythonPrintErrorFunction)(
        ) nogil

    cdef PetscFwkPythonImportConfigureFunction \
        PetscFwkPythonImportConfigure

    cdef PetscFwkPythonConfigureComponentFunction \
        PetscFwkPythonConfigureComponent

    cdef PetscFwkPythonPrintErrorFunction \
        PetscFwkPythonPrintError

PetscFwkPythonImportConfigure    = Fwk_ImportConfigure
PetscFwkPythonConfigureComponent = Fwk_ComponentConfigure
PetscFwkPythonPrintError         = Fwk_PrintError

# -----------------------------------------------------------------------------

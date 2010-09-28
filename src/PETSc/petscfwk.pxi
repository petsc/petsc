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
    int PetscFwkGetComponent(PetscFwk,char[],PetscFwk*,PetscBool*)
    int PetscFwkGetParent(PetscFwk,PetscFwk*)
    int PetscFwkVisit(PetscFwk, char[])
    int PetscFwkDestroy(PetscFwk)
    PetscFwk PETSC_FWK_DEFAULT_(MPI_Comm)


# -----------------------------------------------------------------------------

cdef inline object ref_Fwk(PetscFwk fwk):
    cdef Fwk ob = <Fwk> Fwk()
    PetscIncref(<PetscObject>fwk)
    ob.fwk = fwk
    return ob

cdef dict fwk_cache = {}
__fwk_cache__ = fwk_cache

cdef extern from "Python.h":
    object PyModule_New(char *)
    void Py_XINCREF(object o)
    void Py_XDECREF(object o)

cdef int Fwk_Call(
    PetscFwk   component_p,
    const_char *message_p,
    void       *vtable_p,
    ) except PETSC_ERR_PYTHON with gil:
    assert component_p != NULL
    assert message_p   != NULL
    assert vtable_p    != NULL
    #
    cdef Fwk component = <Fwk> ref_Fwk(component_p)
    cdef object vtable = <object> vtable_p
    cdef object message = bytes2str(message_p)
    #
    cdef function = None
    try:
        function = getattr(vtable, message)
    except AttributeError:
        vtable(component, message)
    else:
        function(component)
    return 0

cdef int Fwk_LoadVTable(
    PetscFwk   component_p,
    const_char *path_p, 
    const_char *name_p,
    void       **vtable_p,
    ) except PETSC_ERR_PYTHON with gil:
    assert component_p != NULL
    assert path_p      != NULL
    assert name_p      != NULL
    assert vtable_p    != NULL
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
    cdef object vtable = getattr(module, name)
    Py_XINCREF(vtable)
    vtable_p[0] = <void*>vtable
    return 0

cdef int Fwk_ClearVTable(
    PetscFwk component_p, 
    void     **vtable_p,
    ) except PETSC_ERR_PYTHON with gil:
    assert component_p != NULL
    assert vtable_p    != NULL
    #
    Py_XDECREF(<object>vtable_p[0])
    vtable_p[0] = NULL
    return 0

# -----------------------------------------------------------------------------

cdef extern from *:

    ctypedef int (*PetscFwkPythonCallFunction)(
        PetscFwk, const_char[], void *
        ) except PETSC_ERR_PYTHON with gil

    ctypedef int (*PetscFwkPythonLoadVTableFunction)(
        PetscFwk, const_char[], const_char[], void**
        ) except PETSC_ERR_PYTHON with gil

    ctypedef int (*PetscFwkPythonClearVTableFunction)(
        PetscFwk, void**
        )  except PETSC_ERR_PYTHON with gil

    cdef PetscFwkPythonCallFunction \
        PetscFwkPythonCall

    cdef PetscFwkPythonLoadVTableFunction \
        PetscFwkPythonLoadVTable

    cdef PetscFwkPythonClearVTableFunction \
        PetscFwkPythonClearVTable

PetscFwkPythonCall          = Fwk_Call
PetscFwkPythonClearVTable   = Fwk_ClearVTable
PetscFwkPythonLoadVTable    = Fwk_LoadVTable

# -----------------------------------------------------------------------------

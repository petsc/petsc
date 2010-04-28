# -----------------------------------------------------------------------------

cdef extern from "petscsys.h" nogil:

    int PetscFwkCreate(MPI_Comm,PetscFwk*)
    int PetscFwkDestroy(PetscFwk)
    int PetscFwkRegisterComponent(PetscFwk,char[])
    int PetscFwkRegisterDependence(PetscFwk,char[],char[])
    int PetscFwkGetComponent(PetscFwk,char[],PetscObject*,PetscTruth*)
    int PetscFwkConfigure(PetscFwk,PetscInt)
    int PetscFwkViewConfigurationOrder(PetscFwk,PetscViewer)

# -----------------------------------------------------------------------------

cdef inline Fwk ref_Fwk(PetscFwk fwk):
    cdef Fwk ob = <Fwk> Fwk()
    PetscIncref(<PetscObject>fwk)
    ob.fwk = fwk
    return ob

# -----------------------------------------------------------------------------

cdef dict Fwk_ConfigureCache = {}

cdef int Fwk_ImportConfigure(
    const_char_p url_p,
    const_char_p path_p, 
    const_char_p name_p,
    void         **configure_p,
    ) except PETSC_ERR_PYTHON with gil:
    configure_p[0] = NULL
    url  = cp2str(url_p)
    #print 'url = ' + url + ', path = ' + path + ', name = ' + name
    if url in Fwk_ConfigureCache:
        #print 'Found url ' + url + ' in cache'
        configure_p[0] = <void*> Fwk_ConfigureCache[url]
        return 0
    #
    import sys, os
    path = cp2str(path_p)
    name = cp2str(name_p)
    #print 'url ' + url + ' not in cache'
    pathpieces = path.split('/')
    #print 'pathpieces = ' + str(pathpieces)
    modname = pathpieces[-1]
    #print 'modname = ' + modname
    #print 'new path pieces = ' + str(pathpieces[:-1])
    path = os.path.join(*pathpieces[:-1])
    if path[0] is not '/':
        path = os.path.relpath(path)
    if path not in sys.path:
        sys.path.insert(-1,path)
    #print 'path = ' + path
    mod = __import__(modname)
    confname = 'PetscFwkConfigure'+name
    if not hasattr(mod, confname):
        raise AttributeError("No configuration method " + confname + 
                             " in module " + modname)
    configure = getattr(mod, confname)
    #print ('Found configure ' + str(configure) + ' with name ' +
    #       confname + ' in module ' + str(mod) + ' with name ' + modname)
    Fwk_ConfigureCache[url] = configure
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
    configure = <object> pconfigure
    #
    assert pfwk != NULL
    cdef Fwk fwk = ref_Fwk(pfwk)
    cdef PetscInt state = asInt(pstate)
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

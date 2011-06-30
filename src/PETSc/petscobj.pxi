# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef int PetscClassId

    int PetscObjectView(PetscObject,PetscViewer)
    int PetscObjectDestroy(PetscObject*)
    int PetscObjectGetReference(PetscObject,PetscInt*)
    int PetscObjectReference(PetscObject)
    int PetscObjectDereference(PetscObject)

    int PetscObjectSetOptionsPrefix(PetscObject,char[])
    int PetscObjectGetOptionsPrefix(PetscObject,char*[])
    int PetscObjectSetFromOptions(PetscObject)

    int PetscObjectGetComm(PetscObject,MPI_Comm*)
    int PetscObjectGetClassId(PetscObject,PetscClassId*)
    int PetscObjectGetType(PetscObject,char*[])
    int PetscObjectGetClassName(PetscObject,char*[])
    int PetscObjectSetName(PetscObject,char[])
    int PetscObjectGetName(PetscObject,char*[])

    int PetscTypeCompare(PetscObject,char[],PetscBool*)
    int PetscObjectCompose(PetscObject,char[],PetscObject)
    int PetscObjectQuery(PetscObject,char[],PetscObject*)

# --------------------------------------------------------------------

cdef inline int PetscINCREF(PetscObject *obj) nogil:
    if obj    == NULL: return 0
    if obj[0] == NULL: return 0
    return PetscObjectReference(obj[0])

cdef inline int PetscCLEAR(PetscObject* obj) nogil:
    if obj    == NULL: return 0
    if obj[0] == NULL: return 0
    cdef PetscObject tmp
    tmp = obj[0]; obj[0] = NULL
    return PetscObjectDestroy(&tmp)

cdef inline int PetscDEALLOC(PetscObject* obj) nogil:
    if obj    == NULL: return 0
    if obj[0] == NULL: return 0
    cdef PetscObject tmp
    tmp = obj[0]; obj[0] = NULL
    if not (<int>PetscInitializeCalled): return 0
    if     (<int>PetscFinalizeCalled):   return 0
    return PetscObjectDestroy(&tmp)

# --------------------------------------------------------------------

cdef extern from *:
    ctypedef struct PyObject
    void _Py_DecRef"Py_DECREF"(PyObject*)
    PyObject* PyDict_New() except NULL
    PyObject* PyDict_GetItem(PyObject*, PyObject*) except *
    int       PyDict_SetItem(PyObject*, PyObject*, PyObject*) except -1
    int       PyDict_DelItem(PyObject*, PyObject*) except -1

cdef extern from * nogil:
    ctypedef struct _p_PetscObject:
        void *python_context
        int (*python_destroy)(void*)

cdef inline void Py_DecRef(PyObject *ob) with gil:
    _Py_DecRef(ob)

cdef int PetscDelPyDict(void* ptr) nogil:
    if ptr != NULL and Py_IsInitialized():
        Py_DecRef(<PyObject*>ptr)
    return 0

cdef object PetscGetPyDict(PetscObject obj, bint create):
    if obj.python_context != NULL:
        return <object>obj.python_context
    if create:
        obj.python_destroy = PetscDelPyDict
        obj.python_context = <void*>PyDict_New()
        return <object>obj.python_context
    return None

cdef object PetscGetPyObj(PetscObject o, char name[]):
    cdef object dct = PetscGetPyDict(o, False)
    if dct is None: return None
    cdef object key = bytes2str(name)
    cdef PyObject *d = <PyObject*>dct
    cdef PyObject *k = <PyObject*>key
    cdef PyObject *v = NULL
    v = PyDict_GetItem(d, k)
    if v != NULL: return <object>v
    return None

cdef object PetscSetPyObj(PetscObject o, char name[], object p):
    cdef object dct
    if p is not None:
        dct = PetscGetPyDict(o, True)
    else:
        dct = PetscGetPyDict(o, False)
        if dct is None: return 0
    cdef str key = bytes2str(name)
    cdef PyObject *d = <PyObject*>dct
    cdef PyObject *k = <PyObject*>key
    cdef PyObject *v = <PyObject*>p
    PyDict_SetItem(d, k, v)
    if v == <PyObject*>None:
        PyDict_DelItem(d, k)
    return None

# --------------------------------------------------------------------

cdef inline long Object_toFortran(PetscObject o) nogil:
    return <long> o

# --------------------------------------------------------------------

cdef inline type subtype_DM(PetscDM dm):
    cdef type klass = DM
    cdef PetscObject obj = <PetscObject> dm
    if obj == NULL: return klass
    cdef PetscBool match = PETSC_FALSE
    # -- DA --
    CHKERR( PetscTypeCompare(obj, b"da", &match) )
    if match == PETSC_FALSE:
        if (PETSC_VERSION_MAJOR == 3 and
            PETSC_VERSION_MINOR == 1): # petsc-3.1
            CHKERR( PetscTypeCompare(obj, b"da1d", &match) )
            if match == PETSC_FALSE:
                CHKERR( PetscTypeCompare(obj, b"da2d", &match) )
                if match == PETSC_FALSE:
                    CHKERR( PetscTypeCompare(obj, b"da3d", &match) )
        if (PETSC_VERSION_MAJOR == 3 and
            PETSC_VERSION_MINOR == 0): # petsc-3.0
            CHKERR( PetscTypeCompare(obj, b"DA", &match) )
    if match == PETSC_TRUE: klass = DA
    # --------
    return klass

cdef inline type subtype_Object(PetscObject obj):
    cdef type klass = Object
    if obj == NULL: return klass
    cdef PetscClassId classid = 0
    CHKERR( PetscObjectGetClassId(obj,&classid) )
    if classid == PETSC_DM_CLASSID:
        klass = subtype_DM(<PetscDM>obj)
    else:
        klass = PyPetscType_Lookup(classid)
    return klass

# --------------------------------------------------------------------

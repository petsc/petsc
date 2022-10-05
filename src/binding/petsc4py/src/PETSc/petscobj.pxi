# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef int PetscClassId
    ctypedef int PetscObjectState
    int PetscObjectView(PetscObject,PetscViewer)
    int PetscObjectDestroy(PetscObject*)
    int PetscObjectGetReference(PetscObject,PetscInt*)
    int PetscObjectReference(PetscObject)
    int PetscObjectDereference(PetscObject)

    int PetscObjectSetOptionsPrefix(PetscObject,char[])
    int PetscObjectAppendOptionsPrefix(PetscObject,char[])
    int PetscObjectGetOptionsPrefix(PetscObject,char*[])
    int PetscObjectSetFromOptions(PetscObject)
    int PetscObjectViewFromOptions(PetscObject,PetscObject,char[])

    int PetscObjectGetComm(PetscObject,MPI_Comm*)
    int PetscObjectGetClassId(PetscObject,PetscClassId*)
    int PetscObjectGetType(PetscObject,char*[])
    int PetscObjectGetClassName(PetscObject,char*[])
    int PetscObjectSetName(PetscObject,char[])
    int PetscObjectGetName(PetscObject,char*[])

    int PetscObjectStateIncrease(PetscObject)
    int PetscObjectStateSet(PetscObject,PetscObjectState)
    int PetscObjectStateGet(PetscObject,PetscObjectState*)
    int PetscObjectTypeCompare(PetscObject,char[],PetscBool*)
    int PetscObjectCompose(PetscObject,char[],PetscObject)
    int PetscObjectQuery(PetscObject,char[],PetscObject*)

    int PetscObjectIncrementTabLevel(PetscObject,PetscObject,PetscInt)
    int PetscObjectGetTabLevel(PetscObject,PetscInt*)
    int PetscObjectSetTabLevel(PetscObject,PetscInt)

cdef extern from "custom.h" nogil:
    int PetscObjectGetDeviceId(PetscObject,PetscInt*)

cdef extern from "petsc/private/garbagecollector.h" nogil:
    int PetscObjectDelayedDestroy(PetscObject*)

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
    return PetscObjectDelayedDestroy(&tmp)

cdef inline int PetscINCSTATE(PetscObject *obj) nogil:
    if obj    == NULL: return 0
    if obj[0] == NULL: return 0
    return PetscObjectStateIncrease(obj[0])

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
        if dct is None: return None
    cdef str key = bytes2str(name)
    cdef PyObject *d = <PyObject*>dct
    cdef PyObject *k = <PyObject*>key
    cdef PyObject *v = <PyObject*>p
    PyDict_SetItem(d, k, v)
    if v == <PyObject*>None:
        PyDict_DelItem(d, k)
    return None

# --------------------------------------------------------------------

cdef extern from *:
    object PyLong_FromVoidPtr(void*)

cdef inline Py_intptr_t Object_toFortran(PetscObject o) nogil:
    return <Py_intptr_t> o

# --------------------------------------------------------------------

cdef inline type subtype_DM(PetscDM dm):
    cdef PetscObject obj = <PetscObject> dm
    if obj == NULL: return DM
    # ---
    cdef PetscBool match = PETSC_FALSE
    CHKERR( PetscObjectTypeCompare(obj, b"da", &match) )
    if match == PETSC_TRUE: return DMDA
    CHKERR( PetscObjectTypeCompare(obj, b"plex", &match) )
    if match == PETSC_TRUE: return DMPlex
    CHKERR( PetscObjectTypeCompare(obj, b"composite", &match) )
    if match == PETSC_TRUE: return DMComposite
    CHKERR( PetscObjectTypeCompare(obj, b"shell", &match) )
    if match == PETSC_TRUE: return DMShell
    CHKERR( PetscObjectTypeCompare(obj, b"stag", &match) )
    if match == PETSC_TRUE: return DMStag
    CHKERR( PetscObjectTypeCompare(obj, b"swarm", &match) )
    if match == PETSC_TRUE: return DMSwarm
    # ---
    return DM

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

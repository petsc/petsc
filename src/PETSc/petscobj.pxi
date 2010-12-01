# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef int PetscClassId

    int PetscObjectView(PetscObject,PetscViewer)
    int PetscObjectDestroy(PetscObject)
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

    int PetscObjectCompose(PetscObject,char[],PetscObject)
    int PetscObjectQuery(PetscObject,char[],PetscObject*)


cdef extern from "context.h":
    int PetscObjectGetPyDict(PetscObject,PetscBool,void**)
    int PetscObjectSetPyObj(PetscObject,char[],void*)
    int PetscObjectGetPyObj(PetscObject,char[],void**)

# --------------------------------------------------------------------

cdef inline int PetscDEALLOC(PetscObject* obj):
    if obj == NULL: return 0
    cdef PetscObject tmp = obj[0]
    if tmp == NULL: return 0
    obj[0] = NULL ## XXX
    if not (<int>PetscInitializeCalled): return 0
    if (<int>PetscFinalizeCalled): return 0
    return PetscObjectDestroy(tmp)

cdef inline int PetscCLEAR(PetscObject* obj):
    if obj == NULL: return 0
    cdef PetscObject tmp = obj[0]
    if tmp == NULL: return 0
    obj[0] = NULL
    return PetscObjectDestroy(tmp)


cdef inline PetscInt PetscRefct(PetscObject obj):
    cdef PetscInt refct = 0
    if obj != NULL:
        PetscObjectGetReference(obj, &refct)
    return refct

cdef inline int PetscIncref(PetscObject obj):
    if obj != NULL:
        return PetscObjectReference(obj)
    return 0

cdef inline int PetscDecref(PetscObject obj):
    if obj != NULL:
        return PetscObjectDereference(obj)
    return 0

# --------------------------------------------------------------------

cdef inline long Object_toFortran(PetscObject o):
    return <long> o

# --------------------------------------------------------------------

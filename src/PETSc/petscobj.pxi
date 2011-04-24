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


cdef extern from "context.h":
    int PetscObjectGetPyDict(PetscObject,PetscBool,void**)
    int PetscObjectSetPyObj(PetscObject,char[],void*)
    int PetscObjectGetPyObj(PetscObject,char[],void**)

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

cdef inline long Object_toFortran(PetscObject o) nogil:
    return <long> o

# --------------------------------------------------------------------

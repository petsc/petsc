cdef extern from * nogil:
    PetscErrorCode PetscMalloc(size_t,void*)
    PetscErrorCode PetscFree(void*)
    PetscErrorCode PetscMemcpy(void*,void*,size_t)
    PetscErrorCode PetscMemmove(void*,void*,size_t)
    PetscErrorCode PetscMemzero(void*,size_t)
    PetscErrorCode PetscMemcmp(void*,void*,size_t,PetscBool*)
    PetscErrorCode PetscStrallocpy(const char[],char*[])


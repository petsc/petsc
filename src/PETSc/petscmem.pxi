cdef extern from * nogil:
    int PetscMalloc(size_t,void*)
    int PetscFree(void*)
    int PetscMemcpy(void*,void*,size_t)
    int PetscMemmove(void*,void*,size_t)
    int PetscMemzero(void*,size_t)
    int PetscMemcmp(void*,void*,size_t,PetscBool*)
    int PetscStrallocpy(const char[],char*[])


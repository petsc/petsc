#---------------------------------------------------------------------

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void* malloc(size_t) nogil
    void* realloc (void*,size_t) nogil
    void free(void*) nogil

cdef extern from "string.h":
    void* memset(void*,int,size_t) nogil
    void* memcpy(void*,void*,size_t) nogil
    char* strdup(char*) nogil

cdef extern from "Python.h":
    object PyCObject_FromVoidPtr(void *, void (*)(void*))

#---------------------------------------------------------------------

cdef inline void *memnew(size_t n):
    if n == 0: n = 1
    return malloc(n)

cdef inline void memdel(void *p):
    if p != NULL: free(p)

cdef inline object allocate(size_t n, void **pp):
    cdef object ob
    cdef void *p = memnew(n)
    if p == NULL: raise MemoryError
    try:    ob = PyCObject_FromVoidPtr(p, memdel)
    except: memdel(p); raise
    pp[0] = p
    return ob

#---------------------------------------------------------------------

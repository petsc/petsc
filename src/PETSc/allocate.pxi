#---------------------------------------------------------------------

cdef extern from "stdlib.h" nogil:
    void* malloc(size_t)
    void* realloc (void*,size_t)
    void free(void*)

cdef extern from "string.h"  nogil:
    void* memset(void*,int,size_t)
    void* memcpy(void*,void*,size_t)
    char* strdup(char*)

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

# --------------------------------------------------------------------

cdef extern from "stdio.h" nogil:
    int printf(char *, ...)

cdef extern from "Python.h":
    ctypedef struct PyObject
    ctypedef struct PyTypeObject
    ctypedef int visitproc(PyObject *, void *)
    ctypedef int traverseproc(PyObject *, visitproc, void *)
    ctypedef int inquiry(PyObject *)
    ctypedef struct PyTypeObject:
       char*        tp_name
       traverseproc tp_traverse
       inquiry      tp_clear
    ctypedef struct PyGC_Head:
       Py_ssize_t gc_refs"gc.gc_refs"
    PyGC_Head *_Py_AS_GC(PyObject*)
    ## PyTypeObject *Py_TYPE(PyObject *)

cdef int traverse(PyObject *o, visitproc visit, void *arg):
    ## printf("%s.tp_traverse(%p)\n", Py_TYPE(o).tp_name, <void*>o)
    cdef PetscObject p = (<Object>o).obj[0]
    if p == NULL: return 0
    cdef PyObject *dct = NULL
    PetscObjectGetPyDict(p, PETSC_FALSE, <void**>&dct)
    if dct == NULL or dct == <PyObject*>None: return 0
    cdef Py_ssize_t gc_refs = _Py_AS_GC(dct).gc_refs
    if gc_refs == 0: return 0
    return visit(dct, arg)

cdef int clear(PyObject *o):
    ## printf("%s.tp_clear(%p)\n", Py_TYPE(o).tp_name, <void*>o)
    cdef PetscObject *p = (<Object>o).obj
    PetscDEALLOC(p)
    return 0

cdef inline void TypeEnableGC(PyTypeObject *t):
    ## printf("%s: enforcing GC support\n", t.tp_name)
    t.tp_traverse = traverse
    t.tp_clear    = clear

# --------------------------------------------------------------------

# --------------------------------------------------------------------

cdef extern from "stdio.h" nogil:
    int printf(char *, ...)

cdef extern from "Python.h":
    ctypedef struct PyObject
    ctypedef struct PyTypeObject
    PyTypeObject *Py_TYPE(PyObject *)
    ctypedef int visitproc(PyObject *, void *)
    ctypedef int traverseproc(PyObject *, visitproc, void *)
    ctypedef int inquiry(PyObject *)
    ctypedef PyObject *allocfunc(PyTypeObject*, Py_ssize_t)
    ctypedef void freefunc(void *)
    ctypedef struct PyTypeObject:
       char*        tp_name
       long         tp_flags
       traverseproc tp_traverse
       inquiry      tp_clear
       allocfunc    tp_alloc
       freefunc     tp_free
       inquiry      tp_is_gc
    enum: Py_TPFLAGS_HAVE_GC
    PyObject *PyType_GenericAlloc(PyTypeObject *, Py_ssize_t)
    void PyObject_GC_Del(void *)

cdef int traverse(PyObject *o, visitproc visit, void *arg):
    ## printf("%s.tp_traverse(%p)\n", Py_TYPE(o).tp_name, <void*>o)
    cdef PetscObject p = (<Object>o).obj[0]
    cdef void *dct = NULL
    cdef int vret = 0
    if not p: return 0
    PetscObjectGetPyDict(p, PETSC_FALSE, &dct)
    if not dct: return 0
    return visit(<PyObject*>dct, arg)

cdef int clear(PyObject *o):
    ## printf("%s.tp_clear(%p)\n", Py_TYPE(o).tp_name, <void*>o)
    cdef PetscObject *p = (<Object>o).obj
    PetscDEALLOC(p)
    return 0

cdef void TypeEnableGC(PyTypeObject *t):
    ## printf("%s: enforcing GC support\n", t.tp_name)
    # this is required
    t.tp_traverse = traverse
    t.tp_clear    = clear
    # and this should not
    t.tp_flags   |= Py_TPFLAGS_HAVE_GC
    t.tp_alloc    = PyType_GenericAlloc
    t.tp_free     = PyObject_GC_Del
    t.tp_is_gc    = NULL

# --------------------------------------------------------------------

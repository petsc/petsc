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
    PyTypeObject *Py_TYPE(PyObject *)

cdef int tp_traverse(PyObject *o, visitproc visit, void *arg):
    ## printf("%s.tp_traverse(%p)\n", Py_TYPE(o).tp_name, <void*>o)
    cdef PetscObject p = (<Object>o).obj[0]
    if p == NULL: return 0
    cdef PyObject *d = <PyObject*>p.python_context
    if d == NULL: return 0
    return visit(d, arg)

cdef int tp_clear(PyObject *o):
    ## printf("%s.tp_clear(%p)\n", Py_TYPE(o).tp_name, <void*>o)
    cdef PetscObject *p = (<Object>o).obj
    PetscDEALLOC(p)
    return 0

cdef inline void TypeEnableGC(PyTypeObject *t):
    ## printf("%s: enforcing GC support\n", t.tp_name)
    t.tp_traverse = tp_traverse
    t.tp_clear    = tp_clear

# --------------------------------------------------------------------

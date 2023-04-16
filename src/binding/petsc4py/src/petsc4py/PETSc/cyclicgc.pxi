# --------------------------------------------------------------------

cdef extern from * nogil:
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

cdef extern from "<petsc/private/garbagecollector.h>" nogil:
    PetscErrorCode PetscGarbageCleanup(MPI_Comm)
    PetscErrorCode PetscGarbageView(MPI_Comm,PetscViewer);

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

def garbage_cleanup(comm=None):
    """Clean up unused PETSc objects.

    Note
    ----
    If the communicator ``comm`` if not provided or it is `None`,
    first clean up on `COMM_WORLD`, then clean up on `COMM_SELF`.

    """
    if not (<int>PetscInitializeCalled): return
    if (<int>PetscFinalizeCalled):   return
    cdef MPI_Comm ccomm = MPI_COMM_NULL
    if comm is None:
        ccomm = GetComm(COMM_WORLD, MPI_COMM_NULL)
        CHKERR( PetscGarbageCleanup(ccomm) )
        ccomm = GetComm(COMM_SELF, MPI_COMM_NULL)
        CHKERR( PetscGarbageCleanup(ccomm) )
    else:
        ccomm = GetComm(comm, MPI_COMM_NULL)
        if ccomm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        CHKERR( PetscGarbageCleanup(ccomm) )

def garbage_view(comm=None):
    """Print summary of the garbage PETSc objects.

    Note
    ----
    Print out garbage summary on each rank of the communicator ``comm``.
    If no communicator is provided then `COMM_WORLD` is used.

    """
    if not (<int>PetscInitializeCalled): return
    if (<int>PetscFinalizeCalled):   return
    cdef MPI_Comm ccomm = MPI_COMM_NULL
    if comm is None:
        comm = COMM_WORLD
    ccomm = GetComm(comm, MPI_COMM_NULL)
    if ccomm == MPI_COMM_NULL:
        raise ValueError("null communicator")
    CHKERR( PetscGarbageView(ccomm, NULL) )

# --------------------------------------------------------------------

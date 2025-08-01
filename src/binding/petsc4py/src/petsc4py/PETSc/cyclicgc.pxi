# --------------------------------------------------------------------

cdef extern from * nogil:
    int printf(char *, ...)

cdef extern from "Python.h":
    """
    #if defined(Py_LIMITED_API)
    #define _pytype_enable_gc(t, traverse, clear) \
    do { (void)(traverse); (void)(clear); } while (0)
    #else
    #define _pytype_enable_gc(t, traverse, clear) \
    do { (t)->tp_traverse = (traverse); (t)->tp_clear = (clear); } while (0)
    #endif
    """
    ctypedef struct PyObject
    ctypedef struct PyTypeObject
    ctypedef int visitproc(PyObject *, void *) noexcept
    ctypedef int traverseproc(PyObject *, visitproc, void *) noexcept
    ctypedef int inquiry(PyObject *) noexcept
    void _pytype_enable_gc(PyTypeObject *, traverseproc, inquiry)

cdef extern from "<petsc/private/garbagecollector.h>" nogil:
    PetscErrorCode PetscGarbageCleanup(MPI_Comm)
    PetscErrorCode PetscGarbageView(MPI_Comm, PetscViewer)

cdef int tp_traverse(PyObject *o, visitproc visit, void *arg) noexcept:
    cdef PetscObject p = (<Object>o).obj[0]
    if p == NULL: return 0
    cdef PyObject *d = <PyObject*>p.python_context
    if d == NULL: return 0
    return visit(d, arg)

cdef int tp_clear(PyObject *o) noexcept:
    cdef PetscObject *p = (<Object>o).obj
    PetscDEALLOC(p)
    return 0

cdef inline void TypeEnableGC(PyTypeObject *t) noexcept:
    _pytype_enable_gc(t, tp_traverse, tp_clear)


def garbage_cleanup(comm: Comm | None = None) -> None:
    """Clean up unused PETSc objects.

    Collective.

    Notes
    -----
    If the communicator ``comm`` if not provided or it is `None`,
    then `COMM_WORLD` is used.

    """
    if not (<int>PetscInitializeCalled): return
    if (<int>PetscFinalizeCalled): return
    cdef MPI_Comm ccomm = MPI_COMM_NULL
    if comm is None:
        ccomm = GetComm(COMM_WORLD, MPI_COMM_NULL)
        CHKERR(PetscGarbageCleanup(ccomm))
    else:
        ccomm = GetComm(comm, MPI_COMM_NULL)
        if ccomm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        CHKERR(PetscGarbageCleanup(ccomm))


def garbage_view(comm: Comm | None = None) -> None:
    """Print summary of the garbage PETSc objects.

    Collective.

    Notes
    -----
    Print out garbage summary on each rank of the communicator ``comm``.
    If no communicator is provided then `COMM_WORLD` is used.

    """
    if not (<int>PetscInitializeCalled): return
    if (<int>PetscFinalizeCalled): return
    cdef MPI_Comm ccomm = MPI_COMM_NULL
    if comm is None:
        comm = COMM_WORLD
    ccomm = GetComm(comm, MPI_COMM_NULL)
    if ccomm == MPI_COMM_NULL:
        raise ValueError("null communicator")
    CHKERR(PetscGarbageView(ccomm, NULL))

# --------------------------------------------------------------------

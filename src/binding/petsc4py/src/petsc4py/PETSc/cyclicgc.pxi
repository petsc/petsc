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
    #if PY_VERSION_HEX < 0x030B0000 && !defined(Py_Version)
    #define Py_Version __Pyx_get_runtime_version()
    #endif
    """
    const unsigned long Py_Version
    enum: Py_TPFLAGS_HEAPTYPE
    ctypedef struct PyObject
    ctypedef struct PyTypeObject
    void Py_VISIT(void*) noexcept
    PyTypeObject *Py_TYPE(PyObject *) noexcept
    unsigned long PyType_GetFlags(PyTypeObject *type) noexcept
    ctypedef int (*visitproc)(PyObject *, void *) noexcept
    ctypedef int (*traverseproc)(PyObject *, visitproc, void *) noexcept
    ctypedef int (*inquiry)(PyObject *) noexcept
    void _pytype_enable_gc(PyTypeObject *, traverseproc, inquiry)

cdef extern from "<petsc/private/garbagecollector.h>" nogil:
    PetscErrorCode PetscGarbageCleanup(MPI_Comm)
    PetscErrorCode PetscGarbageView(MPI_Comm, PetscViewer)

cdef int tp_traverse(PyObject *o, visitproc _visit, void *_arg) noexcept:
    cdef visitproc visit "visit" = _visit
    cdef void *arg "arg" = _arg
    <void> visit
    <void> arg
    if Py_Version >= 0x03090000:
        if not (PyType_GetFlags(Py_TYPE(o)) & Py_TPFLAGS_HEAPTYPE):
            Py_VISIT(Py_TYPE(o))
    cdef PetscObject p = (<Object>o).obj[0]
    if p == NULL: return 0
    cdef PyObject *d = <PyObject*>p.python_context
    if d == NULL: return 0
    Py_VISIT(d)
    return 0

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

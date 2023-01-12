# --------------------------------------------------------------------

cdef extern from * nogil:

    MPI_Comm MPI_COMM_NULL
    MPI_Comm MPI_COMM_SELF
    MPI_Comm MPI_COMM_WORLD

    MPI_Datatype MPI_DATATYPE_NULL
    MPI_Op       MPI_OP_NULL

    enum: MPI_IDENT
    enum: MPI_CONGRUENT
    int MPI_Comm_compare(MPI_Comm,MPI_Comm,int*)

    int MPI_Comm_size(MPI_Comm,int*)
    int MPI_Comm_rank(MPI_Comm,int*)
    int MPI_Barrier(MPI_Comm)

    int MPI_Initialized(int*)
    int MPI_Finalized(int*)

    ctypedef int MPI_Fint
    MPI_Fint MPI_Comm_c2f(MPI_Comm)

cdef extern from * nogil:

    MPI_Comm PETSC_COMM_SELF
    MPI_Comm PETSC_COMM_WORLD

    PetscErrorCode PetscCommDuplicate(MPI_Comm,MPI_Comm*,int*)
    PetscErrorCode PetscCommDestroy(MPI_Comm*)

# --------------------------------------------------------------------

cdef extern from "cython.h":
    void *Cython_ImportFunction(object, char[], char[]) except? NULL

ctypedef MPI_Comm*     PyMPICommGet(object) except NULL
ctypedef object        PyMPICommNew(MPI_Comm)
ctypedef MPI_Datatype* PyMPIDatatypeGet(object) except NULL
ctypedef MPI_Op*       PyMPIOpGet(object) except NULL

cdef inline MPI_Comm mpi4py_Comm_Get(object comm) except *:
    from mpi4py import MPI
    cdef PyMPICommGet *commget = \
        <PyMPICommGet*> Cython_ImportFunction(
        MPI, b"PyMPIComm_Get", b"MPI_Comm *(PyObject *)")
    if commget == NULL: return MPI_COMM_NULL
    cdef MPI_Comm *ptr = commget(comm)
    if ptr == NULL: return MPI_COMM_NULL
    return ptr[0]

cdef inline object mpi4py_Comm_New(MPI_Comm comm):
    from mpi4py import MPI
    cdef PyMPICommNew *commnew = \
        <PyMPICommNew*> Cython_ImportFunction(
        MPI, b"PyMPIComm_New", b"PyObject *(MPI_Comm)")
    if commnew == NULL: return None
    return commnew(comm)

cdef inline MPI_Datatype mpi4py_Datatype_Get(object datatype) except *:
    from mpi4py import MPI
    cdef PyMPIDatatypeGet *datatypeget = \
        <PyMPIDatatypeGet*> Cython_ImportFunction(
        MPI, b"PyMPIDatatype_Get", b"MPI_Datatype *(PyObject *)")
    if datatypeget == NULL: return MPI_DATATYPE_NULL
    cdef MPI_Datatype *ptr = datatypeget(datatype)
    if ptr == NULL: return MPI_DATATYPE_NULL
    return ptr[0]

cdef inline MPI_Op mpi4py_Op_Get(object op) except *:
    from mpi4py import MPI
    cdef PyMPIOpGet *opget = \
        <PyMPIOpGet*> Cython_ImportFunction(
        MPI, b"PyMPIOp_Get", b"MPI_Op *(PyObject *)")
    if opget == NULL: return MPI_OP_NULL
    cdef MPI_Op *ptr = opget(op)
    if ptr == NULL: return MPI_OP_NULL
    return ptr[0]

# --------------------------------------------------------------------

cdef inline PetscErrorCode PetscCommDEALLOC(MPI_Comm* comm):
    if comm == NULL: return PETSC_SUCCESS
    cdef MPI_Comm tmp = comm[0]
    if tmp == MPI_COMM_NULL: return PETSC_SUCCESS
    comm[0] = MPI_COMM_NULL
    if not (<int>PetscInitializeCalled): return PETSC_SUCCESS
    if (<int>PetscFinalizeCalled): return PETSC_SUCCESS
    return PetscCommDestroy(&tmp)

cdef inline MPI_Comm def_Comm(object comm, MPI_Comm defv) except *:
    cdef MPI_Comm retv = MPI_COMM_NULL
    if comm is None:
        retv = defv
    elif isinstance(comm, Comm):
        retv = (<Comm>comm).comm
    elif type(comm).__module__ == 'mpi4py.MPI':
        retv = mpi4py_Comm_Get(comm)
    else:
        retv = (<Comm?>comm).comm
    return retv

cdef inline Comm new_Comm(MPI_Comm comm):
    cdef Comm ob = <Comm> Comm()
    ob.comm = comm
    return ob

# --------------------------------------------------------------------

cdef inline int comm_size(MPI_Comm comm) except ? -1:
    if comm == MPI_COMM_NULL: raise ValueError("null communicator")
    cdef int size = 0
    CHKERR( <PetscErrorCode>MPI_Comm_size(comm, &size) )
    return size

cdef inline int comm_rank(MPI_Comm comm) except ? -1:
    if comm == MPI_COMM_NULL: raise ValueError("null communicator")
    cdef int rank = 0
    CHKERR( <PetscErrorCode>MPI_Comm_rank(comm, &rank) )
    return rank

# --------------------------------------------------------------------

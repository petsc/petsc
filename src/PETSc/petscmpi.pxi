# --------------------------------------------------------------------

cdef extern from "mpi.h":

    MPI_Comm MPI_COMM_NULL
    MPI_Comm MPI_COMM_SELF
    MPI_Comm MPI_COMM_WORLD

    enum: MPI_IDENT
    enum: MPI_CONGRUENT
    int MPI_Comm_compare(MPI_Comm,MPI_Comm,int*)

    int MPI_Comm_size(MPI_Comm,int*)
    int MPI_Comm_rank(MPI_Comm,int*)
    int MPI_Barrier(MPI_Comm)

    int MPI_Initialized(int*)
    int MPI_Finalized(int*)


cdef extern from "petsc.h":

    MPI_Comm PETSC_COMM_NULL "MPI_COMM_NULL"
    MPI_Comm PETSC_COMM_SELF
    MPI_Comm PETSC_COMM_WORLD

    int PetscCommDuplicate(MPI_Comm,MPI_Comm*,int*)
    int PetscCommDestroy(MPI_Comm*)

# --------------------------------------------------------------------

cdef inline Comm new_Comm(MPI_Comm comm):
    cdef Comm ob = <Comm> Comm()
    ob.comm = comm
    return ob

cdef inline MPI_Comm def_Comm(object comm,
                              MPI_Comm deft) except *:
    if comm is None: return deft
    return (<Comm?>comm).comm

# --------------------------------------------------------------------

cdef inline int comm_size(MPI_Comm comm) except ? -1:
    if comm == MPI_COMM_NULL: raise ValueError("null communicator")
    cdef int size = 0
    CHKERR( MPI_Comm_size(comm, &size) )
    return size

cdef inline int comm_rank(MPI_Comm comm) except ? -1:
    if comm == MPI_COMM_NULL: raise ValueError("null communicator")
    cdef int rank = 0
    CHKERR( MPI_Comm_rank(comm, &rank) )
    return rank

# --------------------------------------------------------------------

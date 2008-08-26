# --------------------------------------------------------------------

cdef class Comm:

    #

    def __cinit__(self, comm=None):
        self.comm = def_Comm(comm, MPI_COMM_NULL)
        self.isdup = 0

    def __dealloc__(self):
        if self.isdup:
            if self.comm != MPI_COMM_NULL:
                CHKERR( PetscCommDestroy(&self.comm) )
        self.comm = MPI_COMM_NULL
        self.isdup = 0

    def __richcmp__(Comm self, Comm other, int op):
        if op!=2 and op!=3: raise TypeError("only '==' and '!='")
        cdef int eq = (op == 2)
        cdef MPI_Comm comm1 = self.comm
        cdef MPI_Comm comm2 = other.comm
        cdef int flag = 0
        if comm1 != MPI_COMM_NULL and comm2 != MPI_COMM_NULL:
            MPI_Comm_compare(comm1, comm2, &flag)
            if eq: return (flag==MPI_IDENT or  flag==MPI_CONGRUENT)
            else:  return (flag!=MPI_IDENT and flag!=MPI_CONGRUENT)
        else:
            if eq: return (comm1 == comm2)
            else:  return (comm1 != comm2)

    def __nonzero__(self):
        return self.comm != MPI_COMM_NULL

    def __bool__(self):
        return self.comm != MPI_COMM_NULL

    #

    def destroy(self):
        if self.comm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        if not self.isdup:
            raise ValueError("communicator not owned")
        CHKERR( PetscCommDestroy(&self.comm) )
        self.comm = MPI_COMM_NULL
        self.isdup = 0

    def duplicate(self):
        if self.comm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        cdef Comm comm = type(self)()
        CHKERR( PetscCommDuplicate(self.comm, &comm.comm, NULL) )
        comm.isdup = 1
        return comm

    def getSize(self):
        if self.comm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        cdef int size=0
        MPI_Comm_size(self.comm, &size)
        return size

    def getRank(self):
        if self.comm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        cdef int rank=0
        MPI_Comm_rank(self.comm, &rank)
        return rank

    def barrier(self):
        if self.comm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        MPI_Barrier(self.comm)

    #

    property size:
        def __get__(self):
            return self.getSize()

    property rank:
        def __get__(self):
            return self.getRank()

    # --- mpi4py compatibility API ---

    Free     = destroy
    Clone    = duplicate
    Dup      = duplicate
    Get_size = getSize
    Get_rank = getRank
    Barrier  = barrier

# --------------------------------------------------------------------

COMM_NULL  = Comm()
COMM_SELF  = Comm()
COMM_WORLD = Comm()

# --------------------------------------------------------------------

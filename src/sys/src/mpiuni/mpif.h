C     Trying to provide as little support for fortran code in petsc as needed

c     External objects outside of MPI calls 
       integer MPI_COMM_WORLD
       parameter (MPI_COMM_WORLD = 1)
       integer MPI_COMM_SELF
       parameter (MPI_COMM_SELF = 2)
       integer  MPI_COMM_NULL
       parameter ( MPI_COMM_NULL = 0 )
       integer MPI_SUCCESS 
       parameter (MPI_SUCCESS = 0 )
       integerMPI_IDENT 
       parameter (MPI_IDENT = 0  )
       integer MPI_UNEQUAL 
       parameter (MPI_UNEQUAL = 3  )
       integer MPI_KEYVAL_INVALID
       parameter (MPI_KEYVAL_INVALID = 0 )
       integer MPI_ERR_UNKNOWN
       parameter (MPI_ERR_UNKNOWN = 18 )
       integer MPI_ERR_EXHAUSTED
       parameter (MPI_ERR_EXHAUSTED = 1 )
       integer MPI_ERR_INTERN 
       parameter (MPI_ERR_INTERN = 21 )
       
       INTEGER MPI_SOURCE, MPI_TAG, MPI_ERROR
       PARAMETER(MPI_SOURCE=2, MPI_TAG=3, MPI_ERROR=4)

c     External types 
#define  MPI_Request integer;
#define  MPI_Group integer;
#define  MPI_Errhandler integer;
       double precision MPI_Wtime

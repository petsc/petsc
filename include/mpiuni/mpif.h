!
!

!     Trying to provide as little support for fortran code in petsc as needed

!     External objects outside of MPI calls 
       integer MPI_COMM_WORLD
       parameter (MPI_COMM_WORLD = 1)
       integer MPI_COMM_SELF
       parameter (MPI_COMM_SELF = 2)
       integer  MPI_COMM_NULL
       parameter (MPI_COMM_NULL = 0)
       integer MPI_SUCCESS 
       parameter (MPI_SUCCESS = 0)
       integer MPI_IDENT 
       parameter (MPI_IDENT = 0)
       integer MPI_UNEQUAL 
       parameter (MPI_UNEQUAL = 3)
       integer MPI_KEYVAL_INVALID
       parameter (MPI_KEYVAL_INVALID = 0)
       integer MPI_ERR_UNKNOWN
       parameter (MPI_ERR_UNKNOWN = 18)
       integer MPI_ERR_INTERN 
       parameter (MPI_ERR_INTERN = 21)
       integer MPI_SUM
       parameter (MPI_SUM=0)
       integer MPI_MAX
       parameter (MPI_MAX=0)
       integer MPI_MIN
       parameter (MPI_MIN=0)

       integer MPI_PACKED
       parameter (MPI_PACKED=0)
       integer MPI_ANY_SOURCE
       parameter (MPI_ANY_SOURCE=0)
       integer MPI_ANY_TAG
       parameter (MPI_ANY_TAG=0)
       integer MPI_STATUS_SIZE
       parameter (MPI_STATUS_SIZE=4)
       integer MPI_UNDEFINED
       parameter (MPI_UNDEFINED=-32766)
       INTEGER MPI_INFO_NULL
       PARAMETER (MPI_INFO_NULL=469762048)


       integer MPI_REQUEST_NULL
       parameter (MPI_REQUEST_NULL=0)
       integer MPI_MINLOC
       parameter (MPI_MINLOC=0)

       INTEGER MPI_SOURCE,MPI_TAG,MPI_ERROR
       PARAMETER(MPI_SOURCE=2,MPI_TAG=3,MPI_ERROR=4)

     
!     Data Types. Same Values used in mpi.c
       integer MPI_INTEGER,MPI_LOGICAL
       integer MPI_REAL,MPI_DOUBLE_PRECISION
       integer MPI_COMPLEX, MPI_CHARACTER
       integer MPI_2INTEGER
       integer MPI_DOUBLE_COMPLEX
       integer MPI_INTEGER4
       integer MPI_INTEGER8
       integer MPI_2DOUBLE_PRECISION

       parameter (MPI_INTEGER=0)
       parameter (MPI_LOGICAL=0)
       parameter (MPI_REAL=1)
       parameter (MPI_DOUBLE_PRECISION=2)
       parameter (MPI_COMPLEX=3)
       parameter (MPI_CHARACTER=4)
       parameter (MPI_2INTEGER=5)
       parameter (MPI_DOUBLE_COMPLEX=6)
       parameter (MPI_INTEGER4=7)
       parameter (MPI_INTEGER8=8)
       parameter (MPI_2DOUBLE_PRECISION=9)

       integer MPI_MAXLOC
       parameter (MPI_MAXLOC=5)

       integer MPI_MAX_PROCESSOR_NAME
       parameter (MPI_MAX_PROCESSOR_NAME=128-1)


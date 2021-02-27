!
!     Trying to provide as little support for fortran code in petsc as needed
!
#include "petsc/mpiuni/mpiunifdef.h"
!
!     External objects outside of MPI calls
       MPIUNI_FInt MPI_COMM_WORLD
       parameter (MPI_COMM_WORLD = 2)
       MPIUNI_FInt MPI_COMM_SELF
       parameter (MPI_COMM_SELF = 1)
       MPIUNI_FInt  MPI_COMM_NULL
       parameter (MPI_COMM_NULL = 0)
       MPIUNI_FInt MPI_IDENT
       parameter (MPI_IDENT = 0)
       MPIUNI_FInt MPI_UNEQUAL
       parameter (MPI_UNEQUAL = 3)
       MPIUNI_FInt MPI_KEYVAL_INVALID
       parameter (MPI_KEYVAL_INVALID = 0)
       MPIUNI_FInt MPI_SUCCESS
       parameter (MPI_SUCCESS = 0)
       MPIUNI_FInt MPI_ERR_OTHER
       parameter (MPI_ERR_OTHER = 17)
       MPIUNI_FInt MPI_ERR_UNKNOWN
       parameter (MPI_ERR_UNKNOWN = 18)
       MPIUNI_FInt MPI_ERR_INTERN
       parameter (MPI_ERR_INTERN = 21)

       MPIUNI_FInt MPI_PACKED
       parameter (MPI_PACKED=0)
       MPIUNI_FInt MPI_ANY_SOURCE
       parameter (MPI_ANY_SOURCE=2)
       MPIUNI_FInt MPI_ANY_TAG
       parameter (MPI_ANY_TAG=-1)
       MPIUNI_FInt MPI_UNDEFINED
       parameter (MPI_UNDEFINED=-32766)
       MPIUNI_FInt MPI_INFO_NULL
       PARAMETER (MPI_INFO_NULL=0)


       MPIUNI_FInt MPI_REQUEST_NULL
       parameter (MPI_REQUEST_NULL=0)

       MPIUNI_FInt MPI_STATUS_SIZE
       parameter (MPI_STATUS_SIZE=3)
       MPIUNI_FInt MPI_SOURCE,MPI_TAG,MPI_ERROR
       PARAMETER(MPI_SOURCE=1,MPI_TAG=2,MPI_ERROR=3)

       MPIUNI_FInt MPI_STATUS_IGNORE
       parameter (MPI_STATUS_IGNORE=0)

!     Data Types. Same Values used in mpi.c
       MPIUNI_FInt MPI_INTEGER,MPI_LOGICAL
       MPIUNI_FInt MPI_REAL,MPI_DOUBLE_PRECISION
       MPIUNI_FInt MPI_COMPLEX, MPI_CHARACTER
       MPIUNI_FInt MPI_COMPLEX16
       MPIUNI_FInt MPI_2INTEGER
       MPIUNI_FInt MPI_DOUBLE_COMPLEX
       MPIUNI_FInt MPI_INTEGER4
       MPIUNI_FInt MPI_INTEGER8
       MPIUNI_FInt MPI_2DOUBLE_PRECISION
       MPIUNI_FInt MPI_REAL4,MPI_REAL8

!
!  These should match the values in mpi.h many below are wrong
!
       parameter (MPI_INTEGER=4194564)
       parameter (MPI_DOUBLE_PRECISION=1048840)
       parameter (MPI_COMPLEX16=2097424)
       parameter (MPI_LOGICAL=INT(Z'400104'))
       parameter (MPI_REAL=INT(Z'100104'))
       parameter (MPI_REAL4=INT(Z'100104'))
       parameter (MPI_REAL8=INT(Z'100108'))
       parameter (MPI_COMPLEX=INT(Z'200108'))
       parameter (MPI_CHARACTER=INT(Z'300101'))
       parameter (MPI_2INTEGER=INT(Z'e00108'))
       parameter (MPI_DOUBLE_COMPLEX=INT(Z'200110'))
       parameter (MPI_INTEGER4=INT(Z'400104'))
       parameter (MPI_INTEGER8=INT(Z'400108'))
       parameter (MPI_2DOUBLE_PRECISION=INT(Z'100208'))

       MPIUNI_FInt MPI_SUM
       parameter (MPI_SUM=1)
       MPIUNI_FInt MPI_MAX
       parameter (MPI_MAX=2)
       MPIUNI_FInt MPI_MIN
       parameter (MPI_MIN=3)
       MPIUNI_FInt MPI_REPLACE
       parameter (MPI_REPLACE=4)
       MPIUNI_FInt MPI_MAXLOC
       parameter (MPI_MAXLOC=12)
       MPIUNI_FInt MPI_MINLOC
       parameter (MPI_MINLOC=13)

       MPIUNI_FInt MPI_MAX_PROCESSOR_NAME
       parameter (MPI_MAX_PROCESSOR_NAME=128-1)



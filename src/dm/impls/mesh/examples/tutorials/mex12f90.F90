!
      module mex12f90
#include "finclude/petsc.h90"

!   Data structure used to contain information about the problem
!   You can add physical values etc here

      type appctx
        MPI_Comm :: comm = MPI_COMM_WORLD

      end type appctx

      end module mex12f90




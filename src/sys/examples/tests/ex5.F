C
C  Simple PETSc Program written in Fortran 77, (shudder)
C
#include "include/finclude/petsc.h"
       integer ierr, mytid

       call PetscInitialize(ierr)

       if (ierr .ne. 0) then 
         print*, "Unable to begin PETSc program"
       endif

       call MPI_Comm_rank(MPI_COMM_WORLD,mytid,ierr)
       print*, "Greetings from", mytid

       call PetscFinalize(ierr)
       end

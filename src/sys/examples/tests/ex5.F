C    "$Id: ex5.F,v 1.3 1995/10/12 04:14:40 bsmith Exp curfman $";
C
C  Simple PETSc Program written in Fortran 77, (shudder)
C
#include "include/finclude/petsc.h"
       integer ierr, rank

       call PetscInitialize(ierr)

       if (ierr .ne. 0) then 
         print*, "Unable to begin PETSc program"
       endif

       call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
       print*, "Greetings from", rank

       call PetscFinalize(ierr)
       end

C    "$Id: ex5.F,v 1.6 1996/03/19 21:24:49 bsmith Exp bsmith $";
C
C  Simple PETSc Program written in Fortran 77.
C
#include "include/FINCLUDE/petsc.h"
       integer        ierr, rank
       character*(80) arch

       call PetscInitialize(PETSC_NULL_CHAR,ierr)

       if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
       endif

       call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
       print*, 'Greetings from', rank

       call PetscGetArchType(arch,ierr)
       print*, 'Machine type ',arch

       call PetscFinalize(ierr)
       end

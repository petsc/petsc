C    "$Id: ex5.F,v 1.7 1996/03/20 03:28:28 bsmith Exp curfman $";
C
C  Simple PETSc Program written in Fortran 77.
C
#include "include/FINCLUDE/petsc.h"
       integer        ierr, rank
       character*(80) arch

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

       if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
       endif

       call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
       print*, 'Greetings from', rank

       call PetscGetArchType(arch,ierr)
       print*, 'Machine type ',arch

       call PetscFinalize(ierr)
       end

!    "$Id: ex5.F,v 1.10 1998/03/24 16:42:29 balay Exp balay $";
!
!  Simple PETSc Program written in Fortran 77.
!
#include "include/finclude/petsc.h"
       integer        ierr, rank
       character*(80) arch

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

       if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
       endif

       call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
       print*, 'Greetings from', rank

       call PetscGetArchType(arch,ierr)
       print*, 'Machine type ',arch

       call PetscFinalize(ierr)
       end

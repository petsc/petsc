!    "$Id: ex5.F,v 1.9 1997/09/22 15:19:00 balay Exp balay $";
!
!  Simple PETSc Program written in Fortran 77.
!
#include "include/FINCLUDE/petsc.h"
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

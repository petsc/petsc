!    "$Id: ex5.F,v 1.13 2000/09/27 20:04:20 bsmith Exp $";
!
!  Simple PETSc Program written in Fortran 77.
!
       program main
#include "include/finclude/petsc.h"
       integer        ierr,rank
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

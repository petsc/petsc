!
!  Simple PETSc Program written in Fortran
!
       program main
#include <petsc/finclude/petscsys.h>
       use petscsys
       implicit none

       PetscErrorCode  ierr
       PetscInt f(1)
       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
       endif

       f(1) = 1
       call MPI_Allreduce(MPI_IN_PLACE,f,1,MPIU_INTEGER,MPI_MIN,PETSC_COMM_WORLD,ierr)
       call PetscFinalize(ierr)
       end

!/*TEST
!
!   test:
!
!TEST*/

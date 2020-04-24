!
!  Simple PETSc Program written in Fortran
!
       program main
#include <petsc/finclude/petscsys.h>
       use petscsys
       implicit none

       PetscErrorCode  ierr
       PetscMPIInt     rank
       character*(80) arch

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
       endif

       call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
       print*, 'Greetings from rank', rank

       call PetscGetArchType(arch,ierr)
       write(6,100) arch
 100   format (' PETSC_ARCH ',A)

       call PetscFinalize(ierr)
       end

!
!/*TEST
!
!   test:
!     filter: grep -v PETSC_ARCH
!
!TEST*/

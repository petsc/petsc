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

       PetscCallA(PetscInitialize(ierr))
       PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
       print*, 'Greetings from rank', rank

       PetscCallA(PetscGetArchType(arch,ierr))
       write(6,100) arch
 100   format (' PETSC_ARCH ',A)

       PetscCallA(PetscFinalize(ierr))
       end

!
!/*TEST
!
!   test:
!     filter: grep -v PETSC_ARCH
!
!TEST*/

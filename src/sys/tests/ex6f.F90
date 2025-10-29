!
!  Tests random number generation routines from Fortran.
!
#include <petsc/finclude/petscsys.h>
program main
  use petscsys
  implicit none

  PetscErrorCode ierr
  PetscRandom r
  PetscScalar rand

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD, r, ierr))
  PetscCallA(PetscRandomSetFromOptions(r, ierr))
  PetscCallA(PetscRandomGetValue(r, rand, ierr))
  print *, 'Random value:', rand
  PetscCallA(PetscRandomDestroy(r, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!
!/*TEST
!
!   test:
!      requires: !complex
!
!TEST*/

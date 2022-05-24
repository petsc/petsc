!
!  Program to test random number generation routines from fortran.
!
      program main

#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscErrorCode  ierr
      PetscRandom     r
      PetscScalar     rand

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,ierr))

      PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD,r,ierr))
      PetscCallA(PetscRandomSetFromOptions(r,ierr))
      PetscCallA(PetscRandomGetValue(r,rand,ierr))
      print*, 'Random value:',rand
      PetscCallA(PetscRandomDestroy(r,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   test:
!      requires: !complex
!
!TEST*/

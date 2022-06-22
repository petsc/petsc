!
!  Program to test PetscRandom, PetscObjectReference() and other PetscObjectXXX functions.
!
      program main

#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscErrorCode  ierr
      PetscRandom     r,q,r2
      PetscScalar     rand
      PetscInt        ref

      PetscCallA(PetscInitialize(ierr))

      PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD,r,ierr))
      PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD,r2,ierr))
      PetscCallA(PetscRandomSetFromOptions(r,ierr))
      PetscCallA(PetscRandomGetValue(r,rand,ierr))
      print*, 'Random value:',rand

      PetscCallA(PetscObjectReference(r,ierr))
      PetscCallA(PetscObjectGetReference(r,ref,ierr))
      print*, 'Reference value:',ref
      PetscCallA(PetscObjectDereference(r,ierr))

      PetscCallA(PetscObjectCompose(r,'test',r2,ierr));
      PetscCallA(PetscObjectQuery(r,'test',q,ierr));
      if (q .ne. r2) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Object compose/query failed'); endif

      PetscCallA(PetscRandomDestroy(r,ierr))
      PetscCallA(PetscRandomDestroy(r2,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   test:
!      requires: !complex
!
!TEST*/

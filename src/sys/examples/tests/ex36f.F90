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

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
      endif

      call PetscRandomCreate(PETSC_COMM_WORLD,r,ierr)
      call PetscRandomCreate(PETSC_COMM_WORLD,r2,ierr)
      call PetscRandomSetFromOptions(r,ierr)
      call PetscRandomGetValue(r,rand,ierr)
      print*, 'Random value:',rand

      call PetscObjectReference(r,ierr)
      call PetscObjectGetReference(r,ref,ierr)
      print*, 'Reference value:',ref
      call PetscObjectDereference(r,ierr)

      call PetscObjectCompose(r,'test',r2,ierr);
      call PetscObjectQuery(r,'test',q,ierr);
      if (q .ne. r2) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Object compose/query failed'); endif

      call PetscRandomDestroy(r,ierr)
      call PetscRandomDestroy(r2,ierr)
      call PetscFinalize(ierr)
      end

!
!/*TEST
!
!   test:
!      requires: !complex
!
!TEST*/

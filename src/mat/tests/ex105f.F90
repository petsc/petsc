!
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      PetscInt i,one,twelve,j
      Mat     m
      PetscScalar value

      PetscCallA(PetscInitialize(ierr))
      twelve = 12
      PetscCallA(MatCreate(PETSC_COMM_WORLD,m,ierr))
      PetscCallA(MatSetSizes(m,PETSC_DECIDE,PETSC_DECIDE,twelve,twelve,ierr))
      PetscCallA(MatSetFromOptions(m,ierr))
      PetscCallA(MatMPIAIJSetPreallocation(m,PETSC_DEFAULT_INTEGER,PETSC_NULL_INTEGER,PETSC_DEFAULT_INTEGER,PETSC_NULL_INTEGER,ierr))

      value = 3.0
      i     = 4
      one   = 1
      PetscCallA(MatSetValuesMPIAIJ(m,one,i,one,i,value,ADD_VALUES,ierr))
      i = 5
      j = 7
      PetscCallA(MatSetValuesMPIAIJ(m,one,i,one,j,value,ADD_VALUES,ierr))
      i = 10
      j = 9
      PetscCallA(MatSetValuesMPIAIJ(m,one,i,one,j,value,ADD_VALUES,ierr))
      PetscCallA(MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr))

      PetscCallA(MatDestroy(m,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!
!TEST*/

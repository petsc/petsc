!
!   This program tests MatGetDiagonal()
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      PetscInt i,one,twelve
      Vec     v
      Mat     m
      PetscScalar value

      PetscCallA(PetscInitialize(ierr))

      twelve = 12
      PetscCallA(MatCreate(PETSC_COMM_SELF,m,ierr))
      PetscCallA(MatSetSizes(m,twelve,twelve,twelve,twelve,ierr))
      PetscCallA(MatSetFromOptions(m,ierr))
      PetscCallA(MatSetUp(m,ierr))

      value = 3.0
      i     = 4
      one   = 1
      PetscCallA(MatSetValues(m,one,i,one,i,value,INSERT_VALUES,ierr))
      PetscCallA(MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr))

      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,twelve,v,ierr))
      PetscCallA(MatGetDiagonal(m,v,ierr))
      PetscCallA(VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(MatDestroy(m,ierr))
      PetscCallA(VecDestroy(v,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!
!TEST*/

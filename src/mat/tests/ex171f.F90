!
!   This program tests MatNullSpaceCreate()
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      MatNullSpace nsp
      Vec     v(1)
      PetscInt nloc,on
      PetscScalar one
      PetscReal norm

      PetscCallA(PetscInitialize(ierr))

      nloc = 12
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,nloc,PETSC_DETERMINE,v(1),ierr))
      one = 1.0
      PetscCallA(VecSet(v(1),one,ierr))
      PetscCallA(VecNormalize(v(1),norm,ierr))
      on = 1
      PetscCallA(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,on,v(1),nsp,ierr))
      PetscCallA(MatNullSpaceDestroy(nsp,ierr))
      PetscCallA(VecDestroy(v(1),ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!
!TEST*/

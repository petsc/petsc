program main

#include <petsc/finclude/petscvec.h>

use petscvec
implicit none

  PetscErrorCode ierr
  Vec   v,s
  PetscInt,parameter ::      n   = 20
  PetscScalar,parameter ::   sone = 1.0
  PetscBool :: flg
  PetscInt,parameter :: zero = 0, one = 1, two = 2

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-n",n,flg,ierr))

  ! Create multi-component vector with 2 components
  PetscCallA(VecCreate(PETSC_COMM_WORLD,v,ierr))
  PetscCallA(VecSetSizes(v,PETSC_DECIDE,n,ierr))
  PetscCallA(VecSetBlockSize(v,two,ierr))
  PetscCallA(VecSetFromOptions(v,ierr))

  ! Create single-component vector
  PetscCallA(VecCreate(PETSC_COMM_WORLD,s,ierr))
  PetscCallA(VecSetSizes(s,PETSC_DECIDE,n/2,ierr))
  PetscCallA(VecSetFromOptions(s,ierr))

  !Set the vectors to entries to a constant value.
  PetscCallA(VecSet(v,sone,ierr))

  !Get the first component from the multi-component vector to the single vector
  PetscCallA(VecStrideGather(v,zero,s,INSERT_VALUES,ierr))

  PetscCallA(VecView(s,PETSC_VIEWER_STDOUT_WORLD,ierr))

  !Put the values back into the second component
  PetscCallA(VecStrideScatter(s,one,v,ADD_VALUES,ierr))

  PetscCallA(VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr))

  ! Free work space.All PETSc objects should be destroyed when they are no longer needed.
  PetscCallA(VecDestroy(v,ierr))
  PetscCallA(VecDestroy(s,ierr))
  PetscCallA(PetscFinalize(ierr))

  end program

!/*TEST
!
!     test:
!       nsize: 2
!       output_file: output/ex12_1.out
!
!TEST*/

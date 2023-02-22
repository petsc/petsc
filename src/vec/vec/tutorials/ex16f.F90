program main
#include <petsc/finclude/petscvec.h>
use petscvec
implicit none

  Vec ::           v,s,r
  Vec,pointer,dimension(:) ::  vecs
  PetscInt :: i,start
  PetscInt :: endd
  PetscInt,parameter :: n = 20, four = 4, two = 2, one = 1
  PetscErrorCode ierr
  PetscScalar  ::  myValue
  PetscBool :: flg

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-n",n,flg,ierr))

  !Create multi-component vector with 2 components
  PetscCallA(VecCreate(PETSC_COMM_WORLD,v,ierr))
  PetscCallA(VecSetSizes(v,PETSC_DECIDE,n,ierr))
  PetscCallA(VecSetBlockSize(v,four,ierr))
  PetscCallA(VecSetFromOptions(v,ierr))

  ! Create double-component vectors

  PetscCallA(VecCreate(PETSC_COMM_WORLD,s,ierr))
  PetscCallA(VecSetSizes(s,PETSC_DECIDE,n/two,ierr))
  PetscCallA(VecSetBlockSize(s,two,ierr))
  PetscCallA(VecSetFromOptions(s,ierr))
  PetscCallA(VecDuplicate(s,r,ierr))
  allocate(vecs(0:2))

  vecs(0) = s
  vecs(1) = r

  !Set the vector values

  PetscCallA(VecGetOwnershipRange(v,start,endd,ierr))
  do i=start,endd-1
     myValue = real(i)
     PetscCallA(VecSetValues(v,one,i,myValue,INSERT_VALUES,ierr))
  end do
  PetscCallA(VecAssemblyBegin(v,ierr));
  PetscCallA(VecAssemblyEnd(v,ierr));

  ! Get the components from the multi-component vector to the other vectors

  PetscCallA(VecStrideGatherAll(v,vecs,INSERT_VALUES,ierr))

  PetscCallA(VecView(s,PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(VecView(r,PETSC_VIEWER_STDOUT_WORLD,ierr))

  PetscCallA(VecStrideScatterAll(vecs,v,ADD_VALUES,ierr))

  PetscCallA(VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr))

  !Free work space.All PETSc objects should be destroyed when they are no longer needed.

  deallocate(vecs)
  PetscCallA(VecDestroy(v,ierr))
  PetscCallA(VecDestroy(s,ierr))
  PetscCallA(VecDestroy(r,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!     test:
!       nsize: 2
!       output_file: output/ex16_1.out
!
!TEST*/

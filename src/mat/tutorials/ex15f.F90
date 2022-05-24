program main

#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>

  use petscvec
  use petscmat

  implicit none

  Mat             :: A
  MatPartitioning :: part
  IS              :: is
  PetscInt        :: r,myStart,myEnd
  PetscInt        :: N = 10
  PetscErrorCode  :: ierr
  PetscScalar,pointer,dimension(:) :: vals
  PetscInt,pointer,dimension(:) :: cols
  PetscBool :: flg
  PetscInt,parameter :: one = 1, two = 2, three = 3

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-N",N,flg,ierr))
  PetscCallA(MatCreate(PETSC_COMM_WORLD, A,ierr))
  PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N,ierr))
  PetscCallA(MatSetFromOptions(A,ierr))
  PetscCallA(MatSeqAIJSetPreallocation(A, three, PETSC_NULL_INTEGER,ierr))
  PetscCallA(MatMPIAIJSetPreallocation(A, three, PETSC_NULL_INTEGER, two, PETSC_NULL_INTEGER,ierr))

  !/* Create a linear mesh */
  PetscCallA(MatGetOwnershipRange(A, myStart, myEnd,ierr))

  do r=myStart,myEnd-1
    if (r == 0) then
     allocate(vals(2))
     vals = 1.0
     allocate(cols(2),source=[r,r+1])
     PetscCallA(MatSetValues(A, one, r, two, cols, vals, INSERT_VALUES,ierr))
     deallocate(cols)
     deallocate(vals)
    else if (r == N-1) then
     allocate(vals(2))
     vals = 1.0
     allocate(cols(2),source=[r-1,r])
     PetscCallA(MatSetValues(A, one, r, two, cols, vals, INSERT_VALUES,ierr))
     deallocate(cols)
     deallocate(vals)
    else
     allocate(vals(3))
     vals = 1.0
     allocate(cols(3),source=[r-1,r,r+1])
     PetscCallA(MatSetValues(A, one, r, three, cols, vals, INSERT_VALUES,ierr))
     deallocate(cols)
     deallocate(vals)
    end if
  end do
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatAssemblyend(A, MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatPartitioningCreate(PETSC_COMM_WORLD, part,ierr))
  PetscCallA(MatPartitioningSetAdjacency(part, A,ierr))
  PetscCallA(MatPartitioningSetFromOptions(part,ierr))
  PetscCallA(MatPartitioningApply(part, is,ierr))
  PetscCallA(ISView(is, PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(ISDestroy(is,ierr))
  PetscCallA(MatPartitioningDestroy(part,ierr))
  PetscCallA(MatDestroy(A,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!   test:
!      nsize: 3
!      requires: parmetis
!      args: -mat_partitioning_type parmetis
!      output_file: output/ex15_1.out
!
!   test:
!      suffix: 2
!      nsize: 3
!      requires: ptscotch
!      args: -mat_partitioning_type ptscotch
!      output_file: output/ex15_2.out
!
!   test:
!      suffix: 3
!      nsize: 4
!      requires: party
!      args: -mat_partitioning_type party
!      output_file: output/ex15_3.out
!
!   test:
!      suffix: 4
!      nsize: 3
!      requires: chaco
!      args: -mat_partitioning_type chaco
!      output_file: output/ex15_4.out
!
!   test:
!      suffix: 5
!      nsize: 3
!      requires: parmetis
!      args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 3 -mat_partitioning_nparts 10 -N 100
!      output_file: output/ex15_5.out
!
!TEST*/

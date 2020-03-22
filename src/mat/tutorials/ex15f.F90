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

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-N",N,flg,ierr);CHKERRA(ierr)
  call MatCreate(PETSC_COMM_WORLD, A,ierr);CHKERRA(ierr)
  call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N,ierr);CHKERRA(ierr)
  call MatSetFromOptions(A,ierr);CHKERRA(ierr)
  call MatSeqAIJSetPreallocation(A, three, PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
  call MatMPIAIJSetPreallocation(A, three, PETSC_NULL_INTEGER, two, PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)

  !/* Create a linear mesh */
  call MatGetOwnershipRange(A, myStart, myEnd,ierr);CHKERRA(ierr)
  
  do r=myStart,myEnd-1
    if (r == 0) then
     allocate(vals(2))
     vals = 1.0
     allocate(cols(2),source=[r,r+1])
     call MatSetValues(A, one, r, two, cols, vals, INSERT_VALUES,ierr);CHKERRA(ierr)
     deallocate(cols)
     deallocate(vals)
    else if (r == N-1) then
     allocate(vals(2))
     vals = 1.0
     allocate(cols(2),source=[r-1,r])
     call MatSetValues(A, one, r, two, cols, vals, INSERT_VALUES,ierr);CHKERRA(ierr)
     deallocate(cols)
     deallocate(vals)
    else 
     allocate(vals(3))
     vals = 1.0
     allocate(cols(3),source=[r-1,r,r+1])
     call MatSetValues(A, one, r, three, cols, vals, INSERT_VALUES,ierr);CHKERRA(ierr)
     deallocate(cols)
     deallocate(vals)
    end if
  end do
  call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatAssemblyend(A, MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatPartitioningCreate(PETSC_COMM_WORLD, part,ierr);CHKERRA(ierr)
  call MatPartitioningSetAdjacency(part, A,ierr);CHKERRA(ierr)
  call MatPartitioningSetFromOptions(part,ierr);CHKERRA(ierr)
  call MatPartitioningApply(part, is,ierr);CHKERRA(ierr)
  call ISView(is, PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  call ISDestroy(is,ierr);CHKERRA(ierr)
  call MatPartitioningDestroy(part,ierr);CHKERRA(ierr)
  call MatDestroy(A,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)
 
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


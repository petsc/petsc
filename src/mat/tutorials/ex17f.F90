
program main
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>

use petscvec
use petscmat

implicit none

  Mat             A
  MatPartitioning   part
  IS              is
  PetscInt   ::     i,m,N
  PetscInt   ::     rstart,rend
  PetscInt,pointer,dimension(:) ::   emptyranks,bigranks,cols
  PetscScalar,pointer,dimension(:) :: vals
  PetscInt :: &
    nbigranks   = 10, &
    nemptyranks = 10
  PetscMPIInt   ::  rank,sizef
  PetscErrorCode  ierr
  PetscBool  set
  PetscInt,parameter :: zero = 0, one = 1, two = 2, three = 3

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,sizef,ierr))

  allocate(emptyranks(nemptyranks))
  allocate(bigranks(nbigranks))

  PetscCallA(PetscOptionsGetIntArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-emptyranks",emptyranks,nemptyranks,set,ierr))
  PetscCallA(PetscOptionsGetIntArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-bigranks",bigranks,nbigranks,set,ierr))

  m = 1
  do i=1,nemptyranks
    if (rank == emptyranks(i)) m = 0
  end do
  do i=1,nbigranks
    if (rank == bigranks(i)) m = 5
  end do

  deallocate(emptyranks)
  deallocate(bigranks)

  PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
  PetscCallA(MatSetsizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE,ierr))
  PetscCallA(MatSetFromOptions(A,ierr))
  PetscCallA(MatSeqAIJSetPreallocation(A,three,PETSC_NULL_INTEGER,ierr))
  PetscCallA(MatMPIAIJSetPreallocation(A,three,PETSC_NULL_INTEGER,two,PETSC_NULL_INTEGER,ierr))
  PetscCallA(MatSeqBAIJSetPreallocation(A,one,three,PETSC_NULL_INTEGER,ierr))
  PetscCallA(MatMPIBAIJSetPreallocation(A,one,three,PETSC_NULL_INTEGER,2,PETSC_NULL_INTEGER,ierr))
  PetscCallA(MatSeqSBAIJSetPreallocation(A,one,two,PETSC_NULL_INTEGER,ierr))
  PetscCallA(MatMPISBAIJSetPreallocation(A,one,two,PETSC_NULL_INTEGER,1,PETSC_NULL_INTEGER,ierr))

  PetscCallA(MatGetSize(A,PETSC_NULL_INTEGER,N,ierr))
  PetscCallA(MatGetOwnershipRange(A,rstart,rend,ierr))

  allocate(cols(0:3))
  allocate(vals(0:3))
  do i=rstart,rend-1

    cols = (/mod((i+N-1),N),i,mod((i+1),N)/)
    vals = [1.0,1.0,1.0]
    PetscCallA(MatSetValues(A,one,i,three,cols,vals,INSERT_VALUES,ierr))
  end do
  deallocate(cols)
  deallocate(vals)
  PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))

  PetscCallA(MatPartitioningCreate(PETSC_COMM_WORLD,part,ierr))
  PetscCallA(MatPartitioningSetAdjacency(part,A,ierr))
  PetscCallA(MatPartitioningSetFromOptions(part,ierr))
  PetscCallA(MatPartitioningApply(part,is,ierr))
  PetscCallA(ISView(is,PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(ISDestroy(is,ierr))
  PetscCallA(MatPartitioningDestroy(part,ierr))
  PetscCallA(MatDestroy(A,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!   test:
!      nsize: 8
!      args: -emptyranks 0,2,4 -bigranks 1,3,7 -mat_partitioning_type average
!      output_file: output/ex17_1.out
!      # cannot test with external package partitioners since they produce different results on different systems
!
!TEST*/

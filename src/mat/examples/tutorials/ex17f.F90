
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

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
    print*,'PetscInitialize failed'
    stop
  endif
      
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)
  call MPI_Comm_size(PETSC_COMM_WORLD,sizef,ierr);CHKERRA(ierr)

  allocate(emptyranks(nemptyranks))
  allocate(bigranks(nbigranks))

  call PetscOptionsGetIntArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-emptyranks",emptyranks,nemptyranks,set,ierr);CHKERRA(ierr)
  call PetscOptionsGetIntArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-bigranks",bigranks,nbigranks,set,ierr);CHKERRA(ierr)
  
  m = 1
  do i=1,nemptyranks
    if (rank == emptyranks(i)) m = 0
  end do
  do i=1,nbigranks
    if (rank == bigranks(i)) m = 5
  end do

  deallocate(emptyranks)
  deallocate(bigranks)

  call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
  call MatSetsizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE,ierr);CHKERRA(ierr)
  call MatSetFromOptions(A,ierr);CHKERRA(ierr)
  call MatSeqAIJSetPreallocation(A,three,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
  call MatMPIAIJSetPreallocation(A,three,PETSC_NULL_INTEGER,two,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
  call MatSeqBAIJSetPreallocation(A,one,three,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
  call MatMPIBAIJSetPreallocation(A,one,three,PETSC_NULL_INTEGER,2,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
  call MatSeqSBAIJSetPreallocation(A,one,two,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
  call MatMPISBAIJSetPreallocation(A,one,two,PETSC_NULL_INTEGER,1,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)

  call MatGetSize(A,PETSC_NULL_INTEGER,N,ierr);CHKERRA(ierr)
  call MatGetOwnershipRange(A,rstart,rend,ierr);CHKERRA(ierr)
  
  allocate(cols(0:3))
  allocate(vals(0:3))
  do i=rstart,rend-1 
    
    cols = (/mod((i+N-1),N),i,mod((i+1),N)/)
    vals = [1.0,1.0,1.0]
    call MatSetValues(A,one,i,three,cols,vals,INSERT_VALUES,ierr);CHKERRA(ierr)
  end do
  deallocate(cols)
  deallocate(vals)
  call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

  call MatPartitioningCreate(PETSC_COMM_WORLD,part,ierr);CHKERRA(ierr)
  call MatPartitioningSetAdjacency(part,A,ierr);CHKERRA(ierr)
  call MatPartitioningSetFromOptions(part,ierr);CHKERRA(ierr)
  call MatPartitioningApply(part,is,ierr);CHKERRA(ierr)
  call ISView(is,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  call ISDestroy(is,ierr);CHKERRA(ierr)
  call MatPartitioningDestroy(part,ierr);CHKERRA(ierr)
  call MatDestroy(A,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)
  
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

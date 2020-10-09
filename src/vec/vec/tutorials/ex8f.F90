!
! Description: Demonstrates using a local ordering to set values into a parallel vector
!
!
!   Concepts: vectors^assembling vectors with local ordering;
!   Processors: n
!

  program main
#include <petsc/finclude/petscvec.h>
  use petscvec

  implicit none

  PetscErrorCode ierr
  PetscMPIInt    rank
  PetscInt    ::   i,ng,rstart,rend,M
  PetscInt, pointer, dimension(:) :: gindices
  PetscScalar, parameter :: sone = 1.0
  Vec   ::         x
  ISLocalToGlobalMapping :: ltog
  PetscInt,parameter :: one = 1

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
    print*,'PetscInitialize failed'
    stop
  endif
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

!
!     Create a parallel vector.
!      - In this case, we specify the size of each processor's local
!        portion, and PETSc computes the global size.  Alternatively,
!        PETSc could determine the vector's distribution if we specify
!        just the global size.
!
  call VecCreate(PETSC_COMM_WORLD,x,ierr);CHKERRA(ierr)
  call VecSetSizes(x,rank+one,PETSC_DECIDE,ierr);CHKERRA(ierr)
  call VecSetFromOptions(x,ierr);CHKERRA(ierr)

  call VecSet(x,sone,ierr);CHKERRA(ierr)

!
!     Set the local to global ordering for the vector. Each processor
!     generates a list of the global indices for each local index. Note that
!     the local indices are just whatever is convenient for a particular application.
!     In this case we treat the vector as lying on a one dimensional grid and
!     have one ghost point on each end of the blocks owned by each processor.
!

  call VecGetSize(x,M,ierr);CHKERRA(ierr)
  call VecGetOwnershipRange(x,rstart,rend,ierr);CHKERRA(ierr)
  ng = rend - rstart + 2
  allocate(gindices(0:ng-1))
  gindices(0) = rstart -1

  do i=0,ng-2
   gindices(i+1) = gindices(i) + 1
  end do

! map the first and last point as periodic

  if (gindices(0) == -1) gindices(0) = M - 1

  if (gindices(ng-1) == M) gindices(ng-1) = 0

  call ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,one,ng,gindices,PETSC_COPY_VALUES,ltog,ierr);CHKERRA(ierr)
  call VecSetLocalToGlobalMapping(x,ltog,ierr);CHKERRA(ierr)
  call ISLocalToGlobalMappingDestroy(ltog,ierr);CHKERRA(ierr)
  deallocate(gindices)

     ! Set the vector elements.
     ! - In this case set the values using the local ordering
     ! - Each processor can contribute any vector entries,
     !   regardless of which processor "owns" them; any nonlocal
     !   contributions will be transferred to the appropriate processor
     !   during the assembly process.
     ! - In this example, the flag ADD_VALUES indicates that all
     !   contributions will be added together.

  do i=0,ng-1
   call VecSetValuesLocal(x,one,i,sone,ADD_VALUES,ierr);CHKERRA(ierr)
  end do

  !
  ! Assemble vector, using the 2-step process:
  ! VecAssemblyBegin(), VecAssemblyEnd()
  ! Computations can be done while messages are in transition
  ! by placing code between these two statements.
  !
  call VecAssemblyBegin(x,ierr);CHKERRA(ierr)
  call VecAssemblyEnd(x,ierr);CHKERRA(ierr)
  !
  ! View the vector; then destroy it.
  !
  call VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  call VecDestroy(x,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)

end program

!/*TEST
!
!     test:
!       nsize: 4
!       output_file: output/ex8_1.out
!
!TEST*/

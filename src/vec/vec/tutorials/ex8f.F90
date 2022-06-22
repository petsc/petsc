!
! Description: Demonstrates using a local ordering to set values into a parallel vector
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

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!
!     Create a parallel vector.
!      - In this case, we specify the size of each processor's local
!        portion, and PETSc computes the global size.  Alternatively,
!        PETSc could determine the vector's distribution if we specify
!        just the global size.
!
  PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
  PetscCallA(VecSetSizes(x,rank+one,PETSC_DECIDE,ierr))
  PetscCallA(VecSetFromOptions(x,ierr))

  PetscCallA(VecSet(x,sone,ierr))

!
!     Set the local to global ordering for the vector. Each processor
!     generates a list of the global indices for each local index. Note that
!     the local indices are just whatever is convenient for a particular application.
!     In this case we treat the vector as lying on a one dimensional grid and
!     have one ghost point on each end of the blocks owned by each processor.
!

  PetscCallA(VecGetSize(x,M,ierr))
  PetscCallA(VecGetOwnershipRange(x,rstart,rend,ierr))
  ng = rend - rstart + 2
  allocate(gindices(0:ng-1))
  gindices(0) = rstart -1

  do i=0,ng-2
   gindices(i+1) = gindices(i) + 1
  end do

! map the first and last point as periodic

  if (gindices(0) == -1) gindices(0) = M - 1

  if (gindices(ng-1) == M) gindices(ng-1) = 0

  PetscCallA(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,one,ng,gindices,PETSC_COPY_VALUES,ltog,ierr))
  PetscCallA(VecSetLocalToGlobalMapping(x,ltog,ierr))
  PetscCallA(ISLocalToGlobalMappingDestroy(ltog,ierr))
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
   PetscCallA(VecSetValuesLocal(x,one,i,sone,ADD_VALUES,ierr))
  end do

  !
  ! Assemble vector, using the 2-step process:
  ! VecAssemblyBegin(), VecAssemblyEnd()
  ! Computations can be done while messages are in transition
  ! by placing code between these two statements.
  !
  PetscCallA(VecAssemblyBegin(x,ierr))
  PetscCallA(VecAssemblyEnd(x,ierr))
  !
  ! View the vector; then destroy it.
  !
  PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(VecDestroy(x,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!     test:
!       nsize: 4
!       output_file: output/ex8_1.out
!
!TEST*/

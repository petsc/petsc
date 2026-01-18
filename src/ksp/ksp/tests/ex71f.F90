!     Contributed by leonardo.mutti01@universitadipavia.it
#include <petsc/finclude/petscksp.h>
program main
  use petscksp
  implicit none

  Mat :: A
  PetscInt, parameter :: M = 16, dof = 1, overlap = 0, NSubx = 4
  PetscInt :: NSub, i, j
  PetscMPIInt :: size
  PetscErrorCode :: ierr
  PetscScalar :: v
  PC :: pc
  IS, pointer :: subdomains_IS(:) => null(), inflated_IS(:) => null()
  PetscViewer :: singleton

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
  PetscCallA(MatCreateFromOptions(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 1_PETSC_INT_KIND, PETSC_DECIDE, PETSC_DECIDE, M**2, M**2, A, ierr))
  do i = 1, M**2
    do j = 1, M**2
      v = i*j
      PetscCallA(MatSetValue(A, i - 1, j - 1, v, INSERT_VALUES, ierr))
    end do
  end do

  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(PCCreate(PETSC_COMM_WORLD, pc, ierr))
  PetscCallA(PCSetOperators(pc, A, A, ierr))
  PetscCallA(PCSetType(pc, PCGASM, ierr))

  PetscCallA(PCGASMCreateSubdomains2D(pc, M, M, NSubx, NSubx, dof, overlap, NSub, subdomains_IS, inflated_IS, ierr))
  PetscCallA(PCGASMSetSubdomains(pc, NSub, subdomains_IS, inflated_IS, ierr))
  PetscCallA(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, singleton, ierr))
  PetscCallA(PetscViewerASCIIPrintf(singleton, 'GASM index sets from this MPI process\n', ierr))
  do i = 1, Nsub
    PetscCallA(ISView(subdomains_IS(i), singleton, ierr))
  end do
  PetscCallA(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, singleton, ierr))
  PetscCallA(PCGASMDestroySubdomains(NSub, subdomains_IS, inflated_IS, ierr))

  if (size == 1) then
    ! this routine only works on one rank
    PetscCallA(PCASMCreateSubdomains2D(M, M, NSubx, NSubx, dof, overlap, NSub, subdomains_IS, inflated_IS, ierr))
    do i = 1, Nsub
      PetscCallA(ISView(subdomains_IS(i), PETSC_VIEWER_STDOUT_SELF, ierr))
    end do
    PetscCallA(PCASMDestroySubdomains(NSub, subdomains_IS, inflated_IS, ierr))
  end if

  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(PCDestroy(pc, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!     suffix: 1
!
!   test:
!     suffix: 2
!     nsize: 2
!
!TEST*/

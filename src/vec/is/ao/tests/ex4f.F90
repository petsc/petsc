!
!    Test AO with on IS with 0 entries - Fortran version of ex4.c
!
#include <petsc/finclude/petscao.h>
program main
  use petscao
  implicit none

  PetscErrorCode ierr
  AO ao
  PetscInt localvert(4), nlocal
  PetscMPIInt rank
  IS is

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))

  if (rank == 0) then
    nlocal = 4
    localvert = [0, 1, 2, 3]
  else
    nlocal = 0
  end if

! Test AOCreateBasic()
  PetscCallA(AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert, PETSC_NULL_INTEGER_ARRAY, ao, ierr))
  PetscCallA(AODestroy(ao, ierr))

! Test AOCreateMemoryScalable()
  PetscCallA(AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert, PETSC_NULL_INTEGER_ARRAY, ao, ierr))
  PetscCallA(AODestroy(ao, ierr))

  PetscCallA(AOCreate(PETSC_COMM_WORLD, ao, ierr))
  PetscCallA(ISCreateStride(PETSC_COMM_WORLD, 1_PETSC_INT_KIND, 0_PETSC_INT_KIND, 1_PETSC_INT_KIND, is, ierr))
  PetscCallA(AOSetIS(ao, is, is, ierr))
  PetscCallA(AOSetType(ao, AOMEMORYSCALABLE, ierr))
  PetscCallA(ISDestroy(is, ierr))
  PetscCallA(AODestroy(ao, ierr))

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!     output_file: output/empty.out
!
!   test:
!      suffix: 2
!      nsize: 2
!      output_file: output/empty.out
!
!TEST*/

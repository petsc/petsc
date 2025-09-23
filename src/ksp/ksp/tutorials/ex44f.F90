program main              !   Solves the linear system  J x = f
#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscdmda.h>
  use petscmpi  ! or mpi or mpi_f08
  use petscksp
  use petscdmda
  implicit none

  Vec x, f
  Mat J
  DM da
  KSP ksp
  PetscErrorCode ierr
  PetscInt eight, one

  eight = 8
  one = 1
  PetscCallA(PetscInitialize(ierr))
  PetscCallA(DMDACreate1d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, eight, one, one, PETSC_NULL_INTEGER_ARRAY, da, ierr))
  PetscCallA(DMSetFromOptions(da, ierr))
  PetscCallA(DMSetUp(da, ierr))
  PetscCallA(DMCreateGlobalVector(da, x, ierr))
  PetscCallA(VecDuplicate(x, f, ierr))
  PetscCallA(DMSetMatType(da, MATAIJ, ierr))
  PetscCallA(DMCreateMatrix(da, J, ierr))

  PetscCallA(ComputeRHS(da, f, ierr))
  PetscCallA(ComputeMatrix(da, J, ierr))

  PetscCallA(KSPCreate(MPI_COMM_WORLD, ksp, ierr))
  PetscCallA(KSPSetOperators(ksp, J, J, ierr))
  PetscCallA(KSPSetFromOptions(ksp, ierr))
  PetscCallA(KSPSolve(ksp, f, x, ierr))

  PetscCallA(MatDestroy(J, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(f, ierr))
  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(DMDestroy(da, ierr))
  PetscCallA(PetscFinalize(ierr))

contains
  subroutine ComputeRHS(da, x, ierr)
    use petscdmda
    implicit none

    DM da
    Vec x
    PetscErrorCode ierr
    PetscInt xs, xm, i, mx
    PetscScalar hx
    PetscScalar, pointer :: xx(:)
    PetscCall(DMDAGetInfo(da, PETSC_NULL_INTEGER, mx, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_DMBOUNDARYTYPE, PETSC_NULL_DMBOUNDARYTYPE, PETSC_NULL_DMBOUNDARYTYPE, PETSC_NULL_DMDASTENCILTYPE, ierr))
    PetscCall(DMDAGetCorners(da, xs, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, xm, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr))
    hx = 1.0_PETSC_REAL_KIND/(mx - 1)
    PetscCall(DMDAVecGetArray(da, x, xx, ierr))
    do i = xs, xs + xm - 1
      xx(i) = i*hx
    end do
    PetscCall(DMDAVecRestoreArray(da, x, xx, ierr))
  end

  subroutine ComputeMatrix(da, J, ierr)
    use petscdmda
    use petscmat
    implicit none

    Mat J
    DM da
    PetscErrorCode ierr
    PetscInt xs, xm, i, mx
    PetscScalar hx, one

    one = 1.0
    PetscCall(DMDAGetInfo(da, PETSC_NULL_INTEGER, mx, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_DMBOUNDARYTYPE, PETSC_NULL_DMBOUNDARYTYPE, PETSC_NULL_DMBOUNDARYTYPE, PETSC_NULL_DMDASTENCILTYPE, ierr))
    PetscCall(DMDAGetCorners(da, xs, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, xm, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr))
    hx = 1.0_PETSC_REAL_KIND/(mx - 1)
    do i = xs, xs + xm - 1
      if ((i == 0) .or. (i == mx - 1)) then
        PetscCall(MatSetValue(J, i, i, one, INSERT_VALUES, ierr))
      else
        PetscCall(MatSetValue(J, i, i - 1, -hx, INSERT_VALUES, ierr))
        PetscCall(MatSetValue(J, i, i + 1, -hx, INSERT_VALUES, ierr))
        PetscCall(MatSetValue(J, i, i, 2*hx, INSERT_VALUES, ierr))
      end if
    end do
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY, ierr))
  end
end program
!/*TEST
!
!   test:
!      args: -ksp_converged_reason
!
!TEST*/

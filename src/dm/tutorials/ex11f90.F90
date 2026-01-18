!     Tests DMDAGetVecGetArray()
#include <petsc/finclude/petscdmda.h>
program main
  use petscdmda
  implicit none

  type(tVec) g
  type(tDM) ada

  PetscScalar, pointer :: x1(:), x2(:, :)
  PetscScalar, pointer :: x3(:, :, :), x4(:, :, :, :)
  PetscErrorCode ierr
  PetscInt, parameter :: m = 5, n = 6, p = 4, s = 1, sw = 1
  PetscInt i, j, k, dof
  PetscInt xs, xl, ys, yl, zs, zl

  PetscInt nen, nel
  PetscInt, pointer :: elements(:)

  PetscInt nfields
  character(80), pointer :: namefields(:)
  IS, pointer :: isfields(:)
  DM, pointer :: dmfields(:)

  PetscCallA(PetscInitialize(ierr))

  dof = 1
  PetscCallA(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, m, dof, sw, PETSC_NULL_INTEGER_ARRAY, ada, ierr))
  PetscCallA(DMSetUp(ada, ierr))
  PetscCallA(DMGetGlobalVector(ada, g, ierr))
  PetscCallA(DMDAGetCorners(ada, xs, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, xl, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr))
  PetscCallA(DMDAVecGetArray(ada, g, x1, ierr))
  do i = xs, xs + xl - 1
    x1(i) = i
  end do
  PetscCallA(DMDAVecRestoreArray(ada, g, x1, ierr))
  PetscCallA(VecView(g, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(DMRestoreGlobalVector(ada, g, ierr))
  PetscCallA(DMDestroy(ada, ierr))

  PetscCallA(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, m, n, PETSC_DECIDE, PETSC_DECIDE, dof, s, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ada, ierr))
  PetscCallA(DMSetUp(ada, ierr))
  PetscCallA(DMGetGlobalVector(ada, g, ierr))
  PetscCallA(DMDAGetCorners(ada, xs, ys, PETSC_NULL_INTEGER, xl, yl, PETSC_NULL_INTEGER, ierr))
  PetscCallA(DMDAVecGetArray(ada, g, x2, ierr))
  do i = xs, xs + xl - 1
    do j = ys, ys + yl - 1
      x2(i, j) = i + j
    end do
  end do
  PetscCallA(DMDAVecRestoreArray(ada, g, x2, ierr))
  PetscCallA(VecView(g, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(DMRestoreGlobalVector(ada, g, ierr))

  PetscCallA(DMDAGetElements(ada, nen, nel, elements, ierr))
  do i = 1, nen*nel
    PetscCheckA(elements(i) >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Error getting DMDA elements')
  end do
  PetscCallA(DMDARestoreElements(ada, nen, nel, elements, ierr))
  PetscCallA(DMDestroy(ada, ierr))

  PetscCallA(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, m, n, p, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, s, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ada, ierr))
  PetscCallA(DMSetUp(ada, ierr))
  PetscCallA(DMGetGlobalVector(ada, g, ierr))
  PetscCallA(DMDAGetCorners(ada, xs, ys, zs, xl, yl, zl, ierr))
  PetscCallA(DMDAVecGetArray(ada, g, x3, ierr))
  do i = xs, xs + xl - 1
    do j = ys, ys + yl - 1
      do k = zs, zs + zl - 1
        x3(i, j, k) = i + j + k
      end do
    end do
  end do
  PetscCallA(DMDAVecRestoreArray(ada, g, x3, ierr))
  PetscCallA(VecView(g, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(DMRestoreGlobalVector(ada, g, ierr))
  PetscCallA(DMDestroy(ada, ierr))

!
!  Same tests but now with DOF > 1, so dimensions of array are one higher
!
  dof = 2
  PetscCallA(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, m, dof, sw, PETSC_NULL_INTEGER_ARRAY, ada, ierr))
  PetscCallA(DMSetUp(ada, ierr))
  PetscCallA(DMGetGlobalVector(ada, g, ierr))
  PetscCallA(DMDAGetCorners(ada, xs, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, xl, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr))
  PetscCallA(DMDAVecGetArray(ada, g, x2, ierr))
  do i = xs, xs + xl - 1
    x2(0, i) = i
    x2(1, i) = -i
  end do
  PetscCallA(DMDAVecRestoreArray(ada, g, x1, ierr))
  PetscCallA(VecView(g, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(DMRestoreGlobalVector(ada, g, ierr))

  ! some testing unrelated to the example
  PetscCallA(DMDASetFieldName(ada, 0_PETSC_INT_KIND, 'Field 0', ierr))
  PetscCallA(DMDASetFieldName(ada, 1_PETSC_INT_KIND, 'Field 1', ierr))
  PetscCallA(DMCreateFieldDecomposition(ada, nfields, namefields, PETSC_NULL_IS_POINTER, PETSC_NULL_DM_POINTER, ierr))
  ! print*,nfields,trim(namefields(1)),trim(namefields(2))
  PetscCallA(DMDestroyFieldDecomposition(ada, nfields, namefields, PETSC_NULL_IS_POINTER, PETSC_NULL_DM_POINTER, ierr))
  PetscCallA(DMCreateFieldDecomposition(ada, nfields, namefields, isfields, dmfields, ierr))
  PetscCallA(DMDestroyFieldDecomposition(ada, nfields, namefields, isfields, dmfields, ierr))

  PetscCallA(DMDestroy(ada, ierr))

  PetscCallA(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, m, n, PETSC_DECIDE, PETSC_DECIDE, dof, s, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ada, ierr))
  PetscCallA(DMSetUp(ada, ierr))
  PetscCallA(DMGetGlobalVector(ada, g, ierr))
  PetscCallA(DMDAGetCorners(ada, xs, ys, PETSC_NULL_INTEGER, xl, yl, PETSC_NULL_INTEGER, ierr))
  PetscCallA(DMDAVecGetArray(ada, g, x3, ierr))
  do i = xs, xs + xl - 1
    do j = ys, ys + yl - 1
      x3(0, i, j) = i + j
      x3(1, i, j) = -(i + j)
    end do
  end do
  PetscCallA(DMDAVecRestoreArray(ada, g, x3, ierr))
  PetscCallA(VecView(g, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(DMRestoreGlobalVector(ada, g, ierr))
  PetscCallA(DMDestroy(ada, ierr))

  dof = 3
  PetscCallA(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, m, n, p, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, s, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ada, ierr))
  PetscCallA(DMSetUp(ada, ierr))
  PetscCallA(DMGetGlobalVector(ada, g, ierr))
  PetscCallA(DMDAGetCorners(ada, xs, ys, zs, xl, yl, zl, ierr))
  PetscCallA(DMDAVecGetArray(ada, g, x4, ierr))
  do i = xs, xs + xl - 1
    do j = ys, ys + yl - 1
      do k = zs, zs + zl - 1
        x4(0, i, j, k) = i + j + k
        x4(1, i, j, k) = -(i + j + k)
        x4(2, i, j, k) = i + j + k
      end do
    end do
  end do
  PetscCallA(DMDAVecRestoreArray(ada, g, x4, ierr))
  PetscCallA(VecView(g, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(DMRestoreGlobalVector(ada, g, ierr))
  PetscCallA(DMDestroy(ada, ierr))

  PetscCallA(PetscFinalize(ierr))
end program

!
!/*TEST
!
!   build:
!     requires: !complex
!
!   test:
!     filter: Error: grep -v "Vec Object" | grep -v "Warning: ieee_inexact is signaling"
!
!TEST*/

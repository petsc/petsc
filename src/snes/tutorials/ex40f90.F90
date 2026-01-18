!
!  Demonstrates use of DMDASNESSetFunctionLocal() from Fortran
!
!    Note: the access to the entries of the local arrays below use the Fortran
!   convention of starting at zero. However calls to MatSetValues()  start at 0.
!   Also note that you will have to map the i,j,k coordinates to the local PETSc ordering
!   before calling MatSetValuesLocal(). Often you will find that using PETSc's default
!   code for computing the Jacobian works fine and you will not need to implement
!   your own FormJacobianLocal().
#include <petsc/finclude/petscsnes.h>
#include <petsc/finclude/petscdmda.h>
module ex40module
  use petscdmda
  implicit none
contains
  subroutine FormFunctionLocal(in, x, f, dummy, ierr)
    PetscInt, intent(in) :: dummy
    DMDALocalInfo in
    PetscScalar, intent(in) :: x(in%DOF, in%GXS + 1:in%GXS + in%GXM, in%GYS + 1:in%GYS + in%GYM)
    PetscScalar, intent(out) :: f(in%DOF, in%XS + 1:in%XS + in%XM, in%YS + 1:in%YS + in%YM)
    PetscErrorCode, intent(out) :: ierr

    f = x**2 - 2.0

    ierr = 0
  end
end module ex40module

program ex40f90
  use petscdmda
  use petscsnes
  use ex40module
  implicit none

  SNES snes
  PetscErrorCode ierr
  DM da
  PetscScalar, parameter :: one = 1.0
  Vec x

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 10_PETSC_INT_KIND, 10_PETSC_INT_KIND, PETSC_DECIDE, PETSC_DECIDE, 2_PETSC_INT_KIND, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, da, ierr))
  PetscCallA(DMSetFromOptions(da, ierr))
  PetscCallA(DMSetUp(da, ierr))

!       Create solver object and associate it with the unknowns (on the grid)

  PetscCallA(SNESCreate(PETSC_COMM_WORLD, snes, ierr))
  PetscCallA(SNESSetDM(snes, da, ierr))

  PetscCallA(DMDASNESSetFunctionLocal(da, INSERT_VALUES, FormFunctionLocal, 0, ierr))
  PetscCallA(SNESSetFromOptions(snes, ierr))

!      Solve the nonlinear system
!
  PetscCallA(DMCreateGlobalVector(da, x, ierr))
  PetscCallA(VecSet(x, one, ierr))
  PetscCallA(SNESSolve(snes, PETSC_NULL_VEC, x, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(SNESDestroy(snes, ierr))
  PetscCallA(DMDestroy(da, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!     args: -snes_monitor_short -snes_view -da_refine 1 -pc_type mg -pc_mg_type full -ksp_type fgmres -pc_mg_galerkin pmat
!     requires: !single
!
!TEST*/

!
!  Demonstrates use of DMDASNESSetFunctionLocal() from Fortran
!
!    Note: the access to the entries of the local arrays below use the Fortran
!   convention of starting at zero. However calls to MatSetValues()  start at 0.
!   Also note that you will have to map the i,j,k coordinates to the local PETSc ordering
!   before calling MatSetValuesLocal(). Often you will find that using PETSc's default
!   code for computing the Jacobian works fine and you will not need to implement
!   your own FormJacobianLocal().

program ex40f90
#include <petsc/finclude/petscsnes.h>
#include <petsc/finclude/petscdmda.h>
  use petscdmda
  use petscsnes
  implicit none

  SNES snes
  PetscErrorCode ierr
  DM da
  PetscInt ten, two, one
  PetscScalar sone
  Vec x
  external FormFunctionLocal

  PetscCallA(PetscInitialize(ierr))
  ten = 10
  one = 1
  two = 2
  sone = 1.0

  PetscCallA(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, ten, ten, PETSC_DECIDE, PETSC_DECIDE, two, one, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, da, ierr))
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
  PetscCallA(VecSet(x, sone, ierr))
  PetscCallA(SNESSolve(snes, PETSC_NULL_VEC, x, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(SNESDestroy(snes, ierr))
  PetscCallA(DMDestroy(da, ierr))
  PetscCallA(PetscFinalize(ierr))
end

subroutine FormFunctionLocal(in, x, f, dummy, ierr)
  use petscdmda
  implicit none
  PetscInt i, j, k, dummy
  DMDALocalInfo in
  PetscScalar x(in%DOF, in%GXS + 1:in%GXS + in%GXM, in%GYS + 1:in%GYS + in%GYM)
  PetscScalar f(in%DOF, in%XS + 1:in%XS + in%XM, in%YS + 1:in%YS + in%YM)
  PetscErrorCode ierr

  do i = in%XS + 1, in%XS + in%XM
    do j = in%YS + 1, in%YS + in%YM
      do k = 1, in%DOF
        f(k, i, j) = x(k, i, j)*x(k, i, j) - 2.0
      end do
    end do
  end do

end

!/*TEST
!
!   test:
!     args: -snes_monitor_short -snes_view -da_refine 1 -pc_type mg -pc_mg_type full -ksp_type fgmres -pc_mg_galerkin pmat
!     requires: !single
!
!TEST*/

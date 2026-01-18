!
!
!     Solves the problem A x - x^3 + 1 = 0 via Picard iteration
!
#include <petsc/finclude/petscsnes.h>
module ex21fmodule
  use petscsnes
  implicit none
  type userctx
    Mat A
  end type userctx
contains
  subroutine FormFunction(snes, x, f, user, ierr)
    SNES snes
    Vec x, f
    type(userctx) user
    PetscErrorCode, intent(out) :: ierr
    PetscInt n
    PetscScalar, pointer :: xx(:), ff(:)

    PetscCallA(MatMult(user%A, x, f, ierr))
    PetscCallA(VecGetArray(f, ff, ierr))
    PetscCallA(VecGetArrayRead(x, xx, ierr))
    PetscCallA(VecGetLocalSize(x, n, ierr))
    ff(1:n) = ff(1:n) - xx(1:n)**4 + 1.0
    PetscCallA(VecRestoreArray(f, ff, ierr))
    PetscCallA(VecRestoreArrayRead(x, xx, ierr))
  end subroutine

! The matrix is constant so no need to recompute it
  subroutine FormJacobian(snes, x, jac, jacb, user, ierr)
    SNES snes
    Vec x
    type(userctx) user
    Mat jac, jacb
    PetscErrorCode, intent(out) :: ierr

    ierr = 0
  end subroutine
end module ex21fmodule

program main
  use petscsnes
  use ex21fmodule
  implicit none
  SNES snes
  PetscErrorCode ierr
  Vec res, x
  type(userctx) user
  PetscScalar val

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(MatCreateSeqAIJ(PETSC_COMM_SELF, 2_PETSC_INT_KIND, 2_PETSC_INT_KIND, 2_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, user%A, ierr))
  val = 2.0
  PetscCallA(MatSetValues(user%A, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], [val], INSERT_VALUES, ierr))
  val = -1.0
  PetscCallA(MatSetValues(user%A, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [1_PETSC_INT_KIND], [val], INSERT_VALUES, ierr))
  val = -1.0
  PetscCallA(MatSetValues(user%A, 1_PETSC_INT_KIND, [1_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], [val], INSERT_VALUES, ierr))
  val = 1.0
  PetscCallA(MatSetValues(user%A, 1_PETSC_INT_KIND, [1_PETSC_INT_KIND], 1_PETSC_INT_KIND, [1_PETSC_INT_KIND], [val], INSERT_VALUES, ierr))
  PetscCallA(MatAssemblyBegin(user%A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(user%A, MAT_FINAL_ASSEMBLY, ierr))

  PetscCallA(MatCreateVecs(user%A, x, res, ierr))

  PetscCallA(SNESCreate(PETSC_COMM_SELF, snes, ierr))
  PetscCallA(SNESSetPicard(snes, res, FormFunction, user%A, user%A, FormJacobian, user, ierr))
  PetscCallA(SNESSetFromOptions(snes, ierr))
  PetscCallA(SNESSolve(snes, PETSC_NULL_VEC, x, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(res, ierr))
  PetscCallA(MatDestroy(user%A, ierr))
  PetscCallA(SNESDestroy(snes, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!     nsize: 1
!     requires: !single
!     args: -snes_monitor -snes_converged_reason -snes_view -pc_type lu
!
!TEST*/

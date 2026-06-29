!
!  Description: Test setting several callback functions from Fortran.
!
#include <petsc/finclude/petsc.h>
module ex2fmodule
  use petscsnes
  implicit none

contains
!
! ------------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameter:
!  f - function vector
!
  subroutine FormFunction(snes, x, f, dummy, ierr)
    SNES snes
    Vec x, f
    PetscErrorCode, intent(out) :: ierr
    integer dummy(*)

!  Declarations for use with local arrays
    PetscScalar, pointer :: lx_v(:), lf_v(:)

!  Get pointers to vector data.
!    - VecGetArray() returns a pointer to the data array.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

    PetscCall(VecGetArrayRead(x, lx_v, ierr))
    PetscCall(VecGetArray(f, lf_v, ierr))

!  Compute function

    lf_v(1) = lx_v(1)*lx_v(1) + lx_v(1)*lx_v(2) - 3.0
    lf_v(2) = lx_v(1)*lx_v(2) + lx_v(2)*lx_v(2) - 6.0

!  Restore vectors

    PetscCall(VecRestoreArrayRead(x, lx_v, ierr))
    PetscCall(VecRestoreArray(f, lf_v, ierr))
  end

! ---------------------------------------------------------------------
!
!  MonitorDummy - Does nothing, used to test setting several callback functions in Fortran
!
  subroutine MonitorDummy(snes, its, norm, mctx, ierr)
    SNES, intent(in)  :: snes
    PetscInt, intent(in)  :: its
    PetscReal, intent(in)  :: norm
    integer, intent(in)  :: mctx
    PetscErrorCode, intent(out) :: ierr
    ierr = 0
  end subroutine MonitorDummy

end module

program main
  use ex2fmodule
  implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     snes        - nonlinear solver
!     x, r        - solution, residual vectors
!     its         - iterations for convergence
!
  SNES snes
  Vec x, r
  PetscErrorCode ierr
  PetscInt its
  PetscMPIInt size
  PetscScalar, parameter :: pfive = 0.5
  character(len=256) :: outputString

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
  PetscCheckA(size == 1, PETSC_COMM_SELF, PETSC_ERR_WRONG_MPI_SIZE, 'Uniprocessor example')

! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(SNESCreate(PETSC_COMM_WORLD, snes, ierr))

  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, 2_PETSC_INT_KIND, x, ierr))
  PetscCallA(VecDuplicate(x, r, ierr))

  PetscCallA(SNESSetFunction(snes, r, FormFunction, 0, ierr))

!  Test setting two more callback functions
  PetscCallA(SNESMonitorSet(snes, MonitorDummy, 0, PETSC_NULL_FUNCTION, ierr))

  PetscCallA(SNESSetFromOptions(snes, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(VecSet(x, pfive, ierr))
  PetscCallA(SNESSolve(snes, PETSC_NULL_VEC, x, ierr))

! View solver converged reason; we could instead use the option -snes_converged_reason
  PetscCallA(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(SNESGetIterationNumber(snes, its, ierr))
  write (outputString, '("Number of SNES iterations = ",i5,"\n")') its
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, outputString, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(r, ierr))
  PetscCallA(SNESDestroy(snes, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      args: -snes_type composite -snes_composite_type additiveoptimal -snes_composite_sneses anderson,nrichardson
!      requires: !single
!
!TEST*/

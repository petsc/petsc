! Time-dependent advection-reaction PDE in 1d. Demonstrates IMEX methods
!
!   u_t + a1*u_x = -k1*u + k2*v + s1
!   v_t + a2*v_x = k1*u - k2*v + s2
!   0 < x < 1
!   a1 = 1, k1 = 10^6, s1 = 0,
!   a2 = 0, k2 = 2*k1, s2 = 1
!
!   Initial conditions:
!   u(x,0) = 1 + s2*x
!   v(x,0) = k0/k1*u(x,0) + s1/k1
!
!   Upstream boundary conditions:
!   u(0,t) = 1-sin(12*t)^4
!

#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petscdmda.h>

module ex22f_modctx
  use petscts
  use petscdm
  implicit none
  type AppCtx
    PetscReal a(2), k(2), s(2)
  end type AppCtx
contains

! Small helper to extract the layout, result uses 1-based indexing.
  subroutine GetLayout(da, mx, xs, xe, gxs, gxe, ierr)
    DM da
    PetscInt mx, xs, xe, gxs, gxe
    PetscErrorCode, intent(out) :: ierr
    PetscInt xm, gxm
    PetscCall(DMDAGetInfo(da, PETSC_NULL_INTEGER, mx, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, PETSC_NULL_ENUM, PETSC_NULL_ENUM, PETSC_NULL_ENUM, PETSC_NULL_ENUM, ierr))
    PetscCall(DMDAGetCorners(da, xs, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, xm, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr))
    PetscCall(DMDAGetGhostCorners(da, gxs, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, gxm, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr))
    xs = xs + 1
    gxs = gxs + 1
    xe = xs + xm - 1
    gxe = gxs + gxm - 1
  end subroutine

  subroutine FormIFunctionLocal(mx, xs, xe, gxs, gxe, x, xdot, f, a, k, s, ierr)
    PetscInt, intent(in) :: mx, xs, xe, gxs, gxe
    PetscScalar, dimension(2, xs:xe), intent(in) :: x, xdot
    PetscScalar, dimension(2, xs:xe), intent(out) :: f
    PetscReal, dimension(2) :: a, k, s
    PetscErrorCode, intent(out) :: ierr

    f(1, :) = xdot(1, :) + k(1)*x(1, :) - k(2)*x(2, :) - s(1)
    f(2, :) = xdot(2, :) - k(1)*x(1, :) + k(2)*x(2, :) - s(2)
    ierr = 0
  end subroutine

  subroutine FormIFunction(ts, t, X, Xdot, F, ctx, ierr)
    TS ts
    type(AppCtx) ctx
    PetscReal t
    Vec X, Xdot, F
    PetscErrorCode, intent(out) :: ierr

    DM da
    PetscInt mx, xs, xe, gxs, gxe
    PetscScalar, pointer :: xx(:), xxdot(:), ff(:)

    PetscCall(TSGetDM(ts, da, ierr))
    PetscCall(GetLayout(da, mx, xs, xe, gxs, gxe, ierr))

! Get access to vector data
    PetscCall(VecGetArrayRead(X, xx, ierr))
    PetscCall(VecGetArrayRead(Xdot, xxdot, ierr))
    PetscCall(VecGetArray(F, ff, ierr))

    PetscCall(FormIFunctionLocal(mx, xs, xe, gxs, gxe, xx, xxdot, ff, ctx%a, ctx%k, ctx%s, ierr))

    PetscCall(VecRestoreArrayRead(X, xx, ierr))
    PetscCall(VecRestoreArrayRead(Xdot, xxdot, ierr))
    PetscCall(VecRestoreArray(F, ff, ierr))
  end subroutine

  subroutine FormRHSFunctionLocal(mx, xs, xe, gxs, gxe, t, x, f, a, k, s, ierr)
    PetscInt, intent(in) :: mx, xs, xe, gxs, gxe
    PetscReal, intent(in) :: t
    PetscScalar x(2, gxs:gxe), f(2, xs:xe)
    PetscReal, dimension(2) :: a, k, s, u0t
    PetscErrorCode ierr
    PetscInt i, j
    PetscReal hx
    PetscReal, parameter :: twelfth = 1._PETSC_REAL_KIND/12._PETSC_REAL_KIND, twothird = 2._PETSC_REAL_KIND/3._PETSC_REAL_KIND

    hx = 1.0_PETSC_REAL_KIND/mx
    ! The Fortran standard only allows positive base for power functions; Nag compiler fails on this
    u0t = [1.0_PETSC_REAL_KIND - abs(sin(12._PETSC_REAL_KIND*t))**4, 0._PETSC_REAL_KIND]
    do i = xs, xe
      do j = 1, 2
        if (i == 1) then
          f(j, i) = a(j)/hx*(u0t(j)/3._PETSC_REAL_KIND + .5_PETSC_REAL_KIND*x(j, i) - x(j, i + 1) + x(j, i + 2)/6._PETSC_REAL_KIND)
        else if (i == 2) then
          f(j, i) = a(j)/hx*(-twelfth*u0t(j) + twothird*x(j, i - 1) - twothird*x(j, i + 1) + twelfth*x(j, i + 2))
        else if (i == mx - 1) then
          f(j, i) = a(j)/hx*(-x(j, i - 2)/6._PETSC_REAL_KIND + x(j, i - 1) - .5_PETSC_REAL_KIND*x(j, i) - x(j, i + 1)/3._PETSC_REAL_KIND)
        else if (i == mx) then
          f(j, i) = a(j)/hx*(-x(j, i) + x(j, i - 1))
        else
          f(j, i) = a(j)/hx*(-twelfth*x(j, i - 2) + twothird*x(j, i - 1) - twothird*x(j, i + 1) + twelfth*x(j, i + 2))
        end if
      end do
    end do
  end subroutine

  subroutine FormRHSFunction(ts, t, X, F, ctx, ierr)
    type(AppCtx) ctx
    TS ts
    PetscReal t
    Vec X, F
    PetscErrorCode, intent(out) :: ierr
    DM da
    Vec Xloc
    PetscInt mx, xs, xe, gxs, gxe
    PetscScalar, pointer :: xx(:), ff(:)

    PetscCall(TSGetDM(ts, da, ierr))
    PetscCall(GetLayout(da, mx, xs, xe, gxs, gxe, ierr))

!     Scatter ghost points to local vector,using the 2-step process
!        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
!     By placing code between these two statements, computations can be
!     done while messages are in transition.
    PetscCall(DMGetLocalVector(da, Xloc, ierr))
    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, Xloc, ierr))
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, Xloc, ierr))

! Get access to vector data
    PetscCall(VecGetArrayRead(Xloc, xx, ierr))
    PetscCall(VecGetArray(F, ff, ierr))

    PetscCall(FormRHSFunctionLocal(mx, xs, xe, gxs, gxe, t, xx, ff, ctx%a, ctx%k, ctx%s, ierr))

    PetscCall(VecRestoreArrayRead(Xloc, xx, ierr))
    PetscCall(VecRestoreArray(F, ff, ierr))
    PetscCall(DMRestoreLocalVector(da, Xloc, ierr))
  end subroutine

! ---------------------------------------------------------------------
!
!  IJacobian - Compute IJacobian = dF/dU + shift*dF/dUdot
!
  subroutine FormIJacobian(ts, t, X, Xdot, shift, J, Jpre, ctx, ierr)
    type(AppCtx) ctx
    TS ts
    PetscReal t, shift
    Vec X, Xdot
    Mat J, Jpre
    PetscErrorCode ierr

    DM da
    PetscInt mx, xs, xe, gxs, gxe
    PetscInt i, row, col
    PetscReal k1, k2
    PetscScalar val(4)

    PetscCall(TSGetDM(ts, da, ierr))
    PetscCall(GetLayout(da, mx, xs, xe, gxs, gxe, ierr))

    k1 = ctx%k(1)
    k2 = ctx%k(2)
    do i = xs, xe
      row = i - gxs
      col = i - gxs
      val = [shift + k1, -k2, -k1, shift + k2]
      PetscCall(MatSetValuesBlockedLocal(Jpre, 1_PETSC_INT_KIND, [row], 1_PETSC_INT_KIND, [col], val, INSERT_VALUES, ierr))
    end do
    PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY, ierr))
    if (J /= Jpre) then
      PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY, ierr))
      PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY, ierr))
    end if
  end subroutine

  subroutine FormInitialSolutionLocal(mx, xs, xe, gxs, gxe, x, a, k, s, ierr)
    PetscInt, intent(in) :: mx, xs, xe, gxs, gxe
    PetscScalar, intent(out) :: x(2, xs:xe)
    PetscReal, dimension(2) :: a, k, s
    PetscErrorCode, intent(out) :: ierr
    PetscInt i
    PetscReal hx, r, ik

    hx = 1._PETSC_REAL_KIND/mx
    do i = xs, xe
      r = i*hx
      if (k(2) /= 0.0) then
        ik = 1._PETSC_REAL_KIND/k(2)
      else
        ik = 1._PETSC_REAL_KIND
      end if
      x(1, i) = 1._PETSC_REAL_KIND + s(2)*r
      x(2, i) = k(1)*ik*x(1, i) + s(2)*ik
    end do
    ierr = 0
  end subroutine

  subroutine FormInitialSolution(ts, X, ctx, ierr)
    type(AppCtx) ctx
    TS ts
    Vec X
    PetscErrorCode, intent(out) :: ierr
    DM da
    PetscInt mx, xs, xe, gxs, gxe
    PetscScalar, pointer :: xx(:)

    PetscCall(TSGetDM(ts, da, ierr))
    PetscCall(GetLayout(da, mx, xs, xe, gxs, gxe, ierr))

! Get access to vector data
    PetscCall(VecGetArray(X, xx, ierr))

    PetscCall(FormInitialSolutionLocal(mx, xs, xe, gxs, gxe, xx, ctx%a, ctx%k, ctx%s, ierr))

    PetscCall(VecRestoreArray(X, xx, ierr))
  end subroutine

end module ex22f_modctx
program main
  use ex22f_modctx
  implicit none

!
!  Create an application context to contain data needed by the
!  application-provided call-back routines, FormJacobian() and
!  FormFunction(). We use a double precision array with six
!  entries, two for each problem parameter a, k, s.
!

  TS ts
  SNES snes
  SNESLineSearch linesearch
  Vec X
  Mat J
  PetscInt mx
  PetscErrorCode ierr
  DM da
  PetscReal, parameter :: ftime = 1.0
  PetscReal dt
  PetscBool flg
  type(AppCtx) ctx

  PetscCallA(PetscInitialize(ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create distributed array (DMDA) to manage parallel grid and vectors
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 11_PETSC_INT_KIND, 2_PETSC_INT_KIND, 2_PETSC_INT_KIND, PETSC_NULL_INTEGER, da, ierr))
  PetscCallA(DMSetFromOptions(da, ierr))
  PetscCallA(DMSetUp(da, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!    Extract global vectors from DMDA
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(DMCreateGlobalVector(da, X, ierr))

! Initialize user application context
! Use zero-based indexing for command line parameters to match ex22.c
  ctx%a(1) = 1.0
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-a0', ctx%a(1), flg, ierr))
  ctx%a(2) = 0.0
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-a1', ctx%a(2), flg, ierr))
  ctx%k(1) = 1000000.0
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-k0', ctx%k(1), flg, ierr))
  ctx%k(2) = 2*ctx%k(1)
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-k1', ctx%k(2), flg, ierr))
  ctx%s(1) = 0.0
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-s0', ctx%s(1), flg, ierr))
  ctx%s(2) = 1.0
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-s1', ctx%s(2), flg, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!    Create timestepping solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(TSCreate(PETSC_COMM_WORLD, ts, ierr))
  PetscCallA(TSSetDM(ts, da, ierr))
  PetscCallA(TSSetType(ts, TSARKIMEX, ierr))
  PetscCallA(TSSetRHSFunction(ts, PETSC_NULL_VEC, FormRHSFunction, ctx, ierr))
  PetscCallA(TSSetIFunction(ts, PETSC_NULL_VEC, FormIFunction, ctx, ierr))
  PetscCallA(DMSetMatType(da, MATAIJ, ierr))
  PetscCallA(DMCreateMatrix(da, J, ierr))
  PetscCallA(TSSetIJacobian(ts, J, J, FormIJacobian, ctx, ierr))

  PetscCallA(TSGetSNES(ts, snes, ierr))
  PetscCallA(SNESGetLineSearch(snes, linesearch, ierr))
  PetscCallA(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC, ierr))

  PetscCallA(TSSetMaxTime(ts, ftime, ierr))
  PetscCallA(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set initial conditions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(FormInitialSolution(ts, X, ctx, ierr))
  PetscCallA(TSSetSolution(ts, X, ierr))
  PetscCallA(VecGetSize(X, mx, ierr))
!  Advective CFL, I don't know why it needs so much safety factor.
  dt = .1_PETSC_REAL_KIND*max(ctx%a(1), ctx%a(2))/mx
  PetscCallA(TSSetTimeStep(ts, dt, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   Set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(TSSetFromOptions(ts, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(TSSolve(ts, X, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(MatDestroy(J, ierr))
  PetscCallA(VecDestroy(X, ierr))
  PetscCallA(TSDestroy(ts, ierr))
  PetscCallA(DMDestroy(da, ierr))
  PetscCallA(PetscFinalize(ierr))
end program
!/*TEST
!
!    test:
!      args: -da_grid_x 200 -ts_arkimex_type 4
!      requires: !single
!      output_file: output/empty.out
!
!TEST*/

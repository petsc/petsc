! Time-dependent advection-reaction PDE in 1d. Demonstrates IMEX methods
!
!   u_t + a1*u_x = -k1*u + k2*v + s1
!   v_t + a2*v_x = k1*u - k2*v + s2
!   0 < x < 1;
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

      program main
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petscdmda.h>
      use petscts
      implicit none

!
!  Create an application context to contain data needed by the
!  application-provided call-back routines, FormJacobian() and
!  FormFunction(). We use a double precision array with six
!  entries, two for each problem parameter a, k, s.
!
      PetscReal user(6)
      integer user_a,user_k,user_s
      parameter (user_a = 0,user_k = 2,user_s = 4)

      external FormRHSFunction,FormIFunction,FormIJacobian
      external FormInitialSolution

      TS             ts
      SNES           snes
      SNESLineSearch linesearch
      Vec            X
      Mat            J
      PetscInt       mx
      PetscErrorCode ierr
      DM             da
      PetscReal      ftime,dt
      PetscReal      one,pone
      PetscInt       im11,i2
      PetscBool      flg

      im11 = 11
      i2   = 2
      one = 1.0
      pone = one / 10

      PetscCallA(PetscInitialize(ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create distributed array (DMDA) to manage parallel grid and vectors
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,im11,i2,i2,PETSC_NULL_INTEGER,da,ierr))
      PetscCallA(DMSetFromOptions(da,ierr))
      PetscCallA(DMSetUp(da,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!    Extract global vectors from DMDA;
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(DMCreateGlobalVector(da,X,ierr))

! Initialize user application context
! Use zero-based indexing for command line parameters to match ex22.c
      user(user_a+1) = 1.0
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-a0',user(user_a+1),flg,ierr))
      user(user_a+2) = 0.0
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-a1',user(user_a+2),flg,ierr))
      user(user_k+1) = 1000000.0
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-k0',user(user_k+1),flg,ierr))
      user(user_k+2) = 2*user(user_k+1)
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-k1', user(user_k+2),flg,ierr))
      user(user_s+1) = 0.0
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s0',user(user_s+1),flg,ierr))
      user(user_s+2) = 1.0
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s1',user(user_s+2),flg,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!    Create timestepping solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(TSCreate(PETSC_COMM_WORLD,ts,ierr))
      PetscCallA(TSSetDM(ts,da,ierr))
      PetscCallA(TSSetType(ts,TSARKIMEX,ierr))
      PetscCallA(TSSetRHSFunction(ts,PETSC_NULL_VEC,FormRHSFunction,user,ierr))
      PetscCallA(TSSetIFunction(ts,PETSC_NULL_VEC,FormIFunction,user,ierr))
      PetscCallA(DMSetMatType(da,MATAIJ,ierr))
      PetscCallA(DMCreateMatrix(da,J,ierr))
      PetscCallA(TSSetIJacobian(ts,J,J,FormIJacobian,user,ierr))

      PetscCallA(TSGetSNES(ts,snes,ierr))
      PetscCallA(SNESGetLineSearch(snes,linesearch,ierr))
      PetscCallA(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC,ierr))

      ftime = 1.0
      PetscCallA(TSSetMaxTime(ts,ftime,ierr))
      PetscCallA(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set initial conditions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(FormInitialSolution(ts,X,user,ierr))
      PetscCallA(TSSetSolution(ts,X,ierr))
      PetscCallA(VecGetSize(X,mx,ierr))
!  Advective CFL, I don't know why it needs so much safety factor.
      dt = pone * max(user(user_a+1),user(user_a+2)) / mx;
      PetscCallA(TSSetTimeStep(ts,dt,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   Set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(TSSetFromOptions(ts,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(TSSolve(ts,X,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(MatDestroy(J,ierr))
      PetscCallA(VecDestroy(X,ierr))
      PetscCallA(TSDestroy(ts,ierr))
      PetscCallA(DMDestroy(da,ierr))
      PetscCallA(PetscFinalize(ierr))
      end program

! Small helper to extract the layout, result uses 1-based indexing.
      subroutine GetLayout(da,mx,xs,xe,gxs,gxe,ierr)
      use petscdmda
      implicit none

      DM da
      PetscInt mx,xs,xe,gxs,gxe
      PetscErrorCode ierr
      PetscInt xm,gxm
      PetscCall(DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(da,gxs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,gxm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      xs = xs + 1
      gxs = gxs + 1
      xe = xs + xm - 1
      gxe = gxs + gxm - 1
      end subroutine

      subroutine FormIFunctionLocal(mx,xs,xe,gxs,gxe,x,xdot,f,a,k,s,ierr)
      implicit none
      PetscInt mx,xs,xe,gxs,gxe
      PetscScalar x(2,xs:xe)
      PetscScalar xdot(2,xs:xe)
      PetscScalar f(2,xs:xe)
      PetscReal a(2),k(2),s(2)
      PetscErrorCode ierr
      PetscInt i
      do 10 i = xs,xe
         f(1,i) = xdot(1,i) + k(1)*x(1,i) - k(2)*x(2,i) - s(1)
         f(2,i) = xdot(2,i) - k(1)*x(1,i) + k(2)*x(2,i) - s(2)
 10   continue
      end subroutine

      subroutine FormIFunction(ts,t,X,Xdot,F,user,ierr)
      use petscts
      implicit none

      TS ts
      PetscReal t
      Vec X,Xdot,F
      PetscReal user(6)
      PetscErrorCode ierr
      integer user_a,user_k,user_s
      parameter (user_a = 1,user_k = 3,user_s = 5)

      DM             da
      PetscInt       mx,xs,xe,gxs,gxe
      PetscScalar,pointer :: xx(:),xxdot(:),ff(:)

      PetscCall(TSGetDM(ts,da,ierr))
      PetscCall(GetLayout(da,mx,xs,xe,gxs,gxe,ierr))

! Get access to vector data
      PetscCall(VecGetArrayReadF90(X,xx,ierr))
      PetscCall(VecGetArrayReadF90(Xdot,xxdot,ierr))
      PetscCall(VecGetArrayF90(F,ff,ierr))

      PetscCall(FormIFunctionLocal(mx,xs,xe,gxs,gxe,xx,xxdot,ff,user(user_a),user(user_k),user(user_s),ierr))

      PetscCall(VecRestoreArrayReadF90(X,xx,ierr))
      PetscCall(VecRestoreArrayReadF90(Xdot,xxdot,ierr))
      PetscCall(VecRestoreArrayF90(F,ff,ierr))
      end subroutine

      subroutine FormRHSFunctionLocal(mx,xs,xe,gxs,gxe,t,x,f,a,k,s,ierr)
      implicit none
      PetscInt mx,xs,xe,gxs,gxe
      PetscReal t
      PetscScalar x(2,gxs:gxe),f(2,xs:xe)
      PetscReal a(2),k(2),s(2)
      PetscErrorCode ierr
      PetscInt i,j
      PetscReal hx,u0t(2)
      PetscReal one,two,three,four,six,twelve
      PetscReal half,third,twothird,sixth
      PetscReal twelfth

      one = 1.0
      two = 2.0
      three = 3.0
      four = 4.0
      six = 6.0
      twelve = 12.0
      hx = one / mx
!     The Fortran standard only allows positive base for power functions; Nag compiler fails on this
      u0t(1) = one - abs(sin(twelve*t))**four
      u0t(2) = 0.0
      half = one/two
      third = one / three
      twothird = two / three
      sixth = one / six
      twelfth = one / twelve
      do 20 i = xs,xe
         do 10 j = 1,2
            if (i .eq. 1) then
               f(j,i) = a(j)/hx*(third*u0t(j) + half*x(j,i) - x(j,i+1)  &
     &              + sixth*x(j,i+2))
            else if (i .eq. 2) then
               f(j,i) = a(j)/hx*(-twelfth*u0t(j) + twothird*x(j,i-1)    &
     &              - twothird*x(j,i+1) + twelfth*x(j,i+2))
            else if (i .eq. mx-1) then
               f(j,i) = a(j)/hx*(-sixth*x(j,i-2) + x(j,i-1)             &
     &         - half*x(j,i) -third*x(j,i+1))
            else if (i .eq. mx) then
               f(j,i) = a(j)/hx*(-x(j,i) + x(j,i-1))
            else
               f(j,i) = a(j)/hx*(-twelfth*x(j,i-2)                      &
     &              + twothird*x(j,i-1)                                 &
     &              - twothird*x(j,i+1) + twelfth*x(j,i+2))
            end if
 10      continue
 20   continue
      end subroutine

      subroutine FormRHSFunction(ts,t,X,F,user,ierr)
      use petscts
      implicit none

      TS ts
      PetscReal t
      Vec X,F
      PetscReal user(6)
      PetscErrorCode ierr
      integer user_a,user_k,user_s
      parameter (user_a = 1,user_k = 3,user_s = 5)
      DM             da
      Vec            Xloc
      PetscInt       mx,xs,xe,gxs,gxe
      PetscScalar,pointer :: xx(:),ff(:)

      PetscCall(TSGetDM(ts,da,ierr))
      PetscCall(GetLayout(da,mx,xs,xe,gxs,gxe,ierr))

!     Scatter ghost points to local vector,using the 2-step process
!        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
!     By placing code between these two statements, computations can be
!     done while messages are in transition.
      PetscCall(DMGetLocalVector(da,Xloc,ierr))
      PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc,ierr))
      PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc,ierr))

! Get access to vector data
      PetscCall(VecGetArrayReadF90(Xloc,xx,ierr))
      PetscCall(VecGetArrayF90(F,ff,ierr))

      PetscCall(FormRHSFunctionLocal(mx,xs,xe,gxs,gxe,t,xx,ff,user(user_a),user(user_k),user(user_s),ierr))

      PetscCall(VecRestoreArrayReadF90(Xloc,xx,ierr))
      PetscCall(VecRestoreArrayF90(F,ff,ierr))
      PetscCall(DMRestoreLocalVector(da,Xloc,ierr))
      end subroutine

! ---------------------------------------------------------------------
!
!  IJacobian - Compute IJacobian = dF/dU + shift*dF/dUdot
!
      subroutine FormIJacobian(ts,t,X,Xdot,shift,J,Jpre,user,ierr)
      use petscts
      implicit none

      TS ts
      PetscReal t,shift
      Vec X,Xdot
      Mat J,Jpre
      PetscReal user(6)
      PetscErrorCode ierr
      integer user_a,user_k,user_s
      parameter (user_a = 0,user_k = 2,user_s = 4)

      DM             da
      PetscInt       mx,xs,xe,gxs,gxe
      PetscInt       i,i1,row,col
      PetscReal      k1,k2;
      PetscScalar    val(4)

      PetscCall(TSGetDM(ts,da,ierr))
      PetscCall(GetLayout(da,mx,xs,xe,gxs,gxe,ierr))

      i1 = 1
      k1 = user(user_k+1)
      k2 = user(user_k+2)
      do 10 i = xs,xe
         row = i-gxs
         col = i-gxs
         val(1) = shift + k1
         val(2) = -k2
         val(3) = -k1
         val(4) = shift + k2
         PetscCall(MatSetValuesBlockedLocal(Jpre,i1,row,i1,col,val,INSERT_VALUES,ierr))
 10   continue
      PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY,ierr))
      if (J /= Jpre) then
         PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY,ierr))
         PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY,ierr))
      end if
      end subroutine

      subroutine FormInitialSolutionLocal(mx,xs,xe,gxs,gxe,x,a,k,s,ierr)
      implicit none
      PetscInt mx,xs,xe,gxs,gxe
      PetscScalar x(2,xs:xe)
      PetscReal a(2),k(2),s(2)
      PetscErrorCode ierr

      PetscInt i
      PetscReal one,hx,r,ik
      one = 1.0
      hx = one / mx
      do 10 i=xs,xe
         r = i*hx
         if (k(2) .ne. 0.0) then
            ik = one/k(2)
         else
            ik = one
         end if
         x(1,i) = one + s(2)*r
         x(2,i) = k(1)*ik*x(1,i) + s(2)*ik
 10   continue
      end subroutine

      subroutine FormInitialSolution(ts,X,user,ierr)
      use petscts
      implicit none

      TS ts
      Vec X
      PetscReal user(6)
      PetscErrorCode ierr
      integer user_a,user_k,user_s
      parameter (user_a = 1,user_k = 3,user_s = 5)

      DM             da
      PetscInt       mx,xs,xe,gxs,gxe
      PetscScalar,pointer :: xx(:)

      PetscCall(TSGetDM(ts,da,ierr))
      PetscCall(GetLayout(da,mx,xs,xe,gxs,gxe,ierr))

! Get access to vector data
      PetscCall(VecGetArrayF90(X,xx,ierr))

      PetscCall(FormInitialSolutionLocal(mx,xs,xe,gxs,gxe,xx,user(user_a),user(user_k),user(user_s),ierr))

      PetscCall(VecRestoreArrayF90(X,xx,ierr))
      end subroutine

!/*TEST
!
!    test:
!      args: -da_grid_x 200 -ts_arkimex_type 4
!      requires: !single
!
!TEST*/

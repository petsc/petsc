!     Time-dependent advection-reaction PDE in 1d. Demonstrates IMEX methods
!
!     u_t + a1*u_x = -k1*u + k2*v + s1
!     v_t + a2*v_x = k1*u - k2*v + s2
!     0 < x < 1;
!     a1 = 1, k1 = 10^6, s1 = 0,
!     a2 = 0, k2 = 2*k1, s2 = 1
!
!     Initial conditions:
!     u(x,0) = 1 + s2*x
!     v(x,0) = k0/k1*u(x,0) + s1/k1
!
!     Upstream boundary conditions:
!     u(0,t) = 1-sin(12*t)^4
!

  module PETScShiftMod
#include <petsc/finclude/petscts.h>
    use petscts
    PetscScalar::PETSC_SHIFT
    TS::tscontext
    Mat::Jmat
    PetscReal::MFuser(6)
  end module PETScShiftMod

program main
  use PETScShiftMod
  use petscdmda
  implicit none

  !
  !     Create an application context to contain data needed by the
  !     application-provided call-back routines, FormJacobian() and
  !     FormFunction(). We use a double precision array with six
  !     entries, two for each problem parameter a, k, s.
  !
  PetscReal user(6)
  integer user_a,user_k,user_s
  parameter (user_a = 0,user_k = 2,user_s = 4)

  external FormRHSFunction,FormIFunction
  external FormInitialSolution
  external FormIJacobian
  external MyMult,FormIJacobianMF

  TS             ts
  Vec            X
  Mat            J
  PetscInt       mx
  PetscBool      OptionSaveToDisk
  PetscErrorCode ierr
  DM             da
  PetscReal      ftime,dt
  PetscReal      one,pone
  PetscInt       im11,i2
  PetscBool      flg

  PetscInt       xs,xe,gxs,gxe,dof,gdof
  PetscScalar    shell_shift
  Mat            A

  im11 = 11
  i2   = 2
  one = 1.0
  pone = one / 10

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr .ne. 0) then
    print*,'PetscInitialize failed'
    stop
  endif

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Create distributed array (DMDA) to manage parallel grid and vectors
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,im11,i2,i2,PETSC_NULL_INTEGER,da,ierr);CHKERRA(ierr)
  call DMSetFromOptions(da,ierr);CHKERRA(ierr)
  call DMSetUp(da,ierr);CHKERRA(ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !    Extract global vectors from DMDA;
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call DMCreateGlobalVector(da,X,ierr);CHKERRA(ierr)

  ! Initialize user application context
  ! Use zero-based indexing for command line parameters to match ex22.c
  user(user_a+1) = 1.0
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-a0',user(user_a+1),flg,ierr);CHKERRA(ierr)
  user(user_a+2) = 0.0
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-a1',user(user_a+2),flg,ierr);CHKERRA(ierr)
  user(user_k+1) = 1000000.0
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-k0', user(user_k+1),flg,ierr);CHKERRA(ierr)
  user(user_k+2) = 2*user(user_k+1)
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-k1', user(user_k+2),flg,ierr);CHKERRA(ierr)
  user(user_s+1) = 0.0
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s0',user(user_s+1),flg,ierr);CHKERRA(ierr)
  user(user_s+2) = 1.0
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s1',user(user_s+2),flg,ierr);CHKERRA(ierr)

  OptionSaveToDisk=.FALSE.
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-sdisk',OptionSaveToDisk,flg,ierr);CHKERRA(ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !    Create timestepping solver context
  !     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSCreate(PETSC_COMM_WORLD,ts,ierr);CHKERRA(ierr)
  tscontext=ts
  call TSSetDM(ts,da,ierr);CHKERRA(ierr)
  call TSSetType(ts,TSARKIMEX,ierr);CHKERRA(ierr)
  call TSSetRHSFunction(ts,PETSC_NULL_VEC,FormRHSFunction,user,ierr);CHKERRA(ierr)

  ! - - - - - - - - -- - - - -
  !   Matrix free setup
  call GetLayout(da,mx,xs,xe,gxs,gxe,ierr);CHKERRA(ierr)
  dof=i2*(xe-xs+1)
  gdof=i2*(gxe-gxs+1)
  call MatCreateShell(PETSC_COMM_WORLD,dof,dof,gdof,gdof,shell_shift,A,ierr);CHKERRA(ierr)

  call MatShellSetOperation(A,MATOP_MULT,MyMult,ierr);CHKERRA(ierr)
  ! - - - - - - - - - - - -

  call TSSetIFunction(ts,PETSC_NULL_VEC,FormIFunction,user,ierr);CHKERRA(ierr)
  call DMSetMatType(da,MATAIJ,ierr);CHKERRA(ierr)
  call DMCreateMatrix(da,J,ierr);CHKERRA(ierr)

  Jmat=J

  call TSSetIJacobian(ts,J,J,FormIJacobian,user,ierr);CHKERRA(ierr)
  call TSSetIJacobian(ts,A,A,FormIJacobianMF,user,ierr);CHKERRA(ierr)

  ftime = 1.0
  call TSSetMaxTime(ts,ftime,ierr);CHKERRA(ierr)
  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr);CHKERRA(ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Set initial conditions
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call FormInitialSolution(ts,X,user,ierr);CHKERRA(ierr)
  call TSSetSolution(ts,X,ierr);CHKERRA(ierr)
  call VecGetSize(X,mx,ierr);CHKERRA(ierr)
  !  Advective CFL, I don't know why it needs so much safety factor.
  dt = pone * max(user(user_a+1),user(user_a+2)) / mx;
  call TSSetTimeStep(ts,dt,ierr);CHKERRA(ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !   Set runtime options
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSSetFromOptions(ts,ierr);CHKERRA(ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Solve nonlinear system
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSSolve(ts,X,ierr);CHKERRA(ierr)

  if (OptionSaveToDisk) then
     call GetLayout(da,mx,xs,xe,gxs,gxe,ierr);CHKERRA(ierr)
     dof=i2*(xe-xs+1)
     gdof=i2*(gxe-gxs+1)
     call SaveSolutionToDisk(da,X,gdof,xs,xe);CHKERRA(ierr)
  end if

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Free work space.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call MatDestroy(A,ierr);CHKERRA(ierr)
  call MatDestroy(J,ierr);CHKERRA(ierr)
  call VecDestroy(X,ierr);CHKERRA(ierr)
  call TSDestroy(ts,ierr);CHKERRA(ierr)
  call DMDestroy(da,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr)
end program main

! Small helper to extract the layout, result uses 1-based indexing.
  subroutine GetLayout(da,mx,xs,xe,gxs,gxe,ierr)
  use petscdmda
  implicit none

  DM da
  PetscInt mx,xs,xe,gxs,gxe
  PetscErrorCode ierr
  PetscInt xm,gxm
  call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,    &
       PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
  call DMDAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
  call DMDAGetGhostCorners(da,gxs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,gxm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
  xs = xs + 1
  gxs = gxs + 1
  xe = xs + xm - 1
  gxe = gxs + gxm - 1
end subroutine GetLayout

subroutine FormIFunctionLocal(mx,xs,xe,gxs,gxe,x,xdot,f,a,k,s,ierr)
  implicit none
  PetscInt mx,xs,xe,gxs,gxe
  PetscScalar x(2,xs:xe)
  PetscScalar xdot(2,xs:xe)
  PetscScalar f(2,xs:xe)
  PetscReal a(2),k(2),s(2)
  PetscErrorCode ierr
  PetscInt i
  do  i = xs,xe
     f(1,i) = xdot(1,i) + k(1)*x(1,i) - k(2)*x(2,i) - s(1)
     f(2,i) = xdot(2,i) - k(1)*x(1,i) + k(2)*x(2,i) - s(2)
  end do
end subroutine FormIFunctionLocal

subroutine FormIFunction(ts,t,X,Xdot,F,user,ierr)
  use petscdmda
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
  PetscOffset    ixx,ixxdot,iff
  PetscScalar    xx(0:1),xxdot(0:1),ff(0:1)

  call TSGetDM(ts,da,ierr);CHKERRQ(ierr)
  call GetLayout(da,mx,xs,xe,gxs,gxe,ierr);CHKERRQ(ierr)

  ! Get access to vector data
  call VecGetArrayRead(X,xx,ixx,ierr);CHKERRQ(ierr)
  call VecGetArrayRead(Xdot,xxdot,ixxdot,ierr);CHKERRQ(ierr)
  call VecGetArray(F,ff,iff,ierr);CHKERRQ(ierr)

  call FormIFunctionLocal(mx,xs,xe,gxs,gxe,xx(ixx),xxdot(ixxdot),ff(iff),user(user_a),user(user_k),user(user_s),ierr);CHKERRQ(ierr)

  call VecRestoreArrayRead(X,xx,ixx,ierr);CHKERRQ(ierr)
  call VecRestoreArrayRead(Xdot,xxdot,ixxdot,ierr);CHKERRQ(ierr)
  call VecRestoreArray(F,ff,iff,ierr);CHKERRQ(ierr)
end subroutine FormIFunction

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
  u0t(1) = one - sin(twelve*t)**four
  u0t(2) = 0.0
  half = one/two
  third = one / three
  twothird = two / three
  sixth = one / six
  twelfth = one / twelve
  do  i = xs,xe
     do  j = 1,2
        if (i .eq. 1) then
           f(j,i) = a(j)/hx*(third*u0t(j) + half*x(j,i) - x(j,i+1) + sixth*x(j,i+2))
        else if (i .eq. 2) then
           f(j,i) = a(j)/hx*(-twelfth*u0t(j) + twothird*x(j,i-1) - twothird*x(j,i+1) + twelfth*x(j,i+2))
        else if (i .eq. mx-1) then
           f(j,i) = a(j)/hx*(-sixth*x(j,i-2) + x(j,i-1) - half*x(j,i) -third*x(j,i+1))
        else if (i .eq. mx) then
           f(j,i) = a(j)/hx*(-x(j,i) + x(j,i-1))
        else
           f(j,i) = a(j)/hx*(-twelfth*x(j,i-2) + twothird*x(j,i-1) - twothird*x(j,i+1) + twelfth*x(j,i+2))
        end if
     end do
  end do

#ifdef EXPLICIT_INTEGRATOR22
  do  i = xs,xe
     f(1,i) = f(1,i) -( k(1)*x(1,i) - k(2)*x(2,i) - s(1))
     f(2,i) = f(2,i) -(- k(1)*x(1,i) + k(2)*x(2,i) - s(2))
  end do
#endif

end subroutine FormRHSFunctionLocal

subroutine FormRHSFunction(ts,t,X,F,user,ierr)
  use petscts
  use petscdmda
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
  PetscOffset    ixx,iff
  PetscScalar    xx(0:1),ff(0:1)

  call TSGetDM(ts,da,ierr);CHKERRQ(ierr)
  call GetLayout(da,mx,xs,xe,gxs,gxe,ierr);CHKERRQ(ierr)

  !     Scatter ghost points to local vector,using the 2-step process
  !        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
  !     By placing code between these two statements, computations can be
  !     done while messages are in transition.
  call DMGetLocalVector(da,Xloc,ierr);CHKERRQ(ierr)
  call DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc,ierr);CHKERRQ(ierr)
  call DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc,ierr);CHKERRQ(ierr)

  ! Get access to vector data
  call VecGetArrayRead(Xloc,xx,ixx,ierr);CHKERRQ(ierr)
  call VecGetArray(F,ff,iff,ierr);CHKERRQ(ierr)

  call FormRHSFunctionLocal(mx,xs,xe,gxs,gxe,t,xx(ixx),ff(iff),user(user_a),user(user_k),user(user_s),ierr);CHKERRQ(ierr)

  call VecRestoreArrayRead(Xloc,xx,ixx,ierr);CHKERRQ(ierr)
  call VecRestoreArray(F,ff,iff,ierr);CHKERRQ(ierr)
  call DMRestoreLocalVector(da,Xloc,ierr);CHKERRQ(ierr)
end subroutine FormRHSFunction

! ---------------------------------------------------------------------
!
!  IJacobian - Compute IJacobian = dF/dU + shift*dF/dUdot
!
subroutine FormIJacobian(ts,t,X,Xdot,shift,J,Jpre,user,ierr)
  use petscts
  use petscdmda
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

  call TSGetDM(ts,da,ierr);CHKERRQ(ierr)
  call GetLayout(da,mx,xs,xe,gxs,gxe,ierr);CHKERRQ(ierr)

  i1 = 1
  k1 = user(user_k+1)
  k2 = user(user_k+2)
  do i = xs,xe
     row = i-gxs
     col = i-gxs
     val(1) = shift + k1
     val(2) = -k2
     val(3) = -k1
     val(4) = shift + k2
     call MatSetValuesBlockedLocal(Jpre,i1,row,i1,col,val,INSERT_VALUES,ierr);CHKERRQ(ierr)
  end do
  call MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
  call MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
  if (J /= Jpre) then
     call MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
     call MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
  end if
end subroutine FormIJacobian

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
  do i=xs,xe
     r = i*hx
     if (k(2) .ne. 0.0) then
        ik = one/k(2)
     else
        ik = one
     end if
     x(1,i) = one + s(2)*r
     x(2,i) = k(1)*ik*x(1,i) + s(2)*ik
  end do
end subroutine FormInitialSolutionLocal

subroutine FormInitialSolution(ts,X,user,ierr)
  use petscts
  use petscdmda
  implicit none

  TS ts
  Vec X
  PetscReal user(6)
  PetscErrorCode ierr
  integer user_a,user_k,user_s
  parameter (user_a = 1,user_k = 3,user_s = 5)

  DM             da
  PetscInt       mx,xs,xe,gxs,gxe
  PetscOffset    ixx
  PetscScalar    xx(0:1)

  call TSGetDM(ts,da,ierr);CHKERRQ(ierr)
  call GetLayout(da,mx,xs,xe,gxs,gxe,ierr);CHKERRQ(ierr)

  ! Get access to vector data
  call VecGetArray(X,xx,ixx,ierr);CHKERRQ(ierr)

  call FormInitialSolutionLocal(mx,xs,xe,gxs,gxe,xx(ixx),user(user_a),user(user_k),user(user_s),ierr);CHKERRQ(ierr)

  call VecRestoreArray(X,xx,ixx,ierr);CHKERRQ(ierr)
end subroutine FormInitialSolution

! ---------------------------------------------------------------------
!
!  IJacobian - Compute IJacobian = dF/dU + shift*dF/dUdot
!
subroutine FormIJacobianMF(ts,t,X,Xdot,shift,J,Jpre,user,ierr)
  use PETScShiftMod
  implicit none
  TS ts
  PetscReal t,shift
  Vec X,Xdot
  Mat J,Jpre
  PetscReal user(6)
  PetscErrorCode ierr

  !  call MatShellSetContext(J,shift,ierr)
  PETSC_SHIFT=shift
  MFuser=user

end subroutine FormIJacobianMF

! -------------------------------------------------------------------
!
!   MyMult - user provided matrix multiply
!
!   Input Parameters:
!.  X - input vector
!
!   Output Parameter:
!.  F - function vector
!
subroutine  MyMult(A,X,F,ierr)
  use PETScShiftMod
  implicit none

  Mat     A
  Vec     X,F

  PetscErrorCode ierr
  PetscScalar shift

!  Mat J,Jpre

  PetscReal user(6)

  integer user_a,user_k,user_s
  parameter (user_a = 0,user_k = 2,user_s = 4)

  DM             da
  PetscInt       mx,xs,xe,gxs,gxe
  PetscInt       i,i1,row,col
  PetscReal      k1,k2;
  PetscScalar    val(4)

  !call MatShellGetContext(A,shift,ierr)
  shift=PETSC_SHIFT
  user=MFuser

  call TSGetDM(tscontext,da,ierr)
  call GetLayout(da,mx,xs,xe,gxs,gxe,ierr)

  i1 = 1
  k1 = user(user_k+1)
  k2 = user(user_k+2)

  do i = xs,xe
     row = i-gxs
     col = i-gxs
     val(1) = shift + k1
     val(2) = -k2
     val(3) = -k1
     val(4) = shift + k2
     call MatSetValuesBlockedLocal(Jmat,i1,row,i1,col,val,INSERT_VALUES,ierr)
  end do

!  call MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY,ierr)
!  call MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY,ierr)
!  if (J /= Jpre) then
     call MatAssemblyBegin(Jmat,MAT_FINAL_ASSEMBLY,ierr)
     call MatAssemblyEnd(Jmat,MAT_FINAL_ASSEMBLY,ierr)
!  end if

  call MatMult(Jmat,X,F,ierr)

  return
end subroutine MyMult

!
subroutine SaveSolutionToDisk(da,X,gdof,xs,xe)
  use petscdmda
  implicit none

  Vec X
  DM             da
  PetscInt xs,xe,two
  PetscInt gdof,i
  PetscErrorCode ierr
  PetscOffset    ixx
  PetscScalar data2(2,xs:xe),data(gdof)
  PetscScalar    xx(0:1)

  call VecGetArrayRead(X,xx,ixx,ierr)

  two = 2
  data2=reshape(xx(ixx:ixx+gdof),(/two,xe-xs+1/))
  data=reshape(data2,(/gdof/))
  open(1020,file='solution_out_ex22f_mf.txt')
  do i=1,gdof
     write(1020,'(e24.16,1x)') data(i)
  end do
  close(1020)

  call VecRestoreArrayRead(X,xx,ixx,ierr)
end subroutine SaveSolutionToDisk

!/*TEST
!
!    test:
!      args: -da_grid_x 200 -ts_arkimex_type 4
!      requires: !single
!      output_file: output/ex22f_mf_1.out
!
!TEST*/

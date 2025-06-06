!  Program usage: mpiexec -n <proc> eptorsion2f [all TAO options]
!
!  Description:  This example demonstrates use of the TAO package to solve
!  unconstrained minimization problems in parallel.  This example is based
!  on the Elastic-Plastic Torsion (dept) problem from the MINPACK-2 test suite.
!  The command line options are:
!    -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction
!    -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction
!    -par <param>, where <param> = angle of twist per unit length
!
!
! ----------------------------------------------------------------------
!
!  Elastic-plastic torsion problem.
!
!  The elastic plastic torsion problem arises from the deconverged
!  of the stress field on an infinitely long cylindrical bar, which is
!  equivalent to the solution of the following problem:
!     min{ .5 * integral(||gradient(v(x))||^2 dx) - C * integral(v(x) dx)}
!  where C is the torsion angle per unit length.
!
!  The C version of this code is eptorsion2.c
!
! ----------------------------------------------------------------------

      module eptorsion2fmodule
#include "petsc/finclude/petscdmda.h"
#include "petsc/finclude/petsctao.h"
      use petscdmda
      use petsctao
      implicit none

      type(tVec)       localX
      type(tDM)        dm
      PetscReal        param
      PetscInt         mx, my
      end module

      use eptorsion2fmodule
      implicit none
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  See additional variable declarations in the file eptorsion2f.h
!
      PetscErrorCode   ierr           ! used to check for functions returning nonzeros
      type(tVec)       x              ! solution vector
      type(tMat)       H              ! hessian matrix
      PetscInt         Nx, Ny         ! number of processes in x- and y- directions
      type(tTao)       ta            ! Tao solver context
      PetscBool        flg
      PetscInt         i1
      PetscInt         dummy

!  Note: Any user-defined Fortran routines (such as FormGradient)
!  MUST be declared as external.

      external FormInitialGuess,FormFunctionGradient,ComputeHessian
      external Monitor,ConvergenceTest

      i1 = 1

!     Initialize TAO, PETSc  contexts
      PetscCallA(PetscInitialize(ierr))

!     Specify default parameters
      param = 5.0
      mx = 10
      my = 10
      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE

!     Check for any command line arguments that might override defaults
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-mx',mx,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-my',my,flg,ierr))
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-par',param,flg,ierr))

!     Set up distributed array and vectors
      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,Nx,Ny,i1,i1,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,dm,ierr))
      PetscCallA(DMSetFromOptions(dm,ierr))
      PetscCallA(DMSetUp(dm,ierr))

!     Create vectors
      PetscCallA(DMCreateGlobalVector(dm,x,ierr))
      PetscCallA(DMCreateLocalVector(dm,localX,ierr))

!     Create Hessian
      PetscCallA(DMCreateMatrix(dm,H,ierr))
      PetscCallA(MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE,ierr))

!     The TAO code begins here

!     Create TAO solver
      PetscCallA(TaoCreate(PETSC_COMM_WORLD,ta,ierr))
      PetscCallA(TaoSetType(ta,TAOCG,ierr))

!     Set routines for function and gradient evaluation

      PetscCallA(TaoSetObjectiveAndGradient(ta,PETSC_NULL_VEC,FormFunctionGradient,0,ierr))
      PetscCallA(TaoSetHessian(ta,H,H,ComputeHessian,0,ierr))

!     Set initial guess
      PetscCallA(FormInitialGuess(x,ierr))
      PetscCallA(TaoSetSolution(ta,x,ierr))

      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-testmonitor',flg,ierr))
      if (flg) then
         PetscCallA(TaoMonitorSet(ta,Monitor,dummy,PETSC_NULL_FUNCTION,ierr))
      endif

      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-testconvergence',flg, ierr))
      if (flg) then
         PetscCallA(TaoSetConvergenceTest(ta,ConvergenceTest,dummy,ierr))
      endif

!     Check for any TAO command line options
      PetscCallA(TaoSetFromOptions(ta,ierr))

!     SOLVE THE APPLICATION
      PetscCallA(TaoSolve(ta,ierr))

!     Free TAO data structures
      PetscCallA(TaoDestroy(ta,ierr))

!     Free PETSc data structures
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(localX,ierr))
      PetscCallA(MatDestroy(H,ierr))
      PetscCallA(DMDestroy(dm,ierr))

!     Finalize TAO and PETSc
      PetscCallA(PetscFinalize(ierr))
      end

! ---------------------------------------------------------------------
!
!   FormInitialGuess - Computes an initial approximation to the solution.
!
!   Input Parameters:
!   X    - vector
!
!   Output Parameters:
!   X    - vector
!   ierr - error code
!
      subroutine FormInitialGuess(X,ierr)
      use eptorsion2fmodule
      implicit none

!  Input/output variables:
      Vec              X
      PetscErrorCode   ierr

!  Local variables:
      PetscInt         i, j, k, xe, ye
      PetscReal      temp, val, hx, hy
      PetscInt         xs, ys, xm, ym
      PetscInt         gxm, gym, gxs, gys
      PetscInt         i1

      i1 = 1
      hx = 1.0/real(mx + 1)
      hy = 1.0/real(my + 1)

!  Get corner information
      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

!  Compute initial guess over locally owned part of mesh
      xe = xs+xm
      ye = ys+ym
      do j=ys,ye-1
         temp = min(j+1,my-j)*hy
         do i=xs,xe-1
            k   = (j-gys)*gxm + i-gxs
            val = min((min(i+1,mx-i))*hx,temp)
            PetscCall(VecSetValuesLocal(X,i1,[k],[val],ADD_VALUES,ierr))
         end do
      end do
      PetscCall(VecAssemblyBegin(X,ierr))
      PetscCall(VecAssemblyEnd(X,ierr))
      end

! ---------------------------------------------------------------------
!
!  FormFunctionGradient - Evaluates gradient G(X).
!
!  Input Parameters:
!  tao   - the Tao context
!  X     - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameters:
!  f     - the function value at X
!  G     - vector containing the newly evaluated gradient
!  ierr  - error code
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationGradient", where the actual computations are
!  done using the standard Fortran style of treating the local
!  input vector data as an array over the local mesh.
!
      subroutine FormFunctionGradient(ta,X,f,G,dummy,ierr)
      use eptorsion2fmodule
      implicit none

!  Input/output variables:
      type(tTao)       ta
      type(tVec)       X, G
      PetscReal        f
      PetscErrorCode   ierr
      PetscInt         dummy

!  Declarations for use with local array:

      PetscReal, pointer :: lx_v(:)

!  Local variables:
      PetscReal      zero, p5, area, cdiv3
      PetscReal      val, flin, fquad,floc
      PetscReal      v, vb, vl, vr, vt, dvdx
      PetscReal      dvdy, hx, hy
      PetscInt       xe, ye, xsm, ysm
      PetscInt       xep, yep, i, j, k, ind
      PetscInt       xs, ys, xm, ym
      PetscInt       gxs, gys, gxm, gym
      PetscInt       i1

      i1 = 1
      ierr  = 0
      cdiv3 = param/3.0
      zero = 0.0
      p5   = 0.5
      hx = 1.0/real(mx + 1)
      hy = 1.0/real(my + 1)
      fquad = zero
      flin = zero

!  Initialize gradient to zero
      PetscCall(VecSet(G,zero,ierr))

!  Scatter ghost points to local vector
      PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX,ierr))
      PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX,ierr))

!  Get corner information
      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

!  Get pointer to vector data.
      PetscCall(VecGetArrayRead(localX,lx_v,ierr))

!  Set local loop dimensions
      xe = xs+xm
      ye = ys+ym
      if (xs .eq. 0) then
         xsm = xs-1
      else
         xsm = xs
      endif
      if (ys .eq. 0) then
         ysm = ys-1
      else
         ysm = ys
      endif
      if (xe .eq. mx) then
         xep = xe+1
      else
         xep = xe
      endif
      if (ye .eq. my) then
         yep = ye+1
      else
         yep = ye
      endif

!     Compute local gradient contributions over the lower triangular elements

      do j = ysm, ye-1
         do i = xsm, xe-1
            k  = (j-gys)*gxm + i-gxs
            v  = zero
            vr = zero
            vt = zero
            if (i .ge. 0 .and. j .ge. 0)      v = lx_v(k+1)
            if (i .lt. mx-1 .and. j .gt. -1) vr = lx_v(k+2)
            if (i .gt. -1 .and. j .lt. my-1) vt = lx_v(k+1+gxm)
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            if (i .ne. -1 .and. j .ne. -1) then
               ind = k
               val = - dvdx/hx - dvdy/hy - cdiv3
               PetscCall(VecSetValuesLocal(G,i1,[k],[val],ADD_VALUES,ierr))
            endif
            if (i .ne. mx-1 .and. j .ne. -1) then
               ind = k+1
               val =  dvdx/hx - cdiv3
               PetscCall(VecSetValuesLocal(G,i1,[ind],[val],ADD_VALUES,ierr))
            endif
            if (i .ne. -1 .and. j .ne. my-1) then
              ind = k+gxm
              val = dvdy/hy - cdiv3
              PetscCall(VecSetValuesLocal(G,i1,[ind],[val],ADD_VALUES,ierr))
            endif
            fquad = fquad + dvdx*dvdx + dvdy*dvdy
            flin = flin - cdiv3 * (v+vr+vt)
         end do
      end do

!     Compute local gradient contributions over the upper triangular elements

      do j = ys, yep-1
         do i = xs, xep-1
            k  = (j-gys)*gxm + i-gxs
            vb = zero
            vl = zero
            v  = zero
            if (i .lt. mx .and. j .gt. 0) vb = lx_v(k+1-gxm)
            if (i .gt. 0 .and. j .lt. my) vl = lx_v(k)
            if (i .lt. mx .and. j .lt. my) v = lx_v(1+k)
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            if (i .ne. mx .and. j .ne. 0) then
               ind = k-gxm
               val = - dvdy/hy - cdiv3
               PetscCall(VecSetValuesLocal(G,i1,[ind],[val],ADD_VALUES,ierr))
            endif
            if (i .ne. 0 .and. j .ne. my) then
               ind = k-1
               val =  - dvdx/hx - cdiv3
               PetscCall(VecSetValuesLocal(G,i1,[ind],[val],ADD_VALUES,ierr))
            endif
            if (i .ne. mx .and. j .ne. my) then
               ind = k
               val =  dvdx/hx + dvdy/hy - cdiv3
               PetscCall(VecSetValuesLocal(G,i1,[ind],[val],ADD_VALUES,ierr))
            endif
            fquad = fquad + dvdx*dvdx + dvdy*dvdy
            flin = flin - cdiv3*(vb + vl + v)
         end do
      end do

!  Restore vector
      PetscCall(VecRestoreArrayRead(localX,lx_v,ierr))

!  Assemble gradient vector
      PetscCall(VecAssemblyBegin(G,ierr))
      PetscCall(VecAssemblyEnd(G,ierr))

! Scale the gradient
      area = p5*hx*hy
      floc = area *(p5*fquad+flin)
      PetscCall(VecScale(G,area,ierr))

!  Sum function contributions from all processes
      PetscCallMPI(MPI_Allreduce(floc,f,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD,ierr))
      PetscCall(PetscLogFlops(20.0d0*(ye-ysm)*(xe-xsm)+16.0d0*(xep-xs)*(yep-ys),ierr))
      end

      subroutine ComputeHessian(ta, X, H, Hpre, dummy, ierr)
      use eptorsion2fmodule
      implicit none

      type(tTao)      ta
      type(tVec)      X
      type(tMat)      H,Hpre
      PetscErrorCode  ierr
      PetscInt        dummy

      PetscInt i,j,k
      PetscInt col(0:4),row
      PetscInt xs,xm,gxs,gxm
      PetscInt ys,ym,gys,gym
      PetscReal v(0:4)
      PetscInt i1

      i1 = 1

!     Get local grid boundaries
      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

      do j=ys,ys+ym-1
         do i=xs,xs+xm-1
            row = (j-gys)*gxm + (i-gxs)

            k = 0
            if (j .gt. gys) then
               v(k) = -1.0
               col(k) = row-gxm
               k = k + 1
            endif

            if (i .gt. gxs) then
               v(k) = -1.0
               col(k) = row - 1
               k = k +1
            endif

            v(k) = 4.0
            col(k) = row
            k = k + 1

            if (i+1 .lt. gxs + gxm) then
               v(k) = -1.0
               col(k) = row + 1
               k = k + 1
            endif

            if (j+1 .lt. gys + gym) then
               v(k) = -1.0
               col(k) = row + gxm
               k = k + 1
            endif

            PetscCall(MatSetValuesLocal(H,i1,[row],k,col,v,INSERT_VALUES,ierr))
         enddo
      enddo

!     Assemble matrix
      PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY,ierr))

!     Tell the matrix we will never add a new nonzero location to the
!     matrix.  If we do it will generate an error.

      PetscCall(MatSetOption(H,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr))
      PetscCall(MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE,ierr))

      PetscCall(PetscLogFlops(9.0d0*xm*ym + 49.0d0*xm,ierr))

      ierr = 0
      end

      subroutine Monitor(ta, dummy, ierr)
      use eptorsion2fmodule
      implicit none

      type(tTao)        ta
      PetscInt          dummy
      PetscErrorCode    ierr

      PetscInt           its
      PetscReal          f,gnorm,cnorm,xdiff
      TaoConvergedReason reason

      PetscCall(TaoGetSolutionStatus(ta,its,f,gnorm,cnorm,xdiff,reason,ierr))
      if (mod(its,5) .ne. 0) then
         PetscCall(PetscPrintf(PETSC_COMM_WORLD,'iteration multiple of 5\n',ierr))
      endif

      ierr = 0

      end

      subroutine ConvergenceTest(ta, dummy, ierr)
      use eptorsion2fmodule
      implicit none

      type(tTao)          ta
      PetscInt           dummy
      PetscErrorCode     ierr

      PetscInt           its
      PetscReal          f,gnorm,cnorm,xdiff
      TaoConvergedReason reason

      PetscCall(TaoGetSolutionStatus(ta,its,f,gnorm,cnorm,xdiff,reason,ierr))
      if (its .eq. 7) then
       PetscCall(TaoSetConvergedReason(ta,TAO_DIVERGED_MAXITS,ierr))
      endif

      ierr = 0

      end

!/*TEST
!
!   build:
!      requires: !complex
!
!   test:
!      args: -tao_monitor_short -tao_type nls -tao_gttol 1.e-2
!
!   test:
!      suffix: 2
!      nsize: 2
!      args: -tao_monitor_short -tao_type lmvm -tao_gttol 1.e-2
!
!   test:
!      suffix: 3
!      args: -testmonitor -tao_type lmvm -tao_gttol 1.e-2
!TEST*/

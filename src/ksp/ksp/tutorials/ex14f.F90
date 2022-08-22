!
!
!  Solves a nonlinear system in parallel with a user-defined
!  Newton method that uses KSP to solve the linearized Newton systems.  This solver
!  is a very simplistic inexact Newton method.  The intent of this code is to
!  demonstrate the repeated solution of linear systems with the same nonzero pattern.
!
!  This is NOT the recommended approach for solving nonlinear problems with PETSc!
!  We urge users to employ the SNES component for solving nonlinear problems whenever
!  possible, as it offers many advantages over coding nonlinear solvers independently.
!
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DMDAs) to partition the parallel grid.
!
!  The command line options include:
!  -par <parameter>, where <parameter> indicates the problem's nonlinearity
!     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
!  -mx <xg>, where <xg> = number of grid points in the x-direction
!  -my <yg>, where <yg> = number of grid points in the y-direction
!  -Nx <npx>, where <npx> = number of processors in the x-direction
!  -Ny <npy>, where <npy> = number of processors in the y-direction
!  -mf use matrix free for matrix vector product
!

!  ------------------------------------------------------------------------
!
!    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
!    the partial differential equation
!
!            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
!
!    with boundary conditions
!
!             u = 0  for  x = 0, x = 1, y = 0, y = 1.
!
!    A finite difference approximation with the usual 5-point stencil
!    is used to discretize the boundary value problem to obtain a nonlinear
!    system of equations.
!
!    The SNES version of this problem is:  snes/tutorials/ex5f.F
!
!  -------------------------------------------------------------------------
      module ex14fmodule
#include <petsc/finclude/petscksp.h>
      use petscdmda
      use petscksp
      Vec      localX
      PetscInt mx,my
      Mat B
      DM da
      end module

      program main
      use ex14fmodule
      implicit none

      MPI_Comm comm
      Vec      X,Y,F
      Mat      J
      KSP      ksp

      PetscInt  Nx,Ny,N,ifive,ithree
      PetscBool  flg,nooutput,usemf
!
!      This is the routine to use for matrix-free approach
!
      external mymult

!     --------------- Data to define nonlinear solver --------------
      PetscReal   rtol,ttol
      PetscReal   fnorm,ynorm,xnorm
      PetscInt            max_nonlin_its,one
      PetscInt            lin_its
      PetscInt           i,m
      PetscScalar        mone
      PetscErrorCode ierr

      mone           = -1.0
      rtol           = 1.e-8
      max_nonlin_its = 10
      one            = 1
      ifive          = 5
      ithree         = 3

      PetscCallA(PetscInitialize(ierr))
      comm = PETSC_COMM_WORLD

!  Initialize problem parameters

!
      mx = 4
      my = 4
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-mx',mx,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-my',my,flg,ierr))
      N = mx*my

      nooutput = .false.
      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-no_output',nooutput,ierr))

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Create linear solver context
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(KSPCreate(comm,ksp,ierr))

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Create vector data structures
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!
!  Create distributed array (DMDA) to manage parallel grid and vectors
!
      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-Nx',Nx,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-Ny',Ny,flg,ierr))
      PetscCallA(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx,my,Nx,Ny,one,one,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr))
      PetscCallA(DMSetFromOptions(da,ierr))
      PetscCallA(DMSetUp(da,ierr))
!
!  Extract global and local vectors from DMDA then duplicate for remaining
!  vectors that are the same types
!
       PetscCallA(DMCreateGlobalVector(da,X,ierr))
       PetscCallA(DMCreateLocalVector(da,localX,ierr))
       PetscCallA(VecDuplicate(X,F,ierr))
       PetscCallA(VecDuplicate(X,Y,ierr))

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Create matrix data structure for Jacobian
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!     Note:  For the parallel case, vectors and matrices MUST be partitioned
!     accordingly.  When using distributed arrays (DMDAs) to create vectors,
!     the DMDAs determine the problem partitioning.  We must explicitly
!     specify the local matrix dimensions upon its creation for compatibility
!     with the vector distribution.
!
!     Note: Here we only approximately preallocate storage space for the
!     Jacobian.  See the users manual for a discussion of better techniques
!     for preallocating matrix memory.
!
      PetscCallA(VecGetLocalSize(X,m,ierr))
      PetscCallA(MatCreateAIJ(comm,m,m,N,N,ifive,PETSC_NULL_INTEGER,ithree,PETSC_NULL_INTEGER,B,ierr))

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     if usemf is on then matrix vector product is done via matrix free
!     approach. Note this is just an example, and not realistic because
!     we still use the actual formed matrix, but in reality one would
!     provide their own subroutine that would directly do the matrix
!     vector product and call MatMult()
!     Note: we put B into a module so it will be visible to the
!     mymult() routine
      usemf = .false.
      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-mf',usemf,ierr))
      if (usemf) then
         PetscCallA(MatCreateShell(comm,m,m,N,N,PETSC_NULL_INTEGER,J,ierr))
         PetscCallA(MatShellSetOperation(J,MATOP_MULT,mymult,ierr))
      else
!        If not doing matrix free then matrix operator, J,  and matrix used
!        to construct preconditioner, B, are the same
        J = B
      endif

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Customize linear solver set runtime options
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!     Set runtime options (e.g., -ksp_monitor -ksp_rtol <rtol> -ksp_type <type>)
!
       PetscCallA(KSPSetFromOptions(ksp,ierr))

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Evaluate initial guess
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

       PetscCallA(FormInitialGuess(X,ierr))
       PetscCallA(ComputeFunction(X,F,ierr))
       PetscCallA(VecNorm(F,NORM_2,fnorm,ierr))
       ttol = fnorm*rtol
       if (.not. nooutput) then
         print*, 'Initial function norm ',fnorm
       endif

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Solve nonlinear system with a user-defined method
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  This solver is a very simplistic inexact Newton method, with no
!  no damping strategies or bells and whistles. The intent of this code
!  is merely to demonstrate the repeated solution with KSP of linear
!  systems with the same nonzero structure.
!
!  This is NOT the recommended approach for solving nonlinear problems
!  with PETSc!  We urge users to employ the SNES component for solving
!  nonlinear problems whenever possible with application codes, as it
!  offers many advantages over coding nonlinear solvers independently.

       do 10 i=0,max_nonlin_its

!  Compute the Jacobian matrix.  See the comments in this routine for
!  important information about setting the flag mat_flag.

         PetscCallA(ComputeJacobian(X,B,ierr))

!  Solve J Y = F, where J is the Jacobian matrix.
!    - First, set the KSP linear operators.  Here the matrix that
!      defines the linear system also serves as the preconditioning
!      matrix.
!    - Then solve the Newton system.

         PetscCallA(KSPSetOperators(ksp,J,B,ierr))
         PetscCallA(KSPSolve(ksp,F,Y,ierr))

!  Compute updated iterate

         PetscCallA(VecNorm(Y,NORM_2,ynorm,ierr))
         PetscCallA(VecAYPX(Y,mone,X,ierr))
         PetscCallA(VecCopy(Y,X,ierr))
         PetscCallA(VecNorm(X,NORM_2,xnorm,ierr))
         PetscCallA(KSPGetIterationNumber(ksp,lin_its,ierr))
         if (.not. nooutput) then
           print*,'linear solve iterations = ',lin_its,' xnorm = ',xnorm,' ynorm = ',ynorm
         endif

!  Evaluate nonlinear function at new location

         PetscCallA(ComputeFunction(X,F,ierr))
         PetscCallA(VecNorm(F,NORM_2,fnorm,ierr))
         if (.not. nooutput) then
           print*, 'Iteration ',i+1,' function norm',fnorm
         endif

!  Test for convergence

       if (fnorm .le. ttol) then
         if (.not. nooutput) then
           print*,'Converged: function norm ',fnorm,' tolerance ',ttol
         endif
         goto 20
       endif
 10   continue
 20   continue

      write(6,100) i+1
 100  format('Number of SNES iterations =',I2)

!     Check if mymult() produces a linear operator
      if (usemf) then
         N = 5
         PetscCallA(MatIsLinear(J,N,flg,ierr))
         if (.not. flg) then
            print *, 'IsLinear',flg
         endif
      endif

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Free work space.  All PETSc objects should be destroyed when they
!     are no longer needed.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

       PetscCallA(MatDestroy(B,ierr))
       if (usemf) then
         PetscCallA(MatDestroy(J,ierr))
       endif
       PetscCallA(VecDestroy(localX,ierr))
       PetscCallA(VecDestroy(X,ierr))
       PetscCallA(VecDestroy(Y,ierr))
       PetscCallA(VecDestroy(F,ierr))
       PetscCallA(KSPDestroy(ksp,ierr))
       PetscCallA(DMDestroy(da,ierr))
       PetscCallA(PetscFinalize(ierr))
       end

! -------------------------------------------------------------------
!
!   FormInitialGuess - Forms initial approximation.
!
!   Input Parameters:
!   X - vector
!
!   Output Parameter:
!   X - vector
!
      subroutine FormInitialGuess(X,ierr)
      use ex14fmodule
      implicit none

      PetscErrorCode    ierr
      PetscOffset      idx
      Vec       X
      PetscInt  i,j,row
      PetscInt  xs,ys,xm
      PetscInt  ym
      PetscReal one,lambda,temp1,temp,hx,hy
      PetscScalar      xx(2)

      one    = 1.0
      lambda = 6.0
      hx     = one/(mx-1)
      hy     = one/(my-1)
      temp1  = lambda/(lambda + one)

!  Get a pointer to vector data.
!    - VecGetArray() returns a pointer to the data array.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
       PetscCall(VecGetArray(X,xx,idx,ierr))

!  Get local grid boundaries (for 2-dimensional DMDA):
!    xs, ys   - starting grid indices (no ghost points)
!    xm, ym   - widths of local grid (no ghost points)

       PetscCall(DMDAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))

!  Compute initial guess over the locally owned part of the grid

      do 30 j=ys,ys+ym-1
        temp = (min(j,my-j-1))*hy
        do 40 i=xs,xs+xm-1
          row = i - xs + (j - ys)*xm + 1
          if (i .eq. 0 .or. j .eq. 0 .or. i .eq. mx-1 .or. j .eq. my-1) then
            xx(idx+row) = 0.0
            continue
          endif
          xx(idx+row) = temp1*sqrt(min((min(i,mx-i-1))*hx,temp))
 40     continue
 30   continue

!     Restore vector

       PetscCall(VecRestoreArray(X,xx,idx,ierr))
       return
       end

! -------------------------------------------------------------------
!
!   ComputeFunction - Evaluates nonlinear function, F(x).
!
!   Input Parameters:
!.  X - input vector
!
!   Output Parameter:
!.  F - function vector
!
      subroutine  ComputeFunction(X,F,ierr)
      use ex14fmodule
      implicit none

      Vec              X,F
      PetscInt         gys,gxm,gym
      PetscOffset      idx,idf
      PetscErrorCode ierr
      PetscInt i,j,row,xs,ys,xm,ym,gxs
      PetscInt rowf
      PetscReal two,one,lambda,hx
      PetscReal hy,hxdhy,hydhx,sc
      PetscScalar      u,uxx,uyy,xx(2),ff(2)

      two    = 2.0
      one    = 1.0
      lambda = 6.0

      hx     = one/(mx-1)
      hy     = one/(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.
!
      PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr))
      PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr))

!  Get pointers to vector data

      PetscCall(VecGetArray(localX,xx,idx,ierr))
      PetscCall(VecGetArray(F,ff,idf,ierr))

!  Get local grid boundaries

      PetscCall(DMDAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

!  Compute function over the locally owned part of the grid
      rowf = 0
      do 50 j=ys,ys+ym-1

        row  = (j - gys)*gxm + xs - gxs
        do 60 i=xs,xs+xm-1
          row  = row + 1
          rowf = rowf + 1

          if (i .eq. 0 .or. j .eq. 0 .or. i .eq. mx-1 .or. j .eq. my-1) then
            ff(idf+rowf) = xx(idx+row)
            goto 60
          endif
          u   = xx(idx+row)
          uxx = (two*u - xx(idx+row-1) - xx(idx+row+1))*hydhx
          uyy = (two*u - xx(idx+row-gxm) - xx(idx+row+gxm))*hxdhy
          ff(idf+rowf) = uxx + uyy - sc*exp(u)
 60     continue
 50   continue

!  Restore vectors

       PetscCall(VecRestoreArray(localX,xx,idx,ierr))
       PetscCall(VecRestoreArray(F,ff,idf,ierr))
       return
       end

! -------------------------------------------------------------------
!
!   ComputeJacobian - Evaluates Jacobian matrix.
!
!   Input Parameters:
!   x - input vector
!
!   Output Parameters:
!   jac - Jacobian matrix
!   flag - flag indicating matrix structure
!
!   Notes:
!   Due to grid point reordering with DMDAs, we must always work
!   with the local grid points, and then transform them to the new
!   global numbering with the 'ltog' mapping
!   We cannot work directly with the global numbers for the original
!   uniprocessor grid!
!
      subroutine ComputeJacobian(X,jac,ierr)
      use ex14fmodule
      implicit none

      Vec         X
      Mat         jac
      PetscInt     ltog(2)
      PetscOffset idltog,idx
      PetscErrorCode ierr
      PetscInt xs,ys,xm,ym
      PetscInt gxs,gys,gxm,gym
      PetscInt grow(1),i,j
      PetscInt row,ione
      PetscInt col(5),ifive
      PetscScalar two,one,lambda
      PetscScalar v(5),hx,hy,hxdhy
      PetscScalar hydhx,sc,xx(2)
      ISLocalToGlobalMapping ltogm

      ione   = 1
      ifive  = 5
      one    = 1.0
      two    = 2.0
      hx     = one/(mx-1)
      hy     = one/(my-1)
      sc     = hx*hy
      hxdhy  = hx/hy
      hydhx  = hy/hx
      lambda = 6.0

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.

      PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr))
      PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr))

!  Get pointer to vector data

      PetscCall(VecGetArray(localX,xx,idx,ierr))

!  Get local grid boundaries

      PetscCall(DMDAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

!  Get the global node numbers for all local nodes, including ghost points

      PetscCall(DMGetLocalToGlobalMapping(da,ltogm,ierr))
      PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,ltog,idltog,ierr))

!  Compute entries for the locally owned part of the Jacobian.
!   - Currently, all PETSc parallel matrix formats are partitioned by
!     contiguous chunks of rows across the processors. The 'grow'
!     parameter computed below specifies the global row number
!     corresponding to each local grid point.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global row and columns of matrix entries.
!   - Here, we set all entries for a particular row at once.

      do 10 j=ys,ys+ym-1
        row = (j - gys)*gxm + xs - gxs
        do 20 i=xs,xs+xm-1
          row = row + 1
          grow(1) = ltog(idltog+row)
          if (i .eq. 0 .or. j .eq. 0 .or. i .eq. (mx-1) .or. j .eq. (my-1)) then
             PetscCall(MatSetValues(jac,ione,grow,ione,grow,one,INSERT_VALUES,ierr))
             go to 20
          endif
          v(1)   = -hxdhy
          col(1) = ltog(idltog+row - gxm)
          v(2)   = -hydhx
          col(2) = ltog(idltog+row - 1)
          v(3)   = two*(hydhx + hxdhy) - sc*lambda*exp(xx(idx+row))
          col(3) = grow(1)
          v(4)   = -hydhx
          col(4) = ltog(idltog+row + 1)
          v(5)   = -hxdhy
          col(5) = ltog(idltog+row + gxm)
          PetscCall(MatSetValues(jac,ione,grow,ifive,col,v,INSERT_VALUES,ierr))
 20     continue
 10   continue

      PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm,ltog,idltog,ierr))

!  Assemble matrix, using the 2-step process:
!    MatAssemblyBegin(), MatAssemblyEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.

      PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(VecRestoreArray(localX,xx,idx,ierr))
      PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))
      return
      end

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
      subroutine  MyMult(J,X,F,ierr)
      use ex14fmodule
      implicit none

      Mat     J
      Vec     X,F
      PetscErrorCode ierr
!
!       Here we use the actual formed matrix B; users would
!     instead write their own matrix vector product routine
!
      PetscCall(MatMult(B,X,F,ierr))
      return
      end

!/*TEST
!
!   test:
!      args: -no_output -ksp_gmres_cgs_refinement_type refine_always
!      output_file: output/ex14_1.out
!      requires: !single
!
!TEST*/

!
!    "$Id: ex14f.F,v 1.14 1999/04/01 16:45:35 bsmith Exp bsmith $";
!
!  Solves a nonlinear system in parallel with a user-defined
!  Newton method that uses SLES to solve the linearized Newton sytems.  This solver
!  is a very simplistic inexact Newton method.  The intent of this code is to
!  demonstrate the repeated solution of linear sytems with the same nonzero pattern.
!
!  This is NOT the recommended approach for solving nonlinear problems with PETSc!
!  We urge users to employ the SNES component for solving nonlinear problems whenever
!  possible, as it offers many advantages over coding nonlinear solvers independently.
!
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DAs) to partition the parallel grid.
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
!/*T
!   Concepts: SLES^Writing a user-defined nonlinear solver (parallel Bratu example)
!   Concepts: DA^Using distributed arrays
!   Routines: SLESCreate(); SLESSetOperators(); SLESSolve(); SLESSetFromOptions();
!   Routines: DACreate2d(); DADestroy(); DACreateGlobalVector(); DACreateLocalVector();
!   Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal();
!   Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetGlobalIndices();
!   Routines: MatCreateShell(); MatShellSetOperation();
!   Processors: n
!T*/
!  ------------------------------------------------------------------------
!
!    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
!    the partial differential equation
!  
!            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
!  
!    with boundary conditions
!   
!             u = 0  for  x = 0, x = 1, y = 0, y = 1.
!  
!    A finite difference approximation with the usual 5-point stencil
!    is used to discretize the boundary value problem to obtain a nonlinear 
!    system of equations.
!
!    The SNES version of this problem is:  snes/examples/tutorials/ex5f.F
!
!  ------------------------------------------------------------------------- 

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!     petsc.h  - base PETSc routines   vec.h - vectors
!     sys.h    - system routines       mat.h - matrices
!     is.h     - index sets            ksp.h - Krylov subspace methods
!     viewer.h - viewers               pc.h  - preconditioners

#include "include/finclude/petsc.h"
#include "include/finclude/is.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/da.h"

      MPI_Comm comm
      SLES     sles
      Vec      X, Y, F
      Mat      J, B
      integer  Nx, Ny, flg, N, ierr
      integer  usemf
      common   /mycommon/ B
!
!      user() containes PETSc objects (vectors) hence we need to 
!     declare it as a PetscFortranAddr array rather than as an 
!     integer array to make sure it always has room to hold the 
!     PETSc object; it is an integer or integer(8) array
!     
      PetscFortranAddr  user(6)
!
!      This is the routine to use for matrix-free approach
!
      external mymult

!     --------------- Data to define nonlinear solver -------------- 
      double precision   rtol, xtol, ttol
      double precision   fnorm, ynorm, xnorm
      integer            max_nonlin_its
      integer            lin_its
      integer            i,m
      Scalar             mone

      mone           = -1.d0
      rtol           = 1.d-8
      xtol           = 1.d-8
      max_nonlin_its = 10

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      comm = PETSC_COMM_WORLD

!  Initialize problem parameters
!  
!    user(1) = mx, user(2) = my, user(3) = localX, user(4) = localF, user(5) = da
!
      user(1) = 4
      user(2) = 4
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-mx',user(1),flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-my',user(2),flg,ierr)
      N = user(1)*user(2)


!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Create linear solver context
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SLESCreate(comm,sles,ierr)

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Create vector data structures
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  
!  Create distributed array (DA) to manage parallel grid and vectors
!  
      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Nx',Nx,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Ny',Ny,flg,ierr)
      call DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,user(1),      &
     &     user(2),Nx,Ny,1,1,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,     &
     &     user(5),ierr)

!  
!  Extract global and local vectors from DA then duplicate for remaining
!  vectors that are the same types
!  
       call DACreateGlobalVector(user(5),X,ierr)
       call DACreateLocalVector(user(5),user(3),ierr)
       call VecDuplicate(X,F,ierr)
       call VecDuplicate(X,Y,ierr)
       call VecDuplicate(user(3),user(4),ierr)


!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Create matrix data structure for Jacobian
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!  
!     Note:  For the parallel case, vectors and matrices MUST be partitioned
!     accordingly.  When using distributed arrays (DAs) to create vectors,
!     the DAs determine the problem partitioning.  We must explicitly
!     specify the local matrix dimensions upon its creation for compatibility
!     with the vector distribution.  Thus, the generic MatCreate() routine
!     is NOT sufficient when working with distributed arrays.
!
!     Note: Here we only approximately preallocate storage space for the
!     Jacobian.  See the users manual for a discussion of better techniques
!     for preallocating matrix memory.
!  
      call VecGetLocalSize(X,m,ierr)
      call MatCreateMPIAIJ(comm,m,m,N,N,5,PETSC_NULL_INTEGER,3,         &
     &     PETSC_NULL_INTEGER,B,ierr)

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     if usemf is on then matrix vector product is done via matrix free
!     approach. Note this is just an example, and not realistic because 
!     we still use the actual formed matrix, but in reality one would
!     provide their own subroutine that would directly do the matrix 
!     vector product and not call MatMult()
!     Note: we put B into a common block so it will be visible to the
!     mymult() routine
      usemf = 0
      call OptionsHasName(PETSC_NULL_CHARACTER,'-mf',usemf,flg,ierr)
      if (usemf .eq. 1) then
         call MatCreateShell(comm,m,m,N,N,PETSC_NULL_INTEGER,J,ierr)
         call MatShellSetOperation(J,MATOP_MULT,mymult,ierr)
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
       call SLESSetFromOptions(sles,ierr)

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Evaluate initial guess
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       call FormInitialGuess(user,X,ierr)
       call ComputeFunction(user,X,F,ierr)
       call VecNorm(F,NORM_2,fnorm,ierr)
       ttol = fnorm*rtol
       print*, 'Initial function norm ',fnorm

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Solve nonlinear system with a user-defined method
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  This solver is a very simplistic inexact Newton method, with no
!  no damping strategies or bells and whistles. The intent of this code
!  is merely to demonstrate the repeated solution with SLES of linear
!  sytems with the same nonzero structure.
!
!  This is NOT the recommended approach for solving nonlinear problems
!  with PETSc!  We urge users to employ the SNES component for solving
!  nonlinear problems whenever possible with application codes, as it
!  offers many advantages over coding nonlinear solvers independently.

       do 10 i=0,max_nonlin_its

!  Compute the Jacobian matrix.  See the comments in this routine for
!  important information about setting the flag mat_flag.

         call ComputeJacobian(user,X,B,ierr)

!  Solve J Y = F, where J is the Jacobian matrix.
!    - First, set the SLES linear operators.  Here the matrix that
!      defines the linear system also serves as the preconditioning
!      matrix.
!    - Then solve the Newton system.

         call SLESSetOperators(sles,J,B,SAME_NONZERO_PATTERN,ierr)
         call SLESSolve(sles,F,Y,lin_its,ierr)

!  Compute updated iterate

         call VecNorm(Y,NORM_2,ynorm,ierr)
         call VecAYPX(mone,X,Y,ierr)
         call VecCopy(Y,X,ierr)
         call VecNorm(X,NORM_2,xnorm,ierr)
         print*,'linear solve iterations = ',lin_its,' xnorm = ',       &
     &        xnorm,' ynorm = ',ynorm

!  Evaluate nonlinear function at new location

         call ComputeFunction(user,X,F,ierr)
         call VecNorm(F,NORM_2,fnorm,ierr)
         print*, 'Iteration ',i+1,' function norm',fnorm

!  Test for convergence

       if (fnorm .le. ttol) then 
         print*,'Converged: function norm ',fnorm,' tolerance ',ttol
         goto 20
       endif
 10   continue
 20   continue

      print*, 'Number of Newton iterations = ',i+1

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Free work space.  All PETSc objects should be destroyed when they
!     are no longer needed.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       call MatDestroy(B,ierr)
       if ( usemf .ne. 0) then
         call MatDestroy(J,ierr)
       endif
       call VecDestroy(user(3),ierr)
       call VecDestroy(X,ierr)
       call VecDestroy(Y,ierr)
       call VecDestroy(user(4),ierr)
       call VecDestroy(F,ierr)
       call SLESDestroy(sles,ierr)
       call DADestroy(user(5),ierr)
       call PetscFinalize(ierr)
       end

! ------------------------------------------------------------------- 
! 
!   FormInitialGuess - Forms initial approximation.
!
!   Input Parameters:
!   user - user-defined application context
!   X - vector
!
!   Output Parameter:
!   X - vector
! 
      subroutine FormInitialGuess(user,X,ierr)
      implicit none

!     petsc.h  - base PETSc routines   vec.h - vectors
!     sys.h    - system routines       mat.h - matrices
!     is.h     - index sets            ksp.h - Krylov subspace methods
!     viewer.h - viewers               pc.h  - preconditioners

#include "include/finclude/petsc.h"
#include "include/finclude/is.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/da.h"
      integer          user(6),ierr
      PetscOffset      idx
      Vec              X,localX
      integer          i, j, row, mx, my,  xs, ys, xm
      integer          ym, gxm, gym, gxs, gys
      double precision one, lambda, temp1, temp, hx, hy
      double precision hxdhy, hydhx,sc
      Scalar           xx(1)
      DA               da
 
      da     = user(5)
      localX = user(3)
      one    = 1.d0
      mx     = user(1)
      my     = user(2) 
      lambda = 6.d0
      hx     = one/(mx-1)
      hy     = one/(my-1)
      sc     = hx*hy*lambda  
      hxdhy  = hx/hy 
      hydhx  = hy/hx
      temp1  = lambda/(lambda + one)

!  Get a pointer to vector data.
!    - VecGetArray() returns a pointer to the data array.  
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
       call VecGetArray(localX,xx,idx,ierr)

!  Get local grid boundaries (for 2-dimensional DA):
!    xs, ys   - starting grid indices (no ghost points)
!    xm, ym   - widths of local grid (no ghost points)
!    gxs, gys - starting grid indices (including ghost points)
!    gxm, gym - widths of local grid (including ghost points)

       call DAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,             &
     &      PETSC_NULL_INTEGER,ierr)
       call DAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,    &
     &      PETSC_NULL_INTEGER,ierr)

!  Compute initial guess over the locally owned part of the grid

      do 30 j=ys,ys+ym-1
        temp = (min(j,my-j-1))*hy
        do 40 i=xs,xs+xm-1
          row = i - gxs + (j - gys)*gxm + 1
          if (i .eq. 0 .or. j .eq. 0 .or. i .eq. mx-1 .or.              &
     &        j .eq. my-1) then
            xx(idx+row) = 0.d0 
            continue
          endif
          xx(idx+row) = temp1*sqrt(min( (min(i,mx-i-1))*hx,temp))
 40     continue
 30   continue

!     Restore vector

       call VecRestoreArray(localX,xx,idx,ierr)

!     Insert values into global vector

       call DALocalToGlobal(da,localX,INSERT_VALUES,X,ierr)
       return 
       end

! ------------------------------------------------------------------- 
! 
!   ComputeFunction - Evaluates nonlinear function, F(x).
!
!   Input Parameters:
!.  X - input vector
!.  user - user-defined application context
!
!   Output Parameter:
!.  F - function vector
! 
      subroutine  ComputeFunction(user,X,F,ierr)
      implicit none

!     petsc.h  - base PETSc routines   vec.h - vectors
!     sys.h    - system routines       mat.h - matrices
!     is.h     - index sets            ksp.h - Krylov subspace methods
!     viewer.h - viewers               pc.h  - preconditioners

#include "include/finclude/petsc.h"
#include "include/finclude/is.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/da.h"

      Vec              X,F,localX, localF
      integer          user(6), gys, gxm, gym
      PetscOffset      idx,idf
      integer          ierr, i, j, row, mx, my, xs,ys,xm,ym,gxs 
      double precision two, one, lambda,hx
      double precision hy, hxdhy, hydhx,sc
      Scalar           u, uxx, uyy, xx(1),ff(1)
      DA               da

      two    = 2.d0
      one    = 1.d0
      da     = user(5)
      localX = user(3)
      localF = user(4)
      lambda = 6.d0

      mx     = user(1)
      my     = user(2)
      hx     = one/(mx-1)
      hy     = one/(my-1)
      sc     = hx*hy*lambda 
      hxdhy  = hx/hy  
      hydhx  = hy/hx

!  Scatter ghost points to local vector, using the 2-step process
!     DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.
!  
      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get pointers to vector data

      call VecGetArray(localX,xx,idx,ierr)
      call VecGetArray(localF,ff,idf,ierr)

!  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,              &
     &     PETSC_NULL_INTEGER,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,     &
     &     PETSC_NULL_INTEGER,ierr)

!  Compute function over the locally owned part of the grid

      do 50 j=ys,ys+ym-1

        row = (j - gys)*gxm + xs - gxs
        do 60 i=xs,xs+xm-1
          row = row + 1

          if (i .eq. 0 .or. j .eq. 0 .or. i .eq. mx-1 .or.              &
     &        j .eq. my-1) then
            ff(idf+row) = xx(idx+row)
            goto 60
          endif
          u   = xx(idx+row)
          uxx = (two*u - xx(idx+row-1) - xx(idx+row+1))*hydhx
          uyy = (two*u - xx(idx+row-gxm) - xx(idx+row+gxm))*hxdhy
          ff(idf+row) = uxx + uyy - sc*exp(u)
 60     continue
 50   continue

!  Restore vectors

       call VecRestoreArray(localX,xx,idx,ierr)
       call VecRestoreArray(localF,ff,idf,ierr)

!  Insert values into global vector

       call DALocalToGlobal(da,localF,INSERT_VALUES,F,ierr)
       return 
       end

! ------------------------------------------------------------------- 
!
!   ComputeJacobian - Evaluates Jacobian matrix.
!
!   Input Parameters:
!   x - input vector
!   user - user-defined application context
!
!   Output Parameters:
!   jac - Jacobian matrix
!   flag - flag indicating matrix structure
!
!   Notes:
!   Due to grid point reordering with DAs, we must always work
!   with the local grid points, and then transform them to the new
!   global numbering with the 'ltog' mapping (via DAGetGlobalIndices()).
!   We cannot work directly with the global numbers for the original
!   uniprocessor grid!
!
      subroutine ComputeJacobian(user,X,jac,ierr)
      implicit none

!     petsc.h  - base PETSc routines   vec.h - vectors
!     sys.h    - system routines       mat.h - matrices
!     is.h     - index sets            ksp.h - Krylov subspace methods
!     viewer.h - viewers               pc.h  - preconditioners

#include "include/finclude/petsc.h"
#include "include/finclude/is.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/da.h"

      integer     user(6)
      Vec         X
      Mat         jac
      Vec         localX
      DA          da
      integer     ltog(1)
      PetscOffset idltog,idx
      integer     ierr, i, j, row, mx, my, col(5)
      integer     nloc, xs, ys, xm, ym, gxs, gys, gxm, gym, grow
      Scalar      two, one, lambda, v(5), hx, hy, hxdhy
      Scalar      hydhx, sc, xx(1)

      da     = user(5)
      one    = 1.d0
      two    = 2.d0
      localX = user(3)
      da     = user(5)
      mx     = user(1)
      my     = user(2)
      hx     = one/(mx-1)
      hy     = one/(my-1)
      sc     = hx*hy  
      hxdhy  = hx/hy        
      hydhx  = hy/hx
      lambda = 6.d0

!  Scatter ghost points to local vector, using the 2-step process
!     DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get pointer to vector data

      call VecGetArray(localX,xx,idx,ierr)

!  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,              &
     &     PETSC_NULL_INTEGER,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,     &
     &                        PETSC_NULL_INTEGER,ierr)

!  Get the global node numbers for all local nodes, including ghost points

      call DAGetGlobalIndices(da,nloc,ltog,idltog,ierr)

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
          grow = ltog(idltog+row)
          if (i .eq. 0 .or. j .eq. 0 .or. i .eq. (mx-1) .or.            &
     &        j .eq. (my-1)) then
             call MatSetValues(jac,1,grow,1,grow,one,INSERT_VALUES,ierr)
             go to 20
          endif
          v(1)   = -hxdhy
          col(1) = ltog(idltog+row - gxm)
          v(2)   = -hydhx 
          col(2) = ltog(idltog+row - 1)
          v(3)   = two*(hydhx + hxdhy) - sc*lambda*exp(xx(idx+row))
          col(3) = grow
          v(4)   = -hydhx 
          col(4) = ltog(idltog+row + 1)
          v(5)   = -hxdhy 
          col(5) = ltog(idltog+row + gxm)
          call MatSetValues(jac,1,grow,5,col,v,INSERT_VALUES,ierr)
 20     continue
 10   continue

!  Assemble matrix, using the 2-step process:
!    MatAssemblyBegin(), MatAssemblyEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArray(localX,xx,idx,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return 
      end


! ------------------------------------------------------------------- 
! 
!   MyMult - user provided matrix multiply 
!
!   Input Parameters:
!.  user - user-defined application context
!.  X - input vector
!
!   Output Parameter:
!.  F - function vector
! 
      subroutine  MyMult(J,X,F,ierr)
      implicit none
      Mat     J,B
      Vec     X,F
      integer ierr

      common   /mycommon/ B
!
!       Here we use the actual formed matrix B; users would 
!     instead write their own matrix vector product routine
!
      call MatMult(B,X,F,ierr)
      return
      end






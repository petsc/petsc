C
C    "$Id: ex14f.F,v 1.4 1997/04/25 02:35:47 curfman Exp balay $";
C
C  Solves a nonlinear system in parallel with a user-defined
C  Newton method that uses SLES to solve the linearized Newton sytems.  This solver
C  is a very simplistic inexact Newton method.  The intent of this code is to
C  demonstrate the repeated solution of linear sytems with the same nonzero pattern.
C
C  This is NOT the recommended approach for solving nonlinear problems with PETSc!
C  We urge users to employ the SNES component for solving nonlinear problems whenever
C  possible, as it offers many advantages over coding nonlinear solvers independently.
C
C  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
C  domain, using distributed arrays (DAs) to partition the parallel grid.
C
C  The command line options include:
C  -par <parameter>, where <parameter> indicates the problem's nonlinearity
C     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
C  -mx <xg>, where <xg> = number of grid points in the x-direction
C  -my <yg>, where <yg> = number of grid points in the y-direction
C  -Nx <npx>, where <npx> = number of processors in the x-direction
C  -Ny <npy>, where <npy> = number of processors in the y-directionn'
C
C/*T
C   Concepts: SLES^Writing a user-defined nonlinear solver (parallel Bratu example)
C   Concepts: DA^Using distributed arrays
C   Routines: SLESCreate() SLESSetOperators() SLESSolve() SLESSetFromOptions()
C   Routines: DACreate2d() DADestroy() DAGetDistributedVector() DAGetLocalVector()
C   Routines: DAGetCorners() DAGetGhostCorners() DALocalToGlobal()
C   Routines: DAGlobalToLocalBegin() DAGlobalToLocalEnd() DAGetGlobalIndices()
C   Processors: n
CT*/
C  ------------------------------------------------------------------------
C
C    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
C    the partial differential equation
C  
C            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
C  
C    with boundary conditions
C   
C             u = 0  for  x = 0, x = 1, y = 0, y = 1.
C  
C    A finite difference approximation with the usual 5-point stencil
C    is used to discretize the boundary value problem to obtain a nonlinear 
C    system of equations.
C
C    The SNES version of this problem is:  snes/examples/tutorials/ex5f.F
C
C  ------------------------------------------------------------------------- 

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C     petsc.h  - base PETSc routines   vec.h - vectors
C     sys.h    - system routines       mat.h - matrices
C     is.h     - index sets            ksp.h - Krylov subspace methods
C     viewer.h - viewers               pc.h  - preconditioners

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/is.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/da.h"

      MPI_Comm comm
      SLES     sles
      Vec      X, Y, F
      Mat      J
      integer  Nx, Ny, flg, N, ierr
      integer  user(6)

C     --------------- Data to define nonlinear solver -------------- 
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

C  Initialize problem parameters
C  
C    user(1) = mx, user(2) = my, user(3) = localX, user(4) = localF, user(5) = da
C
      user(1) = 4
      user(2) = 4
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-mx',user(1),flg,
     &                       ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-my',user(2),flg,ierr)
      N = user(1)*user(2)

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Create linear solver context
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SLESCreate(comm,sles,ierr)

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Create vector data structures
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  
C  Create distributed array (DA) to manage parallel grid and vectors
C  
      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Nx',Nx,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Ny',Ny,flg,ierr)
      call DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,user(1),
     &          user(2),Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,user(5),ierr)

C  
C  Extract global and local vectors from DA then duplicate for remaining
C  vectors that are the same types
C  
       call DAGetDistributedVector(user(5),X,ierr)
       call DAGetLocalVector(user(5),user(3),ierr)
       call VecDuplicate(X,F,ierr)
       call VecDuplicate(X,Y,ierr)
       call VecDuplicate(user(3),user(4),ierr)


C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Create matrix data structure for Jacobian
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C  
C     Note:  For the parallel case, vectors and matrices MUST be partitioned
C     accordingly.  When using distributed arrays (DAs) to create vectors,
C     the DAs determine the problem partitioning.  We must explicitly
C     specify the local matrix dimensions upon its creation for compatibility
C     with the vector distribution.  Thus, the generic MatCreate() routine
C     is NOT sufficient when working with distributed arrays.
C
C     Note: Here we only approximately preallocate storage space for the
C     Jacobian.  See the users manual for a discussion of better techniques
C     for preallocating matrix memory.
C  
      call VecGetLocalSize(X,m,ierr)
      call MatCreateMPIAIJ(comm,m,m,N,N,5,PETSC_NULL,3,PETSC_NULL,
     &                     J,ierr)

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Customize linear solver set runtime options
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C  
C     Set runtime options (e.g., -ksp_monitor -ksp_rtol <rtol> -ksp_type <type>)
C  
       call SLESSetFromOptions(sles,ierr)

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Evaluate initial guess
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       call FormInitialGuess(user,X,ierr)
       call ComputeFunction(user,X,F,ierr)
       call VecNorm(F,NORM_2,fnorm,ierr)
       ttol = fnorm*rtol
       print*, 'Initial function norm ',fnorm

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Solve nonlinear system with a user-defined method
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  This solver is a very simplistic inexact Newton method, with no
C  no damping strategies or bells and whistles. The intent of this code
C  is merely to demonstrate the repeated solution with SLES of linear
C  sytems with the same nonzero structure.
C
C  This is NOT the recommended approach for solving nonlinear problems
C  with PETSc!  We urge users to employ the SNES component for solving
C  nonlinear problems whenever possible with application codes, as it
C  offers many advantages over coding nonlinear solvers independently.

       do 10 i=0,max_nonlin_its

C  Compute the Jacobian matrix.  See the comments in this routine for
C  important information about setting the flag mat_flag.

         call ComputeJacobian(user,X,J,ierr)

C  Solve J Y = F, where J is the Jacobian matrix.
C    - First, set the SLES linear operators.  Here the matrix that
C      defines the linear system also serves as the preconditioning
C      matrix.
C    - Then solve the Newton system.

         call SLESSetOperators(sles,J,J,SAME_NONZERO_PATTERN,ierr)
         call SLESSolve(sles,F,Y,lin_its,ierr)

C  Compute updated iterate

         call VecNorm(Y,NORM_2,ynorm,ierr)
         call VecAYPX(mone,X,Y,ierr)
         call VecCopy(Y,X,ierr)
         call VecNorm(X,NORM_2,xnorm,ierr)
         print*,'linear solve iterations = ',lin_its,' xnorm = ',
     &        xnorm,' ynorm = ',ynorm

C  Evaluate nonlinear function at new location

         call ComputeFunction(user,X,F,ierr)
         call VecNorm(F,NORM_2,fnorm,ierr)
         print*, 'Iteration ',i+1,' function norm',fnorm

C  Test for convergence

       if (fnorm .le. ttol) then 
         print*,'Converged: function norm ',fnorm,' tolerance ',ttol
         goto 20
       endif
 10   continue
 20   continue

      print*, 'Number of Newton iterations = ',i+1

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Free work space.  All PETSc objects should be destroyed when they
C     are no longer needed.
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       call MatDestroy(J,ierr)
       call VecDestroy(user(3),ierr)
       call VecDestroy(X,ierr)
       call VecDestroy(user(4),ierr)
       call VecDestroy(F,ierr)
       call SLESDestroy(sles,ierr)
       call DADestroy(user(5),ierr)
       call PetscFinalize(ierr)
       end

C ------------------------------------------------------------------- 
C 
C   FormInitialGuess - Forms initial approximation.
C
C   Input Parameters:
C   user - user-defined application context
C   X - vector
C
C   Output Parameter:
C   X - vector
C 
      subroutine FormInitialGuess(user,X,ierr)
      implicit none

C     petsc.h  - base PETSc routines   vec.h - vectors
C     sys.h    - system routines       mat.h - matrices
C     is.h     - index sets            ksp.h - Krylov subspace methods
C     viewer.h - viewers               pc.h  - preconditioners

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/is.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/da.h"
      integer          user(6),ierr,idx
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

C  Get a pointer to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
       call VecGetArray(localX,xx,idx,ierr)

C  Get local grid boundaries (for 2-dimensional DA):
C    xs, ys   - starting grid indices (no ghost points)
C    xm, ym   - widths of local grid (no ghost points)
C    gxs, gys - starting grid indices (including ghost points)
C    gxm, gym - widths of local grid (including ghost points)

       call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
       call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                        PETSC_NULL,ierr)

C  Compute initial guess over the locally owned part of the grid

      do 30 j=ys,ys+ym-1
        temp = (min(j,my-j-1))*hy
        do 40 i=xs,xs+xm-1
          row = i - gxs + (j - gys)*gxm + 1
          if (i.eq.0 .or. j.eq.0 .or. i.eq.mx-1 .or. j.eq.my-1) then
            xx(idx+row) = 0.0 
            continue
          endif
          xx(idx+row) = temp1*sqrt(min( (min(i,mx-i-1))*hx,temp))
 40     continue
 30   continue

C     Restore vector

       call VecRestoreArray(localX,xx,idx,ierr)

C     Insert values into global vector

       call DALocalToGlobal(da,localX,INSERT_VALUES,X,ierr)
       return 
       end

C ------------------------------------------------------------------- 
C 
C   ComputeFunction - Evaluates nonlinear function, F(x).
C
C   Input Parameters:
C.  X - input vector
C.  user - user-defined application context
C
C   Output Parameter:
C.  F - function vector
C 
      subroutine  ComputeFunction(user,X,F,ierr)
      implicit none

C     petsc.h  - base PETSc routines   vec.h - vectors
C     sys.h    - system routines       mat.h - matrices
C     is.h     - index sets            ksp.h - Krylov subspace methods
C     viewer.h - viewers               pc.h  - preconditioners

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/is.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/da.h"

      Vec              X,F,localX, localF
      integer          user(6), gys, gxm, gym,idx,idf
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

C  Scatter ghost points to local vector, using the 2-step process
C     DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
C  By placing code between these two statements, computations can be
C  done while messages are in transition.
C  
      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

C  Get pointers to vector data

      call VecGetArray(localX,xx,idx,ierr)
      call VecGetArray(localF,ff,idf,ierr)

C  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,PETSC_NULL,
     &                        ierr)

C  Compute function over the locally owned part of the grid

      do 50 j=ys,ys+ym-1

        row = (j - gys)*gxm + xs - gxs
        do 60 i=xs,xs+xm-1
          row = row + 1

          if (i.eq.0 .or. j.eq.0 .or. i.eq.mx-1 .or. j.eq.my-1) then
            ff(idf+row) = xx(idx+row)
            goto 60
          endif
          u   = xx(idx+row)
          uxx = (two*u - xx(idx+row-1) - xx(idx+row+1))*hydhx
          uyy = (two*u - xx(idx+row-gxm) - xx(idx+row+gxm))*hxdhy
          ff(idf+row) = uxx + uyy - sc*exp(u)
 60     continue
 50   continue

C  Restore vectors

       call VecRestoreArray(localX,xx,idx,ierr)
       call VecRestoreArray(localF,ff,idf,ierr)

C  Insert values into global vector

       call DALocalToGlobal(da,localF,INSERT_VALUES,F,ierr)
       return 
       end

C ------------------------------------------------------------------- 
C
C   ComputeJacobian - Evaluates Jacobian matrix.
C
C   Input Parameters:
C   x - input vector
C   user - user-defined application context
C
C   Output Parameters:
C   jac - Jacobian matrix
C   flag - flag indicating matrix structure
C
C   Notes:
C   Due to grid point reordering with DAs, we must always work
C   with the local grid points, and then transform them to the new
C   global numbering with the 'ltog' mapping (via DAGetGlobalIndices()).
C   We cannot work directly with the global numbers for the original
C   uniprocessor grid!
C
      subroutine ComputeJacobian(user,X,jac,ierr)
      implicit none

C     petsc.h  - base PETSc routines   vec.h - vectors
C     sys.h    - system routines       mat.h - matrices
C     is.h     - index sets            ksp.h - Krylov subspace methods
C     viewer.h - viewers               pc.h  - preconditioners

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/is.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/da.h"

      integer user(6)
      Vec     X
      Mat     jac
      Vec     localX
      DA      da
      integer ltog(1),idltog
      integer ierr, i, j, row, mx, my, col(5),idx 
      integer nloc, xs, ys, xm, ym, gxs, gys, gxm, gym, grow
      Scalar  two, one, lambda, v(5), hx, hy, hxdhy
      Scalar  hydhx, sc, xx(1)

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

C  Scatter ghost points to local vector, using the 2-step process
C     DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
C  By placing code between these two statements, computations can be
C  done while messages are in transition.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

C  Get pointer to vector data

      call VecGetArray(localX,xx,idx,ierr)

C  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                        PETSC_NULL,ierr)

C  Get the global node numbers for all local nodes, including ghost points

      call DAGetGlobalIndices(da,nloc,ltog,idltog,ierr)

C  Compute entries for the locally owned part of the Jacobian.
C   - Currently, all PETSc parallel matrix formats are partitioned by
C     contiguous chunks of rows across the processors. The 'grow'
C     parameter computed below specifies the global row number 
C     corresponding to each local grid point.
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Always specify global row and columns of matrix entries.
C   - Here, we set all entries for a particular row at once.

      do 10 j=ys,ys+ym-1
        row = (j - gys)*gxm + xs - gxs 
        do 20 i=xs,xs+xm-1
          row = row + 1
          grow = ltog(idltog+row)
          if (i.eq.0 .or. j.eq.0 .or. i. eq. (mx-1) .or. 
     &        j .eq. (my-1)) then
             call MatSetValues(jac,1,grow,1,grow,one,INSERT_VALUES,
     &                         ierr)
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

C  Assemble matrix, using the 2-step process:
C    MatAssemblyBegin(), MatAssemblyEnd().
C  By placing code between these two statements, computations can be
C  done while messages are in transition.

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArray(localX,xx,idx,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return 
      end







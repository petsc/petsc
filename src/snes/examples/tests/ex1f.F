
C "$Id: ex5f.F,v 1.13 1996/09/30 20:11:03 bsmith Exp curfman $";
C
C  Description: This example solves a nonlinear system on 1 processor with SNES.
C  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
C  domain.  The command line options include:
C    -par <parameter>, where <parameter> indicates the nonlinearity of the problem
C       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
C    -mx <xg>, where <xg> = number of grid points in the x-direction
C    -my <yg>, where <yg> = number of grid points in the y-direction
C
C/*T
C  Concepts: SNES^Solving a system of nonlinear equations (sequential Bratu example);
C  Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
C  Routines: SNESSolve(); SNESSetFromOptions();
C  Routines: DrawOpenX();
C  Processors: 1
CT*/
C
C  --------------------------------------------------------------------------
C 
C  Solid Fuel Ignition (SFI) problem.  This problem is modeled by
C  the partial differential equation
C  
C          -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
C 
C  with boundary conditions
C  
C           u = 0  for  x = 0, x = 1, y = 0, y = 1.
C 
C  A finite difference approximation with the usual 5-point stencil
C  is used to discretize the boundary value problem to obtain a nonlinear 
C  system of equations.
C
C  The parallel version of this code is snes/examples/tutorials/ex5f.F
C
C  --------------------------------------------------------------------------

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  The following include statements are generally used in SNES Fortran
C  programs:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     mat.h    - matrices
C     ksp.h    - Krylov subspace methods
C     pc.h     - preconditioners
C     sles.h   - SLES interface
C     snes.h   - SNES interface
C  In addition, we need the following for use of PETSc drawing routines
C     draw.h   - drawing routines
C  Other include statements may be needed if using additional PETSc
C  routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/draw.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/snes.h"
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  The application context to contain data needed by the 
C  application-provided call-back routines, FormJacobian() and
C  FormFunction().  In this example the application context is
C  simply a Fortran integer array. 
C     user(1) = mx     - discretization in x-direction
C          2  = my     - discretization in y-direction
C
C  Variables:
C     snes        - nonlinear solver
C     x, r        - solution, residual vectors
C     J           - Jacobian matrix
C     user        - user-defined work context
C     its         - iterations for convergence
C     matrix_free - flag - 1 indicates matrix-free version
C     lambda      - nonlinearity parameter
C     draw        - drawing context
C
      SNES     snes
      Vec      x, r
      Draw     draw
      Mat      J
      integer  its, matrix_free, flg, N, ierr
      integer  user(2), mx, my, size, rank
      Double   lambda_max, lambda_min, lambda

C  Store nonlinearity parameter in common block

      common /params/ lambda

C  Note: Any user-defined Fortran routines (such as FormJacobian)
C  MUST be declared as external.

      external FormFunction, FormInitialGuess, FormJacobian

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                   Macro definitions
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  Macros to make clearer the process of setting values in vectors and
C  getting values from vectors.  These vectors are used in the routines
C  FormFunction(), FormInitialGuess(), and FormJacobian().
C
C   - The element lx_a(ib) is element ib+1 in the vector, x
C   - Here we add 1 to the base array index to facilitate the use of
C     conventional Fortran 1-based array indexing.
C
#define lx_a(ib) lx_v(lx_i + (ib) + 1)
#define lf_a(ib) lf_v(lf_i + (ib) + 1)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
         call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
         if (rank .eq. 0)
     &      write(6,*) 'This is a uniprocessor example only!'
         SETERRA(1,0)
      endif

C  Initialize problem parameters

      lambda_max = 6.81
      lambda_min = 0.0
      lambda     = 6.0
      mx         = 4
      my         = 4
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-mx',mx,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-my',my,flg,ierr)
      call OptionsGetDouble(PETSC_NULL_CHARACTER,'-par',lambda,flg,ierr)
      if (lambda .ge. lambda_max .or. lambda .le. lambda_min) then
         if (rank .eq. 0) write(6,*) 'Lambda is out of range'
         SETERRA(1,0)
      endif
      user(1) = mx
      user(2) = my
      N       = mx*my

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create nonlinear solver context
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,ierr)

C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C     Create vector data structures; set function evaluation routine
C  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call VecCreate(MPI_COMM_WORLD,N,x,ierr)
      call VecDuplicate(x,r,ierr)

C  Set function evaluation routine and vector.  Whenever the nonlinear
C  solver needs to evaluate the nonlinear function, it will call this
C  routine.
C   - Note that the final routine argument is the user-defined
C     context that provides application-specific data for the
C     function evaluation routine.

      call SNESSetFunction(snes,r,FormFunction,user,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create matrix data structure; set Jacobian evaluation routine
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Create matrix. Here we only approximately preallocate storage space
C  for the Jacobian.  See the users manual for a discussion of better 
C  techniques for preallocating matrix memory.

      call OptionsHasName(PETSC_NULL_CHARACTER,'-snes_mf',matrix_free,
     &                    ierr)
      if (matrix_free .eq. 0) then
        call MatCreateSeqAIJ(MPI_COMM_WORLD,N,N,5,PETSC_NULL,J,ierr)
      endif

C  Set Jacobian matrix data structure and default Jacobian evaluation
C  routine.  Whenever the nonlinear solver needs to compute the
C  Jacobian matrix, it will call this routine.
C   - Note that the final routine argument is the user-defined
C     context that provides application-specific data for the
C     Jacobian evaluation routine.
C   - The user can override with:
C      -snes_fd : default finite differencing approximation of Jacobian
C      -snes_mf : matrix-free Newton-Krylov method with no preconditioning
C                 (unless user explicitly sets preconditioner) 
C      -snes_mf_operator : form preconditioning matrix as set by the user,
C                          but use matrix-free approx for Jacobian-vector
C                          products within Newton-Krylov method
C
      if (matrix_free .eq. 0) then
        call SNESSetJacobian(snes,J,J,FormJacobian,user,ierr)
      endif

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Customize nonlinear solver; set runtime options
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)

      call SNESSetFromOptions(snes,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Evaluate initial guess; then solve nonlinear system.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Note: The user should initialize the vector, x, with the initial guess
C  for the nonlinear solver prior to calling SNESSolve().  In particular,
C  to employ an initial guess of zero, the user should explicitly set
C  this vector to zero by calling VecSet().

      call FormInitialGuess(user,x,ierr)
      call SNESSolve(snes,x,its,ierr) 
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

C  Draw contour plot of solution

c      call DrawOpenX(MPI_COMM_WORLD,0,"Solution",300,0,300,300,
c     &               draw,ierr)
c      call DrawTensorContour(draw,mx,my,0,0,x,ierr)
c      call DrawSyncFlush(draw,ierr)
c      call DrawPause(draw,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (matrix_free .eq. 0) call MatDestroy(J,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call SNESDestroy(snes,ierr)
c      call DrawDestroy(draw,ierr)
      call PetscFinalize(ierr)

      stop
      end

C ---------------------------------------------------------------------
C
C  FormInitialGuess - Forms initial approximation.
C
C  Input Parameters:
C  user - user-defined application context
C  X - vector
C
C  Output Parameter:
C  X - vector
C
      subroutine FormInitialGuess(user,X,ierr)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/snes.h"

      Vec      X
      integer  user(*)
      integer  i, j, row, mx, my, ierr, hxdhy, hydhx
      Scalar   lambda, temp1, temp, hx, hy, sc, one

C  Declarations for use with local arrays
      Scalar   lx_v(1)
      integer  lx_i

      common /params/ lambda

C  Set parameters

      ierr   = 0
      one    = 1.0
      mx     = user(1)
      my     = user(2)
      hx     = one/(DBLE(mx-1))
      hy     = one/(DBLE(my-1))
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx
      temp1  = lambda/(lambda + one)

C  Get a pointer to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the users manual for details.

      call VecGetArray(X,lx_v,lx_i,ierr)

C  Compute initial guess 

      do 20 j=0,my-1
         temp = DBLE(min(j,my-j-1))*hy
         do 10 i=0,mx-1
            row = i + j*mx
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
              lx_a(row) = 0.0
            else
              lx_a(row) = temp1 *
     &          sqrt(min(DBLE(min(i,mx-i-1))*hx,temp))
            endif
 10      continue
 20   continue


C  Restore vector

      call VecRestoreArray(X,lx_v,lx_i,ierr)

      return 
      end

C ---------------------------------------------------------------------
C
C  FormFunction - Evaluates nonlinear function, F(x).
C
C  Input Parameters:
C  snes - the SNES context
C  X - input vector
C  user - optional user-defined context, as set by SNESSetFunction()
C
C  Output Parameter:
C  F - function vector
C
      subroutine FormFunction(snes,X,F,user)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/snes.h"

      SNES     snes
      Vec      X, F
      Scalar   two, one, lambda, hx, hy, hxdhy, hydhx, sc
      Scalar   u, uxx, uyy
      integer  ierr, i, j, row, mx, my, user(*)

C  Declarations for use with local arrays

      Scalar   lx_v(1), lf_v(1)
      integer  lx_i, lf_i 

      common /params/ lambda

      one    = 1.0
      two    = 2.0
      mx     = user(1)
      my     = user(2)
      hx     = one/DBLE(mx-1)
      hy     = one/DBLE(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

C  Get pointers to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the Fortran chapter of the users manual for details.

      call VecGetArray(X,lx_v,lx_i,ierr)
      call VecGetArray(F,lf_v,lf_i,ierr)

C  Compute function 

      do 20 j=0,my-1
         row = j*mx - 1
         do 10 i=0,mx-1
            row = row + 1
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
               lf_a(row) = lx_a(row)
            else
               u = lx_a(row)
               uxx = hydhx * (two*u
     &               - lx_a(row-1)
     &               - lx_a(row+1))
               uyy = hxdhy * (two*u 
     &               - lx_a(row-mx) 
     &               - lx_a(row+mx))
               lf_a(row) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

C  Restore vectors

      call VecRestoreArray(X,lx_v,lx_i,ierr)
      call VecRestoreArray(F,lf_v,lf_i,ierr)

      call PLogFlops(11*mx*my,ierr)

      return 
      end

C ---------------------------------------------------------------------
C
C  FormJacobian - Evaluates Jacobian matrix.
C
C  Input Parameters:
C  snes - the SNES context
C  x - input vector
C  user - optional user-defined context, as set by SNESSetJacobian()
C
C  Output Parameters:
C  A - Jacobian matrix
C  B - optionally different preconditioning matrix
C  flag - flag indicating matrix structure
C
      subroutine FormJacobian(snes,X,jac,B,flag,user)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/snes.h"

      SNES         snes
      Vec          X
      Mat          jac, B
      MatStructure flag
      integer      user(*), ierr, i, j, row, mx, my, col(5)
      Scalar       two, one, lambda, hx, hy, hxdhy, hydhx
      Scalar       sc, v(5)

      common /params/ lambda

C  Declarations for use with local array

      Scalar   lx_v(1)
      integer  lx_i

C  Set parameters

      one    = 1.0
      two    = 2.0
      mx     = user(1)
      my     = user(2)
      hx     = one/DBLE(mx-1)
      hy     = one/DBLE(my-1)
      sc     = hx*hy
      hxdhy  = hx/hy
      hydhx  = hy/hx

C  Get a pointer to vector data

      call VecGetArray(X,lx_v,lx_i,ierr)


C  Compute entries of the Jacobian matrix
C   - Here, we set all entries for a particular row at once.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C.

      do 20 j=0,my-1
         row = j*mx - 1
         do 10 i=0,mx-1
            row = row + 1
C           boundary points
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
               call MatSetValues(jac,1,row,1,row,one,
     &                           INSERT_VALUES,ierr)
C           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy) 
     &                  - sc*lambda*exp(lx_a(row))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - mx
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + mx
               call MatSetValues(jac,1,row,5,col,v,
     &                           INSERT_VALUES,ierr)
            endif
 10      continue
 20   continue

C  Restore vector

      call VecRestoreArray(X,lx_v,lx_i,ierr)

C  Assemble matrix

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)

C  Set flag to indicate that the Jacobian matrix retains an identical
C  nonzero structure throughout all nonlinear iterations (although the
C  values of the entries change). Thus, we can save some work in setting
C  up the preconditioner (e.g., no need to redo symbolic factorization for
C  ILU/ICC preconditioners).
C   - If the nonzero structure of the matrix is different during
C     successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
C     must be used instead.  If you are unsure whether the matrix
C     structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
C   - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
C     believes your assertion and does not check the structure
C     of the matrix.  If you erroneously claim that the structure
C     is the same when it actually is not, the new preconditioner
C     will not function correctly.  Thus, use this optimization
C     feature with caution!

      flag = SAME_NONZERO_PATTERN

      return
      end

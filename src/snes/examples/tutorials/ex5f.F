
C "$Id: ex5f.F,v 1.12 1996/09/30 20:10:27 bsmith Exp bsmith $";
C
C  Description: This example solves a nonlinear system in parallel with SNES.
C  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
C  domain, using distributed arrays (DAs) to partition the parallel grid.
C  The command line options include:
C    -par <parameter>, where <parameter> indicates the nonlinearity of the problem
C       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
C    -mx <xg>, where <xg> = number of grid points in the x-direction
C    -my <yg>, where <yg> = number of grid points in the y-direction
C    -Nx <npx>, where <npx> = number of processors in the x-direction
C    -Ny <npy>, where <npy> = number of processors in the y-direction
C
C/*T
C  Concepts: SNES^Solving a system of nonlinear equations (parallel Bratu example);
C  Concepts: DA^Using distributed arrays;
C  Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
C  Routines: SNESSolve(); SNESSetFromOptions();
C  Routines: DACreate2d(); DADestroy(); DAGetDistributedVector(); DAGetLocalVector();
C  Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal();
C  Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetGlobalIndices();
C  Processors: n
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
C  In addition, we need the following for use of distributed arrays
C     da.h     - distributed arrays (DAs)
C  Other include statements may be needed if using additional PETSc
C  routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"
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
C  a Fortran integer array.  Note that we can store (pointers to)
C  PETSc objects within this array.
C     user(1) = mx     - discretization in x-direction
C          2  = my     - discretization in y-direction
C          3  = da     - distributed array
C          4  = localX - local work vector (including ghost points)
C          5  = localF - local work vector (including ghost points)
C          6  = rank   - processor rank within communicator
C
C  Variables:
C     snes        - nonlinear solver
C     x, r        - solution, residual vectors
C     J           - Jacobian matrix
C     user        - user-defined work context
C     its         - iterations for convergence
C     Nx, Ny      - number of preocessors in x- and y- directions
C     matrix_free - flag - 1 indicates matrix-free version
C     size        - number of processors
C     rank        - processor rank
C     lambda      - nonlinearity parameter
C
      SNES     snes
      Vec      x, r, localX, localF
      DA       da
      Mat      J
      integer  its, Nx, Ny, matrix_free, size, flg, N, ierr, m
      integer  user(6), mx, my, rank
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
C   - The element lx_a(ib) is element ib+1 in the vector localX
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
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

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

C  Create distributed array (DA) to manage parallel grid and vectors

      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Nx',Nx,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Ny',Ny,flg,ierr)
      if (Nx*Ny .ne. size .and.
     &      (Nx .ne. PETSC_DECIDE .or. Ny .ne. PETSC_DECIDE)) then
         if (rank .eq. 0) 
     &      write(6,*) 'Incompatible number of procs: Nx * Ny != size'
         SETERRA(1,0)
      endif
      call DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,mx,
     &                my,Nx,Ny,1,1,da,ierr)
      user(3) = da

C  Extract global and local vectors from DA; then duplicate for remaining
C  vectors that are the same types

      call DAGetDistributedVector(da,x,ierr)
      call DAGetLocalVector(da,localX,ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(localX,localF,ierr)
      user(4) = localX
      user(5) = localF
      user(6) = rank

C  Create nonlinear solver context

      call SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,ierr)

C  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,user,ierr)

C  Set Jacobian matrix data structure and default Jacobian evaluation
C  routine. User can override with:
C     -snes_fd : default finite differencing approximation of Jacobian
C     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
C                (unless user explicitly sets preconditioner) 
C     -snes_mf_operator : form preconditioning matrix as set by the user,
C                         but use matrix-free approx for Jacobian-vector
C                         products within Newton-Krylov method
C
C  Note:  For the parallel case, vectors and matrices MUST be partitioned
C     accordingly.  When using distributed arrays (DAs) to create vectors,
C     the DAs determine the problem partitioning.  We must explicitly
C     specify the local matrix dimensions upon its creation for compatibility
C     with the vector distribution.  Thus, the generic MatCreate() routine
C     is NOT sufficient when working with distributed arrays.

      call OptionsHasName(PETSC_NULL_CHARACTER,'-snes_mf',matrix_free,
     &                    ierr)
      if (matrix_free .eq. 0) then
        if (size .eq. 1) then
          call MatCreateSeqAIJ(MPI_COMM_WORLD,N,N,5,PETSC_NULL,J,ierr)
        else
          call VecGetLocalSize(x,m,ierr)
          call MatCreateMPIAIJ(MPI_COMM_WORLD,m,m,N,N,5,PETSC_NULL,
     &                         3,PETSC_NULL,J,ierr)
        endif
        call SNESSetJacobian(snes,J,J,FormJacobian,user,ierr)
      endif

C  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)

      call SNESSetFromOptions(snes,ierr)

C  Evaluate initial guess; then solve nonlinear system.
C   - The user should initialize the vector, x, with the initial guess
C     for the nonlinear solver prior to calling SNESSolve().  In particular,
C     to employ an initial guess of zero, the user should explicitly set
C     this vector to zero by calling VecSet().

      call FormInitialGuess(user,x,ierr)
      call SNESSolve(snes,x,its,ierr) 
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

      if (matrix_free .eq. 0) call MatDestroy(J,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call VecDestroy(localX,ierr)
      call VecDestroy(localF,ierr)
      call SNESDestroy(snes,ierr)
      call DADestroy(da,ierr)
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
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/snes.h"

      Vec      X, localX
      DA       da
      integer  user(*), xs, ys, xm, ym, gxm, gym, gxs, gys, xe, ye
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
      da     = user(3)
      localX = user(4)
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

      call VecGetArray(localX,lx_v,lx_i,ierr)

C  Get local grid boundaries (for 2-dimensional DA):
C       xs, ys   - starting grid indices (no ghost points)
C       xm, ym   - widths of local grid (no ghost points)
C       gxs, gys - starting grid indices (including ghost points)
C       gxm, gym - widths of local grid (including ghost points)

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      ye = ys+ym-1
      xe = xs+xm-1

C  Compute initial guess over the locally owned part of the grid

      do 20 j=ys,ye
         temp = DBLE(min(j,my-j-1))*hy
         do 10 i=xs,xe
            row = i - gxs + (j - gys)*gxm
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

      call VecRestoreArray(localX,lx_v,lx_i,ierr)

C  Insert values into global vector

      call DALocalToGlobal(da,localX,INSERT_VALUES,X,ierr)

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
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/snes.h"

      SNES     snes
      Vec      X, F, localX, localF
      DA       da
      Scalar   two, one, lambda, hx, hy, hxdhy, hydhx, sc
      Scalar   u, uxx, uyy
      integer  ierr, i, j, row, mx, my, xs, ys, xm, ym, xe, ye
      integer  user(*), gxs, gys, gxm, gym

C  Declarations for use with local arrays

      Scalar   lx_v(1), lf_v(1)
      integer  lx_i, lf_i 

      common /params/ lambda

      one    = 1.0
      two    = 2.0
      mx     = user(1)
      my     = user(2)
      da     = user(3)
      localX = user(4)
      localF = user(5)
      hx     = one/DBLE(mx-1)
      hy     = one/DBLE(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

C  Scatter ghost points to local vector, using the 2-step process
C     DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
C  By placing code between these two statements, computations can
C  be done while messages are in transition.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

C  Get pointers to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the Fortran chapter of the users manual for details.

      call VecGetArray(localX,lx_v,lx_i,ierr)
      call VecGetArray(localF,lf_v,lf_i,ierr)

C  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      ye = ys+ym-1
      xe = xs+xm-1

C  Compute function over the locally owned part of the grid

      do 20 j=ys,ye
         row = (j - gys)*gxm + xs - gxs - 1
         do 10 i=xs,xe
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
     &               - lx_a(row-gxm) 
     &               - lx_a(row+gxm))
               lf_a(row) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

C  Restore vectors

      call VecRestoreArray(localX,lx_v,lx_i,ierr)
      call VecRestoreArray(localF,lf_v,lf_i,ierr)

C  Insert values into global vector

      call DALocalToGlobal(da,localF,INSERT_VALUES,F,ierr)
      call PLogFlops(11*ym*xm,ierr)

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
C  Notes:
C  Due to grid point reordering with DAs, we must always work
C  with the local grid points, and then transform them to the new
C  global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
C  We cannot work directly with the global numbers for the original
C  uniprocessor grid!
C
      subroutine FormJacobian(snes,X,jac,B,flag,user)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/snes.h"

      SNES         snes
      DA           da
      Vec          X, localX
      Mat          jac, B
      MatStructure flag
      integer      user(*), ierr, i, j, row, mx, my
      integer      col(5), grow, xe, ye
      integer      nloc, xs, ys, xm, ym, gxs, gys, gxm, gym
      Scalar       two, one, lambda, hx, hy, hxdhy, hydhx
      Scalar       sc, v(5)

      common /params/ lambda

C  Declarations for use with local arrays

      Scalar   lx_v(1)
      integer  ltog_a(0:1), ltog_i, lx_i

C  Macro to facilitate use of the local-to-global mapping
C  (obtained via DAGetGlobalIndices) in Fortran.  Note
C  that this mapping is always 0-based (following the
C  C convention) rather than 1-based.

#define ltog_v(ib)  ltog_a(ltog_i + (ib))

C  Set parameters

      one    = 1.0
      two    = 2.0
      mx     = user(1)
      my     = user(2)
      da     = user(3)
      localX = user(4)
      hx     = one/DBLE(mx-1)
      hy     = one/DBLE(my-1)
      sc     = hx*hy
      hxdhy  = hx/hy
      hydhx  = hy/hx

C  Scatter ghost points to local vector, using the 2-step process
C     DAGlobalToLocalBegin(), DAGlobalToLocalEnd()
C  Computations can be done while messages are in transition,
C  by placing code between these two statements.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

C  Get a pointer to vector data

      call VecGetArray(localX,lx_v,lx_i,ierr)

C  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      ye = ys+ym-1
      xe = xs+xm-1

C  Get the global node numbers for all local nodes, including ghost points
C    - Note that the Fortran interface to DAGetGlobalIndices() differs from
C      the C version.  See the Fortran chapter of the users manual for details.

       call DAGetGlobalIndices(da,nloc,ltog_a,ltog_i,ierr)

C  Compute entries for the locally owned part of the Jacobian.
C   - Currently, all PETSc parallel matrix formats are partitioned by
C     contiguous chunks of rows across the processors. The "grow"
C     parameter computed below specifies the global row number 
C     corresponding to each local grid point.
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Always specify global row and columns of matrix entries.
C   - Here, we set all entries for a particular row at once.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C.

      do 20 j=ys,ye
         row = (j - gys)*gxm + xs - gxs - 1
         do 10 i=xs,xe
            row = row + 1
            grow = ltog_v(row)
C           boundary points
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
               call MatSetValues(jac,1,grow,1,grow,one,
     &                           INSERT_VALUES,ierr)
C           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy) 
     &                  - sc*lambda*exp(lx_a(row))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = ltog_v(row - gxm)
               col(2) = ltog_v(row - 1)
               col(3) = grow
               col(4) = ltog_v(row + 1)
               col(5) = ltog_v(row + gxm)
               call MatSetValues(jac,1,grow,5,col,v,
     &                           INSERT_VALUES,ierr)
            endif
 10      continue
 20   continue

C  Assemble matrix, using the 2-step process:
C     MatAssemblyBegin(), MatAssemblyEnd()
C  Computations can be done while messages are in transition,
C  by placing code between these two statements.

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArray(localX,lx_v,lx_i,ierr)
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

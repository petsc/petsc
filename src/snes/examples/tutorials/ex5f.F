
C "$Id: ex5f.F,v 1.22 1997/05/01 18:42:15 curfman Exp curfman $";
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
C  Routines: SNESSolve(); SNESSetFromOptions(); DAView();
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
C  The uniprocessor version of this code is snes/examples/tutorials/ex4f.F
C
C  --------------------------------------------------------------------------

      program main
      implicit none

#include "ex5f.h"

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  The application context to contain data needed by the 
C  application-provided call-back routines, FormJacobian() and
C  FormFunction().  In this example the application context is
C  a Fortran common block.  Note that we can store (pointers to)
C  PETSc objects within this common block.
C              mx     - discretization in x-direction
C              my     - discretization in y-direction
C              da     - distributed array
C              localX - local work vector (including ghost points)
C              localF - local work vector (including ghost points)
C              rank   - processor rank within communicator
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
      Vec      x, r
      Mat      J
      integer  its, Nx, Ny, matrix_free, size, flg, N, ierr, m
      Double   lambda_max, lambda_min, lambda
      integer  ltog_a(0:1), ltog_i, nloc

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
         SETERRA(1,0,' ')
      endif
      N = mx*my

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create nonlinear solver context
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create vector data structures; set function evaluation routine
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Create distributed array (DA) to manage parallel grid and vectors

      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Nx',Nx,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Ny',Ny,flg,ierr)
      if (Nx*Ny .ne. size .and.
     &      (Nx .ne. PETSC_DECIDE .or. Ny .ne. PETSC_DECIDE)) then
         if (rank .eq. 0) 
     &      write(6,*) 'Incompatible number of procs: Nx * Ny != size'
         SETERRA(1,0,' ')
      endif
      call DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,mx,
     &                my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,da,ierr)

C
C   Visualize the distribution of the array across the processors
C
C     call DAView(da,VIEWER_DRAWX_WORLD,ierr)

C  Extract global and local vectors from DA; then duplicate for remaining
C  vectors that are the same types

      call DAGetDistributedVector(da,x,ierr)
      call DAGetLocalVector(da,localX,ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(localX,localF,ierr)

C  Get local grid boundaries (for 2-dimensional DA):
C       xs, ys   - starting grid indices (no ghost points)
C       xm, ym   - widths of local grid (no ghost points)
C       gxs, gys - starting grid indices (including ghost points)
C       gxm, gym - widths of local grid (including ghost points)

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      xs  = xs+1
      ys  = ys+1
      gxs = gxs+1
      gys = gys+1

      ye  = ys+ym-1
      xe  = xs+xm-1
      gye = gys+gym-1
      gxe = gxs+gxm-1

C  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,PETSC_NULL,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create matrix data structure; set Jacobian evaluation routine
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
C
C     Note: Here we only approximately preallocate storage space for the
C     Jacobian.  See the users manual for a discussion of better techniques
C     for preallocating matrix memory.

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
        call SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL,ierr)

C       Get the global node numbers for all local nodes, including ghost points.
C       Associate this mapping with the matrix for later use in setting matrix
C       entries via MatSetValuesLocal().
C        - Note that the Fortran interface to DAGetGlobalIndices() differs from
C          the C version.  See the Fortran chapter of the users manual for details.

        call DAGetGlobalIndices(da,nloc,ltog_a,ltog_i,ierr)
        call MatSetLocalToGlobalMapping(J,nloc,ltog_a(ltog_i),ierr)

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

      call FormInitialGuess(x,ierr)
      call SNESSolve(snes,x,its,ierr) 
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
C  X - vector
C
C  Output Parameter:
C  X - vector
C
C  Notes:
C  This routine serves as a wrapper to the lower-level routine
C  "ApplicationInitialGuess", where the actual computations are 
C  done using standard Fortran-style computations over a 2-dim array.
C  This routine merely handles ghost point scatters and accesses
C  the local vector data via VecGetArray() and VecRestoreArray().
C
      subroutine FormInitialGuess(X,ierr)
      implicit none

#include "ex5f.h"

      Vec      X, localX
      integer  ierr

C  Declarations for use with local arrays
      Scalar   lx_v(0:1)
      common   lx_v
      integer  lx_i

C  Set parameters

C  Get a pointer to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the users manual for details.

      call VecGetArray(localX,lx_v,lx_i,ierr)

C  Compute initial guess over the locally owned part of the grid

      call ApplicationInitialGuess(lx_v(lx_i),ierr)

C  Restore vector

      call VecRestoreArray(localX,lx_v,lx_i,ierr)

C  Insert values into global vector

      call DALocalToGlobal(da,localX,INSERT_VALUES,X,ierr)

      return 
      end

C ---------------------------------------------------------------------
C
C  ApplicationInitialGuess - Computes initial approximation, called by
C  the higher level routine FormInitialGuess().
C
C  Input Parameter:
C  x - local vector data
C
C  Output Parameters:
C  x - local vector data
C  ierr - error code 
C
C  Notes:
C  This routine uses standard Fortran-style computations over a 2-dim array.
C
      subroutine ApplicationInitialGuess(x,ierr)

      implicit none

C  Input/output variables:
      Scalar  x(gxs:gxe,gys:gye)
      integer ierr

#include "ex5f.h"

C  Local variables:
      integer  i, j, hxdhy, hydhx
      Scalar   temp1, temp, hx, hy, sc, one

C  Set parameters

      ierr   = 0
      one    = 1.0
      hx     = one/(DBLE(mx-1))
      hy     = one/(DBLE(my-1))
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx
      temp1  = lambda/(lambda + one)

      do 20 j=ys,ye
         temp = DBLE(min(j-1,my-j))*hy
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1
     &             .or. i .eq. mx .or. j .eq. my) then
              x(i,j) = 0.0
            else
              x(i,j) = temp1 *
     &          sqrt(min(DBLE(min(i-1,mx-i)*hx),DBLE(temp)))
            endif
 10      continue
 20   continue

      return 
      end

C ---------------------------------------------------------------------
C
C  FormFunction - Evaluates nonlinear function, F(x).
C
C  Input Parameters:
C  snes - the SNES context
C  X - input vector
C  dummy - optional user-defined context, as set by SNESSetFunction()
C          (not used here)
C
C  Output Parameter:
C  F - function vector
C
C  Notes:
C  This routine serves as a wrapper to the lower-level routine
C  "ApplicationFunction", where the actual computations are done
C  using standard Fortran-style computations over a 2-dim array.
C  This routine merely handles ghost point scatters and accesses
C  the local vector data via VecGetArray() and VecRestoreArray().
C
      subroutine FormFunction(snes,X,F,dummy)
      implicit none

#include "ex5f.h"

      SNES     snes
      Vec      X, F
      integer  dummy,ierr

C  Declarations for use with local arrays

      Scalar   lx_v(0:1), lf_v(0:1)
      common   lx_v, lf_v
      integer  lx_i, lf_i 


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

C  Compute function over the locally owned part of the grid

      call ApplicationFunction(lx_v(lx_i),lf_v(lf_i),ierr)

C  Restore vectors

      call VecRestoreArray(localX,lx_v,lx_i,ierr)
      call VecRestoreArray(localF,lf_v,lf_i,ierr)

C  Insert values into global vector

      call DALocalToGlobal(da,localF,INSERT_VALUES,F,ierr)
      call PLogFlops(11*ym*xm,ierr)

c      call VecView(X,VIEWER_STDOUT_WORLD)
c      call VecView(F,VIEWER_STDOUT_WORLD)

      return 
      end

C ---------------------------------------------------------------------
C
C  ApplicationFunction - Computes nonlinear function, called by
C  the higher level routine FormFunction().
C
C  Input Parameter:
C  x - local vector data
C
C  Output Parameters:
C  f - local vector data, f(x)
C  ierr - error code 
C
C  Notes:
C  This routine uses standard Fortran-style computations over a 2-dim array.
C
      subroutine ApplicationFunction(x,f,ierr)

      implicit none

C  Input/output variables:

      Scalar   x(gxs:gxe,gys:gye), f(gxs:gxe,gys:gye)
      integer  ierr

#include "ex5f.h"

C  Local variables:

      Scalar   two, one, hx, hy, hxdhy, hydhx, sc
      Scalar   u, uxx, uyy
      integer  ierr, i, j

      one    = 1.0
      two    = 2.0
      hx     = one/DBLE(mx-1)
      hy     = one/DBLE(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

C  Compute function over the locally owned part of the grid

      do 20 j=ys,ye
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1
     &             .or. i .eq. mx .or. j .eq. my) then
               f(i,j) = x(i,j)
            else
               u = x(i,j) 
               uxx = hydhx * (two*u
     &                - x(i-1,j) - x(i+1,j))
               uyy = hxdhy * (two*u - x(i,j-1) - x(i,j+1))
               f(i,j) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

      return
      end

C ---------------------------------------------------------------------
C
C  FormJacobian - Evaluates Jacobian matrix.
C
C  Input Parameters:
C  snes - the SNES context
C  x - input vector
C  dummy - optional user-defined context, as set by SNESSetJacobian()
C          (not used here)
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
C  Two methods are available for imposing this transformation
C  when setting matrix entries:
C    (A) MatSetValuesLocal(), using the local ordering (including
C        ghost points!)
C        - Use DAGetGlobalIndices() to extract the local-to-global map
C        - Associate this map with the matrix by calling
C          MatSetLocalToGlobalMapping() once
C        - Set matrix entries using the local ordering
C          by calling MatSetValuesLocal()
C    (B) MatSetValues(), using the global ordering 
C        - Use DAGetGlobalIndices() to extract the local-to-global map
C        - Then apply this map explicitly yourself
C        - Set matrix entries using the global ordering by calling
C          MatSetValues()
C  Option (A) seems cleaner/easier in many cases, and is the procedure
C  used in this example.
C
      subroutine FormJacobian(snes,X,jac,B,flag,dummy)
      implicit none

#include "ex5f.h"

      SNES         snes
      Vec          X
      Mat          jac, B
      MatStructure flag
      integer      dummy, ierr, row, col(5), i, j
      Scalar       two, one, hx, hy, hxdhy, hydhx
      Scalar       sc, v(5)

C  Declarations for use with local arrays

      Scalar   lx_v(1)
      common   lx_v
      integer  lx_i

C  Set parameters

      one    = 1.0
      two    = 2.0
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

C  Compute entries for the locally owned part of the Jacobian.
C   - Currently, all PETSc parallel matrix formats are partitioned by
C     contiguous chunks of rows across the processors. The "grow"
C     parameter computed below specifies the global row number 
C     corresponding to each local grid point.
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Here, we set all entries for a particular row at once.
C   - We can set matrix entries either using either
C     MatSetValuesLocal() or MatSetValues(), as discussed above.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C.

      do 20 j=ys,ye
         row = (j - gys)*gxm + xs - gxs - 1
         do 10 i=xs,xe
            row = row + 1
c            write(6,*) 'j,i,row = ',j,i,row
C           boundary points
            if (i .eq. 1 .or. j .eq. 1
     &             .or. i .eq. mx .or. j .eq. my) then
               call MatSetValuesLocal(jac,1,row,1,row,one,
     &                           INSERT_VALUES,ierr)
C           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy) 
     &                  - sc*lambda*exp(lx_a(row))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - gxm
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + gxm
               call MatSetValuesLocal(jac,1,row,5,col,v,
     &                                INSERT_VALUES,ierr)
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
C
C      Tell the matrix we will never add a new nonzero location to the
C    matrix. If we do it will generate an error.
C
       call MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERROR,ierr)

      return
      end

! "$Id: ex5f90.F,v 1.12 1998/03/20 22:52:53 bsmith Exp balay $";
!
!  Description: This example solves a nonlinear system in parallel with SNES.
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DAs) to partition the parallel grid.
!  The command line options include:
!    -par <parameter>, where <parameter> indicates the nonlinearity of the problem
!       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
!    -mx <xg>, where <xg> = number of grid points in the x-direction
!    -my <yg>, where <yg> = number of grid points in the y-direction
!    -Nx <npx>, where <npx> = number of processors in the x-direction
!    -Ny <npy>, where <npy> = number of processors in the y-direction
!
!/*T
!  Concepts: SNES^Solving a system of nonlinear equations (parallel Bratu example);
!  Concepts: DA^Using distributed arrays;
!  Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
!  Routines: SNESSolve(); SNESSetFromOptions(); DAView();
!  Routines: DACreate2d(); DADestroy(); DACreateGlobalVector(); DACreateLocalVector();
!  Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal();
!  Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetGlobalIndicesF90();
!  Processors: n
!T*/
!
!  --------------------------------------------------------------------------
! 
!  Solid Fuel Ignition (SFI) problem.  This problem is modeled by
!  the partial differential equation
!  
!          -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
! 
!  with boundary conditions
!  
!           u = 0  for  x = 0, x = 1, y = 0, y = 1.
! 
!  A finite difference approximation with the usual 5-point stencil
!  is used to discretize the boundary value problem to obtain a nonlinear 
!  system of equations.
!
!  The uniprocessor version of this code is snes/examples/tutorials/ex4f.F
!
!  --------------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  This program uses CPP for preprocessing, as indicated by the use of
!  PETSc include files in the directory petsc/include/FINCLUDE.  This
!  convention enables use of the CPP preprocessor, which allows the use
!  of the #include statements that define PETSc objects and variables.
!
!  Use of the conventional Fortran include statements is also supported
!  In this case, the PETsc include files are located in the directory
!  petsc/include/finclude.
!         
!  Since one must be very careful to include each file no more than once
!  in a Fortran routine, application programmers must exlicitly list
!  each file needed for the various PETSc components within their
!  program (unlike the C/C++ interface).
!
!  See the Fortran section of the PETSc users manual for details.
!
!  The following include statements are generally used in SNES Fortran
!  programs:
!     petsc.h  - base PETSc routines
!     vec.h    - vectors
!     vec.h90  - Fortran 90 interface to vectors
!     mat.h    - matrices
!     ksp.h    - Krylov subspace methods
!     pc.h     - preconditioners
!     sles.h   - SLES interface
!     snes.h   - SNES interface
!  In addition, we need the following for use of distributed arrays
!     da.h     - distributed arrays (DAs)
!     da.h90   - Fortran 90 DA interface
!  Other include statements may be needed if using additional PETSc
!  routines in a Fortran program, e.g.,
!     viewer.h - viewers
!     is.h     - index sets
!
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/vec.h90"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/da.h90"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/snes.h"
#include "include/FINCLUDE/is.h"
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  The application context to contain data needed by the 
!  application-provided call-back routines, FormJacobian() and
!  FormFunction().  In this example the application context is
!  a Fortran integer array.  Note that we can store (pointers to)
!  PETSc objects within this array.
!     user(1) = mx     - discretization in x-direction
!          2  = my     - discretization in y-direction
!          3  = da     - distributed array
!          4  = localX - local work vector (including ghost points)
!          5  = localF - local work vector (including ghost points)
!          6  = rank   - processor rank within communicator
!
!  Variables:
!     snes        - nonlinear solver
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!     user        - user-defined work context
!     its         - iterations for convergence
!     Nx, Ny      - number of preocessors in x- and y- directions
!     matrix_free - flag - 1 indicates matrix-free version
!     size        - number of processors
!     rank        - processor rank
!     lambda      - nonlinearity parameter
!
      SNES                   snes
      Vec                    x, r, localX, localF
      DA                     da
      Mat                    J
      ISLocalToGlobalMapping isltog
      integer                its, Nx, Ny, matrix_free, size, flg
      integer                N, ierr, m
      integer                user(6), mx, my, rank
      double precision       lambda_max, lambda_min, lambda
      integer                ltog_a(0:1),  nloc
      PetscOffset            ltog_i

!  Store nonlinearity parameter in common block

      common /params/ lambda

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction, FormInitialGuess, FormJacobian

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Macro definitions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Macros to make clearer the process of setting values in vectors and
!  getting values from vectors.  These vectors are used in the routines
!  FormFunction(), FormInitialGuess(), and FormJacobian().
!
!   - The element lx_a(ib) is element ib+1 in the vector localX
!   - Here we add 1 to the base array index to facilitate the use of
!     conventional Fortran 1-based array indexing.
!
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

!  Initialize problem parameters

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
      user(1) = mx
      user(2) = my
      N       = mx*my

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,
     &               ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DA) to manage parallel grid and vectors

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
      call DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,
     &                mx,my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,da,ierr)
      user(3) = da

!
!   Visualize the distribution of the array across the processors
!
!     call DAView(da,VIEWER_DRAWX_WORLD,ierr)

!  Extract global and local vectors from DA; then duplicate for remaining
!  vectors that are the same types

      call DACreateGlobalVector(da,x,ierr)
      call DACreateLocalVector(da,localX,ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(localX,localF,ierr)
      user(4) = localX
      user(5) = localF
      user(6) = rank

!  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,user,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix data structure; set Jacobian evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set Jacobian matrix data structure and default Jacobian evaluation
!  routine. User can override with:
!     -snes_fd : default finite differencing approximation of Jacobian
!     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
!                (unless user explicitly sets preconditioner) 
!     -snes_mf_operator : form preconditioning matrix as set by the user,
!                         but use matrix-free approx for Jacobian-vector
!                         products within Newton-Krylov method
!
!  Note:  For the parallel case, vectors and matrices MUST be partitioned
!     accordingly.  When using distributed arrays (DAs) to create vectors,
!     the DAs determine the problem partitioning.  We must explicitly
!     specify the local matrix dimensions upon its creation for compatibility
!     with the vector distribution.  Thus, the generic MatCreate() routine
!     is NOT sufficient when working with distributed arrays.
!
!     Note: Here we only approximately preallocate storage space for the
!     Jacobian.  See the users manual for a discussion of better techniques
!     for preallocating matrix memory.

      call OptionsHasName(PETSC_NULL_CHARACTER,'-snes_mf',matrix_free,
     &                    ierr)
      if (matrix_free .eq. 0) then
        if (size .eq. 1) then
          call MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,PETSC_NULL,J,ierr)
        else
          call VecGetLocalSize(x,m,ierr)
          call MatCreateMPIAIJ(PETSC_COMM_WORLD,m,m,N,N,5,PETSC_NULL,
     &                         3,PETSC_NULL,J,ierr)
        endif
        call SNESSetJacobian(snes,J,J,FormJacobian,user,ierr)

!       Get the global node numbers for all local nodes, including ghost points.
!       Associate this mapping with the matrix for later use in setting matrix
!       entries via MatSetValuesLocal().
!        - Note that the Fortran interface to DAGetGlobalIndices() differs from
!          the C version.  See the Fortran chapter of the users manual for details.

        call DAGetGlobalIndices(da,nloc,ltog_a,ltog_i,ierr)
        call ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,nloc,
     &                                   ltog_a(ltog_i),isltog,ierr)
        call MatSetLocalToGlobalMapping(J,isltog,ierr)
        call ISLocalToGlobalMappingDestroy(isltog,ierr)

      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)

      call SNESSetFromOptions(snes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      call FormInitialGuess(user,x,ierr)
      call SNESSolve(snes,x,its,ierr) 
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (matrix_free .eq. 0) call MatDestroy(J,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call VecDestroy(localX,ierr)
      call VecDestroy(localF,ierr)
      call SNESDestroy(snes,ierr)
      call DADestroy(da,ierr)
      call PetscFinalize(ierr)
      end

! ---------------------------------------------------------------------
!
!  FormInitialGuess - Forms initial approximation.
!
!  Input Parameters:
!  user - user-defined application context
!  X - vector
!
!  Output Parameter:
!  X - vector
!
      subroutine FormInitialGuess(user,X,ierr)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/vec.h90"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/snes.h"

      Vec      X, localX
      DA       da
      integer  user(*), xs, ys, xm, ym, gxm, gym, gxs, gys, xe, ye
      integer  i, j, row, mx, my, ierr, hxdhy, hydhx
      Scalar   lambda, temp1, temp, hx, hy, sc, one

!  Declarations for use with local arrays
      Scalar, pointer ::  lx_v(:)

      common /params/ lambda

!  Set parameters

      ierr   = 0
      one    = 1.0
      mx     = user(1)
      my     = user(2)
      da     = user(3)
      localX = user(4)
      hx     = one/(dble(mx-1))
      hy     = one/(dble(my-1))
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx
      temp1  = lambda/(lambda + one)

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

      call VecGetArrayF90(localX,lx_v,ierr)

!  Get local grid boundaries (for 2-dimensional DA):
!       xs, ys   - starting grid indices (no ghost points)
!       xm, ym   - widths of local grid (no ghost points)
!       gxs, gys - starting grid indices (including ghost points)
!       gxm, gym - widths of local grid (including ghost points)

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      ye = ys+ym-1
      xe = xs+xm-1

!  Compute initial guess over the locally owned part of the grid

      do 20 j=ys,ye
         temp = dble(min(j,my-j-1))*hy
         do 10 i=xs,xe
            row = i - gxs + (j - gys)*gxm + 1
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
              lx_v(row) = 0.0
            else
              lx_v(row) = temp1 *
     &          sqrt(min(dble(min(i,mx-i-1))*hx,temp))
            endif
 10      continue
 20   continue


!  Restore vector

      call VecRestoreArrayF90(localX,lx_v,ierr)

!  Insert values into global vector

      call DALocalToGlobal(da,localX,INSERT_VALUES,X,ierr)

      return 
      end

! ---------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  X - input vector
!  user - optional user-defined context, as set by SNESSetFunction()
!
!  Output Parameter:
!  F - function vector
!
      subroutine FormFunction(snes,X,F,user)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/vec.h90"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/snes.h"

      SNES     snes
      Vec      X, F, localX, localF
      DA       da
      Scalar   two, one, lambda, hx, hy, hxdhy, hydhx, sc
      Scalar   u, uxx, uyy
      integer  ierr, i, j, row, mx, my, xs, ys, xm, ym, xe, ye
      integer  user(*), gxs, gys, gxm, gym

!  Declarations for use with local arrays

      Scalar, pointer ::   lx_v(:), lf_v(:)

      common /params/ lambda

      one    = 1.0
      two    = 2.0
      mx     = user(1)
      my     = user(2)
      da     = user(3)
      localX = user(4)
      localF = user(5)
      hx     = one/dble(mx-1)
      hy     = one/dble(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Scatter ghost points to local vector, using the 2-step process
!     DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
!  By placing code between these two statements, computations can
!  be done while messages are in transition.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get pointers to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

      call VecGetArrayF90(localX,lx_v,ierr)
      call VecGetArrayF90(localF,lf_v,ierr)

!  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      ye = ys+ym-1
      xe = xs+xm-1

!  Compute function over the locally owned part of the grid

      do 20 j=ys,ye
         row = (j - gys)*gxm + xs - gxs
         do 10 i=xs,xe
            row = row + 1
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
               lf_v(row) = lx_v(row)
            else
               u = lx_v(row)
               uxx = hydhx * (two*u
     &               - lx_v(row-1)
     &               - lx_v(row+1))
               uyy = hxdhy * (two*u 
     &               - lx_v(row-gxm) 
     &               - lx_v(row+gxm))
               lf_v(row) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

!  Restore vectors

      call VecRestoreArrayF90(localX,lx_v,ierr)
      call VecRestoreArrayF90(localF,lf_v,ierr)

!  Insert values into global vector

      call DALocalToGlobal(da,localF,INSERT_VALUES,F,ierr)
      call PLogFlops(11*ym*xm,ierr)

      return 
      end

! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  user - optional user-defined context, as set by SNESSetJacobian()
!
!  Output Parameters:
!  A - Jacobian matrix
!  B - optionally different preconditioning matrix
!  flag - flag indicating matrix structure
!
!  Notes:
!  Due to grid point reordering with DAs, we must always work
!  with the local grid points, and then transform them to the new
!  global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
!  We cannot work directly with the global numbers for the original
!  uniprocessor grid!
!
!  Two methods are available for imposing this transformation
!  when setting matrix entries:
!    (A) MatSetValuesLocal(), using the local ordering (including
!        ghost points!)
!        - Use DAGetGlobalIndices() to extract the local-to-global map
!        - Associate this map with the matrix by calling
!          MatSetLocalToGlobalMapping() once
!        - Set matrix entries using the local ordering
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering 
!        - Use DAGetGlobalIndices() to extract the local-to-global map
!        - Then apply this map explicitly yourself
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobian(snes,X,jac,B,flag,user)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/vec.h90"
#include "include/FINCLUDE/da.h"
#include "include/FINCLUDE/da.h90"
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

!  Declarations for use with local arrays

      Scalar, pointer ::  lx_v(:)
      integer, pointer :: ltog_a(:)

!  Set parameters

      one    = 1.0
      two    = 2.0
      mx     = user(1)
      my     = user(2)
      da     = user(3)
      localX = user(4)
      hx     = one/dble(mx-1)
      hy     = one/dble(my-1)
      sc     = hx*hy
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Scatter ghost points to local vector, using the 2-step process
!     DAGlobalToLocalBegin(), DAGlobalToLocalEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get a pointer to vector data

      call VecGetArrayF90(localX,lx_v,ierr)

!  Get local grid boundaries

      call DAGetCorners(da,xs,ys,PETSC_NULL,xm,ym,PETSC_NULL,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL,gxm,gym,
     &                                         PETSC_NULL,ierr)
      ye = ys+ym-1
      xe = xs+xm-1

!  Get the global node numbers for all local nodes, including ghost points

       call DAGetGlobalIndicesF90(da,nloc,ltog_a,ierr)

!  Compute entries for the locally owned part of the Jacobian.
!   - Currently, all PETSc parallel matrix formats are partitioned by
!     contiguous chunks of rows across the processors. The "grow"
!     parameter computed below specifies the global row number 
!     corresponding to each local grid point.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly). 
!   - Always specify global row and columns of matrix entries.
!   - Here, we set all entries for a particular row at once.
!   - We can set matrix entries either using either
!     MatSetValuesLocal() or MatSetValues(), as discussed above.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.

      do 20 j=ys,ye
         row = (j - gys)*gxm + xs - gxs
         do 10 i=xs,xe
            row = row + 1
!           boundary points
            if (i .eq. 0 .or. j .eq. 0 
     &             .or. i .eq. mx-1 .or. j .eq. my-1) then
               call MatSetValuesLocal(jac,1,row,1,row,one,
     &                           INSERT_VALUES,ierr)
!           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy) 
     &                  - sc*lambda*exp(lx_v(row))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - gxm
               col(2) = row - 1
               col(3) = grow
               col(4) = row + 1
               col(5) = row + gxm
               call MatSetValuesLocal(jac,1,row,5,col,v,
     &                           INSERT_VALUES,ierr)
            endif
 10      continue
 20   continue

!  Assemble matrix, using the 2-step process:
!     MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArrayF90(localX,lx_v,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)

!  Set flag to indicate that the Jacobian matrix retains an identical
!  nonzero structure throughout all nonlinear iterations (although the
!  values of the entries change). Thus, we can save some work in setting
!  up the preconditioner (e.g., no need to redo symbolic factorization for
!  ILU/ICC preconditioners).
!   - If the nonzero structure of the matrix is different during
!     successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
!     must be used instead.  If you are unsure whether the matrix
!     structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
!   - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
!     believes your assertion and does not check the structure
!     of the matrix.  If you erroneously claim that the structure
!     is the same when it actually is not, the new preconditioner
!     will not function correctly.  Thus, use this optimization
!     feature with caution!

      flag = SAME_NONZERO_PATTERN

      return
      end

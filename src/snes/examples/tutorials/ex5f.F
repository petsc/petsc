! "$Id: ex5f.F,v 1.51 2000/08/08 15:48:23 balay Exp balay $";
!
!  Program usage:  mpirun -np <procs> ex5f [-help] [all PETSc options]
!
!  Description: This example solves a nonlinear system in parallel with SNES.
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DAs) to partition the parallel grid.
!  The command line options include:
!    -par <param>, where <param> indicates the nonlinearity of the problem
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
!  Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetISLocalToGlobalMapping();
!  Routines: MatSetLocalToGlobalMapping();
!  Processors: n
!T*/
!
!  --------------------------------------------------------------------------
! 
!  Solid Fuel Ignition (SFI) problem.  This problem is modeled by
!  the partial differential equation
!  
!          -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
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
!
!  We place common blocks, variable declarations, and other include files
!  needed for this code in the single file ex5f.h.  We then need to include
!  only this file throughout the various routines in this program.  See
!  additional comments in the file ex5f.h.
!
#include "ex5f.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     snes        - nonlinear solver
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!     its         - iterations for convergence
!     Nx, Ny      - number of preocessors in x- and y- directions
!     matrix_free - flag - 1 indicates matrix-free version
!
!  See additional variable declarations in the file ex5f.h
!
      SNES                   snes
      Vec                    x,r
      Mat                    J
      ISLocalToGlobalMapping isltog
      integer                its,Nx,Ny,matrix_free,flg,N,ierr,m
      double precision       lambda_max,lambda_min

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction,FormInitialGuess,FormJacobian

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!  Initialize program
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
         SETERRA(1,0,' ',ierr)
      endif
      N = mx*my

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,        &
     &                snes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DA) to manage parallel grid and vectors

      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Nx',Nx,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-Ny',Ny,flg,ierr)
      if (Nx*Ny .ne. size .and.                                         &
     &      (Nx .ne. PETSC_DECIDE .or. Ny .ne. PETSC_DECIDE)) then
         if (rank .eq. 0)  then
            write(6,*) 'Incompatible number of procs: Nx * Ny != size'
         endif
         SETERRA(1,0,' ',ierr)
      endif
! This really needs only the star-type stencil, but we use the box
! stencil temporarily.
      call DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,mx,&
     &     my,Nx,Ny,1,1,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr)

!
!   Visualize the distribution of the array across the processors
!
!     call DAView(da,VIEWER_DRAW_WORLD,ierr)

!  Extract global and local vectors from DA; then duplicate for remaining
!  vectors that are the same types

      call DACreateGlobalVector(da,x,ierr)
      call DACreateLocalVector(da,localX,ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(localX,localF,ierr)

!  Get local grid boundaries (for 2-dimensional DA)

      call DAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,              &
     &     PETSC_NULL_INTEGER,ierr)
      call DAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,     &
     &     PETSC_NULL_INTEGER,ierr)

!  Here we shift the starting indices up by one so that we can easily
!  use the Fortran convention of 1-based indices (rather 0-based indices).

      xs  = xs+1
      ys  = ys+1
      gxs = gxs+1
      gys = gys+1

      ye  = ys+ym-1
      xe  = xs+xm-1
      gye = gys+gym-1
      gxe = gxs+gxm-1

!  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,PETSC_NULL_OBJECT,ierr)

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

      call OptionsHasName(PETSC_NULL_CHARACTER,'-snes_mf',matrix_free,  &
     &                    ierr)
      if (matrix_free .eq. 0) then
         if (size .eq. 1) then
            call MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,                &
     &           PETSC_NULL_INTEGER,J,ierr)
        else
          call VecGetLocalSize(x,m,ierr)
          call MatCreateMPIAIJ(PETSC_COMM_WORLD,m,m,N,N,5,              &
     &         PETSC_NULL_INTEGER,3,PETSC_NULL_INTEGER,J,ierr)
        endif
        call SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL_OBJECT,   &
     &       ierr)

!       Get the mapping from local-to-global node numbers for all local nodes,
!       including ghost points.  Associate this mapping with the matrix for later
!       use in setting matrix entries via MatSetValuesLocal().

        call DAGetISLocalToGlobalMapping(da,isltog,ierr)
        call MatSetLocalToGlobalMapping(J,isltog,ierr)

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

      call FormInitialGuess(x,ierr)
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
!  X - vector
!
!  Output Parameter:
!  X - vector
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationInitialGuess", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely handles ghost point scatters and accesses
!  the local vector data via VecGetArray() and VecRestoreArray().
!
      subroutine FormInitialGuess(X,ierr)
      implicit none

#include "ex5f.h"

!  Input/output variables:
      Vec      X
      integer  ierr

!  Declarations for use with local arrays:
      Scalar      lx_v(0:1)
      PetscOffset lx_i

      ierr = 0

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

      call VecGetArray(localX,lx_v,lx_i,ierr)

!  Compute initial guess over the locally owned part of the grid

      call ApplicationInitialGuess(lx_v(lx_i),ierr)

!  Restore vector

      call VecRestoreArray(localX,lx_v,lx_i,ierr)

!  Insert values into global vector

      call DALocalToGlobal(da,localX,INSERT_VALUES,X,ierr)

      return 
      end

! ---------------------------------------------------------------------
!
!  ApplicationInitialGuess - Computes initial approximation, called by
!  the higher level routine FormInitialGuess().
!
!  Input Parameter:
!  x - local vector data
!
!  Output Parameters:
!  x - local vector data
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
      subroutine ApplicationInitialGuess(x,ierr)
      implicit none

#include "ex5f.h"

!  Input/output variables:
      Scalar  x(gxs:gxe,gys:gye)
      integer ierr

!  Local variables:
      integer  i,j,hxdhy,hydhx
      Scalar   temp1,temp,hx,hy,sc,one

!  Set parameters

      ierr   = 0
      one    = 1.0
      hx     = one/(dble(mx-1))
      hy     = one/(dble(my-1))
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx
      temp1  = lambda/(lambda + one)

      do 20 j=ys,ye
         temp = dble(min(j-1,my-j))*hy
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. mx .or. j .eq. my) then
              x(i,j) = 0.0
            else
              x(i,j) = temp1 *                                          &
     &          sqrt(min(dble(min(i-1,mx-i)*hx),dble(temp)))
            endif
 10      continue
 20   continue

      return 
      end

! ---------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  X - input vector
!  dummy - optional user-defined context, as set by SNESSetFunction()
!          (not used here)
!
!  Output Parameter:
!  F - function vector
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationFunction", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely handles ghost point scatters and accesses
!  the local vector data via VecGetArray() and VecRestoreArray().
!
      subroutine FormFunction(snes,X,F,dummy,ierr)
      implicit none

#include "ex5f.h"

!  Input/output variables:
      SNES     snes
      Vec      X,F
      integer  dummy
      integer  ierr

!  Declarations for use with local arrays:
      Scalar      lx_v(0:1),lf_v(0:1)
      PetscOffset lx_i,lf_i 

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
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the Fortran chapter of the users manual for details.

      call VecGetArray(localX,lx_v,lx_i,ierr)
      call VecGetArray(localF,lf_v,lf_i,ierr)

!  Compute function over the locally owned part of the grid

      call ApplicationFunction(lx_v(lx_i),lf_v(lf_i),ierr)

!  Restore vectors

      call VecRestoreArray(localX,lx_v,lx_i,ierr)
      call VecRestoreArray(localF,lf_v,lf_i,ierr)

!  Insert values into global vector

      call DALocalToGlobal(da,localF,INSERT_VALUES,F,ierr)
      call PLogFlops(11*ym*xm)

!      call VecView(X,VIEWER_STDOUT_WORLD,ierr)
!      call VecView(F,VIEWER_STDOUT_WORLD,ierr)

      return 
      end

! ---------------------------------------------------------------------
!
!  ApplicationFunction - Computes nonlinear function, called by
!  the higher level routine FormFunction().
!
!  Input Parameter:
!  x - local vector data
!
!  Output Parameters:
!  f - local vector data, f(x)
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
      subroutine ApplicationFunction(x,f,ierr)

      implicit none

#include "ex5f.h"

!  Input/output variables:
      Scalar   x(gxs:gxe,gys:gye),f(gxs:gxe,gys:gye)
      integer  ierr

!  Local variables:
      Scalar   two,one,hx,hy,hxdhy,hydhx,sc
      Scalar   u,uxx,uyy
      integer  i,j

      one    = 1.0
      two    = 2.0
      hx     = one/dble(mx-1)
      hy     = one/dble(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute function over the locally owned part of the grid

      do 20 j=ys,ye
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. mx .or. j .eq. my) then
               f(i,j) = x(i,j)
            else
               u = x(i,j) 
               uxx = hydhx * (two*u                                     &
     &                - x(i-1,j) - x(i+1,j))
               uyy = hxdhy * (two*u - x(i,j-1) - x(i,j+1))
               f(i,j) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

      return
      end

! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  snes     - the SNES context
!  x        - input vector
!  dummy    - optional user-defined context, as set by SNESSetJacobian()
!             (not used here)
!
!  Output Parameters:
!  jac      - Jacobian matrix
!  jac_prec - optionally different preconditioning matrix (not used here)
!  flag     - flag indicating matrix structure
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationJacobian", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
!  Notes:
!  Due to grid point reordering with DAs, we must always work
!  with the local grid points, and then transform them to the new
!  global numbering with the local-to-global mapping.  We
!  cannot work directly with the global numbers for the original
!  uniprocessor grid!
!
!  Two methods are available for imposing this transformation
!  when setting matrix entries:
!    (A) MatSetValuesLocal(), using the local ordering (including
!        ghost points!)
!         - Do the following two steps once, before calling SNESSolve()
!           - Use DAGetISLocalToGlobalMapping() to extract the
!             local-to-global map from the DA
!           - Associate this map with the matrix by calling
!             MatSetLocalToGlobalMapping() 
!        - Then set matrix entries using the local ordering
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering 
!        - Use DAGetGlobalIndices() to extract the local-to-global map
!        - Then apply this map explicitly yourself
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobian(snes,X,jac,jac_prec,flag,dummy,ierr)
      implicit none

#include "ex5f.h"

!  Input/output variables:
      SNES         snes
      Vec          X
      Mat          jac,jac_prec
      MatStructure flag
      integer      dummy
      integer      ierr

!  Declarations for use with local arrays:
      Scalar      lx_v(0:1)
      PetscOffset lx_i

!  Scatter ghost points to local vector,using the 2-step process
!     DAGlobalToLocalBegin(), DAGlobalToLocalEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr)
      call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get a pointer to vector data

      call VecGetArray(localX,lx_v,lx_i,ierr)

!  Compute entries for the locally owned part of the Jacobian.

      call ApplicationJacobian(lx_v(lx_i),jac,jac_prec,ierr)

!  Assemble matrix, using the 2-step process:
!     MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArray(localX,lx_v,lx_i,ierr)
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

!  Tell the matrix we will never add a new nonzero location to the
!  matrix. If we do it will generate an error.

       call MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,ierr)

      return
      end

! ---------------------------------------------------------------------
!
!  ApplicationJacobian - Computes Jacobian matrix, called by
!  the higher level routine FormJacobian().
!
!  Input Parameters:
!  x        - local vector data
!
!  Output Parameters:
!  jac      - Jacobian matrix
!  jac_prec - optionally different preconditioning matrix (not used here)
!  ierr     - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
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
      subroutine ApplicationJacobian(x,jac,jac_prec,ierr)
      implicit none

#include "ex5f.h"

!  Input/output variables:
      Scalar   x(gxs:gxe,gys:gye)
      Mat      jac,jac_prec
      integer  ierr

!  Local variables:
      integer  row,col(5),i,j
      Scalar   two,one,hx,hy,hxdhy,hydhx,sc,v(5)

!  Set parameters

      one    = 1.0
      two    = 2.0
      hx     = one/dble(mx-1)
      hy     = one/dble(my-1)
      sc     = hx*hy
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute entries for the locally owned part of the Jacobian.
!   - Currently, all PETSc parallel matrix formats are partitioned by
!     contiguous chunks of rows across the processors. 
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly). 
!   - Here, we set all entries for a particular row at once.
!   - We can set matrix entries either using either
!     MatSetValuesLocal() or MatSetValues(), as discussed above.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.

      do 20 j=ys,ye
         row = (j - gys)*gxm + xs - gxs - 1
         do 10 i=xs,xe
            row = row + 1
!           boundary points
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. mx .or. j .eq. my) then
               call MatSetValuesLocal(jac,1,row,1,row,one,              &
     &                           INSERT_VALUES,ierr)
!           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy)                               &
     &                  - sc*lambda*exp(x(i,j))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - gxm
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + gxm
               call MatSetValuesLocal(jac,1,row,5,col,v,                &
     &                                INSERT_VALUES,ierr)
            endif
 10      continue
 20   continue

      return
      end


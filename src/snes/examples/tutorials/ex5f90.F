!
!  Description: Solves a nonlinear system in parallel with SNES.
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DMDAs) to partition the parallel grid.
!  The command line options include:
!    -par <parameter>, where <parameter> indicates the nonlinearity of the problem
!       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
!
!/*T
!  Concepts: SNES^parallel Bratu example
!  Concepts: DMDA^using distributed arrays;
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
!  The following define must be used before including any PETSc include files
!  into a module or interface. This is because they can't handle declarations
!  in them
!

      module f90module
      type userctx
#include <finclude/petscsysdef.h>
#include <finclude/petscvecdef.h>
#include <finclude/petscdmdef.h>
        PetscInt xs,xe,xm,gxs,gxe,gxm
        PetscInt ys,ye,ym,gys,gye,gym
        PetscInt mx,my
        PetscMPIInt rank
        double precision lambda
      end type userctx

      contains
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
!  "FormFunctionLocal", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely handles ghost point scatters and accesses
!  the local vector data via VecGetArrayF90() and VecRestoreArrayF90().
!
      subroutine FormFunction(snes,X,F,user,ierr)
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>
#include <finclude/petscvec.h90>
#include <finclude/petscsnes.h90>

!  Input/output variables:
      SNES           snes
      Vec            X,F
      PetscErrorCode ierr
      type (userctx) user
      DM             da

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:),lf_v(:)
      Vec            localX

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can
!  be done while messages are in transition.
      call SNESGetDM(snes,da,ierr)
      call DMGetLocalVector(da,localX,ierr)
      call DMGlobalToLocalBegin(da,X,INSERT_VALUES,                     &
     &     localX,ierr)
      call DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray90() returns a pointer to
!      the data array. Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArrayF90() when you no longer need access to
!      the array.
!    - Note that the interface to VecGetArrayF90() differs from VecGetArray(),
!      and is useable from Fortran-90 Only.

      call VecGetArrayF90(localX,lx_v,ierr)
      call VecGetArrayF90(F,lf_v,ierr)

!  Compute function over the locally owned part of the grid
      call FormFunctionLocal(lx_v,lf_v,user,ierr)

!  Restore vectors
      call VecRestoreArrayF90(localX,lx_v,ierr)
      call VecRestoreArrayF90(F,lf_v,ierr)

!  Insert values into global vector

      call DMRestoreLocalVector(da,localX,ierr)
      call PetscLogFlops(11.0d0*user%ym*user%xm,ierr)

!      call VecView(X,PETSC_VIEWER_STDOUT_WORLD,ierr)
!      call VecView(F,PETSC_VIEWER_STDOUT_WORLD,ierr)
      return 
      end subroutine formfunction
      end module f90module

      module f90moduleinterfaces
        use f90module
 
      Interface SNESSetApplicationContext
        Subroutine SNESSetApplicationContext(snes,ctx,ierr)
        use f90module
          SNES snes
          type(userctx) ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESSetApplicationContext

      Interface SNESGetApplicationContext
        Subroutine SNESGetApplicationContext(snes,ctx,ierr)
        use f90module
          SNES snes
          type(userctx), pointer :: ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESGetApplicationContext
      end module f90moduleinterfaces

      program main
      use f90module
      use f90moduleinterfaces
      implicit none
!
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>
#include <finclude/petscvec.h90>
#include <finclude/petscdmda.h90>

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
      SNES             snes
      Vec              x,r
      Mat              J
      PetscErrorCode   ierr
      PetscInt         its
      PetscBool        flg,matrix_free
      PetscInt         ione,nfour
      double precision lambda_max,lambda_min
      type (userctx)   user
      DM               da

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
      external FormInitialGuess,FormJacobian

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!  Initialize program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,user%rank,ierr)

!  Initialize problem parameters
      lambda_max  = 6.81
      lambda_min  = 0.0
      user%lambda = 6.0
      ione = 1
      nfour = -4
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-par',             &
     &     user%lambda,flg,ierr)
      if (user%lambda .ge. lambda_max .or. user%lambda .le. lambda_min) &
     &     then
         if (user%rank .eq. 0) write(6,*) 'Lambda is out of range'
         SETERRQ(PETSC_COMM_SELF,1,' ',ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DMDA) to manage parallel grid and vectors

! This really needs only the star-type stencil, but we use the box
! stencil temporarily.
      call DMDACreate2d(PETSC_COMM_WORLD,                               &
     &     DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,                      &
     &     DMDA_STENCIL_BOX,nfour,nfour,PETSC_DECIDE,PETSC_DECIDE,          &
     &     ione,ione,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr)
      call DMDAGetInfo(da,PETSC_NULL_INTEGER,user%mx,user%my,           &
     &               PETSC_NULL_INTEGER,                                &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,ierr)
      
!
!   Visualize the distribution of the array across the processors
!
!     call DMView(da,PETSC_VIEWER_DRAW_WORLD,ierr)

!  Extract global and local vectors from DMDA; then duplicate for remaining
!  vectors that are the same types
      call DMCreateGlobalVector(da,x,ierr)
      call VecDuplicate(x,r,ierr)

!  Get local grid boundaries (for 2-dimensional DMDA)
      call DMDAGetCorners(da,user%xs,user%ys,PETSC_NULL_INTEGER,        &
     &     user%xm,user%ym,PETSC_NULL_INTEGER,ierr)
      call DMDAGetGhostCorners(da,user%gxs,user%gys,                    &
     &     PETSC_NULL_INTEGER,user%gxm,user%gym,                        &
     &     PETSC_NULL_INTEGER,ierr)

!  Here we shift the starting indices up by one so that we can easily
!  use the Fortran convention of 1-based indices (rather 0-based indices).
      user%xs  = user%xs+1
      user%ys  = user%ys+1
      user%gxs = user%gxs+1
      user%gys = user%gys+1

      user%ye  = user%ys+user%ym-1
      user%xe  = user%xs+user%xm-1
      user%gye = user%gys+user%gym-1
      user%gxe = user%gxs+user%gxm-1

      call SNESSetApplicationContext(snes,user,ierr)

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
!     accordingly.  When using distributed arrays (DMDAs) to create vectors,
!     the DMDAs determine the problem partitioning.  We must explicitly
!     specify the local matrix dimensions upon its creation for compatibility
!     with the vector distribution.  Thus, the generic MatCreate() routine
!     is NOT sufficient when working with distributed arrays.
!
!     Note: Here we only approximately preallocate storage space for the
!     Jacobian.  See the users manual for a discussion of better techniques
!     for preallocating matrix memory.
      
      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-snes_mf',         &
     &     matrix_free,ierr)
      if (.not. matrix_free) then
        call DMCreateMatrix(da,MATAIJ,J,ierr)
        call SNESSetJacobian(snes,J,J,FormJacobian,user,ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
      call SNESSetDM(snes,da,ierr)
      call SNESSetFromOptions(snes,ierr)


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      call FormInitialGuess(snes,x,ierr)
      call SNESSolve(snes,PETSC_NULL_OBJECT,x,ierr) 
      call SNESGetIterationNumber(snes,its,ierr);
      if (user%rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (.not. matrix_free) call MatDestroy(J,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call SNESDestroy(snes,ierr)
      call DMDestroy(da,ierr)

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
!  "InitialGuessLocal", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely handles ghost point scatters and accesses
!  the local vector data via VecGetArrayF90() and VecRestoreArrayF90().
!
      subroutine FormInitialGuess(snes,X,ierr)
      use f90module
      use f90moduleinterfaces
      implicit none

#include <finclude/petscvec.h90>
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>

!  Input/output variables:
      SNES           snes
      type(userctx), pointer:: puser
      Vec            X
      PetscErrorCode ierr
      DM             da

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:)
      Vec               localX

      ierr = 0
      call SNESGetDM(snes,da,ierr)
      call SNESGetApplicationContext(snes,puser,ierr)
!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray90() returns a pointer to
!      the data array. Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArrayF90() when you no longer need access to
!      the array.
!    - Note that the interface to VecGetArrayF90() differs from VecGetArray(),
!      and is useable from Fortran-90 Only.

      call DMGetLocalVector(da,localX,ierr)
      call VecGetArrayF90(localX,lx_v,ierr)

!  Compute initial guess over the locally owned part of the grid
      call InitialGuessLocal(puser,lx_v,ierr)

!  Restore vector
      call VecRestoreArrayF90(localX,lx_v,ierr)

!  Insert values into global vector
      call DMLocalToGlobalBegin(da,localX,INSERT_VALUES,X,ierr)
      call DMLocalToGlobalEnd(da,localX,INSERT_VALUES,X,ierr)
      call DMRestoreLocalVector(da,localX,ierr)

      return 
      end

! ---------------------------------------------------------------------
!
!  InitialGuessLocal - Computes initial approximation, called by
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
      subroutine InitialGuessLocal(user,x,ierr)
      use f90module
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>

!  Input/output variables:
      type (userctx)         user
      PetscScalar  x(user%gxs:user%gxe,                                 &                         
     &              user%gys:user%gye)
      PetscErrorCode ierr

!  Local variables:
      PetscInt  i,j
      PetscScalar   temp1,temp,hx,hy
      PetscScalar   one

!  Set parameters

      ierr   = 0
      one    = 1.0
      hx     = one/(dble(user%mx-1))
      hy     = one/(dble(user%my-1))
      temp1  = user%lambda/(user%lambda + one)

      do 20 j=user%ys,user%ye
         temp = dble(min(j-1,user%my-j))*hy
         do 10 i=user%xs,user%xe
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. user%mx .or. j .eq. user%my) then
              x(i,j) = 0.0
            else
              x(i,j) = temp1 *                                          &
     &          sqrt(min(dble(min(i-1,user%mx-i)*hx),dble(temp)))
            endif
 10      continue
 20   continue

      return 
      end

! ---------------------------------------------------------------------
!
!  FormFunctionLocal - Computes nonlinear function, called by
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
      subroutine FormFunctionLocal(x,f,user,ierr)
      use f90module

      implicit none

!  Input/output variables:
      type (userctx) user
      PetscScalar  x(user%gxs:user%gxe,                                         &
     &              user%gys:user%gye)
      PetscScalar  f(user%xs:user%xe,                                           &
     &              user%ys:user%ye)
      PetscErrorCode ierr

!  Local variables:
      PetscScalar two,one,hx,hy,hxdhy,hydhx,sc
      PetscScalar u,uxx,uyy
      PetscInt  i,j

      one    = 1.0
      two    = 2.0
      hx     = one/dble(user%mx-1)
      hy     = one/dble(user%my-1)
      sc     = hx*hy*user%lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute function over the locally owned part of the grid

      do 20 j=user%ys,user%ye
         do 10 i=user%xs,user%xe
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. user%mx .or. j .eq. user%my) then
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
!  "FormJacobianLocal", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArrayF90() and VecRestoreArrayF90().
!
!  Notes:
!  Due to grid point reordering with DMDAs, we must always work
!  with the local grid points, and then transform them to the new
!  global numbering with the "ltog" mapping (via DMDAGetGlobalIndicesF90()).
!  We cannot work directly with the global numbers for the original
!  uniprocessor grid!
!
!  Two methods are available for imposing this transformation
!  when setting matrix entries:
!    (A) MatSetValuesLocal(), using the local ordering (including
!        ghost points!)
!        - Use DMDAGetGlobalIndicesF90() to extract the local-to-global map
!        - Associate this map with the matrix by calling
!          MatSetLocalToGlobalMapping() once
!        - Set matrix entries using the local ordering
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering 
!        - Use DMDAGetGlobalIndicesF90() to extract the local-to-global map
!        - Then apply this map explicitly yourself
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobian(snes,X,jac,jac_prec,flag,user,ierr)
      use f90module
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>

#include <finclude/petscvec.h90>

!  Input/output variables:
      SNES         snes
      Vec          X
      Mat          jac,jac_prec
      MatStructure flag
      type(userctx)  user
      PetscErrorCode ierr
      DM             da

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:)
      Vec            localX

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call SNESGetDM(snes,da,ierr)
      call DMGetLocalVector(da,localX,ierr)
      call DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX,              &
     &     ierr)
      call DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr)

!  Get a pointer to vector data
      call VecGetArrayF90(localX,lx_v,ierr)

!  Compute entries for the locally owned part of the Jacobian preconditioner.
      call FormJacobianLocal(lx_v,jac_prec,user,ierr)

!  Assemble matrix, using the 2-step process:
!     MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      if (jac .ne. jac_prec) then
         call MatAssemblyBegin(jac_prec,MAT_FINAL_ASSEMBLY,ierr)
      endif
      call VecRestoreArrayF90(localX,lx_v,ierr)
      call DMRestoreLocalVector(da,localX,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      if (jac .ne. jac_prec) then
        call MatAssemblyEnd(jac_prec,MAT_FINAL_ASSEMBLY,ierr)
      endif
        
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

      call MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,      &
     &                  ierr)

      return
      end

! ---------------------------------------------------------------------
!
!  FormJacobianLocal - Computes Jacobian preconditioner matrix, 
!  called by the higher level routine FormJacobian().
!
!  Input Parameters:
!  x        - local vector data
!
!  Output Parameters:
!  jac_prec - Jacobian preconditioner matrix
!  ierr     - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
!  Notes:
!  Due to grid point reordering with DMDAs, we must always work
!  with the local grid points, and then transform them to the new
!  global numbering with the "ltog" mapping (via DMDAGetGlobalIndicesF90()).
!  We cannot work directly with the global numbers for the original
!  uniprocessor grid!
!
!  Two methods are available for imposing this transformation
!  when setting matrix entries:
!    (A) MatSetValuesLocal(), using the local ordering (including
!        ghost points!)
!        - Use DMDAGetGlobalIndicesF90() to extract the local-to-global map
!        - Associate this map with the matrix by calling
!          MatSetLocalToGlobalMapping() once
!        - Set matrix entries using the local ordering
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering 
!        - Use DMDAGetGlobalIndicesF90() to extract the local-to-global map
!        - Then apply this map explicitly yourself
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobianLocal(x,jac_prec,user,ierr)
      use f90module
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>

!  Input/output variables:
      type (userctx) user
      PetscScalar    x(user%gxs:user%gxe,                                      &
     &               user%gys:user%gye)
      Mat            jac_prec
      PetscErrorCode ierr

!  Local variables:
      PetscInt    row,col(5),i,j
      PetscInt    ione,ifive
      PetscScalar two,one,hx,hy,hxdhy
      PetscScalar hydhx,sc,v(5)

!  Set parameters
      ione   = 1
      ifive  = 5
      one    = 1.0
      two    = 2.0
      hx     = one/dble(user%mx-1)
      hy     = one/dble(user%my-1)
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

      do 20 j=user%ys,user%ye
         row = (j - user%gys)*user%gxm + user%xs - user%gxs - 1
         do 10 i=user%xs,user%xe
            row = row + 1
!           boundary points
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. user%mx .or. j .eq. user%my) then
               col(1) = row
               v(1)   = one
               call MatSetValuesLocal(jac_prec,ione,row,ione,col,v,          &
     &                           INSERT_VALUES,ierr)
!           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy)                               &
     &                  - sc*user%lambda*exp(x(i,j))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - user%gxm
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + user%gxm
               call MatSetValuesLocal(jac_prec,ione,row,ifive,col,v,         &
     &                                INSERT_VALUES,ierr)
            endif
 10      continue
 20   continue

      return
      end




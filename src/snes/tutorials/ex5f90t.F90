!
!  Description: Solves a nonlinear system in parallel with SNES.
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DMDAs) to partition the parallel grid.
!  The command line options include:
!    -par <parameter>, where <parameter> indicates the nonlinearity of the problem
!       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
!
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
!  The uniprocessor version of this code is snes/tutorials/ex4f.F
!
!  --------------------------------------------------------------------------
!  The following define must be used before including any PETSc include files
!  into a module or interface. This is because they can't handle declarations
!  in them
!

      module ex5f90tmodule
#include <petsc/finclude/petscdmda.h>
      use petscdmda
      type userctx
        type(tDM) da
        PetscInt xs,xe,xm,gxs,gxe,gxm
        PetscInt ys,ye,ym,gys,gye,gym
        PetscInt mx,my
        PetscMPIInt rank
        PetscReal lambda
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
!  the local vector data via VecGetArray() and VecRestoreArray().
!
      subroutine FormFunction(snesIn,X,F,user,ierr)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      use petscdmda

!  Input/output variables:
      type(tSNES)     snesIn
      type(tVec)      X,F
      PetscErrorCode ierr
      type (userctx) user

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:),lf_v(:)
      type(tVec)              localX

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can
!  be done while messages are in transition.
      PetscCall(DMGetLocalVector(user%da,localX,ierr))
      PetscCall(DMGlobalToLocalBegin(user%da,X,INSERT_VALUES,localX,ierr))
      PetscCall(DMGlobalToLocalEnd(user%da,X,INSERT_VALUES,localX,ierr))

!  Get a pointer to vector data.
!    - VecGetArray90() returns a pointer to the data array.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

      PetscCall(VecGetArray(localX,lx_v,ierr))
      PetscCall(VecGetArray(F,lf_v,ierr))

!  Compute function over the locally owned part of the grid
      PetscCall(FormFunctionLocal(lx_v,lf_v,user,ierr))

!  Restore vectors
      PetscCall(VecRestoreArray(localX,lx_v,ierr))
      PetscCall(VecRestoreArray(F,lf_v,ierr))

!  Insert values into global vector

      PetscCall(DMRestoreLocalVector(user%da,localX,ierr))
      PetscCall(PetscLogFlops(11.0d0*user%ym*user%xm,ierr))

!      PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD,ierr))
!      PetscCall(VecView(F,PETSC_VIEWER_STDOUT_WORLD,ierr))
      end subroutine formfunction
      end module ex5f90tmodule

      module f90moduleinterfacest
        use ex5f90tmodule

      Interface SNESSetApplicationContext
        Subroutine SNESSetApplicationContext(snesIn,ctx,ierr)
#include <petsc/finclude/petscsnes.h>
        use petscsnes
        use ex5f90tmodule
          type(tSNES)    snesIn
          type(userctx) ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESSetApplicationContext

      Interface SNESGetApplicationContext
        Subroutine SNESGetApplicationContext(snesIn,ctx,ierr)
#include <petsc/finclude/petscsnes.h>
        use petscsnes
        use ex5f90tmodule
          type(tSNES)     snesIn
          type(userctx), pointer :: ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESGetApplicationContext
      end module f90moduleinterfacest

      program main
#include <petsc/finclude/petscdmda.h>
#include <petsc/finclude/petscsnes.h>
      use petscdmda
      use petscsnes
      use ex5f90tmodule
      use f90moduleinterfacest
      implicit none
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     mysnes      - nonlinear solver
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!     its         - iterations for convergence
!     Nx, Ny      - number of preocessors in x- and y- directions
!     matrix_free - flag - 1 indicates matrix-free version
!
      type(tSNES)       mysnes
      type(tVec)        x,r
      type(tMat)        J
      PetscErrorCode   ierr
      PetscInt         its
      PetscBool        flg,matrix_free,set
      PetscInt         ione,nfour
      PetscReal lambda_max,lambda_min
      type(userctx)    user
      type(userctx), pointer:: puser
      type(tPetscOptions) :: options

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
      external FormInitialGuess,FormJacobian

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Initialize program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,user%rank,ierr))

!  Initialize problem parameters
      options%v = 0
      lambda_max  = 6.81
      lambda_min  = 0.0
      user%lambda = 6.0
      ione = 1
      nfour = 4
      PetscCallA(PetscOptionsGetReal(options,PETSC_NULL_CHARACTER,'-par',user%lambda,flg,ierr))
      PetscCheckA(user%lambda .lt. lambda_max .and. user%lambda .gt. lambda_min,PETSC_COMM_SELF,PETSC_ERR_USER,'Lambda provided with -par is out of range')

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(SNESCreate(PETSC_COMM_WORLD,mysnes,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DMDA) to manage parallel grid and vectors

! This really needs only the star-type stencil, but we use the box
! stencil temporarily.
      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nfour,nfour,PETSC_DECIDE,PETSC_DECIDE,ione,ione,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,user%da,ierr))
      PetscCallA(DMSetFromOptions(user%da,ierr))
      PetscCallA(DMSetUp(user%da,ierr))
      PetscCallA(DMDAGetInfo(user%da,PETSC_NULL_INTEGER,user%mx,user%my,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_DMBOUNDARYTYPE,PETSC_NULL_DMBOUNDARYTYPE,PETSC_NULL_DMBOUNDARYTYPE,PETSC_NULL_DMDASTENCILTYPE,ierr))

!
!   Visualize the distribution of the array across the processors
!
!     PetscCallA(DMView(user%da,PETSC_VIEWER_DRAW_WORLD,ierr))

!  Extract global and local vectors from DMDA; then duplicate for remaining
!  vectors that are the same types
      PetscCallA(DMCreateGlobalVector(user%da,x,ierr))
      PetscCallA(VecDuplicate(x,r,ierr))

!  Get local grid boundaries (for 2-dimensional DMDA)
      PetscCallA(DMDAGetCorners(user%da,user%xs,user%ys,PETSC_NULL_INTEGER,user%xm,user%ym,PETSC_NULL_INTEGER,ierr))
      PetscCallA(DMDAGetGhostCorners(user%da,user%gxs,user%gys,PETSC_NULL_INTEGER,user%gxm,user%gym,PETSC_NULL_INTEGER,ierr))

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

      PetscCallA(SNESSetApplicationContext(mysnes,user,ierr))

!  Set function evaluation routine and vector
      PetscCallA(SNESSetFunction(mysnes,r,FormFunction,user,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix data structure; set Jacobian evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set Jacobian matrix data structure and default Jacobian evaluation
!  routine. User can override with:
!     -snes_fd : default finite differencing approximation of Jacobian
!     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
!                (unless user explicitly sets preconditioner)
!     -snes_mf_operator : form matrix used to construct the preconditioner as set by the user,
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

      PetscCallA(PetscOptionsHasName(options,PETSC_NULL_CHARACTER,'-snes_mf',matrix_free,ierr))
      if (.not. matrix_free) then
        PetscCallA(DMSetMatType(user%da,MATAIJ,ierr))
        PetscCallA(DMCreateMatrix(user%da,J,ierr))
        PetscCallA(SNESSetJacobian(mysnes,J,J,FormJacobian,user,ierr))
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
      PetscCallA(SNESSetFromOptions(mysnes,ierr))

!     Test Fortran90 wrapper for SNESSet/Get ApplicationContext()
      PetscCallA(PetscOptionsGetBool(options,PETSC_NULL_CHARACTER,'-test_appctx',flg,set,ierr))
      if (flg) then
        PetscCallA(SNESGetApplicationContext(mysnes,puser,ierr))
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      PetscCallA(FormInitialGuess(mysnes,x,ierr))
      PetscCallA(SNESSolve(mysnes,PETSC_NULL_VEC,x,ierr))
      PetscCallA(SNESGetIterationNumber(mysnes,its,ierr))
      if (user%rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (.not. matrix_free) PetscCallA(MatDestroy(J,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(r,ierr))
      PetscCallA(SNESDestroy(mysnes,ierr))
      PetscCallA(DMDestroy(user%da,ierr))

      PetscCallA(PetscFinalize(ierr))
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
!  the local vector data via VecGetArray() and VecRestoreArray().
!
      subroutine FormInitialGuess(mysnes,X,ierr)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      use ex5f90tmodule
      use f90moduleinterfacest
!  Input/output variables:
      type(tSNES)     mysnes
      type(userctx), pointer:: puser
      type(tVec)      X
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:)

      ierr = 0
      PetscCallA(SNESGetApplicationContext(mysnes,puser,ierr))
!  Get a pointer to vector data.
!    - VecGetArray90() returns a pointer to the data array.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

      PetscCallA(VecGetArray(X,lx_v,ierr))

!  Compute initial guess over the locally owned part of the grid
      PetscCallA(InitialGuessLocal(puser,lx_v,ierr))

!  Restore vector
      PetscCallA(VecRestoreArray(X,lx_v,ierr))

!  Insert values into global vector

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
#include <petsc/finclude/petscsys.h>
      use petscsys
      use ex5f90tmodule
!  Input/output variables:
      type (userctx)         user
      PetscScalar  x(user%xs:user%xe,user%ys:user%ye)
      PetscErrorCode ierr

!  Local variables:
      PetscInt  i,j
      PetscScalar   temp1,temp,hx,hy
      PetscScalar   one

!  Set parameters

      ierr   = 0
      one    = 1.0
      hx     = one/(PetscIntToReal(user%mx-1))
      hy     = one/(PetscIntToReal(user%my-1))
      temp1  = user%lambda/(user%lambda + one)

      do 20 j=user%ys,user%ye
         temp = PetscIntToReal(min(j-1,user%my-j))*hy
         do 10 i=user%xs,user%xe
            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. user%mx .or. j .eq. user%my) then
              x(i,j) = 0.0
            else
              x(i,j) = temp1 * sqrt(min(PetscIntToReal(min(i-1,user%mx-i)*hx),PetscIntToReal(temp)))
            endif
 10      continue
 20   continue

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
#include <petsc/finclude/petscsys.h>
      use petscsys
      use ex5f90tmodule
!  Input/output variables:
      type (userctx) user
      PetscScalar  x(user%gxs:user%gxe,user%gys:user%gye)
      PetscScalar  f(user%xs:user%xe,user%ys:user%ye)
      PetscErrorCode ierr

!  Local variables:
      PetscScalar two,one,hx,hy,hxdhy,hydhx,sc
      PetscScalar u,uxx,uyy
      PetscInt  i,j

      one    = 1.0
      two    = 2.0
      hx     = one/PetscIntToReal(user%mx-1)
      hy     = one/PetscIntToReal(user%my-1)
      sc     = hx*hy*user%lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute function over the locally owned part of the grid

      do 20 j=user%ys,user%ye
         do 10 i=user%xs,user%xe
            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. user%mx .or. j .eq. user%my) then
               f(i,j) = x(i,j)
            else
               u = x(i,j)
               uxx = hydhx * (two*u - x(i-1,j) - x(i+1,j))
               uyy = hxdhy * (two*u - x(i,j-1) - x(i,j+1))
               f(i,j) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue
      ierr = 0
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
!  jac_prec - optionally different matrix used to construct the preconditioner (not used here)
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "FormJacobianLocal", where the actual computations are
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
!  Notes:
!  Due to grid point reordering with DMDAs, we must always work
!  with the local grid points, and then transform them to the new
!  global numbering with the "ltog" mapping
!  We cannot work directly with the global numbers for the original
!  uniprocessor grid!
!
!  Two methods are available for imposing this transformation
!  when setting matrix entries:
!    (A) MatSetValuesLocal(), using the local ordering (including
!        ghost points!)
!        - Set matrix entries using the local ordering
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering
!        - Use DMGetLocalToGlobalMapping() then
!          ISLocalToGlobalMappingGetIndices() to extract the local-to-global map
!        - Then apply this map explicitly yourself
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobian(mysnes,X,jac,jac_prec,user,ierr)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      use ex5f90tmodule
!  Input/output variables:
      type(tSNES)     mysnes
      type(tVec)      X
      type(tMat)      jac,jac_prec
      type(userctx)  user
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:)
      type(tVec)      localX

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      PetscCallA(DMGetLocalVector(user%da,localX,ierr))
      PetscCallA(DMGlobalToLocalBegin(user%da,X,INSERT_VALUES,localX,ierr))
      PetscCallA(DMGlobalToLocalEnd(user%da,X,INSERT_VALUES,localX,ierr))

!  Get a pointer to vector data
      PetscCallA(VecGetArray(localX,lx_v,ierr))

!  Compute entries for the locally owned part of the Jacobian preconditioner.
      PetscCallA(FormJacobianLocal(lx_v,jac_prec,user,ierr))

!  Assemble matrix, using the 2-step process:
!     MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      PetscCallA(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
!      if (jac .ne. jac_prec) then
         PetscCallA(MatAssemblyBegin(jac_prec,MAT_FINAL_ASSEMBLY,ierr))
!      endif
      PetscCallA(VecRestoreArray(localX,lx_v,ierr))
      PetscCallA(DMRestoreLocalVector(user%da,localX,ierr))
      PetscCallA(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))
!      if (jac .ne. jac_prec) then
        PetscCallA(MatAssemblyEnd(jac_prec,MAT_FINAL_ASSEMBLY,ierr))
!      endif

!  Tell the matrix we will never add a new nonzero location to the
!  matrix. If we do it will generate an error.

      PetscCallA(MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr))

      end

! ---------------------------------------------------------------------
!
!  FormJacobianLocal - Computes Jacobian matrix used to compute the preconditioner,
!  called by the higher level routine FormJacobian().
!
!  Input Parameters:
!  x        - local vector data
!
!  Output Parameters:
!  jac_prec - Jacobian matrix used to compute the preconditioner
!  ierr     - error code
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
!  Notes:
!  Due to grid point reordering with DMDAs, we must always work
!  with the local grid points, and then transform them to the new
!  global numbering with the "ltog" mapping
!  We cannot work directly with the global numbers for the original
!  uniprocessor grid!
!
!  Two methods are available for imposing this transformation
!  when setting matrix entries:
!    (A) MatSetValuesLocal(), using the local ordering (including
!        ghost points!)
!        - Set matrix entries using the local ordering
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobianLocal(x,jac_prec,user,ierr)
#include <petsc/finclude/petscmat.h>
      use petscmat
      use ex5f90tmodule
!  Input/output variables:
      type (userctx) user
      PetscScalar    x(user%gxs:user%gxe,user%gys:user%gye)
      type(tMat)      jac_prec
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
      hx     = one/PetscIntToReal(user%mx-1)
      hy     = one/PetscIntToReal(user%my-1)
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
            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. user%mx .or. j .eq. user%my) then
               col(1) = row
               v(1)   = one
               PetscCallA(MatSetValuesLocal(jac_prec,ione,[row],ione,col,v,INSERT_VALUES,ierr))
!           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy) - sc*user%lambda*exp(x(i,j))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - user%gxm
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + user%gxm
               PetscCallA(MatSetValuesLocal(jac_prec,ione,[row],ifive,col,v,INSERT_VALUES,ierr))
            endif
 10      continue
 20   continue
      end

!/*TEST
!
!   test:
!      nsize: 4
!      args: -snes_mf -pc_type none -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always
!
!TEST*/

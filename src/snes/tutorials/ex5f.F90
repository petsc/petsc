!
!  This example shows how to avoid Fortran line lengths larger than 132 characters.
!  It avoids used of certain macros such as PetscCallA() and PetscCheckA() that
!  generate very long lines
!
!  We recommend starting from src/snes/tutorials/ex5f90.F90 instead of this example
!  because that does not have the restricted formatting that makes this version
!  more difficult to read
!
!  Description: This example solves a nonlinear system in parallel with SNES.
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DMDAs) to partition the parallel grid.
!  The command line options include:
!    -par <param>, where <param> indicates the nonlinearity of the problem
!       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
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
!  --------------------------------------------------------------------------
      module ex5fmodule
      use petscsnes
      use petscdmda
#include <petsc/finclude/petscsnes.h>
#include <petsc/finclude/petscdmda.h>
      PetscInt xs,xe,xm,gxs,gxe,gxm
      PetscInt ys,ye,ym,gys,gye,gym
      PetscInt mx,my
      PetscMPIInt rank,size
      PetscReal lambda
      end module ex5fmodule

      program main
      use ex5fmodule
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     snes        - nonlinear solver
!     x, r        - solution, residual vectors
!     its         - iterations for convergence
!
!  See additional variable declarations in the file ex5f.h
!
      SNES           snes
      Vec            x,r
      PetscInt       its,i1,i4
      PetscErrorCode ierr
      PetscReal      lambda_max,lambda_min
      PetscBool      flg
      DM             da

!  Note: Any user-defined Fortran routines (such as FormJacobianLocal)
!  MUST be declared as external.

      external FormInitialGuess
      external FormFunctionLocal,FormJacobianLocal
      external MySNESConverged

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Initialize program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(ierr)
      CHKERRA(ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      CHKERRMPIA(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      CHKERRMPIA(ierr)
!  Initialize problem parameters

      i1 = 1
      i4 = 4
      lambda_max = 6.81
      lambda_min = 0.0
      lambda     = 6.0
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-par',lambda,PETSC_NULL_BOOL,ierr)
      CHKERRA(ierr)

! this statement is split into multiple-lines to keep lines under 132 char limit - required by 'make check'
      if (lambda .ge. lambda_max .or. lambda .le. lambda_min) then
         ierr = PETSC_ERR_ARG_OUTOFRANGE
         SETERRA(PETSC_COMM_WORLD,ierr,'Lambda')
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)
      CHKERRA(ierr)

!  Set convergence test routine if desired

      call PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-my_snes_convergence',flg,ierr)
      CHKERRA(ierr)
      if (flg) then
        call SNESSetConvergenceTest(snes,MySNESConverged,0,PETSC_NULL_FUNCTION,ierr)
        CHKERRA(ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DMDA) to manage parallel grid and vectors

!     This really needs only the star-type stencil, but we use the box stencil

      call DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,i4,i4,PETSC_DECIDE,PETSC_DECIDE, &
                        i1,i1, PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,da,ierr)
      CHKERRA(ierr)
      call DMSetFromOptions(da,ierr)
      CHKERRA(ierr)
      call DMSetUp(da,ierr)
      CHKERRA(ierr)

!  Extract global and local vectors from DMDA; then duplicate for remaining
!  vectors that are the same types

      call DMCreateGlobalVector(da,x,ierr)
      CHKERRA(ierr)
      call VecDuplicate(x,r,ierr)
      CHKERRA(ierr)

!  Get local grid boundaries (for 2-dimensional DMDA)

      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,my,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
                       PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_DMBOUNDARYTYPE,PETSC_NULL_DMBOUNDARYTYPE, &
                       PETSC_NULL_DMBOUNDARYTYPE,PETSC_NULL_DMDASTENCILTYPE,ierr)
      CHKERRA(ierr)
      call DMDAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr)
      CHKERRA(ierr)
      call DMDAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr)
      CHKERRA(ierr)

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

      call DMDASNESSetFunctionLocal(da,INSERT_VALUES,FormFunctionLocal,da,ierr)
      CHKERRA(ierr)
      call DMDASNESSetJacobianLocal(da,FormJacobianLocal,da,ierr)
      CHKERRA(ierr)
      call SNESSetDM(snes,da,ierr)
      CHKERRA(ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)

      call SNESSetFromOptions(snes,ierr)
      CHKERRA(ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      call FormInitialGuess(x,ierr)
      CHKERRA(ierr)
      call SNESSolve(snes,PETSC_NULL_VEC,x,ierr)
      CHKERRA(ierr)
      call SNESGetIterationNumber(snes,its,ierr)
      CHKERRA(ierr)
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call VecDestroy(x,ierr)
      CHKERRA(ierr)
      call VecDestroy(r,ierr)
      CHKERRA(ierr)
      call SNESDestroy(snes,ierr)
      CHKERRA(ierr)
      call DMDestroy(da,ierr)
      CHKERRA(ierr)
      call PetscFinalize(ierr)
      CHKERRA(ierr)
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
      use ex5fmodule
      implicit none

!  Input/output variables:
      Vec      X
      PetscErrorCode  ierr
!  Declarations for use with local arrays:
      PetscScalar, pointer :: lx_v(:)

      ierr = 0

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

       call VecGetArray(X,lx_v,ierr)
      CHKERRQ(ierr)

!  Compute initial guess over the locally owned part of the grid

      call InitialGuessLocal(lx_v,ierr)
      CHKERRQ(ierr)

!  Restore vector

      call VecRestoreArray(X,lx_v,ierr)
      CHKERRQ(ierr)

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
      subroutine InitialGuessLocal(x,ierr)
      use ex5fmodule
      implicit none

!  Input/output variables:
      PetscScalar    x(xs:xe,ys:ye)
      PetscErrorCode ierr

!  Local variables:
      PetscInt  i,j
      PetscReal temp1,temp,one,hx,hy

!  Set parameters

      ierr   = 0
      one    = 1.0
      hx     = one/((real(mx)-1))
      hy     = one/((real(my)-1))
      temp1  = lambda/(lambda + one)

      do 20 j=ys,ye
         temp = (real(min(j-1,my-j)))*hy
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. mx .or. j .eq. my) then
              x(i,j) = 0.0
            else
              x(i,j) = temp1 * sqrt(min(real(min(i-1,mx-i))*hx,(temp)))
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
!
      subroutine FormFunctionLocal(info,x,f,da,ierr)
      use ex5fmodule
      implicit none

      DM da

!  Input/output variables:
      DMDALocalInfo info
      PetscScalar x(gxs:gxe,gys:gye)
      PetscScalar f(xs:xe,ys:ye)
      PetscErrorCode     ierr

!  Local variables:
      PetscScalar two,one,hx,hy
      PetscScalar hxdhy,hydhx,sc
      PetscScalar u,uxx,uyy
      PetscInt  i,j

      xs     = info%XS+1
      xe     = xs+info%XM-1
      ys     = info%YS+1
      ye     = ys+info%YM-1
      mx     = info%MX
      my     = info%MY

      one    = 1.0
      two    = 2.0
      hx     = one/(real(mx)-1)
      hy     = one/(real(my)-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute function over the locally owned part of the grid

      do 20 j=ys,ye
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. mx .or. j .eq. my) then
               f(i,j) = x(i,j)
            else
               u = x(i,j)
               uxx = hydhx * (two*u - x(i-1,j) - x(i+1,j))
               uyy = hxdhy * (two*u - x(i,j-1) - x(i,j+1))
               f(i,j) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

      call PetscLogFlops(11.0d0*ym*xm,ierr)
      CHKERRQ(ierr)

      end

! ---------------------------------------------------------------------
!
!  FormJacobianLocal - Computes Jacobian matrix, called by
!  the higher level routine FormJacobian().
!
!  Input Parameters:
!  x        - local vector data
!
!  Output Parameters:
!  jac      - Jacobian matrix
!  jac_prec - optionally different matrix used to construct the preconditioner (not used here)
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
!          by calling MatSetValuesLocal()
!    (B) MatSetValues(), using the global ordering
!        - Use DMDAGetGlobalIndices() to extract the local-to-global map
!        - Then apply this map explicitly yourself
!        - Set matrix entries using the global ordering by calling
!          MatSetValues()
!  Option (A) seems cleaner/easier in many cases, and is the procedure
!  used in this example.
!
      subroutine FormJacobianLocal(info,x,A,jac,da,ierr)
      use ex5fmodule
      implicit none

      DM da

!  Input/output variables:
      PetscScalar x(gxs:gxe,gys:gye)
      Mat         A,jac
      PetscErrorCode  ierr
      DMDALocalInfo info

!  Local variables:
      PetscInt  row,col(5),i,j,i1,i5
      PetscScalar two,one,hx,hy,v(5)
      PetscScalar hxdhy,hydhx,sc

!  Set parameters

      i1     = 1
      i5     = 5
      one    = 1.0
      two    = 2.0
      hx     = one/(real(mx)-1)
      hy     = one/(real(my)-1)
      sc     = hx*hy
      hxdhy  = hx/hy
      hydhx  = hy/hx
! -Wmaybe-uninitialized
      v      = 0.0
      col    = 0

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
            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. mx .or. j .eq. my) then
!       Some f90 compilers need 4th arg to be of same type in both calls
               col(1) = row
               v(1)   = one
               call MatSetValuesLocal(jac,i1,[row],i1,[col],[v],INSERT_VALUES,ierr)
               CHKERRQ(ierr)
!           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy) - sc*lambda*exp(x(i,j))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - gxm
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + gxm
               call MatSetValuesLocal(jac,i1,[row],i5,[col],[v], INSERT_VALUES,ierr)
               CHKERRQ(ierr)
            endif
 10      continue
 20   continue
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      CHKERRQ(ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      CHKERRQ(ierr)
      if (A .ne. jac) then
         call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
         CHKERRQ(ierr)
         call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
         CHKERRQ(ierr)
      endif
      end

!
!     Simple convergence test based on the infinity norm of the residual being small
!
      subroutine MySNESConverged(snes,it,xnorm,snorm,fnorm,reason,dummy,ierr)
      use ex5fmodule
      implicit none

      SNES snes
      PetscInt it,dummy
      PetscReal xnorm,snorm,fnorm,nrm
      SNESConvergedReason reason
      Vec f
      PetscErrorCode ierr

      call SNESGetFunction(snes,f,PETSC_NULL_FUNCTION,dummy,ierr)
      CHKERRQ(ierr)
      call VecNorm(f,NORM_INFINITY,nrm,ierr)
      CHKERRQ(ierr)
      if (nrm .le. 1.e-5) reason = SNES_CONVERGED_FNORM_ABS

      end

!/*TEST
!
!   build:
!      requires: !complex !single
!
!   test:
!      nsize: 4
!      args: -snes_mf -pc_type none -da_processors_x 4 -da_processors_y 1 -snes_monitor_short \
!            -ksp_gmres_cgs_refinement_type refine_always
!
!   test:
!      suffix: 2
!      nsize: 4
!      args: -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always
!
!   test:
!      suffix: 3
!      nsize: 3
!      args: -snes_fd -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always
!
!   test:
!      suffix: 6
!      nsize: 1
!      args: -snes_monitor_short -my_snes_convergence
!
!TEST*/

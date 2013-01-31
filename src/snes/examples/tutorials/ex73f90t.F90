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
      module petsc_kkt_solver_module
#include <finclude/petscdmdef.h>
      use petscdmdef
      type petsc_kkt_solver
        type(DM) da
!     temp A block stuff 
        PetscInt xs,xe,xm,gxs,gxe,gxm
        PetscInt ys,ye,ym,gys,gye,gym
        PetscInt mx,my
        PetscMPIInt rank
        double precision lambda
!     Mats
        type(Mat) Amat,Bmat,CMat,Dmat
      end type petsc_kkt_solver

      end module petsc_kkt_solver_module

      module petsc_kkt_solver_moduleinterfaces
        use petsc_kkt_solver_module

      Interface SNESSetApplicationContext
        Subroutine SNESSetApplicationContext(snesIn,ctx,ierr)
#include <finclude/petscsnesdef.h>
        use petscsnes
        use petsc_kkt_solver_module
          type(SNES)    snesIn
          type(petsc_kkt_solver) ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESSetApplicationContext

      Interface SNESGetApplicationContext
        Subroutine SNESGetApplicationContext(snesIn,ctx,ierr)
#include <finclude/petscsnesdef.h>
        use petscsnes
        use petsc_kkt_solver_module
          type(SNES)     snesIn
          type(petsc_kkt_solver), pointer :: ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESGetApplicationContext
      end module petsc_kkt_solver_moduleinterfaces

      program main
#include <finclude/petscdmdef.h>
#include <finclude/petscsnesdef.h>
      use petscdm
      use petscdmda
      use petscsnes
      use petsc_kkt_solver_module
      use petsc_kkt_solver_moduleinterfaces
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
!
      type(SNES)       mysnes
      type(Vec)        x,r,x2,x1,x2loc,vecArray(2)
      type(Mat)        Amat,Bmat,Cmat,Dmat,KKTMat,matArray(4)
      type(DM)         daphi,dalam
      PetscErrorCode   ierr
      PetscInt         its,N1,N2,ii,jj,idx
      PetscBool        flg
      PetscInt         ione,nfour,itwo
      double precision lambda_max,lambda_min
      type(petsc_kkt_solver)  solver
      PetscScalar      bval,cval,hx,hy,one

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
      external FormInitialGuess,FormJacobian,FormFunction

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Initialize program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,solver%rank,ierr)

!  Initialize problem parameters
      lambda_max  = 6.81
      lambda_min  = 0.0
      solver%lambda = 6.0
      ione = 1
      nfour = -4
      itwo = 2
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-par',             &
     &     solver%lambda,flg,ierr)
      if (solver%lambda .ge. lambda_max .or. solver%lambda .lt. lambda_min) &
     &     then
         if (solver%rank .eq. 0) write(6,*) 'Lambda is out of range'
         SETERRQ(PETSC_COMM_SELF,1,' ',ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DMDA) to manage parallel grid and vectors

! This really needs only the star-type stencil, but we use the box
! stencil temporarily.
      call DMDACreate2d(PETSC_COMM_WORLD,                               &
     &     DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,                      &
     &     DMDA_STENCIL_BOX,nfour,nfour,PETSC_DECIDE,PETSC_DECIDE,          &
     &     ione,ione,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,daphi,ierr)
      call DMDAGetInfo(daphi,PETSC_NULL_INTEGER,solver%mx,solver%my,        &
     &               PETSC_NULL_INTEGER,                                &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,             &
     &               PETSC_NULL_INTEGER,ierr)

      N1 = solver%my*solver%mx
      print *, 'n=',N1, ', M=',solver%my, ', N=',solver%mx

!
!   Visualize the distribution of the array across the processors
!
!     call DMView(daphi,PETSC_VIEWER_DRAW_WORLD,ierr)

!  Get local grid boundaries (for 2-dimensional DMDA)
      call DMDAGetCorners(daphi,solver%xs,solver%ys,PETSC_NULL_INTEGER,     &
     &     solver%xm,solver%ym,PETSC_NULL_INTEGER,ierr)
      call DMDAGetGhostCorners(daphi,solver%gxs,solver%gys,                 &
     &     PETSC_NULL_INTEGER,solver%gxm,solver%gym,                        &
     &     PETSC_NULL_INTEGER,ierr)

!  Here we shift the starting indices up by one so that we can easily
!  use the Fortran convention of 1-based indices (rather 0-based indices).
      solver%xs  = solver%xs+1
      solver%ys  = solver%ys+1
      solver%gxs = solver%gxs+1
      solver%gys = solver%gys+1

      solver%ye  = solver%ys+solver%ym-1
      solver%xe  = solver%xs+solver%xm-1
      solver%gye = solver%gys+solver%gym-1
      solver%gxe = solver%gxs+solver%gxm-1

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix data structure; set Jacobian evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call DMCreateMatrix(daphi,MATAIJ,Amat,ierr)
      call DMSetOptionsPrefix(daphi,'phi_',ierr)
      call DMSetFromOptions(daphi,ierr)

      call VecCreate(PETSC_COMM_WORLD,x1,ierr)
      call VecSetSizes(x1,PETSC_DECIDE,N1,ierr)
      call VecSetFromOptions(x1,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create B, C, & D matrices
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call MatCreate(PETSC_COMM_WORLD,Cmat,ierr)
      call MatSetSizes(Cmat,PETSC_DECIDE,PETSC_DECIDE,solver%my,N1,ierr)
      call MatSetFromOptions(Cmat,ierr)
      call MatSetUp(Cmat,ierr)
!      create data for C and B
      call MatCreate(PETSC_COMM_WORLD,Bmat,ierr)
      call MatSetSizes(Bmat,PETSC_DECIDE,PETSC_DECIDE,N1,solver%my,ierr)
      call MatSetFromOptions(Bmat,ierr)
      call MatSetUp(Bmat,ierr)
!     create data for D
      call MatCreate(PETSC_COMM_WORLD,Dmat,ierr)
      call MatSetSizes(Dmat,PETSC_DECIDE,PETSC_DECIDE,solver%my,solver%my,ierr)
      call MatSetFromOptions(Dmat,ierr)
      call MatSetUp(Dmat,ierr)

      call MatSetOption(Bmat,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE,ierr)
      call MatSetOption(Cmat,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE,ierr)
      call MatSetOption(Dmat,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set fake B and C
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      one    = 1.0
      hx     = one/dble(solver%mx-1)
      hy     = one/dble(solver%my-1)
      bval = -hx
      cval = -hx*hy
      idx = 0
      do 20 jj=0,solver%my-1
         do 10 ii=0,solver%mx-1
            call MatSetValues(Cmat,ione,jj,ione,idx,cval,INSERT_VALUES,ierr)
            call MatSetValues(Bmat,ione,idx,ione,jj,bval,INSERT_VALUES,ierr)
            idx = idx + 1
 10      continue
 20   continue
      call MatAssemblyBegin(Bmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Bmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyBegin(Cmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Cmat,MAT_FINAL_ASSEMBLY,ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set D (indentity)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      do 30 jj=0,solver%my-1
         call MatSetValues(Dmat,ione,jj,ione,jj,one,INSERT_VALUES,ierr)
 30   continue 
      call MatAssemblyBegin(Dmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Dmat,MAT_FINAL_ASSEMBLY,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  DM for lambda (dalam) : temp driver for A block, setup A block solver data
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call VecCreate(PETSC_COMM_WORLD,x2,ierr)
      call VecSetSizes(x2,PETSC_DECIDE,solver%my,ierr)
      call VecSetFromOptions(x2,ierr)

      call DMShellCreate(PETSC_COMM_WORLD,dalam,ierr)
      call DMShellSetGlobalVector(dalam,x2,ierr)
      call DMShellSetMatrix(dalam,Dmat,ierr)

      call VecCreate(PETSC_COMM_SELF,x2loc,ierr)
      call VecSetSizes(x2loc,PETSC_DECIDE,solver%my,ierr)
      call VecSetFromOptions(x2loc,ierr)
      call DMShellSetLocalVector(dalam,x2loc,ierr)

      call DMSetOptionsPrefix(dalam,'lambda_',ierr)
      call DMSetFromOptions(dalam,ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create field split DA
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call DMCompositeCreate(PETSC_COMM_WORLD,solver%da,ierr)
      call DMSetOptionsPrefix(solver%da,'flux_',ierr)
      call DMCompositeAddDM(solver%da,daphi,ierr)
      call DMCompositeAddDM(solver%da,dalam,ierr)
      call PetscObjectSetName(daphi,"phi",ierr) 
      call PetscObjectSetName(dalam,"lambda",ierr)
      call DMSetFromOptions(solver%da,ierr)
      call DMSetUp(solver%da,ierr)
      
!     cache matrices
      solver%Amat = Amat
      solver%Bmat = Bmat
      solver%Cmat = Cmat
      solver%Dmat = Dmat

      matArray(1) = Amat
      matArray(2) = Bmat
      matArray(3) = Cmat
      matArray(4) = Dmat
      call MatCreateNest(PETSC_COMM_WORLD,itwo,PETSC_NULL_OBJECT,itwo, &
     &     PETSC_NULL_OBJECT, matArray, KKTmat, ierr )

!  Extract global and local vectors from DMDA; then duplicate for remaining
!     vectors that are the same types
      vecArray(1) = x1
      vecArray(2) = x2
      call VecCreateNest(PETSC_COMM_WORLD,itwo,PETSC_NULL_OBJECT, &
     &     vecArray,x,ierr)

      call VecDuplicate(x,r,ierr)

      call SNESCreate(PETSC_COMM_WORLD,mysnes,ierr)

      call SNESSetDM(mysnes,solver%da,ierr)

      call SNESSetApplicationContext(mysnes,solver,ierr)

      call SNESSetDM(mysnes,solver%da,ierr)

!  Set function evaluation routine and vector
      call SNESSetFunction(mysnes,r,FormFunction,solver,ierr)

      call SNESSetJacobian(mysnes,KKTmat,KKTmat,FormJacobian,solver,&
     &     ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
      call SNESSetFromOptions(mysnes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      call FormInitialGuess(mysnes,x,ierr)
      call SNESSolve(mysnes,PETSC_NULL_OBJECT,x,ierr)
      call SNESGetIterationNumber(mysnes,its,ierr)
      if (solver%rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call MatDestroy(KKTmat,ierr)
      call MatDestroy(Amat,ierr)
      call MatDestroy(Dmat,ierr)
      call MatDestroy(Bmat,ierr)
      call MatDestroy(Cmat,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(x2,ierr)
      call VecDestroy(x1,ierr)
      call VecDestroy(x2loc,ierr)
      call VecDestroy(r,ierr)
      call SNESDestroy(mysnes,ierr)
      call DMDestroy(solver%da,ierr)
      call DMDestroy(daphi,ierr)
      call DMDestroy(dalam,ierr)

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
      subroutine FormInitialGuess(mysnes,Xnest,ierr)
#include <finclude/petscsnesdef.h>
      use petscsnes
      use petsc_kkt_solver_module
      use petsc_kkt_solver_moduleinterfaces
      implicit none
!  Input/output variables:
      type(SNES)     mysnes
      type(petsc_kkt_solver), pointer:: psolver
      type(Vec)      Xnest,X_1,lam
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:)
      type(Vec)      localX
      PetscInt       izero
      type(DM)       daphi,dmarray(2)

      izero = 0
      ierr = 0
      call SNESGetApplicationContext(mysnes,psolver,ierr)
!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray90() returns a pointer to
!      the data array. Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArrayF90() when you no longer need access to
!      the array.
!    - Note that the interface to VecGetArrayF90() differs from VecGetArray(),
!      and is useable from Fortran-90 Only.

      call VecNestGetSubVec(Xnest,izero,X_1,ierr)
      call DMCompositeGetEntriesArray(psolver%da,dmarray,ierr)
      daphi = dmarray(1)

      call DMGetLocalVector(daphi,localX,ierr)
      call VecGetArrayF90(localX,lx_v,ierr)

!  Compute initial guess over the locally owned part of the grid
      call InitialGuessLocal(psolver,lx_v,ierr)

!  Restore vector
      call VecRestoreArrayF90(localX,lx_v,ierr)

!  Insert values into global vector
      call DMLocalToGlobalBegin(daphi,localX,INSERT_VALUES,X_1,ierr)
      call DMLocalToGlobalEnd(daphi,localX,INSERT_VALUES,X_1,ierr)
      call DMRestoreLocalVector(daphi,localX,ierr)

!     zero out lambda
      daphi = dmarray(2)
      call DMGetLocalVector(daphi,localX,ierr)
      call VecZeroEntries(localX,ierr)
      call DMRestoreLocalVector(daphi,localX,ierr)

      return
      end subroutine FormInitialGuess

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
      subroutine InitialGuessLocal(solver,x,ierr)
#include <finclude/petscsysdef.h>
      use petscsys
      use petsc_kkt_solver_module
      implicit none
!  Input/output variables:
      type (petsc_kkt_solver)         solver
      PetscScalar  x(solver%gxs:solver%gxe,                                 &
     &              solver%gys:solver%gye)
      PetscErrorCode ierr

!  Local variables:
      PetscInt  i,j
      PetscScalar   temp1,temp,hx,hy
      PetscScalar   one

!  Set parameters

      ierr   = 0
      one    = 1.0
      hx     = one/(dble(solver%mx-1))
      hy     = one/(dble(solver%my-1))
      temp1  = solver%lambda/(solver%lambda + one) + one

      do 20 j=solver%ys,solver%ye
         temp = dble(min(j-1,solver%my-j))*hy
         do 10 i=solver%xs,solver%xe
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. solver%mx .or. j .eq. solver%my) then
              x(i,j) = 0.0
            else
              x(i,j) = temp1 *                                          &
     &          sqrt(min(dble(min(i-1,solver%mx-i)*hx),dble(temp)))
            endif
 10      continue
 20   continue

      return
      end subroutine InitialGuessLocal


! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  dummy     - the SNES context
!  x         - input vector
!  solver    - solver data
!
!  Output Parameters:
!  jac      - Jacobian matrix
!  jac_prec - optionally different preconditioning matrix (not used here)
!  flag     - flag indicating matrix structure
!
      subroutine FormJacobian(dummy,X,jac,jac_prec,flag,solver,ierr)
#include <finclude/petscsnesdef.h>
      use petscsnes
      use petsc_kkt_solver_module
      implicit none
!  Input/output variables:
      type(SNES)     dummy
      type(Vec)      X
      type(Mat)      jac,jac_prec
      MatStructure   flag
      type(petsc_kkt_solver)  solver
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:)
      type(Vec)      X_1
      type(Mat)      Amat

      if( jac .ne. jac_prec) stop 'jac != jac_prec'

      call VecNestGetSubVec(X,0,X_1,ierr)

!     Get a pointer to vector data
      call VecGetArrayF90(X_1,lx_v,ierr)

!     Compute entries for the locally owned part of the Jacobian preconditioner.
      call MatNestGetSubMat( jac, 0, 0, Amat, ierr ) ! this will not work with any Mat

      call FormJacobianLocal(lx_v,Amat,solver,ierr)

      call VecRestoreArrayF90(X_1,lx_v,ierr)

      ! the rest of the matrix is not touched
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)

!     Set flag to indicate that the Jacobian matrix retains an identical
!     nonzero structure throughout all nonlinear iterations 
      
      flag = SAME_NONZERO_PATTERN

!     Tell the matrix we will never add a new nonzero location to the
!     matrix. If we do it will generate an error.
      call MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,      &
     &     ierr)

      return
      end subroutine FormJacobian
      
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
      subroutine FormJacobianLocal(x,jac_prec,solver,ierr)
#include <finclude/petscmatdef.h>
      use petscmat
      use petsc_kkt_solver_module
      implicit none
!  Input/output variables:
      type (petsc_kkt_solver) solver
      PetscScalar    x(solver%gxs:solver%gxe,                                      &
     &               solver%gys:solver%gye)
      type(Mat)      jac_prec
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
      hx     = one/dble(solver%mx-1)
      hy     = one/dble(solver%my-1)
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

      do 20 j=solver%ys,solver%ye
         row = (j - solver%gys)*solver%gxm + solver%xs - solver%gxs - 1
         do 10 i=solver%xs,solver%xe
            row = row + 1
!           boundary points
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. solver%mx .or. j .eq. solver%my) then
               col(1) = row
               v(1)   = one
               call MatSetValuesLocal(jac_prec,ione,row,ione,col,v,          &
     &                           INSERT_VALUES,ierr)
!           interior grid points
            else
               v(1) = -hxdhy
               v(2) = -hydhx
               v(3) = two*(hydhx + hxdhy)                               &
     &                  - sc*solver%lambda*exp(x(i,j))
               v(4) = -hydhx
               v(5) = -hxdhy
               col(1) = row - solver%gxm
               col(2) = row - 1
               col(3) = row
               col(4) = row + 1
               col(5) = row + solver%gxm
               call MatSetValuesLocal(jac_prec,ione,row,ifive,col,v,         &
     &                                INSERT_VALUES,ierr)
            endif
 10      continue
 20   continue
      return
      end subroutine FormJacobianLocal


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
      subroutine FormFunction(snesIn,X,F,solver,ierr)
#include <finclude/petscsnesdef.h>
      use petscsnes
      use petsc_kkt_solver_module
      implicit none
!  Input/output variables:
      type(SNES)     snesIn
      type(Vec)      X,F
      PetscErrorCode ierr
      type (petsc_kkt_solver) solver

!  Declarations for use with local arrays:
      PetscScalar,pointer :: lx_v(:),lf_v(:)
      type(Vec)              localX
      type(Vec)              X_1,X_2,F_1,F_2
      type(DM)               daphi,dmarray(2)
      PetscInt               izero,ione

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can
!  be done while messages are in transition.
      
      izero = 0
      ione = 1
      call VecNestGetSubVec(X,izero,X_1,ierr)
      call VecNestGetSubVec(X,ione,X_2,ierr)
      call VecNestGetSubVec(F,izero,F_1,ierr)
      call VecNestGetSubVec(F,ione,F_2,ierr)

      call DMCompositeGetEntriesArray(solver%da,dmarray,ierr)
      daphi = dmarray(1)

      call DMGetLocalVector(daphi,localX,ierr)
      call DMGlobalToLocalBegin(daphi,X_1,INSERT_VALUES,localX,ierr)
      call DMGlobalToLocalEnd(daphi,X_1,INSERT_VALUES,localX,ierr)

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray90() returns a pointer to
!      the data array. Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArrayF90() when you no longer need access to
!      the array.
!    - Note that the interface to VecGetArrayF90() differs from VecGetArray(),
!      and is useable from Fortran-90 Only.

      call VecGetArrayF90(localX,lx_v,ierr)
      call VecGetArrayF90(F_1,lf_v,ierr)

!  Compute function over the locally owned part of the grid
      call FormFunctionLocal(lx_v,lf_v,solver,ierr)

!  Restore vectors
      call VecRestoreArrayF90(localX,lx_v,ierr)
      call VecRestoreArrayF90(F_1,lf_v,ierr)

!  Insert values into global vector

      call DMRestoreLocalVector(daphi,localX,ierr)
      call PetscLogFlops(11.0d0*solver%ym*solver%xm,ierr)

!      call VecView(X,PETSC_VIEWER_STDOUT_WORLD,ierr)
!      call VecView(F,PETSC_VIEWER_STDOUT_WORLD,ierr)

!     do rest of operator (linear)
      call MatMult(    solver%Cmat, X_1,      F_2, ierr)
      call MatMultAdd( solver%Bmat, X_2, F_1, F_1, ierr)
      call MatMultAdd( solver%Dmat, X_2, F_2, F_2, ierr)

      return
      end subroutine formfunction


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
      subroutine FormFunctionLocal(x,f,solver,ierr)
#include <finclude/petscsysdef.h>
      use petscsys
      use petsc_kkt_solver_module
      implicit none
!  Input/output variables:
      type (petsc_kkt_solver) solver
      PetscScalar  x(solver%gxs:solver%gxe,                                         &
     &              solver%gys:solver%gye)
      PetscScalar  f(solver%xs:solver%xe,                                           &
     &              solver%ys:solver%ye)
      PetscErrorCode ierr

!  Local variables:
      PetscScalar two,one,hx,hy,hxdhy,hydhx,sc
      PetscScalar u,uxx,uyy
      PetscInt  i,j

      one    = 1.0
      two    = 2.0
      hx     = one/dble(solver%mx-1)
      hy     = one/dble(solver%my-1)
      sc     = hx*hy*solver%lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute function over the locally owned part of the grid

      do 20 j=solver%ys,solver%ye
         do 10 i=solver%xs,solver%xe
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. solver%mx .or. j .eq. solver%my) then
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
      ierr = 0
      return
      end subroutine FormFunctionLocal

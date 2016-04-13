!
!  Description: Solves a nonlinear system in parallel with SNES.
!  We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular
!  domain, using distributed arrays (DMDAs) to partition the parallel grid.
!  The command line options include:
!    -par <parameter>, where <parameter> indicates the nonlinearity of the problem
!       problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)
!
!  This system (A) is augmented with constraints:
!
!    A -B   *  phi  =  rho
!   -C  I      lam  = 0
!
!  where I is the identity, A is the "normal" Poisson equation, B is the "distributor" of the
!  total flux (the first block equation is the flux surface averaging equation).  The second
!  equation  lambda = C * x enforces the surface flux auxiliary equation.  B and C have all
!  positive entries, areas in C and fraction of area in B.
!
!/*T
!  Concepts: SNES^parallel Bratu example
!  Concepts: MatNest
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
!  --------------------------------------------------------------------------
!  The following define must be used before including any PETSc include files
!  into a module or interface. This is because they can't handle declarations
!  in them
!
      module petsc_kkt_solver
#include <petsc/finclude/petscdmdef.h>
      use petscdmdef
      type petsc_kkt_solver_type
        DM::da
!     temp A block stuff
        PetscInt mx,my
        PetscMPIInt rank
        PetscReal lambda
!     Mats
        Mat::Amat,AmatLin,Bmat,CMat,Dmat
        IS::isPhi,isLambda
      end type petsc_kkt_solver_type

      end module petsc_kkt_solver

      module petsc_kkt_solver_interfaces
        use petsc_kkt_solver

      Interface SNESSetApplicationContext
        Subroutine SNESSetApplicationContext(snesIn,ctx,ierr)
#include <petsc/finclude/petscsnesdef.h>
        use petsc_kkt_solver
          SNES::    snesIn
          type(petsc_kkt_solver_type) ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESSetApplicationContext

      Interface SNESGetApplicationContext
        Subroutine SNESGetApplicationContext(snesIn,ctx,ierr)
#include <petsc/finclude/petscsnesdef.h>
        use petsc_kkt_solver
          SNES::     snesIn
          type(petsc_kkt_solver_type), pointer :: ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESGetApplicationContext
      end module petsc_kkt_solver_interfaces

      program main
#include <petsc/finclude/petscdmdef.h>
#include <petsc/finclude/petscsnesdef.h>
      use petscdm
      use petscdmda
      use petscsnes
      use petsc_kkt_solver
      use petsc_kkt_solver_interfaces
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
      SNES::       mysnes
      Vec::        x,r,x2,x1,x1loc,x2loc
      Mat::       Amat,Bmat,Cmat,Dmat,KKTMat,matArray(4)
!      Mat::       tmat
      DM::       daphi,dalam
      IS::        isglobal(2)
      PetscErrorCode   ierr
      PetscInt         its,N1,N2,i,j,irow,row(1)
      PetscInt         col(1),low,high,lamlow,lamhigh
      PetscBool        flg
      PetscInt         ione,nfour,itwo,nloc,nloclam
      PetscReal lambda_max,lambda_min
      type(petsc_kkt_solver_type)  solver
      PetscScalar      bval(1),cval(1),one

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
      call PetscOptionsGetReal(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,     &
     &                         '-par', solver%lambda,flg,ierr)
      if (solver%lambda .ge. lambda_max .or. solver%lambda .lt. lambda_min) then
         if (solver%rank .eq. 0) write(6,*) 'Lambda is out of range'
         SETERRQ(PETSC_COMM_SELF,1,' ',ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!     just get size
      call DMDACreate2d(PETSC_COMM_WORLD,                               &
     &      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &
     &      DMDA_STENCIL_BOX,nfour,nfour,PETSC_DECIDE,PETSC_DECIDE,       &
     &      ione,ione,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,daphi,ierr)
      call DMDAGetInfo(daphi,PETSC_NULL_INTEGER,solver%mx,solver%my,     &
     &      PETSC_NULL_INTEGER,                                            &
     &      PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                         &
     &      PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                        &
     &      PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                         &
     &      PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                         &
     &      PETSC_NULL_INTEGER,ierr)
      N1 = solver%my*solver%mx
      N2 = solver%my
      flg = .false.
      call PetscOptionsGetBool(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,   &
     &                          '-no_constraints',flg,flg,ierr)
      if (flg) then
         N2 = 0
      endif

      call DMDestroy(daphi,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix data structure; set Jacobian evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call DMShellCreate(PETSC_COMM_WORLD,daphi,ierr)
      call DMSetOptionsPrefix(daphi,'phi_',ierr)
      call DMSetFromOptions(daphi,ierr)

      call VecCreate(PETSC_COMM_WORLD,x1,ierr)
      call VecSetSizes(x1,PETSC_DECIDE,N1,ierr)
      call VecSetFromOptions(x1,ierr)

      call VecGetOwnershipRange(x1,low,high,ierr)
      nloc = high - low

      call MatCreate(PETSC_COMM_WORLD,Amat,ierr)
      call MatSetSizes(Amat,PETSC_DECIDE,PETSC_DECIDE,N1,N1,ierr)
      call MatSetUp(Amat,ierr)

      call MatCreate(PETSC_COMM_WORLD,solver%AmatLin,ierr)
      call MatSetSizes(solver%AmatLin,PETSC_DECIDE,PETSC_DECIDE,N1,N1,ierr)
      call MatSetUp(solver%AmatLin,ierr)

      call FormJacobianLocal(x1,solver%AmatLin,solver,.false.,ierr)
      call MatAssemblyBegin(solver%AmatLin,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(solver%AmatLin,MAT_FINAL_ASSEMBLY,ierr)

      call DMShellSetGlobalVector(daphi,x1,ierr)
      call DMShellSetMatrix(daphi,Amat,ierr)

      call VecCreate(PETSC_COMM_SELF,x1loc,ierr)
      call VecSetSizes(x1loc,nloc,nloc,ierr)
      call VecSetFromOptions(x1loc,ierr)
      call DMShellSetLocalVector(daphi,x1loc,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create B, C, & D matrices
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call MatCreate(PETSC_COMM_WORLD,Cmat,ierr)
      call MatSetSizes(Cmat,PETSC_DECIDE,PETSC_DECIDE,N2,N1,ierr)
      call MatSetUp(Cmat,ierr)
!      create data for C and B
      call MatCreate(PETSC_COMM_WORLD,Bmat,ierr)
      call MatSetSizes(Bmat,PETSC_DECIDE,PETSC_DECIDE,N1,N2,ierr)
      call MatSetUp(Bmat,ierr)
!     create data for D
      call MatCreate(PETSC_COMM_WORLD,Dmat,ierr)
      call MatSetSizes(Dmat,PETSC_DECIDE,PETSC_DECIDE,N2,N2,ierr)
      call MatSetUp(Dmat,ierr)

      call VecCreate(PETSC_COMM_WORLD,x2,ierr)
      call VecSetSizes(x2,PETSC_DECIDE,N2,ierr)
      call VecSetFromOptions(x2,ierr)

      call VecGetOwnershipRange(x2,lamlow,lamhigh,ierr)
      nloclam = lamhigh-lamlow

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set fake B and C
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      one    = 1.0
      if( N2 .gt. 0 ) then
         bval(1) = -one/(solver%mx-2)
!     cval = -one/(solver%my*solver%mx)
         cval(1) = -one
         do 20 irow=low,high-1
            j = irow/solver%mx   ! row in domain
            i = mod(irow,solver%mx)
            row(1) = irow
            col(1) = j
            if (i .eq. 0 .or. j .eq. 0 .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1 ) then
               !     no op
            else
               call MatSetValues(Bmat,ione,row,ione,col,bval,INSERT_VALUES,ierr)
            endif
            row(1) = j
            call MatSetValues(Cmat,ione,row,ione,row,cval,INSERT_VALUES,ierr)
 20   continue
      endif
      call MatAssemblyBegin(Bmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Bmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyBegin(Cmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Cmat,MAT_FINAL_ASSEMBLY,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set D (indentity)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      do 30 j=lamlow,lamhigh-1
         row(1) = j
         cval(1) = one
         call MatSetValues(Dmat,ione,row,ione,row,cval,INSERT_VALUES,ierr)
 30   continue
      call MatAssemblyBegin(Dmat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Dmat,MAT_FINAL_ASSEMBLY,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  DM for lambda (dalam) : temp driver for A block, setup A block solver data
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call DMShellCreate(PETSC_COMM_WORLD,dalam,ierr)
      call DMShellSetGlobalVector(dalam,x2,ierr)
      call DMShellSetMatrix(dalam,Dmat,ierr)

      call VecCreate(PETSC_COMM_SELF,x2loc,ierr)
      call VecSetSizes(x2loc,nloclam,nloclam,ierr)
      call VecSetFromOptions(x2loc,ierr)
      call DMShellSetLocalVector(dalam,x2loc,ierr)

      call DMSetOptionsPrefix(dalam,'lambda_',ierr)
      call DMSetFromOptions(dalam,ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create field split DA
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call DMCompositeCreate(PETSC_COMM_WORLD,solver%da,ierr)
      call DMCompositeAddDM(solver%da,daphi,ierr)
      call DMCompositeAddDM(solver%da,dalam,ierr)
      call DMSetFromOptions(solver%da,ierr)
      call DMSetUp(solver%da,ierr)
      call DMCompositeGetGlobalISs(solver%da,isglobal,ierr)
      solver%isPhi = isglobal(1)
      solver%isLambda = isglobal(2)

!     cache matrices
      solver%Amat = Amat
      solver%Bmat = Bmat
      solver%Cmat = Cmat
      solver%Dmat = Dmat

      matArray(1) = Amat
      matArray(2) = Bmat
      matArray(3) = Cmat
      matArray(4) = Dmat

      call MatCreateNest(PETSC_COMM_WORLD,itwo,isglobal,itwo,isglobal,matArray,KKTmat,ierr)
      call MatSetFromOptions(KKTmat,ierr)

!  Extract global and local vectors from DMDA; then duplicate for remaining
!     vectors that are the same types
      call MatCreateVecs(KKTmat,x,PETSC_NULL_OBJECT,ierr)
      call VecDuplicate(x,r,ierr)

      call SNESCreate(PETSC_COMM_WORLD,mysnes,ierr)

      call SNESSetDM(mysnes,solver%da,ierr)

      call SNESSetApplicationContext(mysnes,solver,ierr)

      call SNESSetDM(mysnes,solver%da,ierr)

!  Set function evaluation routine and vector
      call SNESSetFunction(mysnes,r,FormFunction,solver,ierr)

      call SNESSetJacobian(mysnes,KKTmat,KKTmat,FormJacobian,solver,ierr)

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

      ! convert test
!!&      print *,'[',solver%rank,'] ',N1,' primary global rows, ',N2,' global constraints, ',nloclam,' local constraints'
!!&      call MatConvert(KKTmat,MATAIJ,MAT_INITIAL_MATRIX,tmat,ierr);CHKERRQ(ierr)
!!&      if (ierr==0) then
!!&         call MatDestroy(KKTmat, ierr);CHKERRQ(ierr)
!!&         KKTmat = tmat
!!&         print *,'MatConvert Nest-->AIJ worked!!!!!'
!!&      end if

      call SNESGetIterationNumber(mysnes,its,ierr)
      if (solver%rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)
!      call VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr)
!      call MatView(Amat,PETSC_VIEWER_STDOUT_WORLD,ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call MatDestroy(KKTmat,ierr)
      call MatDestroy(Amat,ierr)
      call MatDestroy(Dmat,ierr)
      call MatDestroy(Bmat,ierr)
      call MatDestroy(Cmat,ierr)
      call MatDestroy(solver%AmatLin,ierr)
      call ISDestroy(solver%isPhi,ierr)
      call ISDestroy(solver%isLambda,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(x2,ierr)
      call VecDestroy(x1,ierr)
      call VecDestroy(x1loc,ierr)
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
#include <petsc/finclude/petscsnesdef.h>
      use petscsnes
      use petsc_kkt_solver
      use petsc_kkt_solver_interfaces
      implicit none
!  Input/output variables:
      SNES::     mysnes
      Vec::      Xnest
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      type(petsc_kkt_solver_type), pointer:: solver
      Vec::      Xsub(2)
      PetscInt::  izero,ione,itwo

      izero = 0
      ione = 1
      itwo = 2
      ierr = 0
      call SNESGetApplicationContext(mysnes,solver,ierr)
      call DMCompositeGetAccessArray(solver%da,Xnest,itwo,PETSC_NULL_INTEGER,Xsub,ierr)

      call InitialGuessLocal(solver,Xsub(1),ierr)
      call VecAssemblyBegin(Xsub(1),ierr)
      call VecAssemblyEnd(Xsub(1),ierr)

!     zero out lambda
      call VecZeroEntries(Xsub(2),ierr)
      call DMCompositeRestoreAccessArray(solver%da,Xnest,itwo,PETSC_NULL_INTEGER,Xsub,ierr)

      return
      end subroutine FormInitialGuess

! ---------------------------------------------------------------------
!
!  InitialGuessLocal - Computes initial approximation, called by
!  the higher level routine FormInitialGuess().
!
!  Input Parameter:
!  X1 - local vector data
!
!  Output Parameters:
!  x - local vector data
!  ierr - error code
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
      subroutine InitialGuessLocal(solver,X1,ierr)
#include <petsc/finclude/petscsysdef.h>
      use petscsys
      use petsc_kkt_solver
      implicit none
!  Input/output variables:
      type (petsc_kkt_solver_type)         solver
      Vec::      X1
      PetscErrorCode ierr

!  Local variables:
      PetscInt      row,i,j,ione,low,high
      PetscReal   temp1,temp,hx,hy,v
      PetscReal   one

!  Set parameters
      ione = 1
      ierr   = 0
      one    = 1.0
      hx     = one/(solver%mx-1)
      hy     = one/(solver%my-1)
      temp1  = solver%lambda/(solver%lambda + one) + one

      call VecGetOwnershipRange(X1,low,high,ierr)

      do 20 row=low,high-1
         j = row/solver%mx
         i = mod(row,solver%mx)
         temp = min(j,solver%my-j+1)*hy
         if (i .eq. 0 .or. j .eq. 0  .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1 ) then
            v = 0.0
         else
            v = temp1 * sqrt(min(min(i,solver%mx-i+1)*hx,temp))
         endif
         call VecSetValues(X1,ione,row,v,INSERT_VALUES,ierr)
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
      subroutine FormJacobian(dummy,X,jac,jac_prec,solver,ierr)
#include <petsc/finclude/petscsnesdef.h>
      use petscsnes
      use petsc_kkt_solver
      implicit none
!  Input/output variables:
      SNES::     dummy
      Vec::      X
     Mat::     jac,jac_prec
      type(petsc_kkt_solver_type)  solver
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      Vec::      Xsub(1)
     Mat::     Amat
      PetscInt       izero,ione

      izero = 0
      ione = 1

      call DMCompositeGetAccessArray(solver%da,X,ione,PETSC_NULL_INTEGER,Xsub,ierr)

!     Compute entries for the locally owned part of the Jacobian preconditioner.
      call MatGetSubMatrix(jac_prec,solver%isPhi,solver%isPhi,MAT_INITIAL_MATRIX,Amat,ierr)

      call FormJacobianLocal(Xsub(1),Amat,solver,.true.,ierr)
      call MatDestroy(Amat,ierr) ! discard our reference
      call DMCompositeRestoreAccessArray(solver%da,X,ione,PETSC_NULL_INTEGER,Xsub,ierr)

      ! the rest of the matrix is not touched
      call MatAssemblyBegin(jac_prec,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac_prec,MAT_FINAL_ASSEMBLY,ierr)
      if (jac .ne. jac_prec) then
         call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
         call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      end if

!     Tell the matrix we will never add a new nonzero location to the
!     matrix. If we do it will generate an error.
      call MatSetOption(jac_prec,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr)

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
!  jac - Jacobian preconditioner matrix
!  ierr     - error code
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
      subroutine FormJacobianLocal(X1,jac,solver,add_nl_term,ierr)
#include <petsc/finclude/petscmatdef.h>
      use petscmat
      use petsc_kkt_solver
      implicit none
!  Input/output variables:
      type (petsc_kkt_solver_type) solver
      Vec::      X1
     Mat::     jac
      logical        add_nl_term
      PetscErrorCode ierr

!  Local variables:
      PetscInt    irow,row(1),col(5),i,j
      PetscInt    ione,ifive,low,high,ii
      PetscScalar two,one,hx,hy,hy2inv
      PetscScalar hx2inv,sc,v(5)
      PetscScalar,pointer :: lx_v(:)

!  Set parameters
      ione   = 1
      ifive  = 5
      one    = 1.0
      two    = 2.0
      hx     = one/(solver%mx-1)
      hy     = one/(solver%my-1)
      sc     = solver%lambda
      hx2inv = one/(hx*hx)
      hy2inv = one/(hy*hy)

      call VecGetOwnershipRange(X1,low,high,ierr)
      call VecGetArrayReadF90(X1,lx_v,ierr)

      ii = 0
      do 20 irow=low,high-1
         j = irow/solver%mx
         i = mod(irow,solver%mx)
         ii = ii + 1            ! one based local index
!     boundary points
         if (i .eq. 0 .or. j .eq. 0 .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1) then
            col(1) = irow
            row(1) = irow
            v(1)   = one
            call MatSetValues(jac,ione,row,ione,col,v,INSERT_VALUES,ierr)
!     interior grid points
         else
            v(1) = -hy2inv
            if(j-1==0) v(1) = 0.0
            v(2) = -hx2inv
            if(i-1==0) v(2) = 0.0
            v(3) = two*(hx2inv + hy2inv) 
            if(add_nl_term) v(3) = v(3) - sc*exp(lx_v(ii))
            v(4) = -hx2inv
            if(i+1==solver%mx-1) v(4) = 0.0
            v(5) = -hy2inv
            if(j+1==solver%my-1) v(5) = 0.0
            col(1) = irow - solver%mx
            col(2) = irow - 1
            col(3) = irow
            col(4) = irow + 1
            col(5) = irow + solver%mx
            row(1) = irow
            call MatSetValues(jac,ione,row,ifive,col,v,INSERT_VALUES,ierr)
         endif
 20   continue

      call VecRestoreArrayReadF90(X1,lx_v,ierr)

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
#include <petsc/finclude/petscsnesdef.h>
      use petscsnes
      use petsc_kkt_solver
      implicit none
!  Input/output variables:
      SNES::     snesIn
     Vec::      X,F
      PetscErrorCode ierr
      type (petsc_kkt_solver_type) solver

!  Declarations for use with local arrays:
     Vec::              Xsub(2),Fsub(2)
      PetscInt               izero,ione,itwo

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can
!  be done while messages are in transition.
 
      izero = 0
      ione = 1
      itwo = 2
      call DMCompositeGetAccessArray(solver%da,X,itwo,PETSC_NULL_INTEGER,Xsub,ierr)
      call DMCompositeGetAccessArray(solver%da,F,itwo,PETSC_NULL_INTEGER,Fsub,ierr)

      call FormFunctionNLTerm( Xsub(1), Fsub(1), solver, ierr )
      call MatMultAdd( solver%AmatLin, Xsub(1), Fsub(1), Fsub(1), ierr)

!     do rest of operator (linear)
      call MatMult(    solver%Cmat, Xsub(1),      Fsub(2), ierr)
      call MatMultAdd( solver%Bmat, Xsub(2), Fsub(1), Fsub(1), ierr)
      call MatMultAdd( solver%Dmat, Xsub(2), Fsub(2), Fsub(2), ierr)

      call DMCompositeRestoreAccessArray(solver%da,X,itwo,PETSC_NULL_INTEGER,Xsub,ierr)
      call DMCompositeRestoreAccessArray(solver%da,F,itwo,PETSC_NULL_INTEGER,Fsub,ierr)
      return
      end subroutine formfunction


! ---------------------------------------------------------------------
!
!  FormFunctionNLTerm - Computes nonlinear function, called by
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
      subroutine FormFunctionNLTerm(X1,F1,solver,ierr)
#include <petsc/finclude/petscvecdef.h>
      use petscvec
      use petsc_kkt_solver
      implicit none
!  Input/output variables:
      type (petsc_kkt_solver_type) solver
     Vec::      X1,F1
      PetscErrorCode ierr
!  Local variables:
      PetscScalar one,sc
      PetscScalar u,v(1)
      PetscInt  i,j,low,high,ii,ione,irow,row(1)
      PetscScalar,pointer :: lx_v(:)

      one    = 1.0
      sc     = solver%lambda
      ione   = 1

      call VecGetArrayReadF90(X1,lx_v,ierr)
      call VecGetOwnershipRange(X1,low,high,ierr)

!     Compute function over the locally owned part of the grid
      ii = 0
      do 20 irow=low,high-1
         j = irow/solver%mx
         i = mod(irow,solver%mx)
         ii = ii + 1            ! one based local index
         row(1) = irow
         if (i .eq. 0 .or. j .eq. 0 .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1 ) then
            v(1) = 0.0
         else
            u = lx_v(ii)
            v(1) = -sc*exp(u)
         endif
         call VecSetValues(F1,ione,row,v,INSERT_VALUES,ierr)
 20   continue

      call VecRestoreArrayReadF90(X1,lx_v,ierr)

      call VecAssemblyBegin(F1,ierr)
      call VecAssemblyEnd(F1,ierr)

      ierr = 0
      return
      end subroutine FormFunctionNLTerm

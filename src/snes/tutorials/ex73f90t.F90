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
      module ex73f90tmodule
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscmat.h>
      use petscdm
      use petscmat
      type ex73f90tmodule_type
        DM::da
!     temp A block stuff
        PetscInt mx,my
        PetscMPIInt rank
        PetscReal lambda
!     Mats
        Mat::Amat,AmatLin,Bmat,CMat,Dmat
        IS::isPhi,isLambda
      end type ex73f90tmodule_type

      end module ex73f90tmodule

      module ex73f90tmodule_interfaces
        use ex73f90tmodule

      Interface SNESSetApplicationContext
        Subroutine SNESSetApplicationContext(snesIn,ctx,ierr)
#include <petsc/finclude/petscsnes.h>
        use petscsnes
        use ex73f90tmodule
          SNES::    snesIn
          type(ex73f90tmodule_type) ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESSetApplicationContext

      Interface SNESGetApplicationContext
        Subroutine SNESGetApplicationContext(snesIn,ctx,ierr)
#include <petsc/finclude/petscsnes.h>
        use petscsnes
        use ex73f90tmodule
          SNES::     snesIn
          type(ex73f90tmodule_type), pointer :: ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface SNESGetApplicationContext
      end module ex73f90tmodule_interfaces

      program main
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscsnes.h>
      use petscdm
      use petscdmda
      use petscsnes
      use ex73f90tmodule
      use ex73f90tmodule_interfaces
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
      type(ex73f90tmodule_type)  solver
      PetscScalar      bval(1),cval(1),one

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
      external FormInitialGuess,FormJacobian,FormFunction

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Initialize program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,solver%rank,ierr))

!  Initialize problem parameters
      lambda_max  = 6.81_PETSC_REAL_KIND
      lambda_min  = 0.0
      solver%lambda = 6.0
      ione = 1
      nfour = 4
      itwo = 2
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-par', solver%lambda,flg,ierr))
      if (solver%lambda .ge. lambda_max .or. solver%lambda .lt. lambda_min) then
         SETERRA(PETSC_COMM_SELF,PETSC_ERR_USER,'Lambda provided with -par is out of range')
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!     just get size
      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nfour,nfour,PETSC_DECIDE,PETSC_DECIDE,ione,ione,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,daphi,ierr))
      PetscCallA(DMSetFromOptions(daphi,ierr))
      PetscCallA(DMSetUp(daphi,ierr))
      PetscCallA(DMDAGetInfo(daphi,PETSC_NULL_INTEGER,solver%mx,solver%my,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      N1 = solver%my*solver%mx
      N2 = solver%my
      flg = .false.
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-no_constraints',flg,flg,ierr))
      if (flg) then
         N2 = 0
      endif

      PetscCallA(DMDestroy(daphi,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix data structure; set Jacobian evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(DMShellCreate(PETSC_COMM_WORLD,daphi,ierr))
      PetscCallA(DMSetOptionsPrefix(daphi,'phi_',ierr))
      PetscCallA(DMSetFromOptions(daphi,ierr))

      PetscCallA(VecCreate(PETSC_COMM_WORLD,x1,ierr))
      PetscCallA(VecSetSizes(x1,PETSC_DECIDE,N1,ierr))
      PetscCallA(VecSetFromOptions(x1,ierr))

      PetscCallA(VecGetOwnershipRange(x1,low,high,ierr))
      nloc = high - low

      PetscCallA(MatCreate(PETSC_COMM_WORLD,Amat,ierr))
      PetscCallA(MatSetSizes(Amat,PETSC_DECIDE,PETSC_DECIDE,N1,N1,ierr))
      PetscCallA(MatSetUp(Amat,ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,solver%AmatLin,ierr))
      PetscCallA(MatSetSizes(solver%AmatLin,PETSC_DECIDE,PETSC_DECIDE,N1,N1,ierr))
      PetscCallA(MatSetUp(solver%AmatLin,ierr))

      PetscCallA(FormJacobianLocal(x1,solver%AmatLin,solver,.false.,ierr))
      PetscCallA(MatAssemblyBegin(solver%AmatLin,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(solver%AmatLin,MAT_FINAL_ASSEMBLY,ierr))

      PetscCallA(DMShellSetGlobalVector(daphi,x1,ierr))
      PetscCallA(DMShellSetMatrix(daphi,Amat,ierr))

      PetscCallA(VecCreate(PETSC_COMM_SELF,x1loc,ierr))
      PetscCallA(VecSetSizes(x1loc,nloc,nloc,ierr))
      PetscCallA(VecSetFromOptions(x1loc,ierr))
      PetscCallA(DMShellSetLocalVector(daphi,x1loc,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create B, C, & D matrices
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(MatCreate(PETSC_COMM_WORLD,Cmat,ierr))
      PetscCallA(MatSetSizes(Cmat,PETSC_DECIDE,PETSC_DECIDE,N2,N1,ierr))
      PetscCallA(MatSetUp(Cmat,ierr))
!      create data for C and B
      PetscCallA(MatCreate(PETSC_COMM_WORLD,Bmat,ierr))
      PetscCallA(MatSetSizes(Bmat,PETSC_DECIDE,PETSC_DECIDE,N1,N2,ierr))
      PetscCallA(MatSetUp(Bmat,ierr))
!     create data for D
      PetscCallA(MatCreate(PETSC_COMM_WORLD,Dmat,ierr))
      PetscCallA(MatSetSizes(Dmat,PETSC_DECIDE,PETSC_DECIDE,N2,N2,ierr))
      PetscCallA(MatSetUp(Dmat,ierr))

      PetscCallA(VecCreate(PETSC_COMM_WORLD,x2,ierr))
      PetscCallA(VecSetSizes(x2,PETSC_DECIDE,N2,ierr))
      PetscCallA(VecSetFromOptions(x2,ierr))

      PetscCallA(VecGetOwnershipRange(x2,lamlow,lamhigh,ierr))
      nloclam = lamhigh-lamlow

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set fake B and C
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      one    = 1.0
      if (N2 .gt. 0) then
         bval(1) = -one/(solver%mx-2)
!     cval = -one/(solver%my*solver%mx)
         cval(1) = -one
         do 20 irow=low,high-1
            j = irow/solver%mx   ! row in domain
            i = mod(irow,solver%mx)
            row(1) = irow
            col(1) = j
            if (i .eq. 0 .or. j .eq. 0 .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1) then
               !     no op
            else
               PetscCallA(MatSetValues(Bmat,ione,row,ione,col,bval,INSERT_VALUES,ierr))
            endif
            row(1) = j
            PetscCallA(MatSetValues(Cmat,ione,row,ione,row,cval,INSERT_VALUES,ierr))
 20   continue
      endif
      PetscCallA(MatAssemblyBegin(Bmat,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(Bmat,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyBegin(Cmat,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(Cmat,MAT_FINAL_ASSEMBLY,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set D (indentity)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      do 30 j=lamlow,lamhigh-1
         row(1) = j
         cval(1) = one
         PetscCallA(MatSetValues(Dmat,ione,row,ione,row,cval,INSERT_VALUES,ierr))
 30   continue
      PetscCallA(MatAssemblyBegin(Dmat,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(Dmat,MAT_FINAL_ASSEMBLY,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  DM for lambda (dalam) : temp driver for A block, setup A block solver data
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(DMShellCreate(PETSC_COMM_WORLD,dalam,ierr))
      PetscCallA(DMShellSetGlobalVector(dalam,x2,ierr))
      PetscCallA(DMShellSetMatrix(dalam,Dmat,ierr))

      PetscCallA(VecCreate(PETSC_COMM_SELF,x2loc,ierr))
      PetscCallA(VecSetSizes(x2loc,nloclam,nloclam,ierr))
      PetscCallA(VecSetFromOptions(x2loc,ierr))
      PetscCallA(DMShellSetLocalVector(dalam,x2loc,ierr))

      PetscCallA(DMSetOptionsPrefix(dalam,'lambda_',ierr))
      PetscCallA(DMSetFromOptions(dalam,ierr))
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create field split DA
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(DMCompositeCreate(PETSC_COMM_WORLD,solver%da,ierr))
      PetscCallA(DMCompositeAddDM(solver%da,daphi,ierr))
      PetscCallA(DMCompositeAddDM(solver%da,dalam,ierr))
      PetscCallA(DMSetFromOptions(solver%da,ierr))
      PetscCallA(DMSetUp(solver%da,ierr))
      PetscCallA(DMCompositeGetGlobalISs(solver%da,isglobal,ierr))
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

      PetscCallA(MatCreateNest(PETSC_COMM_WORLD,itwo,isglobal,itwo,isglobal,matArray,KKTmat,ierr))
      PetscCallA(MatSetFromOptions(KKTmat,ierr))

!  Extract global and local vectors from DMDA; then duplicate for remaining
!     vectors that are the same types
      PetscCallA(MatCreateVecs(KKTmat,x,PETSC_NULL_VEC,ierr))
      PetscCallA(VecDuplicate(x,r,ierr))

      PetscCallA(SNESCreate(PETSC_COMM_WORLD,mysnes,ierr))

      PetscCallA(SNESSetDM(mysnes,solver%da,ierr))

      PetscCallA(SNESSetApplicationContext(mysnes,solver,ierr))

      PetscCallA(SNESSetDM(mysnes,solver%da,ierr))

!  Set function evaluation routine and vector
      PetscCallA(SNESSetFunction(mysnes,r,FormFunction,solver,ierr))

      PetscCallA(SNESSetJacobian(mysnes,KKTmat,KKTmat,FormJacobian,solver,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
      PetscCallA(SNESSetFromOptions(mysnes,ierr))

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
      if (solver%rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(MatDestroy(KKTmat,ierr))
      PetscCallA(MatDestroy(Amat,ierr))
      PetscCallA(MatDestroy(Dmat,ierr))
      PetscCallA(MatDestroy(Bmat,ierr))
      PetscCallA(MatDestroy(Cmat,ierr))
      PetscCallA(MatDestroy(solver%AmatLin,ierr))
      PetscCallA(ISDestroy(solver%isPhi,ierr))
      PetscCallA(ISDestroy(solver%isLambda,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(x2,ierr))
      PetscCallA(VecDestroy(x1,ierr))
      PetscCallA(VecDestroy(x1loc,ierr))
      PetscCallA(VecDestroy(x2loc,ierr))
      PetscCallA(VecDestroy(r,ierr))
      PetscCallA(SNESDestroy(mysnes,ierr))
      PetscCallA(DMDestroy(solver%da,ierr))
      PetscCallA(DMDestroy(daphi,ierr))
      PetscCallA(DMDestroy(dalam,ierr))

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
!  the local vector data via VecGetArrayF90() and VecRestoreArrayF90().
!
      subroutine FormInitialGuess(mysnes,Xnest,ierr)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      use ex73f90tmodule
      use ex73f90tmodule_interfaces
      implicit none
!  Input/output variables:
      SNES::     mysnes
      Vec::      Xnest
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      type(ex73f90tmodule_type), pointer:: solver
      Vec::      Xsub(2)
      PetscInt::  izero,ione,itwo

      izero = 0
      ione = 1
      itwo = 2
      ierr = 0
      PetscCall(SNESGetApplicationContext(mysnes,solver,ierr))
      PetscCall(DMCompositeGetAccessArray(solver%da,Xnest,itwo,PETSC_NULL_INTEGER,Xsub,ierr))

      PetscCall(InitialGuessLocal(solver,Xsub(1),ierr))
      PetscCall(VecAssemblyBegin(Xsub(1),ierr))
      PetscCall(VecAssemblyEnd(Xsub(1),ierr))

!     zero out lambda
      PetscCall(VecZeroEntries(Xsub(2),ierr))
      PetscCall(DMCompositeRestoreAccessArray(solver%da,Xnest,itwo,PETSC_NULL_INTEGER,Xsub,ierr))

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
#include <petsc/finclude/petscsys.h>
      use petscsys
      use ex73f90tmodule
      implicit none
!  Input/output variables:
      type (ex73f90tmodule_type)         solver
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

      PetscCall(VecGetOwnershipRange(X1,low,high,ierr))

      do 20 row=low,high-1
         j = row/solver%mx
         i = mod(row,solver%mx)
         temp = min(j,solver%my-j+1)*hy
         if (i .eq. 0 .or. j .eq. 0  .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1) then
            v = 0.0
         else
            v = temp1 * sqrt(min(min(i,solver%mx-i+1)*hx,temp))
         endif
         PetscCall(VecSetValues(X1,ione,row,v,INSERT_VALUES,ierr))
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
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      use ex73f90tmodule
      implicit none
!  Input/output variables:
      SNES::     dummy
      Vec::      X
     Mat::     jac,jac_prec
      type(ex73f90tmodule_type)  solver
      PetscErrorCode ierr

!  Declarations for use with local arrays:
      Vec::      Xsub(1)
     Mat::     Amat
      PetscInt       ione

      ione = 1

      PetscCall(DMCompositeGetAccessArray(solver%da,X,ione,PETSC_NULL_INTEGER,Xsub,ierr))

!     Compute entries for the locally owned part of the Jacobian preconditioner.
      PetscCall(MatCreateSubMatrix(jac_prec,solver%isPhi,solver%isPhi,MAT_INITIAL_MATRIX,Amat,ierr))

      PetscCall(FormJacobianLocal(Xsub(1),Amat,solver,.true.,ierr))
      PetscCall(MatDestroy(Amat,ierr)) ! discard our reference
      PetscCall(DMCompositeRestoreAccessArray(solver%da,X,ione,PETSC_NULL_INTEGER,Xsub,ierr))

      ! the rest of the matrix is not touched
      PetscCall(MatAssemblyBegin(jac_prec,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(jac_prec,MAT_FINAL_ASSEMBLY,ierr))
      if (jac .ne. jac_prec) then
         PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
         PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))
      end if

!     Tell the matrix we will never add a new nonzero location to the
!     matrix. If we do it will generate an error.
      PetscCall(MatSetOption(jac_prec,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr))

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
#include <petsc/finclude/petscmat.h>
      use petscmat
      use ex73f90tmodule
      implicit none
!  Input/output variables:
      type (ex73f90tmodule_type) solver
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

      PetscCall(VecGetOwnershipRange(X1,low,high,ierr))
      PetscCall(VecGetArrayReadF90(X1,lx_v,ierr))

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
            PetscCall(MatSetValues(jac,ione,row,ione,col,v,INSERT_VALUES,ierr))
!     interior grid points
         else
            v(1) = -hy2inv
            if (j-1==0) v(1) = 0.0
            v(2) = -hx2inv
            if (i-1==0) v(2) = 0.0
            v(3) = two*(hx2inv + hy2inv)
            if (add_nl_term) v(3) = v(3) - sc*exp(lx_v(ii))
            v(4) = -hx2inv
            if (i+1==solver%mx-1) v(4) = 0.0
            v(5) = -hy2inv
            if (j+1==solver%my-1) v(5) = 0.0
            col(1) = irow - solver%mx
            col(2) = irow - 1
            col(3) = irow
            col(4) = irow + 1
            col(5) = irow + solver%mx
            row(1) = irow
            PetscCall(MatSetValues(jac,ione,row,ifive,col,v,INSERT_VALUES,ierr))
         endif
 20   continue

      PetscCall(VecRestoreArrayReadF90(X1,lx_v,ierr))

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
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      use ex73f90tmodule
      implicit none
!  Input/output variables:
      SNES::     snesIn
     Vec::      X,F
      PetscErrorCode ierr
      type (ex73f90tmodule_type) solver

!  Declarations for use with local arrays:
     Vec::              Xsub(2),Fsub(2)
      PetscInt               itwo

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can
!  be done while messages are in transition.

      itwo = 2
      PetscCall(DMCompositeGetAccessArray(solver%da,X,itwo,PETSC_NULL_INTEGER,Xsub,ierr))
      PetscCall(DMCompositeGetAccessArray(solver%da,F,itwo,PETSC_NULL_INTEGER,Fsub,ierr))

      PetscCall(FormFunctionNLTerm( Xsub(1), Fsub(1), solver, ierr))
      PetscCall(MatMultAdd( solver%AmatLin, Xsub(1), Fsub(1), Fsub(1), ierr))

!     do rest of operator (linear)
      PetscCall(MatMult(    solver%Cmat, Xsub(1),      Fsub(2), ierr))
      PetscCall(MatMultAdd( solver%Bmat, Xsub(2), Fsub(1), Fsub(1), ierr))
      PetscCall(MatMultAdd( solver%Dmat, Xsub(2), Fsub(2), Fsub(2), ierr))

      PetscCall(DMCompositeRestoreAccessArray(solver%da,X,itwo,PETSC_NULL_INTEGER,Xsub,ierr))
      PetscCall(DMCompositeRestoreAccessArray(solver%da,F,itwo,PETSC_NULL_INTEGER,Fsub,ierr))
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
#include <petsc/finclude/petscvec.h>
      use petscvec
      use ex73f90tmodule
      implicit none
!  Input/output variables:
      type (ex73f90tmodule_type) solver
     Vec::      X1,F1
      PetscErrorCode ierr
!  Local variables:
      PetscScalar sc
      PetscScalar u,v(1)
      PetscInt  i,j,low,high,ii,ione,irow,row(1)
      PetscScalar,pointer :: lx_v(:)

      sc     = solver%lambda
      ione   = 1

      PetscCall(VecGetArrayReadF90(X1,lx_v,ierr))
      PetscCall(VecGetOwnershipRange(X1,low,high,ierr))

!     Compute function over the locally owned part of the grid
      ii = 0
      do 20 irow=low,high-1
         j = irow/solver%mx
         i = mod(irow,solver%mx)
         ii = ii + 1            ! one based local index
         row(1) = irow
         if (i .eq. 0 .or. j .eq. 0 .or. i .eq. solver%mx-1 .or. j .eq. solver%my-1) then
            v(1) = 0.0
         else
            u = lx_v(ii)
            v(1) = -sc*exp(u)
         endif
         PetscCall(VecSetValues(F1,ione,row,v,INSERT_VALUES,ierr))
 20   continue

      PetscCall(VecRestoreArrayReadF90(X1,lx_v,ierr))

      PetscCall(VecAssemblyBegin(F1,ierr))
      PetscCall(VecAssemblyEnd(F1,ierr))

      ierr = 0
      return
      end subroutine FormFunctionNLTerm

!/*TEST
!
!   build:
!      requires: !single !complex
!
!   test:
!      nsize: 4
!      args: -par 5.0 -da_grid_x 10 -da_grid_y 10 -snes_monitor_short -snes_linesearch_type basic -snes_converged_reason -ksp_type fgmres -ksp_norm_type unpreconditioned -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper -ksp_monitor_short -fieldsplit_lambda_ksp_type preonly -fieldsplit_lambda_pc_type jacobi -fieldsplit_phi_pc_type gamg -fieldsplit_phi_pc_gamg_esteig_ksp_type cg -fieldsplit_phi_pc_gamg_esteig_ksp_max_it 10  -fieldsplit_phi_pc_gamg_agg_nsmooths 1 -fieldsplit_phi_pc_gamg_threshold 0.
!
!TEST*/

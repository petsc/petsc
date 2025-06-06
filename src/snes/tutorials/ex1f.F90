!
!  Description: Uses the Newton method to solve a two-variable system.
!

      program main
#include <petsc/finclude/petsc.h>
        use petsc
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     snes        - nonlinear solver
!     ksp        - linear solver
!     pc          - preconditioner context
!     ksp         - Krylov subspace method context
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!     its         - iterations for convergence
!
      SNES     snes
      PC       pc
      KSP      ksp
      Vec      x,r
      Mat      J
      SNESLineSearch linesearch
      PetscErrorCode  ierr
      PetscInt its,i2,i20
      PetscMPIInt size,rank
      PetscScalar   pfive
      PetscReal   tol
      PetscBool   setls
      PetscReal, pointer :: rhistory(:)
      PetscInt, pointer :: itshistory(:)
      PetscInt nhistory
#if defined(PETSC_USE_LOG)
      PetscViewer viewer
#endif
      double precision threshold,oldthreshold

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction, FormJacobian, MyLineSearch

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscLogNestedBegin(ierr))
      threshold = 1.0
      PetscCallA(PetscLogSetThreshold(threshold,oldthreshold,ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
      PetscCheckA(size .eq. 1,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,'Uniprocessor example')

      i2  = 2
      i20 = 20
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(SNESCreate(PETSC_COMM_WORLD,snes,ierr))

      PetscCallA(SNESSetConvergenceHistory(snes,PETSC_NULL_REAL_ARRAY,PETSC_NULL_INTEGER_ARRAY,PETSC_DECIDE,PETSC_FALSE,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix and vector data structures; set corresponding routines
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create vectors for solution and nonlinear function

      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,i2,x,ierr))
      PetscCallA(VecDuplicate(x,r,ierr))

!  Create Jacobian matrix data structure

      PetscCallA(MatCreate(PETSC_COMM_SELF,J,ierr))
      PetscCallA(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,i2,i2,ierr))
      PetscCallA(MatSetFromOptions(J,ierr))
      PetscCallA(MatSetUp(J,ierr))

!  Set function evaluation routine and vector

      PetscCallA(SNESSetFunction(snes,r,FormFunction,0,ierr))

!  Set Jacobian matrix data structure and Jacobian evaluation routine

      PetscCallA(SNESSetJacobian(snes,J,J,FormJacobian,0,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set linear solver defaults for this problem. By extracting the
!  KSP, KSP, and PC contexts from the SNES context, we can then
!  directly call any KSP, KSP, and PC routines to set various options.

      PetscCallA(SNESGetKSP(snes,ksp,ierr))
      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,PCNONE,ierr))
      tol = 1.e-4
      PetscCallA(KSPSetTolerances(ksp,tol,PETSC_CURRENT_REAL,PETSC_CURRENT_REAL,i20,ierr))

!  Set SNES/KSP/KSP/PC runtime options, e.g.,
!      -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
!  These options will override those specified above as long as
!  SNESSetFromOptions() is called _after_ any other customization
!  routines.

      PetscCallA(SNESSetFromOptions(snes,ierr))

      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-setls',setls,ierr))

      if (setls) then
        PetscCallA(SNESGetLineSearch(snes, linesearch, ierr))
        PetscCallA(SNESLineSearchSetType(linesearch, 'shell', ierr))
        PetscCallA(SNESLineSearchShellSetApply(linesearch, MyLineSearch,0,ierr))
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      pfive = 0.5
      PetscCallA(VecSet(x,pfive,ierr))
      PetscCallA(SNESSolve(snes,PETSC_NULL_VEC,x,ierr))

      PetscCallA(SNESGetConvergenceHistory(snes,rhistory,itshistory,nhistory,ierr))
      PetscCallA(SNESRestoreConvergenceHistory(snes,rhistory,itshistory,nhistory,ierr))

! View solver converged reason; we could instead use the option -snes_converged_reason
      PetscCallA(SNESConvergedReasonView(snes,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(SNESGetIterationNumber(snes,its,ierr))
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(r,ierr))
      PetscCallA(MatDestroy(J,ierr))
      PetscCallA(SNESDestroy(snes,ierr))
#if defined(PETSC_USE_LOG)
      PetscCallA(PetscViewerASCIIOpen(PETSC_COMM_WORLD,'filename.xml',viewer,ierr))
      PetscCallA(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_XML,ierr))
      PetscCallA(PetscLogView(viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
#endif
      PetscCallA(PetscFinalize(ierr))
      end
!
! ------------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameter:
!  f - function vector
!
      subroutine FormFunction(snes,x,f,dummy,ierr)
      use petscvec
      use petscsnesdef
      implicit none

      SNES     snes
      Vec      x,f
      PetscErrorCode ierr
      integer dummy(*)

!  Declarations for use with local arrays
      PetscScalar,pointer :: lx_v(:),lf_v(:)

!  Get pointers to vector data.
!    - VecGetArray() returns a pointer to the data array.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

      PetscCall(VecGetArrayRead(x,lx_v,ierr))
      PetscCall(VecGetArray(f,lf_v,ierr))

!  Compute function

      lf_v(1) = lx_v(1)*lx_v(1) + lx_v(1)*lx_v(2) - 3.0
      lf_v(2) = lx_v(1)*lx_v(2) + lx_v(2)*lx_v(2) - 6.0

!  Restore vectors

      PetscCall(VecRestoreArrayRead(x,lx_v,ierr))
      PetscCall(VecRestoreArray(f,lf_v,ierr))

      end

! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameters:
!  A - Jacobian matrix
!  B - optionally different matrix used to construct the preconditioner
!
      subroutine FormJacobian(snes,X,jac,B,dummy,ierr)
      use petscvec
      use petscmat
      use petscsnesdef
      implicit none

      SNES         snes
      Vec          X
      Mat          jac,B
      PetscScalar  A(4)
      PetscErrorCode ierr
      PetscInt idx(2),i2
      integer dummy(*)

!  Declarations for use with local arrays

      PetscScalar,pointer :: lx_v(:)

!  Get pointer to vector data

      i2 = 2
      PetscCall(VecGetArrayRead(x,lx_v,ierr))

!  Compute Jacobian entries and insert into matrix.
!   - Since this is such a small problem, we set all entries for
!     the matrix at once.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array idx).

      idx(1) = 0
      idx(2) = 1
      A(1) = 2.0*lx_v(1) + lx_v(2)
      A(2) = lx_v(1)
      A(3) = lx_v(2)
      A(4) = lx_v(1) + 2.0*lx_v(2)
      PetscCall(MatSetValues(B,i2,idx,i2,idx,A,INSERT_VALUES,ierr))

!  Restore vector

      PetscCall(VecRestoreArrayRead(x,lx_v,ierr))

!  Assemble matrix

      PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY,ierr))
      if (B .ne. jac) then
        PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
        PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))
      endif

      end

      subroutine MyLineSearch(linesearch, lctx, ierr)
      use petscsnes
      use petscvec
      implicit none

      SNESLineSearch    linesearch
      SNES              snes
      integer           lctx
      Vec               x, f,g, y, w
      PetscReal         ynorm,gnorm,xnorm
      PetscErrorCode    ierr

      PetscScalar       mone

      mone = -1.0
      PetscCall(SNESLineSearchGetSNES(linesearch, snes, ierr))
      PetscCall(SNESLineSearchGetVecs(linesearch, x, f, y, w, g, ierr))
      PetscCall(VecNorm(y,NORM_2,ynorm,ierr))
      PetscCall(VecAXPY(x,mone,y,ierr))
      PetscCall(SNESComputeFunction(snes,x,f,ierr))
      PetscCall(VecNorm(f,NORM_2,gnorm,ierr))
      PetscCall(VecNorm(x,NORM_2,xnorm,ierr))
      PetscCall(VecNorm(y,NORM_2,ynorm,ierr))
      PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, gnorm, ynorm,ierr))
      end

!/*TEST
!
!   test:
!      args: -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short
!      requires: !single
!
!TEST*/

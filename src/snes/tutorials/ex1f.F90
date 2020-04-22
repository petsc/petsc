!
!
!  Description: Uses the Newton method to solve a two-variable system.
!
!!/*T
!  Concepts: SNES^basic uniprocessor example
!  Processors: 1
!T*/


!
! -----------------------------------------------------------------------

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
      PetscViewer viewer
      double precision threshold,oldthreshold

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction, FormJacobian, MyLineSearch

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Macro definitions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Macros to make clearer the process of setting values in vectors and
!  getting values from vectors.  These vectors are used in the routines
!  FormFunction() and FormJacobian().
!   - The element lx_a(ib) is element ib in the vector x
!
#define lx_a(ib) lx_v(lx_i + (ib))
#define lf_a(ib) lf_v(lf_i + (ib))
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call PetscLogNestedBegin(ierr);CHKERRA(ierr)
      threshold = 1.0
      call PetscLogSetThreshold(threshold,oldthreshold,ierr)
! dummy test of logging a reduction
      ierr = PetscAReduce()
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      if (size .ne. 1) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,'Uniprocessor example'); endif

      i2  = 2
      i20 = 20
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix and vector data structures; set corresponding routines
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create vectors for solution and nonlinear function

      call VecCreateSeq(PETSC_COMM_SELF,i2,x,ierr)
      call VecDuplicate(x,r,ierr)

!  Create Jacobian matrix data structure

      call MatCreate(PETSC_COMM_SELF,J,ierr)
      call MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,i2,i2,ierr)
      call MatSetFromOptions(J,ierr)
      call MatSetUp(J,ierr)

!  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,0,ierr)

!  Set Jacobian matrix data structure and Jacobian evaluation routine

      call SNESSetJacobian(snes,J,J,FormJacobian,0,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set linear solver defaults for this problem. By extracting the
!  KSP, KSP, and PC contexts from the SNES context, we can then
!  directly call any KSP, KSP, and PC routines to set various options.

      call SNESGetKSP(snes,ksp,ierr)
      call KSPGetPC(ksp,pc,ierr)
      call PCSetType(pc,PCNONE,ierr)
      tol = 1.e-4
      call KSPSetTolerances(ksp,tol,PETSC_DEFAULT_REAL,                  &
     &                      PETSC_DEFAULT_REAL,i20,ierr)

!  Set SNES/KSP/KSP/PC runtime options, e.g.,
!      -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
!  These options will override those specified above as long as
!  SNESSetFromOptions() is called _after_ any other customization
!  routines.


      call SNESSetFromOptions(snes,ierr)

      call PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,   &
     &                         '-setls',setls,ierr)

      if (setls) then
        call SNESGetLineSearch(snes, linesearch, ierr)
        call SNESLineSearchSetType(linesearch, 'shell', ierr)
        call SNESLineSearchShellSetUserFunc(linesearch, MyLineSearch,   &
     &                                      0, ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      pfive = 0.5
      call VecSet(x,pfive,ierr)
      call SNESSolve(snes,PETSC_NULL_VEC,x,ierr)
      call SNESGetIterationNumber(snes,its,ierr);
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of SNES iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call MatDestroy(J,ierr)
      call SNESDestroy(snes,ierr)
      call PetscViewerASCIIOpen(PETSC_COMM_WORLD,'filename.xml',viewer,ierr)
      call PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_XML,ierr)
      call PetscLogView(viewer,ierr)
      call PetscViewerDestroy(viewer,ierr)
      call PetscFinalize(ierr)
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
      use petscsnes
      implicit none

      SNES     snes
      Vec      x,f
      PetscErrorCode ierr
      integer dummy(*)

!  Declarations for use with local arrays

      PetscScalar  lx_v(2),lf_v(2)
      PetscOffset  lx_i,lf_i

!  Get pointers to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the Fortran chapter of the users manual for details.

      call VecGetArrayRead(x,lx_v,lx_i,ierr)
      call VecGetArray(f,lf_v,lf_i,ierr)

!  Compute function

      lf_a(1) = lx_a(1)*lx_a(1)                                         &
     &          + lx_a(1)*lx_a(2) - 3.0
      lf_a(2) = lx_a(1)*lx_a(2)                                         &
     &          + lx_a(2)*lx_a(2) - 6.0

!  Restore vectors

      call VecRestoreArrayRead(x,lx_v,lx_i,ierr)
      call VecRestoreArray(f,lf_v,lf_i,ierr)

      return
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
!  B - optionally different preconditioning matrix
!  flag - flag indicating matrix structure
!
      subroutine FormJacobian(snes,X,jac,B,dummy,ierr)
      use petscsnes
      implicit none

      SNES         snes
      Vec          X
      Mat          jac,B
      PetscScalar  A(4)
      PetscErrorCode ierr
      PetscInt idx(2),i2
      integer dummy(*)

!  Declarations for use with local arrays

      PetscScalar lx_v(2)
      PetscOffset lx_i

!  Get pointer to vector data

      i2 = 2
      call VecGetArrayRead(x,lx_v,lx_i,ierr)

!  Compute Jacobian entries and insert into matrix.
!   - Since this is such a small problem, we set all entries for
!     the matrix at once.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array idx).

      idx(1) = 0
      idx(2) = 1
      A(1) = 2.0*lx_a(1) + lx_a(2)
      A(2) = lx_a(1)
      A(3) = lx_a(2)
      A(4) = lx_a(1) + 2.0*lx_a(2)
      call MatSetValues(B,i2,idx,i2,idx,A,INSERT_VALUES,ierr)

!  Restore vector

      call VecRestoreArrayRead(x,lx_v,lx_i,ierr)

!  Assemble matrix

      call MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY,ierr)
      if (B .ne. jac) then
        call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
        call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      endif

      return
      end


      subroutine MyLineSearch(linesearch, lctx, ierr)
      use petscsnes
      implicit none

      SNESLineSearch    linesearch
      SNES              snes
      integer           lctx
      Vec               x, f,g, y, w
      PetscReal         ynorm,gnorm,xnorm
      PetscBool         flag
      PetscErrorCode    ierr

      PetscScalar       mone

      mone = -1.0
      call SNESLineSearchGetSNES(linesearch, snes, ierr)
      call SNESLineSearchGetVecs(linesearch, x, f, y, w, g, ierr)
      call VecNorm(y,NORM_2,ynorm,ierr)
      call VecAXPY(x,mone,y,ierr)
      call SNESComputeFunction(snes,x,f,ierr)
      call VecNorm(f,NORM_2,gnorm,ierr)
      call VecNorm(x,NORM_2,xnorm,ierr)
      call VecNorm(y,NORM_2,ynorm,ierr)
      call SNESLineSearchSetNorms(linesearch, xnorm, gnorm, ynorm,      &
     & ierr)
      flag = PETSC_FALSE
      return
      end

!/*TEST
!
!   test:
!      args: -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short
!      requires: !single
!
!TEST*/

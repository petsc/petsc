!
!  Description: Solves a linear system in parallel with KSP (Fortran code).
!               Also shows how to set a user-defined monitoring routine.
!
! -----------------------------------------------------------------------

program main
#include <petsc/finclude/petscksp.h>
  use petscksp
  implicit none
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     ksp     - linear solver context
!     ksp      - Krylov subspace method context
!     pc       - preconditioner context
!     x, b, u  - approx solution, right-hand side, exact solution vectors
!     A        - matrix that defines linear system
!     its      - iterations for convergence
!     norm     - norm of error in solution
!     rctx     - random number generator context
!
!  Note that vectors are declared as PETSc "Vec" objects.  These vectors
!  are mathematical objects that contain more than just an array of
!  double precision numbers. I.e., vectors in PETSc are not just
!        double precision x(*).
!  However, local vector data can be easily accessed via VecGetArray().
!  See the Fortran section of the PETSc users manual for details.
!
  PetscReal norm
  PetscInt i, j, II, JJ, m, n, its
  PetscInt Istart, Iend, izero, ione, itwo, ithree, col(3)
  PetscErrorCode ierr
  PetscMPIInt rank, size
  PetscBool flg
  PetscScalar v, one, neg_one, val(3)
  Vec x, b, u, xx, bb, uu
  Mat A, AA
  KSP ksp, kksp
  PetscRandom rctx
  PetscViewerAndFormat vf, vf2
  PetscClassId classid
  PetscViewer viewer
  PetscLogEvent petscEventNo

!  These variables are not currently used.
!      PC          pc
!      PCType      ptype
!      PetscReal tol

!  Note: Any user-defined Fortran routines (such as MyKSPMonitor)
!  MUST be declared as external.

  external MyKSPMonitor, MyKSPConverged

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  m = 3
  n = 3
  one = 1.0
  neg_one = -1.0
  izero = 0
  ione = 1
  itwo = 2
  ithree = 3

  PetscCallA(PetscLogNestedBegin(ierr))
  PetscCallA(PetscLogEventRegister("myFirstEvent", classid, petscEventNo, ierr))

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-m', m, flg, ierr))
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!      Compute the matrix and right-hand-side vector that define
!      the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n, ierr))
  PetscCallA(MatSetFromOptions(A, ierr))
  PetscCallA(MatSetUp(A, ierr))

!  Currently, all PETSc parallel matrix formats are partitioned by
!  contiguous chunks of rows across the processors.  Determine which
!  rows of the matrix are locally owned.

  PetscCallA(MatGetOwnershipRange(A, Istart, Iend, ierr))

!  Set matrix elements for the 2-D, five-point stencil in parallel.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global row and columns of matrix entries.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.

!     Note: this uses the less common natural ordering that orders first
!     all the unknowns for x = h then for x = 2h etc; Hence you see JH = II +- n
!     instead of JJ = II +- m as you might expect. The more standard ordering
!     would first do all variables for y = h, then y = 2h etc.

  do 10, II = Istart, Iend - 1
    v = -1.0
    i = II/n
    j = II - i*n
    if (i > 0) then
      JJ = II - n
      PetscCallA(MatSetValues(A, ione, [II], ione, [JJ], [v], INSERT_VALUES, ierr))
    end if
    if (i < m - 1) then
      JJ = II + n
      PetscCallA(MatSetValues(A, ione, [II], ione, [JJ], [v], INSERT_VALUES, ierr))
    end if
    if (j > 0) then
      JJ = II - 1
      PetscCallA(MatSetValues(A, ione, [II], ione, [JJ], [v], INSERT_VALUES, ierr))
    end if
    if (j < n - 1) then
      JJ = II + 1
      PetscCallA(MatSetValues(A, ione, [II], ione, [JJ], [v], INSERT_VALUES, ierr))
    end if
    v = 4.0
    PetscCallA(MatSetValues(A, ione, [II], ione, [II], [v], INSERT_VALUES, ierr))
10  continue

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

    PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

!  Create parallel vectors.
!   - Here, the parallel partitioning of the vector is determined by
!     PETSc at runtime.  We could also specify the local dimensions
!     if desired -- or use the more general routine VecCreate().
!   - When solving a linear system, the vectors and matrices MUST
!     be partitioned accordingly.  PETSc automatically generates
!     appropriately partitioned matrices and vectors when MatCreate()
!     and VecCreate() are used with the same communicator.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

    PetscCallA(VecCreateFromOptions(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, ione, PETSC_DECIDE, m*n, u, ierr))
    PetscCallA(VecSetFromOptions(u, ierr))
    PetscCallA(VecDuplicate(u, b, ierr))
    PetscCallA(VecDuplicate(b, x, ierr))

!  Set exact solution; then compute right-hand-side vector.
!  By default we use an exact solution of a vector with all
!  elements of 1.0;  Alternatively, using the runtime option
!  -random_sol forms a solution vector with random components.

    PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-random_exact_sol', flg, ierr))
    if (flg) then
      PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD, rctx, ierr))
      PetscCallA(PetscRandomSetFromOptions(rctx, ierr))
      PetscCallA(VecSetRandom(u, rctx, ierr))
      PetscCallA(PetscRandomDestroy(rctx, ierr))
    else
      PetscCallA(VecSet(u, one, ierr))
    end if
    PetscCallA(MatMult(A, u, b, ierr))

!  View the exact solution vector if desired

    PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-view_exact_sol', flg, ierr))
    if (flg) then
      PetscCallA(VecView(u, PETSC_VIEWER_STDOUT_WORLD, ierr))
    end if

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

    PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))

!  Set operators. Here the matrix that defines the linear system
!  also serves as the matrix from which the preconditioner is constructed.

    PetscCallA(KSPSetOperators(ksp, A, A, ierr))

!  Set linear solver defaults for this problem (optional).
!   - By extracting the KSP and PC contexts from the KSP context,
!     we can then directly call any KSP and PC routines
!     to set various options.
!   - The following four statements are optional; all of these
!     parameters could alternatively be specified at runtime via
!     KSPSetFromOptions(). All of these defaults can be
!     overridden at runtime, as indicated below.

!     We comment out this section of code since the Jacobi
!     preconditioner is not a good general default.

!      PetscCallA(KSPGetPC(ksp,pc,ierr))
!      ptype = PCJACOBI
!      PetscCallA(PCSetType(pc,ptype,ierr))
!      tol = 1.e-7
!      PetscCallA(KSPSetTolerances(ksp,tol,PETSC_CURRENT_REAL,PETSC_CURRENT_REAL,PETSC_CURRENT_INTEGER,ierr))

!  Set user-defined monitoring routine if desired

    PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-my_ksp_monitor', flg, ierr))
    if (flg) then
      PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
      PetscCallA(KSPMonitorSet(ksp, MyKSPMonitor, vf, PetscViewerAndFormatDestroy, ierr))
!
      PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf2, ierr))
      PetscCallA(KSPMonitorSet(ksp, KSPMonitorResidual, vf2, PetscViewerAndFormatDestroy, ierr))
    end if

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

    PetscCallA(KSPSetFromOptions(ksp, ierr))

!  Set convergence test routine if desired

    PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-my_ksp_convergence', flg, ierr))
    if (flg) then
      PetscCallA(KSPSetConvergenceTest(ksp, MyKSPConverged, 0, PETSC_NULL_FUNCTION, ierr))
    end if
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    PetscCallA(PetscLogEventBegin(petscEventNo, ierr))
    PetscCallA(KSPSolve(ksp, b, x, ierr))
    PetscCallA(PetscLogEventEnd(petscEventNo, ierr))

!  Solve small system on master

    if (rank == 0) then

      PetscCallA(MatCreate(PETSC_COMM_SELF, AA, ierr))
      PetscCallA(MatSetSizes(AA, PETSC_DECIDE, PETSC_DECIDE, m, m, ierr))
      PetscCallA(MatSetFromOptions(AA, ierr))

      val = [-1.0, 2.0, -1.0]
      PetscCallA(MatSetValues(AA, ione, [izero], itwo, [izero, ione], val(2:3), INSERT_VALUES, ierr))
      do i = 1, m - 2
        col = [i - 1, i, i + 1]
        PetscCallA(MatSetValues(AA, ione, [i], itwo, col, val, INSERT_VALUES, ierr))
      end do
      PetscCallA(MatSetValues(AA, ione, [m - 1], itwo, [m - 2, m - 1], val(1:2), INSERT_VALUES, ierr))
      PetscCallA(MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY, ierr))

      PetscCallA(VecCreate(PETSC_COMM_SELF, xx, ierr))
      PetscCallA(VecSetSizes(xx, PETSC_DECIDE, m, ierr))
      PetscCallA(VecSetFromOptions(xx, ierr))
      PetscCallA(VecDuplicate(xx, bb, ierr))
      PetscCallA(VecDuplicate(xx, uu, ierr))
      PetscCallA(VecSet(uu, one, ierr))
      PetscCallA(MatMult(AA, uu, bb, ierr))
      PetscCallA(KSPCreate(PETSC_COMM_SELF, kksp, ierr))
      PetscCallA(KSPSetOperators(kksp, AA, AA, ierr))
      PetscCallA(KSPSetFromOptions(kksp, ierr))
      PetscCallA(KSPSolve(kksp, bb, xx, ierr))

    end if

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Check the error
    PetscCallA(VecAXPY(x, neg_one, u, ierr))
    PetscCallA(VecNorm(x, NORM_2, norm, ierr))
    PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
    if (rank == 0) then
      if (norm > 1.e-12) then
        write (6, 100) norm, its
      else
        write (6, 110) its
      end if
    end if
100 format('Norm of error ', e11.4, ' iterations ', i5)
110 format('Norm of error < 1.e-12 iterations ', i5)

!  nested log view
    PetscCallA(PetscViewerASCIIOpen(PETSC_COMM_WORLD, 'report_performance.xml', viewer, ierr))
    PetscCallA(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_XML, ierr))
    PetscCallA(PetscLogView(viewer, ierr))
    PetscCallA(PetscViewerDestroy(viewer, ierr))

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

    PetscCallA(KSPDestroy(ksp, ierr))
    PetscCallA(VecDestroy(u, ierr))
    PetscCallA(VecDestroy(x, ierr))
    PetscCallA(VecDestroy(b, ierr))
    PetscCallA(MatDestroy(A, ierr))

    if (rank == 0) then
      PetscCallA(KSPDestroy(kksp, ierr))
      PetscCallA(VecDestroy(uu, ierr))
      PetscCallA(VecDestroy(xx, ierr))
      PetscCallA(VecDestroy(bb, ierr))
      PetscCallA(MatDestroy(AA, ierr))
    end if

!  Always call PetscFinalize() before exiting a program.  This routine
!    - finalizes the PETSc libraries as well as MPI
!    - provides summary and diagnostic information if certain runtime
!      options are chosen (e.g., -log_view).  See PetscFinalize()
!      manpage for more information.

    PetscCallA(PetscFinalize(ierr))
  end

! --------------------------------------------------------------
!
!  MyKSPMonitor - This is a user-defined routine for monitoring
!  the KSP iterative solvers.
!
!  Input Parameters:
!    ksp   - iterative context
!    n     - iteration number
!    rnorm - 2-norm (preconditioned) residual value (may be estimated)
!    dummy - optional user-defined monitor context (unused here)
!
  subroutine MyKSPMonitor(ksp, n, rnorm, vf, ierr)
    use petscksp
    implicit none

    KSP ksp
    Vec x
    PetscErrorCode ierr
    PetscInt n
    PetscViewerAndFormat vf
    PetscMPIInt rank
    PetscReal rnorm

!  Build the solution vector
    PetscCallA(KSPBuildSolution(ksp, PETSC_NULL_VEC, x, ierr))
    PetscCallA(KSPMonitorTrueResidual(ksp, n, rnorm, vf, ierr))

!  Write the solution vector and residual norm to stdout
!  Since the Fortran IO may be flushed differently than C
!  cannot reliably print both together in CI

    PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
    if (rank == 0) write (6, 100) n
!      PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr))
    if (rank == 0) write (6, 200) n, rnorm

100 format('iteration ', i5, ' solution vector:')
200 format('iteration ', i5, ' residual norm ', e11.4)
    ierr = 0
  end

! --------------------------------------------------------------
!
!  MyKSPConverged - This is a user-defined routine for testing
!  convergence of the KSP iterative solvers.
!
!  Input Parameters:
!    ksp   - iterative context
!    n     - iteration number
!    rnorm - 2-norm (preconditioned) residual value (may be estimated)
!    dummy - optional user-defined monitor context (unused here)
!
  subroutine MyKSPConverged(ksp, n, rnorm, flag, dummy, ierr)
    use petscksp
    implicit none

    KSP ksp
    PetscErrorCode ierr
    PetscInt n, dummy
    KSPConvergedReason flag
    PetscReal rnorm

    if (rnorm <= .05) then
      flag = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS
    else
      flag = KSP_CONVERGED_ITERATING
    end if
    ierr = 0

  end

!/*TEST
!
!   test:
!      nsize: 2
!      filter: sort -b
!      args: -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always
!
!   test:
!      suffix: 2
!      nsize: 2
!      filter: sort -b
!      args: -pc_type jacobi -my_ksp_monitor -ksp_gmres_cgs_refinement_type refine_always
!   test:
!      suffix: 3
!      nsize: 2
!
!
!TEST*/


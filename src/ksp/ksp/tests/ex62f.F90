!
!   Solves a linear system in parallel with KSP.  Also indicates
!   use of a user-provided preconditioner.  Input parameters include:
!
!

!
!  -------------------------------------------------------------------------
#include <petsc/finclude/petscksp.h>
module ex62fmodule
  use petscksp
  implicit none
  PC jacobi, sor
  Vec work
  Mat matwork
  PetscInt matapply_calls, matapplyrichardson_calls

contains
!/***********************************************************************/
!/*          Routines for a user-defined shell preconditioner           */
!/***********************************************************************/

!
!   SampleShellPCSetUp - This routine sets up a user-defined
!   preconditioner context.
!
!   Input Parameters:
!   pc    - preconditioner object
!   x     - vector
!
!   Output Parameter:
!   ierr  - error code (nonzero if error has been detected)
!
!   Notes:
!   In this example, we define the shell preconditioner to be Jacobi
!   method.  Thus, here we create a work vector for storing the reciprocal
!   of the diagonal of the matrix used to compute the preconditioner; this vector is then
!   used within the routine SampleShellPCApply().
!
  subroutine SampleShellPCSetUp(pc, x, ierr)

    PC pc
    Vec x
    Mat pmat
    PetscErrorCode ierr

    PetscCallA(PCGetOperators(pc, PETSC_NULL_MAT, pmat, ierr))
    PetscCallA(PCCreate(PETSC_COMM_WORLD, jacobi, ierr))
    PetscCallA(PCSetType(jacobi, PCJACOBI, ierr))
    PetscCallA(PCSetOperators(jacobi, pmat, pmat, ierr))
    PetscCallA(PCSetUp(jacobi, ierr))

    PetscCallA(PCCreate(PETSC_COMM_WORLD, sor, ierr))
    PetscCallA(PCSetType(sor, PCSOR, ierr))
    PetscCallA(PCSetOperators(sor, pmat, pmat, ierr))
!      PetscCallA(PCSORSetSymmetric(sor,SOR_LOCAL_SYMMETRIC_SWEEP,ierr))
    PetscCallA(PCSetUp(sor, ierr))

    PetscCallA(VecDuplicate(x, work, ierr))
  end

! -------------------------------------------------------------------
!
!   SampleShellPCApply - This routine demonstrates the use of a
!   user-provided preconditioner.
!
!   Input Parameters:
!   pc - preconditioner object
!   x - input vector
!
!   Output Parameters:
!   y - preconditioned vector
!   ierr  - error code (nonzero if error has been detected)
!
!   Notes:
!   This code implements the Jacobi preconditioner plus the
!   SOR preconditioner
!
! YOU CAN GET THE EXACT SAME EFFECT WITH THE PCCOMPOSITE preconditioner using
! mpiexec -n 1 ex21f -ksp_monitor -pc_type composite -pc_composite_pcs jacobi,sor -pc_composite_type additive
!
  subroutine SampleShellPCApply(pc, x, y, ierr)

    PC pc
    Vec x, y
    PetscErrorCode ierr
    PetscScalar, parameter :: one = 1.0

    PetscCallA(PCApply(jacobi, x, y, ierr))
    PetscCallA(PCApply(sor, x, work, ierr))
    PetscCallA(VecAXPY(y, one, work, ierr))
  end

! -------------------------------------------------------------------
!
!   SampleShellPCMatApply - Block analog of SampleShellPCApply(), the
!   preconditioner is applied to a whole block of vectors at once
!
!   Input Parameters:
!   pc - preconditioner object
!   x - input block of vectors
!
!   Output Parameters:
!   y - preconditioned block of vectors
!   ierr  - error code (nonzero if error has been detected)
!
  subroutine SampleShellPCMatApply(pc, x, y, ierr)

    PC pc
    Mat x, y
    PetscErrorCode ierr
    PetscScalar, parameter :: one = 1.0

    PetscCallA(PCMatApply(jacobi, x, y, ierr))
    PetscCallA(PCMatApply(sor, x, matwork, ierr))
    PetscCallA(MatAXPY(y, one, matwork, SAME_NONZERO_PATTERN, ierr))
    matapply_calls = matapply_calls + 1
  end

! -------------------------------------------------------------------
!
!   SampleShellPCMatApplyRichardson - Richardson iteration on a whole block
!   of right-hand sides, x <- x + M (b - A x) with M the preconditioner of
!   SampleShellPCMatApply(), so that it computes the same solutions as the
!   block iteration of KSPRICHARDSON
!
!   Input Parameters:
!   pc        - preconditioner object
!   b         - block of right-hand sides
!   x         - block of initial guesses
!   w         - block of work vectors, KSPRICHARDSON passes a null Mat so the work blocks are created here
!   rtol      - relative tolerance, unused since the iteration is not tested for convergence
!   abstol    - absolute tolerance, unused
!   dtol      - divergence tolerance, unused
!   maxits    - number of iterations to perform
!   guesszero - PETSC_TRUE if x is zero on entry
!
!   Output Parameters:
!   x      - block of solutions
!   outits - number of iterations performed
!   reason - why the iteration stopped
!   ierr   - error code (nonzero if error has been detected)
!
  subroutine SampleShellPCMatApplyRichardson(pc, b, x, w, rtol, abstol, dtol, maxits, guesszero, outits, reason, ierr)

    PC pc
    Mat b, x, w
    PetscReal rtol, abstol, dtol
    PetscInt maxits, outits
    PetscBool guesszero
    PCRichardsonConvergedReason reason
    PetscErrorCode ierr
    Mat amat, r, z
    PetscInt i
    PetscScalar, parameter :: one = 1.0, neg_one = -1.0

    PetscCallA(PCGetOperators(pc, amat, PETSC_NULL_MAT, ierr))
    PetscCallA(MatDuplicate(b, MAT_DO_NOT_COPY_VALUES, z, ierr))
    PetscCallA(MatDuplicate(b, MAT_DO_NOT_COPY_VALUES, r, ierr))
!   set the product A x up once so that only its numeric phase is run in the loop below
    PetscCallA(MatProductCreateWithMat(amat, x, PETSC_NULL_MAT, r, ierr))
    PetscCallA(MatProductSetType(r, MATPRODUCT_AB, ierr))
    PetscCallA(MatProductSetFromOptions(r, ierr))
    PetscCallA(MatProductSymbolic(r, ierr))
    do i = 1, maxits
      if (i == 1 .and. guesszero) then
        PetscCallA(MatCopy(b, r, SAME_NONZERO_PATTERN, ierr))
      else
        PetscCallA(MatProductNumeric(r, ierr))
        PetscCallA(MatAYPX(r, neg_one, b, SAME_NONZERO_PATTERN, ierr))
      end if
      PetscCallA(PCMatApply(jacobi, r, z, ierr))
      PetscCallA(PCMatApply(sor, r, matwork, ierr))
      PetscCallA(MatAXPY(z, one, matwork, SAME_NONZERO_PATTERN, ierr))
      PetscCallA(MatAXPY(x, one, z, SAME_NONZERO_PATTERN, ierr))
    end do
    PetscCallA(MatProductClear(r, ierr))
    PetscCallA(MatDestroy(r, ierr))
    PetscCallA(MatDestroy(z, ierr))
    outits = maxits
    reason = PCRICHARDSON_CONVERGED_ITS
    matapplyrichardson_calls = matapplyrichardson_calls + 1
  end

end module

program main
  use ex62fmodule
  implicit none

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
!     norm     - norm of solution error

  Vec x, b, u
  Mat A, blockb, blockx, blocky
  PC pc, blockpc
  KSP ksp, blockksp
  PetscScalar v
  PetscScalar, parameter :: one = 1.0, neg_one = -1.0
  PetscReal, parameter :: tol = 1e-7
  PetscReal norm, blocknorm
  PetscInt i, j, II, JJ, Istart, Iend, its
  PetscInt m, n, mloc, nrhs, nits
  PetscMPIInt rank
  PetscBool flg, blocksolve
  PetscErrorCode ierr

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(ierr))
  m = 8
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-m', m, flg, ierr))
  n = 7
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))
  blocksolve = PETSC_FALSE
  PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-matsolve', blocksolve, flg, ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))

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

  do II = Istart, Iend - 1
    v = -1.0
    i = II/n
    j = II - i*n
    if (i > 0) then
      JJ = II - n
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    if (i < m - 1) then
      JJ = II + n
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    if (j > 0) then
      JJ = II - 1
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    if (j < n - 1) then
      JJ = II + 1
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    v = 4.0
    PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [II], [v], ADD_VALUES, ierr))
  end do

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

  PetscCallA(VecCreateFromOptions(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 1_PETSC_INT_KIND, PETSC_DECIDE, m*n, u, ierr))
  PetscCallA(VecDuplicate(u, b, ierr))
  PetscCallA(VecDuplicate(b, x, ierr))

!  Set exact solution; then compute right-hand-side vector.

  PetscCallA(VecSet(u, one, ierr))
  PetscCallA(MatMult(A, u, b, ierr))

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

  PetscCallA(KSPGetPC(ksp, pc, ierr))
  PetscCallA(KSPSetTolerances(ksp, tol, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, PETSC_CURRENT_INTEGER, ierr))

!
!  Set a user-defined shell preconditioner
!

! (Required) Indicate to PETSc that we are using a shell preconditioner
  PetscCallA(PCSetType(pc, PCSHELL, ierr))

! (Required) Set the user-defined routine for applying the preconditioner
  PetscCallA(PCShellSetApply(pc, SampleShellPCApply, ierr))

! (Optional) Do any setup required for the preconditioner
!    Note: if you use PCShellSetSetUp, this will be done for your
  PetscCallA(SampleShellPCSetUp(pc, x, ierr))

! Set runtime options, e.g.,
!     -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
! These options will override those specified above as long as
! KSPSetFromOptions() is called _after_ any other customization
! routines.

  PetscCallA(KSPSetFromOptions(ksp, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(KSPSolve(ksp, b, x, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

! Check the error
  PetscCallA(VecAXPY(x, neg_one, u, ierr))
  PetscCallA(VecNorm(x, NORM_2, norm, ierr))
  PetscCallA(KSPGetIterationNumber(ksp, its, ierr))

  if (rank == 0) then
    write (6, 100) norm, its
  end if
100 format('Norm of error ', 1pe11.4, ' iterations ', i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!       Solve with a block of right-hand sides using the block
!       callbacks of the shell preconditioner
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  if (blocksolve) then
    nrhs = 3
    nits = 5
    matapply_calls = 0
    matapplyrichardson_calls = 0

    PetscCallA(MatGetLocalSize(A, mloc, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatCreateDense(PETSC_COMM_WORLD, mloc, PETSC_DECIDE, PETSC_DECIDE, nrhs, PETSC_NULL_SCALAR_ARRAY, blockb, ierr))
    PetscCallA(MatSetRandom(blockb, PETSC_NULL_RANDOM, ierr))
    PetscCallA(MatDuplicate(blockb, MAT_DO_NOT_COPY_VALUES, blockx, ierr))
    PetscCallA(MatDuplicate(blockb, MAT_DO_NOT_COPY_VALUES, blocky, ierr))
    PetscCallA(MatDuplicate(blockb, MAT_DO_NOT_COPY_VALUES, matwork, ierr))

!   A fixed number of Richardson iterations, so that the two block solves below perform the same operations
    PetscCallA(KSPCreate(PETSC_COMM_WORLD, blockksp, ierr))
    PetscCallA(KSPSetOperators(blockksp, A, A, ierr))
    PetscCallA(KSPSetType(blockksp, KSPRICHARDSON, ierr))
    PetscCallA(KSPSetNormType(blockksp, KSP_NORM_NONE, ierr))
    PetscCallA(KSPSetTolerances(blockksp, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, nits, ierr))
    PetscCallA(KSPGetPC(blockksp, blockpc, ierr))
    PetscCallA(PCSetType(blockpc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(blockpc, SampleShellPCApply, ierr))
    PetscCallA(PCShellSetMatApply(blockpc, SampleShellPCMatApply, ierr))

!   Without PCMatApplyRichardson(), the block iteration of KSPRICHARDSON calls PCMatApply() once per iteration
    PetscCallA(KSPMatSolve(blockksp, blockb, blockx, ierr))

!   With PCMatApplyRichardson(), KSPRICHARDSON delegates the whole block iteration to the preconditioner
    PetscCallA(PCShellSetMatApplyRichardson(blockpc, SampleShellPCMatApplyRichardson, ierr))
    PetscCallA(KSPMatSolve(blockksp, blockb, blocky, ierr))

    PetscCallA(MatAXPY(blocky, neg_one, blockx, SAME_NONZERO_PATTERN, ierr))
    PetscCallA(MatNorm(blocky, NORM_FROBENIUS, blocknorm, ierr))
    if (rank == 0) then
      write (6, 120) matapply_calls, matapplyrichardson_calls
      if (blocknorm < 1.e-12) then
        write (6, 130)
      else
        write (6, 140)
      end if
    end if

    PetscCallA(KSPDestroy(blockksp, ierr))
    PetscCallA(MatDestroy(matwork, ierr))
    PetscCallA(MatDestroy(blocky, ierr))
    PetscCallA(MatDestroy(blockx, ierr))
    PetscCallA(MatDestroy(blockb, ierr))
  end if
120 format('SampleShellPCMatApply() calls ', i5, ' SampleShellPCMatApplyRichardson() calls ', i5)
130 format('KSPMatSolve() with and without PCMatApplyRichardson() agree')
140 format('KSPMatSolve() with and without PCMatApplyRichardson() disagree')

! Free work space.  All PETSc objects should be destroyed when they
! are no longer needed.
  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(VecDestroy(u, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(b, ierr))
  PetscCallA(MatDestroy(A, ierr))

! Free up PCShell data
  PetscCallA(PCDestroy(sor, ierr))
  PetscCallA(PCDestroy(jacobi, ierr))
  PetscCallA(VecDestroy(work, ierr))

! Always call PetscFinalize() before exiting a program.
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!     requires: !single
!
!   test:
!     suffix: matsolve
!     requires: !single
!     args: -matsolve
!
!TEST*/

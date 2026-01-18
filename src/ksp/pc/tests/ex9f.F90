!
!   Description: Tests PCFieldSplitGetIS and PCFieldSplitSetIS from Fortran.
!
#include <petsc/finclude/petscksp.h>
program main
  use petscksp
  implicit none

  Vec x, b, u
  Mat A
  KSP ksp
  PC pc
  PetscReal norm
  PetscErrorCode ierr
  PetscInt i, n, col(3), its
  PetscInt :: nksp
  PetscBool flg
  PetscMPIInt size
  PetscScalar, parameter :: one = 1.0, none = -1.0
  PetscScalar value(3)
  KSP, pointer :: subksp(:)

  IS isin, isout

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
  PetscCheckA(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, 'This is a uniprocessor example only')
  n = 10
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Compute the matrix and right-hand-side vector that define
!         the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create matrix.  When using MatCreate(), the matrix format can
!  be specified at runtime.

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n, ierr))
  PetscCallA(MatSetFromOptions(A, ierr))
  PetscCallA(MatSetUp(A, ierr))

!  Assemble matrix.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array "col").

  value = [-1.0, 2.0, -1.0]
  do i = 1, n - 2
    col(1) = i - 1
    col(2) = i
    col(3) = i + 1
    PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [i], 3_PETSC_INT_KIND, col, value, INSERT_VALUES, ierr))
  end do
  i = n - 1
  col(1) = n - 2
  col(2) = n - 1
  PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [i], 2_PETSC_INT_KIND, col, value, INSERT_VALUES, ierr))
  i = 0
  col(1) = 0
  col(2) = 1
  value(1) = 2.0
  value(2) = -1.0
  PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [i], 2_PETSC_INT_KIND, col, value, INSERT_VALUES, ierr))
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

!  Create vectors.  Note that we form 1 vector from scratch and
!  then duplicate as needed.

  PetscCallA(VecCreate(PETSC_COMM_WORLD, x, ierr))
  PetscCallA(VecSetSizes(x, PETSC_DECIDE, n, ierr))
  PetscCallA(VecSetFromOptions(x, ierr))
  PetscCallA(VecDuplicate(x, b, ierr))
  PetscCallA(VecDuplicate(x, u, ierr))

!  Set exact solution; then compute right-hand-side vector.

  PetscCallA(VecSet(u, one, ierr))
  PetscCallA(MatMult(A, u, b, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!          Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

  PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))

!  Set operators. Here the matrix that defines the linear system
!  also serves as the matrix used to construct the preconditioner.

  PetscCallA(KSPSetOperators(ksp, A, A, ierr))

!  Set linear solver defaults for this problem (optional).
!   - By extracting the KSP and PC contexts from the KSP context,
!     we can then directly call any KSP and PC routines
!     to set various options.
!   - The following four statements are optional; all of these
!     parameters could alternatively be specified at runtime via
!     KSPSetFromOptions()

  PetscCallA(KSPGetPC(ksp, pc, ierr))
  PetscCallA(PCSetType(pc, PCFIELDSPLIT, ierr))
  PetscCallA(ISCreateStride(PETSC_COMM_SELF, n, 0_PETSC_INT_KIND, 1_PETSC_INT_KIND, isin, ierr))
  PetscCallA(PCFieldSplitSetIS(pc, 'splitname', isin, ierr))
  PetscCallA(PCFieldSplitGetIS(pc, 'splitname', isout, ierr))
  PetscCheckA(isin == isout, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCFieldSplitGetIS() failed')

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

  PetscCallA(KSPSetFromOptions(ksp, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(PCSetUp(pc, ierr))
  PetscCallA(PCFieldSplitGetSubKSP(pc, nksp, subksp, ierr))
  PetscCheckA(nksp == 2, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Number of KSP should be two')
  PetscCallA(KSPView(subksp(1), PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PCFieldSplitRestoreSubKSP(pc, nksp, subksp, ierr))

  PetscCallA(KSPSolve(ksp, b, x, ierr))

!  View solver info; we could instead use the option -ksp_view

  PetscCallA(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Check the error

  PetscCallA(VecAXPY(x, none, u, ierr))
  PetscCallA(VecNorm(x, NORM_2, norm, ierr))
  PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
  if (norm > 1.e-12) then
    write (6, 100) norm, its
  else
    write (6, 200) its
  end if
100 format('Norm of error ', e11.4, ',  Iterations = ', i5)
200 format('Norm of error < 1.e-12, Iterations = ', i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

  PetscCallA(ISDestroy(isin, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(u, ierr))
  PetscCallA(VecDestroy(b, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(PetscFinalize(ierr))

end

!/*TEST
!
!     test:
!       args: -ksp_monitor
!
!TEST*/

!
!     Creates a tridiagonal sparse matrix explicitly in Fortran and solves a linear system with it
!
!     The matrix is provided in triples in a way that supports new nonzero values with the same nonzero structure
!
program main
#include <petsc/finclude/petscksp.h>
  use petscksp
  implicit none

  PetscInt i, n
  PetscCount nz
  PetscBool flg
  PetscErrorCode ierr
  PetscScalar, ALLOCATABLE :: a(:)
  PetscScalar, pointer :: b(:)

  PetscInt, ALLOCATABLE :: rows(:)
  PetscInt, ALLOCATABLE :: cols(:)

  Mat J
  Vec rhs, solution
  KSP ksp

  PetscCallA(PetscInitialize(ierr))

  n = 3
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))
  nz = 3*n - 4

  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, n, rhs, ierr))
  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, n, solution, ierr))
  ALLOCATE (rows(nz), cols(nz), a(nz))

  PetscCallA(VecGetArray(rhs, b, ierr))
  do i = 1, n
    b(i) = 1.0
  end do
  PetscCallA(VecRestoreArray(rhs, b, ierr))

  rows(1) = 0; cols(1) = 0
  a(1) = 1.0
  do i = 2, n - 1
    rows(2 + 3*(i - 2)) = i - 1; cols(2 + 3*(i - 2)) = i - 2
    a(2 + 3*(i - 2)) = -1.0
    rows(2 + 3*(i - 2) + 1) = i - 1; cols(2 + 3*(i - 2) + 1) = i - 1
    a(2 + 3*(i - 2) + 1) = 2.0
    rows(2 + 3*(i - 2) + 2) = i - 1; cols(2 + 3*(i - 2) + 2) = i
    a(2 + 3*(i - 2) + 2) = -1.0
  end do
  rows(nz) = n - 1; cols(nz) = n - 1
  a(nz) = 1.0

  PetscCallA(MatCreate(PETSC_COMM_SELF, J, ierr))
  PetscCallA(MatSetSizes(J, n, n, n, n, ierr))
  PetscCallA(MatSetType(J, MATSEQAIJ, ierr))
  PetscCallA(MatSetPreallocationCOO(J, nz, rows, cols, ierr))
  PetscCallA(MatSetValuesCOO(J, a, INSERT_VALUES, ierr))

  PetscCallA(KSPCreate(PETSC_COMM_SELF, ksp, ierr))
  PetscCallA(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE, ierr))
  PetscCallA(KSPSetFromOptions(ksp, ierr))
  PetscCallA(KSPSetOperators(ksp, J, J, ierr))

  PetscCallA(KSPSolve(ksp, rhs, solution, ierr))

!     Keep the same size and nonzero structure of the matrix but change its numerical entries
  do i = 2, n - 1
    a(2 + 3*(i - 2) + 1) = 4.0
  end do
  PetscCallA(MatSetValuesCOO(J, a, INSERT_VALUES, ierr))

  PetscCallA(KSPSolve(ksp, rhs, solution, ierr))

  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(VecDestroy(rhs, ierr))
  PetscCallA(VecDestroy(solution, ierr))
  PetscCallA(MatDestroy(J, ierr))

  DEALLOCATE (rows, cols, a)

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!       requires: defined(PETSC_USE_SINGLE_LIBRARY)
!       nsize: 3
!       filter: sed 's?ATOL?RTOL?g' | grep -v HERMITIAN | grep -v "shared memory" | grep -v "Mat_0"
!       # use the MPI Linear Solver Server
!       args: -n 20 -mpi_linear_solver_server -mpi_linear_solver_server_view -mpi_linear_solver_server_use_shared_memory false
!       # controls for the use of PCMPI on a particular system
!       args: -mpi_linear_solver_server_minimum_count_per_rank 5 -mpi_linear_solver_server_ksp_view
!       # the usual options for the linear solver (in this case using the server)
!       args: -ksp_monitor -ksp_converged_reason -ksp_view
!
!     test:
!       suffix: 2
!       requires: defined(PETSC_USE_SINGLE_LIBRARY)
!       nsize: 3
!       filter: sed 's?ATOL?RTOL?g' | grep -v HERMITIAN | grep -v "shared memory" | grep -v "Mat_0"
!       # use the MPI Linear Solver Server
!       args: -n 20 -mpi_linear_solver_server -mpi_linear_solver_server_view -mpi_linear_solver_server_use_shared_memory false
!       # controls for the use of PCMPI on a particular system
!       args: -mpi_linear_solver_server_ksp_view
!       # the usual options for the linear solver (in this case using the server)
!       args: -ksp_monitor -ksp_converged_reason -ksp_view
!
!TEST*/

!
!     Creates a tridiagonal sparse matrix explicitly in Fortran and solves a linear system with it
!
!     The matrix is provided in CSR format by the user
!
program main
#include <petsc/finclude/petscksp.h>
  use petscksp
  implicit none

  PetscInt i, n, nz, cnt
  PetscBool flg
  PetscErrorCode ierr
  PetscScalar, pointer :: b(:)
  PetscInt :: zero

  PetscInt, pointer :: rowptr(:)
  PetscInt, pointer :: colind(:)
  PetscScalar, pointer :: a(:)

  Mat J
  Vec rhs, solution
  KSP ksp

  PetscCallA(PetscInitialize(ierr))

  zero = 0
  n = 3
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))
  nz = 3*n - 4

  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, n, rhs, ierr))
  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, n, solution, ierr))
  PetscCallA(PetscShmgetAllocateArrayInt(zero, n + 1, rowptr, ierr))
  PetscCallA(PetscShmgetAllocateArrayInt(zero, nz, colind, ierr))
  PetscCallA(PetscShmgetAllocateArrayScalar(zero, nz, a, ierr))

  PetscCallA(VecGetArray(rhs, b, ierr))
  do i = 1, n
    b(i) = 1.0
  end do
  PetscCallA(VecRestoreArray(rhs, b, ierr))

  rowptr(0) = 0
  colind(0) = 0
  a(0) = 1.0
  rowptr(1) = 1
  cnt = 1
  do i = 1, n - 2
    colind(cnt) = i - 1
    a(cnt) = -1
    cnt = cnt + 1
    colind(cnt) = i
    a(cnt) = 2
    cnt = cnt + 1
    colind(cnt) = i + 1
    a(cnt) = -1
    cnt = cnt + 1
    rowptr(i + 1) = 3 + rowptr(i)
  end do
  colind(cnt) = n - 1
  a(cnt) = 1.0
  rowptr(n) = cnt + 1

  PetscCallA(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, rowptr, colind, a, J, ierr))

  PetscCallA(KSPCreate(PETSC_COMM_SELF, ksp, ierr))
  PetscCallA(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE, ierr))
  PetscCallA(KSPSetFromOptions(ksp, ierr))
  PetscCallA(KSPSetOperators(ksp, J, J, ierr))

  PetscCallA(KSPSolve(ksp, rhs, solution, ierr))

!     Keep the same size and nonzero structure of the matrix but change its numerical entries
  do i = 2, n - 1
    a(2 + 3*(i - 2)) = 4.0
  end do
  PetscCallA(PetscObjectStateIncrease(J, ierr))

  PetscCallA(KSPSolve(ksp, rhs, solution, ierr))

  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(VecDestroy(rhs, ierr))
  PetscCallA(VecDestroy(solution, ierr))
  PetscCallA(MatDestroy(J, ierr))

  PetscCallA(PetscShmgetDeallocateArrayInt(rowptr, ierr))
  PetscCallA(PetscShmgetDeallocateArrayInt(colind, ierr))
  PetscCallA(PetscShmgetDeallocateArrayScalar(a, ierr))
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

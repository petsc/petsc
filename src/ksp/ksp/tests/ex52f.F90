!
#include <petsc/finclude/petscksp.h>
program main
  use petscksp
  implicit none
!
!  Demonstrates using MatFactorGetError() and MatFactorGetErrorZeroPivot()
!

  PetscErrorCode ierr
  PetscInt, parameter :: m = 2, n = 2
  PetscInt row, col
  Vec x, b
  Mat A, F
  KSP ksp
  PetscScalar, parameter :: two = 2.0, zero = 0.0
  KSPConvergedReason reason
  PCFailedReason pcreason
  PC pc
  MatFactorError ferr
  PetscReal pivot

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, m, n, m, n, ierr))
  PetscCallA(MatSetType(A, MATSEQAIJ, ierr))
  PetscCallA(MatSetUp(A, ierr))
  row = 0
  col = 0
  PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [row], 1_PETSC_INT_KIND, [col], [two], INSERT_VALUES, ierr))
  row = 1
  col = 1
  PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [row], 1_PETSC_INT_KIND, [col], [zero], INSERT_VALUES, ierr))
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

  PetscCallA(VecCreate(PETSC_COMM_WORLD, b, ierr))
  PetscCallA(VecSetSizes(b, m, m, ierr))
  PetscCallA(VecSetType(b, VECSEQ, ierr))

! Set up solution
  PetscCallA(VecDuplicate(b, x, ierr))

! Solve system
  PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
  PetscCallA(KSPSetOperators(ksp, A, A, ierr))
  PetscCallA(KSPSetFromOptions(ksp, ierr))
  PetscCallA(KSPSolve(ksp, b, x, ierr))
  PetscCallA(KSPGetConvergedReason(ksp, reason, ierr))
  PetscCallA(KSPGetPC(ksp, pc, ierr))
  PetscCallA(PCGetFailedReason(pc, pcreason, ierr))
  PetscCallA(PCFactorGetMatrix(pc, F, ierr))
  PetscCallA(MatFactorGetError(F, ferr, ierr))
  PetscCallA(MatFactorGetErrorZeroPivot(F, pivot, row, ierr))
  write (6, 101) ferr, pivot, row
101 format('MatFactorError ', i4, ' Pivot value ', 1pe9.2, ' row ', i4)

! Cleanup
  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(VecDestroy(b, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(MatDestroy(A, ierr))

  PetscCallA(PetscFinalize(ierr))
end

!  Nag compiler automatically turns on catching of floating point exceptions and prints message at
!  end of run about the exceptions seen
!
!/*TEST
!
!   test:
!     args: -fp_trap 0
!     filter: grep -v "Warning: Floating"
!
!TEST*/

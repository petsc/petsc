!
! This program is modified from a user's contribution.
! It illustrates how to PetscCallA(MUMPS's LU solver
!
#include <petsc/finclude/petscmat.h>
program main
  use petscmat
  implicit none

  Vec x, b, u
  Mat A, fact
  PetscInt i, j, II, JJ
  PetscInt Istart, Iend
  PetscInt m
  PetscBool wmumps
  PetscBool flg
  PetscScalar, parameter :: one = 1.0
  PetscScalar v
  IS perm, iperm
  PetscErrorCode ierr
  MatFactorInfo info

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))

  wmumps = PETSC_FALSE

  m = 10
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-m', m, flg, ierr))
  PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-use_mumps', wmumps, flg, ierr))

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m**2, m**2, ierr))
  PetscCallA(MatSetType(A, MATAIJ, ierr))
  PetscCallA(MatSetFromOptions(A, ierr))
  PetscCallA(MatSeqAIJSetPreallocation(A, 5_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(MatMPIAIJSetPreallocation(A, 5_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, 5_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, ierr))

  PetscCallA(MatGetOwnershipRange(A, Istart, Iend, ierr))

  do II = Istart, Iend - 1
    v = -1.0
    i = II/m
    j = II - i*m
    if (i > 0) then
      JJ = II - m
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], INSERT_VALUES, ierr))
    end if
    if (i < m - 1) then
      JJ = II + m
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], INSERT_VALUES, ierr))
    end if
    if (j > 0) then
      JJ = II - 1
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], INSERT_VALUES, ierr))
    end if
    if (j < m - 1) then
      JJ = II + 1
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], INSERT_VALUES, ierr))
    end if
    v = 4.0
    PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [II], [v], INSERT_VALUES, ierr))
  end do

  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

  PetscCallA(VecCreate(PETSC_COMM_WORLD, u, ierr))
  PetscCallA(VecSetSizes(u, PETSC_DECIDE, m*m, ierr))
  PetscCallA(VecSetFromOptions(u, ierr))
  PetscCallA(VecDuplicate(u, b, ierr))
  PetscCallA(VecDuplicate(b, x, ierr))
  PetscCallA(VecSet(u, one, ierr))
  PetscCallA(MatMult(A, u, b, ierr))

  PetscCallA(MatFactorInfoInitialize(info, ierr))
  PetscCallA(MatGetOrdering(A, MATORDERINGNATURAL, perm, iperm, ierr))
  if (wmumps) then
    write (*, *) 'use MUMPS LU...'
    PetscCallA(MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, fact, ierr))
  else
    write (*, *) 'use PETSc LU...'
    PetscCallA(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU, fact, ierr))
  end if
  PetscCallA(MatLUFactorSymbolic(fact, A, perm, iperm, info, ierr))
  PetscCallA(ISDestroy(perm, ierr))
  PetscCallA(ISDestroy(iperm, ierr))

  PetscCallA(MatLUFactorNumeric(fact, A, info, ierr))
  PetscCallA(MatSolve(fact, b, x, ierr))
  PetscCallA(MatDestroy(fact, ierr))

  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(VecDestroy(u, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(b, ierr))

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!
!   test:
!     suffix: 2
!     args: -use_mumps
!     requires: mumps
!
!TEST*/

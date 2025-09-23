!
!  Description: Demonstrates error handling with incorrect Fortran objects
!
! -----------------------------------------------------------------------

program main
#include <petsc/finclude/petscksp.h>
  use petscksp
  implicit none
  PetscErrorCode ierr
  PetscInt test
  KSP ksp

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))

  test = 1
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-test', test, PETSC_NULL_BOOL, ierr))
  if (test == 1) then
    PetscCallA(KSPSolve(PETSC_NULL_KSP, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr))
  else if (test == 2) then
    PetscCallA(KSPCreate(PETSC_COMM_WORLD, PETSC_NULL_KSP, ierr))
  else if (test == 3) then
    PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
    PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
  else if (test == 4) then
    PetscCallA(KSPDestroy(PETSC_NULL_KSP, ierr))
  end if

!     These should error but do not when ksp has not been created
!     PetscCallA(KSPSolve(ksp,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))
!     PetscCallA(KSPDestroy(ksp,ierr))

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      requires: defined(PETSC_USE_DEBUG) !defined(PETSCTEST_VALGRIND) defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE) !defined(PETSC_HAVE_SANITIZER)
!      args: -petsc_ci_portable_error_output -error_output_stdout -test 1
!      filter: grep -E "(PETSC ERROR)" | sed s"?KSPCREATE?kspcreate?" | sed s"?kspcreate_?kspcreate?"
!
!   test:
!      suffix: 2
!      requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE) !defined(PETSC_HAVE_SANITIZER)
!      args: -petsc_ci_portable_error_output -error_output_stdout -test 2
!      filter: grep -E "(PETSC ERROR)" | sed s"?KSPCREATE?kspcreate?" | sed s"?kspcreate_?kspcreate?"
!
!   test:
!      suffix: 3
!      requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE) !defined(PETSC_HAVE_SANITIZER)
!      args: -petsc_ci_portable_error_output -error_output_stdout -test 3
!      filter: grep -E "(PETSC ERROR)" | sed s"?KSPCREATE?kspcreate?" | sed s"?kspcreate_?kspcreate?"
!
!
!   test:
!      suffix: 4
!      requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE) !defined(PETSC_HAVE_SANITIZER)
!      args: -petsc_ci_portable_error_output -error_output_stdout -test 4
!      filter: grep -E "(PETSC ERROR)" | sed s"?KSPDESTROY?kspdestroy?" | sed s"?kspdestroy_?kspdestroy?"
!      output_file: output/empty.out
!
!TEST*/

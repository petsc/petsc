!
!     Demonstrates use of MatShellSetContext() and MatShellGetContext()
!
!     Contributed by:  Samuel Lanthaler
!
#include "petsc/finclude/petscmat.h"
module solver_context_ex6f
  use petscsys
  implicit none
  type :: MatCtx
    PetscReal :: lambda, kappa
    PetscReal :: h
  end type MatCtx
end module solver_context_ex6f

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
program main
  use petscmat
  use solver_context_ex6f
  implicit none

  Mat :: F
  type(MatCtx) :: ctxF
  type(MatCtx), pointer :: ctxF_pt
  PetscErrorCode :: ierr
  PetscInt :: n = 128

  PetscCallA(PetscInitialize(ierr))
  ctxF%lambda = 3.14d0
  PetscCallA(MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, ctxF, F, ierr))
  PetscCallA(MatShellSetContext(F, ctxF, ierr))
  print *, 'ctxF%lambda = ', ctxF%lambda

  PetscCallA(MatShellGetContext(F, ctxF_pt, ierr))
  print *, 'ctxF_pt%lambda = ', ctxF_pt%lambda

  PetscCallA(MatDestroy(F, ierr))
  PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!     build:
!       requires: double
!
!     test:
!
!TEST*/

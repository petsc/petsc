!
!     Demonstrates use of MatShellSetContext() and MatShellGetContext()
!
!     Contributed by:  Samuel Lanthaler
!
#include "petsc/finclude/petscmat.h"
MODULE solver_context_ex6f
  use petscsys
  IMPLICIT NONE
  TYPE :: MatCtx
    PetscReal :: lambda, kappa
    PetscReal :: h
  END TYPE MatCtx

! ----------------------------------------------------
  INTERFACE
    SUBROUTINE MatCreateShell(comm, mloc, nloc, m, n, ctx, mat, ierr)
      use petscmat
      import MatCtx
      implicit none
      MPI_Comm :: comm
      PetscInt :: mloc, nloc, m, n
      TYPE(MatCtx) :: ctx
      Mat :: mat
      PetscErrorCode :: ierr
    END SUBROUTINE MatCreateShell
! ----------------------------------------------------
    SUBROUTINE MatShellSetContext(mat, ctx, ierr)
      use petscmat
      import MatCtx
      implicit none
      MPI_Comm :: comm
      Mat :: mat
      TYPE(MatCtx) :: ctx
      PetscErrorCode :: ierr
    END SUBROUTINE MatShellSetContext
! ----------------------------------------------------
    SUBROUTINE MatShellGetContext(mat, ctx, ierr)
      use petscmat
      import MatCtx
      implicit none
      MPI_Comm :: comm
      Mat :: mat
      TYPE(MatCtx), POINTER :: ctx
      PetscErrorCode :: ierr
    END SUBROUTINE MatShellGetContext
  END INTERFACE

END MODULE solver_context_ex6f

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
PROGRAM main
  use petscmat
  USE solver_context_ex6f
  IMPLICIT NONE
  Mat :: F
  TYPE(MatCtx) :: ctxF
  TYPE(MatCtx), POINTER :: ctxF_pt
  PetscErrorCode :: ierr
  PetscInt :: n = 128

  PetscCallA(PetscInitialize(ierr))
  ctxF%lambda = 3.14d0
  PetscCallA(MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, ctxF, F, ierr))
  PetscCallA(MatShellSetContext(F, ctxF, ierr))
  PRINT *, 'ctxF%lambda = ', ctxF%lambda

  PetscCallA(MatShellGetContext(F, ctxF_pt, ierr))
  PRINT *, 'ctxF_pt%lambda = ', ctxF_pt%lambda

  PetscCallA(MatDestroy(F, ierr))
  PetscCallA(PetscFinalize(ierr))
END PROGRAM main

!/*TEST
!
!     build:
!       requires: double
!
!     test:
!
!TEST*/

!
!     Demonstrates use of MatShellSetContext() and MatShellGetContext()
!
!     Contributed by:  Samuel Lanthaler
!
MODULE solver_context_ex6f
#include "petsc/finclude/petsc.h"
  USE petscsys
  USE petscmat
  IMPLICIT NONE
  TYPE :: MatCtx
    PetscReal :: lambda, kappa
    PetscReal :: h
  END TYPE MatCtx
END MODULE solver_context_ex6f

MODULE solver_context_interfaces_ex6f
  USE solver_context_ex6f
  IMPLICIT NONE

! ----------------------------------------------------
  INTERFACE MatCreateShell
    SUBROUTINE MatCreateShell(comm, mloc, nloc, m, n, ctx, mat, ierr)
      USE solver_context_ex6f
      MPI_Comm :: comm
      PetscInt :: mloc, nloc, m, n
      TYPE(MatCtx) :: ctx
      Mat :: mat
      PetscErrorCode :: ierr
    END SUBROUTINE MatCreateShell
  END INTERFACE MatCreateShell
! ----------------------------------------------------

! ----------------------------------------------------
  INTERFACE MatShellSetContext
    SUBROUTINE MatShellSetContext(mat, ctx, ierr)
      USE solver_context_ex6f
      Mat :: mat
      TYPE(MatCtx) :: ctx
      PetscErrorCode :: ierr
    END SUBROUTINE MatShellSetContext
  END INTERFACE MatShellSetContext
! ----------------------------------------------------

! ----------------------------------------------------
  INTERFACE MatShellGetContext
    SUBROUTINE MatShellGetContext(mat, ctx, ierr)
      USE solver_context_ex6f
      Mat :: mat
      TYPE(MatCtx), POINTER :: ctx
      PetscErrorCode :: ierr
    END SUBROUTINE MatShellGetContext
  END INTERFACE MatShellGetContext

END MODULE solver_context_interfaces_ex6f

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
PROGRAM main
#include "petsc/finclude/petsc.h"
  USE solver_context_interfaces_ex6f
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

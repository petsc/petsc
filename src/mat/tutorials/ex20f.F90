!
!     Demonstrates use of MatDuplicate() for a shell matrix with a context
!
#include "petsc/finclude/petscmat.h"
MODULE ex20fmodule
  USE petscmat
  IMPLICIT NONE
  TYPE :: MatCtx
    PetscReal :: lambda
  END TYPE MatCtx

  interface
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

    SUBROUTINE MatShellSetContext(mat, ctx, ierr)
      use petscmat
      import MatCtx
      implicit none
      Mat :: mat
      TYPE(MatCtx) :: ctx
      PetscErrorCode :: ierr
    END SUBROUTINE MatShellSetContext

    SUBROUTINE MatShellGetContext(mat, ctx, ierr)
      use petscmat
      import MatCtx
      implicit none
      Mat :: mat
      TYPE(MatCtx), POINTER :: ctx
      PetscErrorCode :: ierr
    END SUBROUTINE MatShellGetContext
  end interface

contains
  SUBROUTINE MatDuplicate_F(F, opt, M, ierr)

    Mat                  :: F, M
    MatDuplicateOption   :: opt
    PetscErrorCode       :: ierr
    PetscInt             :: ml, nl
    TYPE(MatCtx), POINTER :: ctxM, ctxF_pt

    PetscCall(MatGetLocalSize(F, ml, nl, ierr))
    PetscCall(MatShellGetContext(F, ctxF_pt, ierr))
    allocate (ctxM)
    ctxM%lambda = ctxF_pt%lambda
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, ml, nl, PETSC_DETERMINE, PETSC_DETERMINE, ctxM, M, ierr))
!        PetscCall(MatShellSetOperation(M,MATOP_DUPLICATE,MatDuplicate_F,ierr))
    PetscCall(MatShellSetOperation(M, MATOP_DESTROY, MatDestroy_F, ierr))
  END SUBROUTINE MatDuplicate_F

  SUBROUTINE MatDestroy_F(F, ierr)

    Mat                  :: F
    PetscErrorCode       :: ierr
    TYPE(MatCtx), POINTER :: ctxF_pt
    PetscCall(MatShellGetContext(F, ctxF_pt, ierr))
    deallocate (ctxF_pt)
  END SUBROUTINE MatDestroy_F

END MODULE ex20fmodule

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
PROGRAM main
  use ex20fmodule
  implicit none
  Mat                  :: F, Fcopy
  TYPE(MatCtx)         :: ctxF
  TYPE(MatCtx), POINTER :: ctxF_pt, ctxFcopy_pt
  PetscErrorCode       :: ierr
  PetscInt             :: n = 128

  PetscCallA(PetscInitialize(ierr))
  ctxF%lambda = 3.14d0
  PetscCallA(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, ctxF, F, ierr))
  PetscCallA(MatShellSetOperation(F, MATOP_DUPLICATE, MatDuplicate_F, ierr))
  PRINT *, 'ctxF%lambda = ', ctxF%lambda

  PetscCallA(MatShellGetContext(F, ctxF_pt, ierr))
  PRINT *, 'ctxF_pt%lambda = ', ctxF_pt%lambda

  PetscCallA(MatDuplicate(F, MAT_DO_NOT_COPY_VALUES, Fcopy, ierr))
  PetscCallA(MatShellGetContext(Fcopy, ctxFcopy_pt, ierr))
  PRINT *, 'ctxFcopy_pt%lambda = ', ctxFcopy_pt%lambda

  PetscCallA(MatDestroy(F, ierr))
  PetscCallA(MatDestroy(Fcopy, ierr))
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

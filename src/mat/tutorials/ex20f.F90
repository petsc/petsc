!
!     Demonstrates use of MatDuplicate() for a shell matrix with a context
!
#include "petsc/finclude/petscmat.h"
module ex20fmodule
  use petscmat
  implicit none
  type :: MatCtx
    PetscReal :: lambda
  end type MatCtx

contains
  subroutine MatDuplicate_F(F, opt, M, ierr)

    Mat                  :: F, M
    MatDuplicateOption   :: opt
    PetscErrorCode       :: ierr
    PetscInt             :: ml, nl
    type(MatCtx), pointer :: ctxM, ctxF_pt

    PetscCall(MatGetLocalSize(F, ml, nl, ierr))
    PetscCall(MatShellGetContext(F, ctxF_pt, ierr))
    allocate (ctxM)
    ctxM%lambda = ctxF_pt%lambda
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, ml, nl, PETSC_DETERMINE, PETSC_DETERMINE, ctxM, M, ierr))
    PetscCall(MatShellSetOperation(M, MATOP_DESTROY, MatDestroy_F, ierr))
  end subroutine MatDuplicate_F

  subroutine MatDestroy_F(F, ierr)

    Mat                  :: F
    PetscErrorCode       :: ierr
    type(MatCtx), pointer :: ctxF_pt
    PetscCall(MatShellGetContext(F, ctxF_pt, ierr))
    deallocate (ctxF_pt)
  end subroutine MatDestroy_F

end module ex20fmodule

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
program main
  use ex20fmodule
  implicit none
  Mat                  :: F, Fcopy
  type(MatCtx)         :: ctxF
  type(MatCtx), pointer :: ctxF_pt, ctxFcopy_pt
  PetscErrorCode       :: ierr
  PetscInt             :: n = 128

  PetscCallA(PetscInitialize(ierr))
  ctxF%lambda = 3.14d0
  PetscCallA(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, ctxF, F, ierr))
  PetscCallA(MatShellSetOperation(F, MATOP_DUPLICATE, MatDuplicate_F, ierr))
  print *, 'ctxF%lambda = ', ctxF%lambda

  PetscCallA(MatShellGetContext(F, ctxF_pt, ierr))
  print *, 'ctxF_pt%lambda = ', ctxF_pt%lambda

  PetscCallA(MatDuplicate(F, MAT_DO_NOT_COPY_VALUES, Fcopy, ierr))
  PetscCallA(MatShellGetContext(Fcopy, ctxFcopy_pt, ierr))
  print *, 'ctxFcopy_pt%lambda = ', ctxFcopy_pt%lambda

  PetscCallA(MatDestroy(F, ierr))
  PetscCallA(MatDestroy(Fcopy, ierr))
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

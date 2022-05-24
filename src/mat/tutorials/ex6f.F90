!
!     Demonstrates use of MatShellSetContext() and MatShellGetContext()
!
!     Contributed by:  Samuel Lanthaler
!
     MODULE solver_context
#include "petsc/finclude/petsc.h"
       USE petscsys
       USE petscmat
       IMPLICIT NONE
       TYPE :: MatCtx
         PetscReal :: lambda,kappa
         PetscReal :: h
       END TYPE MatCtx
     END MODULE solver_context

     MODULE solver_context_interfaces
       USE solver_context
       IMPLICIT NONE

! ----------------------------------------------------
       INTERFACE MatCreateShell
         SUBROUTINE MatCreateShell(comm,mloc,nloc,m,n,ctx,mat,ierr)
           USE solver_context
           MPI_Comm :: comm
           PetscInt :: mloc,nloc,m,n
           TYPE(MatCtx) :: ctx
           Mat :: mat
           PetscErrorCode :: ierr
         END SUBROUTINE MatCreateShell
       END INTERFACE MatCreateShell
! ----------------------------------------------------

! ----------------------------------------------------
       INTERFACE MatShellSetContext
         SUBROUTINE MatShellSetContext(mat,ctx,ierr)
           USE solver_context
           Mat :: mat
           TYPE(MatCtx) :: ctx
           PetscErrorCode :: ierr
         END SUBROUTINE MatShellSetContext
       END INTERFACE MatShellSetContext
! ----------------------------------------------------

! ----------------------------------------------------
       INTERFACE MatShellGetContext
         SUBROUTINE MatShellGetContext(mat,ctx,ierr)
           USE solver_context
           Mat :: mat
           TYPE(MatCtx),  POINTER :: ctx
           PetscErrorCode :: ierr
         END SUBROUTINE MatShellGetContext
       END INTERFACE MatShellGetContext

     END MODULE solver_context_interfaces

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
     PROGRAM main
#include "petsc/finclude/petsc.h"
       USE solver_context_interfaces
       IMPLICIT NONE
       Mat :: F
       TYPE(MatCtx) :: ctxF
       TYPE(MatCtx),POINTER :: ctxF_pt
       PetscErrorCode :: ierr
       PetscInt :: n=128

       PetscCallA(PetscInitialize(ierr))
       ctxF%lambda = 3.14d0
       PetscCallA(MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,ctxF,F,ierr))
       PetscCallA(MatShellSetContext(F,ctxF,ierr))
       PRINT*,'ctxF%lambda = ',ctxF%lambda

       PetscCallA(MatShellGetContext(F,ctxF_pt,ierr))
       PRINT*,'ctxF_pt%lambda = ',ctxF_pt%lambda

       PetscCallA(MatDestroy(F,ierr))
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

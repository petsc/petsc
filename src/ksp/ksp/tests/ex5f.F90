!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
!
!      Solves a linear system matrix free
!

      Mat         A
      Vec         x,y
      PetscInt  m
      PetscErrorCode ierr
      KSP        ksp
      external    mymatmult
      PetscScalar one

      m = 10

      PetscCallA(PetscInitialize(ierr))
      one = 1.0
      PetscCallA(KSPCreate(PETSC_COMM_SELF,ksp,ierr))

      PetscCallA(MatCreateShell(PETSC_COMM_SELF,m,m,m,m,0,A,ierr))
      PetscCallA(MatShellSetOperation(A,MATOP_MULT,mymatmult,ierr))

      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,m,x,ierr))
      PetscCallA(VecDuplicate(x,y,ierr))
      PetscCallA(VecSet(x,one,ierr))

      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      PetscCallA(KSPSetFromOptions(ksp,ierr))

      PetscCallA(KSPSolve(ksp,x,y,ierr))

      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(y,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!  This is a bogus multiply that copies the vector. This corresponds to
!  an identity matrix A

      subroutine mymatmult(A,x,y,ierr)
      use petscksp
      implicit none

      Mat A
      Vec x,y
      PetscErrorCode ierr

      PetscCallA(VecCopy(x,y,ierr))

      return
      end

!/*TEST
!
!    test:
!      args:  -ksp_monitor_short
!
!TEST*/

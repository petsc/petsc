!
!   This program tests MatCreateVecs() for Shell Matrix
!
      subroutine mymatgetvecs(A,x,y,ierr)
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      Mat A
      Vec x,y
      PetscInt tw

      tw = 12
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,tw,x,ierr))
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,tw,y,ierr))
      return
      end

      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      Vec     x,y
      Mat     m
      PetscInt tw
      external  mymatgetvecs

      PetscCallA(PetscInitialize(ierr))

      tw = 12
      PetscCallA(MatCreateShell(PETSC_COMM_SELF,tw,tw,tw,tw,0,m,ierr))
      PetscCallA(MatAssemblyBegin(m,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(m,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatShellSetOperation(m,MATOP_CREATE_VECS,mymatgetvecs,ierr))
      PetscCallA(MatCreateVecs(m,x,y,ierr))
      PetscCallA(MatDestroy(m,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(y,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!
!TEST*/

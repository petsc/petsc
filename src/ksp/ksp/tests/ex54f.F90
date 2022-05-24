! Solve the system for (x,y,z):
!   x + y + z = 6
!   x - y + z = 2
!   x + y - z = 0
!   x + y + 2*z = 9    This equation is used if DMS=4 (else set DMS=3)
! => x=1 , y=2 , z=3

      program main
#include "petsc/finclude/petsc.h"
      use petsc
      implicit none

      PetscInt:: IR(1),IC(1),I,J,DMS=4 ! Set DMS=3 for a 3x3 squared system
      PetscErrorCode ierr;
      PetscReal :: MV(12),X(3),B(4),BI(1)
      Mat:: MTX
      Vec:: PTCB,PTCX
      KSP:: KK
      PetscInt one,three

      PetscCallA(PetscInitialize(ierr))
      one=1
      three=3
      PetscCallA(MatCreate(PETSC_COMM_WORLD,mtx,ierr))
      PetscCallA(MatSetSizes(mtx,PETSC_DECIDE,PETSC_DECIDE,DMS,three,ierr))
      PetscCallA(MatSetFromOptions(mtx,ierr))
      PetscCallA(MatSetUp(mtx,ierr))
      PetscCallA(MatSetOption(mtx,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE,ierr))
      MV=(/1.,1.,1.,1.,-1.,1.,1.,1.,-1.,1.,1.,2./)

      do J=1,3
         do I=1,DMS
            IR(1)=I-1; IC(1)=J-1; X(1)=MV(J+(I-1)*3)
            PetscCallA(MatSetValues(MTX,one,IR,one,IC,X,INSERT_VALUES,ierr))
         end do
      end do

      PetscCallA(MatAssemblyBegin(MTX,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(MTX,MAT_FINAL_ASSEMBLY,ierr))

      X=0.; B=(/6.,2.,0.,9./)
      PetscCallA(VecCreate(PETSC_COMM_WORLD,PTCB,ierr))   ! RHS vector
      PetscCallA(VecSetSizes(PTCB,PETSC_DECIDE,DMS,ierr))
      PetscCallA(VecSetFromOptions(PTCB,ierr))

      do I=1,DMS
         IR(1)=I-1
         BI(1)=B(i)
         PetscCallA(VecSetValues(PTCB,one,IR,BI,INSERT_VALUES,ierr))
      end do

      PetscCallA(vecAssemblyBegin(PTCB,ierr))
      PetscCallA(vecAssemblyEnd(PTCB,ierr))

      PetscCallA(VecCreate(PETSC_COMM_WORLD,PTCX,ierr))   ! Solution vector
      PetscCallA(VecSetSizes(PTCX,PETSC_DECIDE,three,ierr))
      PetscCallA(VecSetFromOptions(PTCX,ierr))
      PetscCallA(vecAssemblyBegin(PTCX,ierr))
      PetscCallA(vecAssemblyEnd(PTCX,ierr))

      PetscCallA(KSPCreate(PETSC_COMM_WORLD,KK,ierr))
      PetscCallA(KSPSetOperators(KK,MTX,MTX,ierr))
      PetscCallA(KSPSetFromOptions(KK,ierr))
      PetscCallA(KSPSetUp(KK,ierr))
      PetscCallA(KSPSolve(KK,PTCB,PTCX,ierr))
      PetscCallA(VecView(PTCX,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(MatDestroy(MTX,ierr))
      PetscCallA(KSPDestroy(KK,ierr))
      PetscCallA(VecDestroy(PTCB,ierr))
      PetscCallA(VecDestroy(PTCX,ierr))
      PetscCallA(PetscFinalize(ierr))
      end program main

!/*TEST
!     build:
!       requires: !complex
!     test:
!       args: -ksp_type cgls -pc_type none
!
!TEST*/

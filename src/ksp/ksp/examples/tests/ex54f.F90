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

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      one=1
      three=3
      call MatCreate(PETSC_COMM_WORLD,mtx,ierr)
      call MatSetSizes(mtx,PETSC_DECIDE,PETSC_DECIDE,DMS,three,ierr)
      call MatSetFromOptions(mtx,ierr)
      call MatSetUp(mtx,ierr)
      call MatSetOption(mtx,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE,ierr)
      MV=(/1.,1.,1.,1.,-1.,1.,1.,1.,-1.,1.,1.,2./)

      do J=1,3
         do I=1,DMS
            IR(1)=I-1; IC(1)=J-1; X(1)=MV(J+(I-1)*3)
            call MatSetValues(MTX,one,IR,one,IC,X,INSERT_VALUES,ierr)
         end do
      end do

      call MatAssemblyBegin(MTX,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(MTX,MAT_FINAL_ASSEMBLY,ierr)

      X=0.; B=(/6.,2.,0.,9./)
      call VecCreate(PETSC_COMM_WORLD,PTCB,ierr)   ! RHS vector
      call VecSetSizes(PTCB,PETSC_DECIDE,DMS,ierr)
      call VecSetFromOptions(PTCB,ierr)

      do I=1,DMS
         IR(1)=I-1
         BI(1)=B(i)
         call VecSetValues(PTCB,one,IR,BI,INSERT_VALUES,ierr)
      end do

      call vecAssemblyBegin(PTCB,ierr);
      call vecAssemblyEnd(PTCB,ierr)

      call VecCreate(PETSC_COMM_WORLD,PTCX,ierr)   ! Solution vector
      call VecSetSizes(PTCX,PETSC_DECIDE,three,ierr)
      call VecSetFromOptions(PTCX,ierr)
      call vecAssemblyBegin(PTCX,ierr);
      call vecAssemblyEnd(PTCX,ierr)

      call KSPCreate(PETSC_COMM_WORLD,KK,ierr)
      call KSPSetOperators(KK,MTX,MTX,ierr)
      call KSPSetFromOptions(KK,ierr)
      call KSPSetUp(KK,ierr);CHKERRA(ierr)
      call KSPSolve(KK,PTCB,PTCX,ierr)
      call VecView(PTCX,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatDestroy(MTX,ierr)
      call KSPDestroy(KK,ierr)
      call VecDestroy(PTCB,ierr)
      call VecDestroy(PTCX,ierr)
      call PetscFinalize(ierr)
      end program main

!/*TEST
!     build:
!       requires: !complex
!     test:
!       args: -ksp_type cgls -pc_type none
!
!TEST*/

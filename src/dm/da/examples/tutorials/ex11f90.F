!-----------------------------------------------------------------------
!
!    Tests DAGetVecGetArray()
!-----------------------------------------------------------------------
!

      IMPLICIT NONE

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscda.h90"
#include "finclude/petscviewer.h"

      Vec  g,l
      DA  da
      PetscScalar,pointer :: x1(:),x2(:,:)
      PetscScalar,pointer :: x3(:,:,:)
      PetscErrorCode ierr
      PetscInt m,n,p,dof,s,i,j,k,xs,xe,xl,ys,yl
      PetscInt zs,zl

      m = 5
      n = 6
      p = 4;
      dof = 1
      s = 1
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,m,dof,1,             &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DAGetGlobalVector(da,g,ierr)
      call DAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,       &
     &                  xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DAVecGetArrayF90(da,g,x1,ierr)
      do i=xs+1,xs+xl
         CHKMEMQ
         x1(i) = i
         CHKMEMQ
      enddo
      call DAVecRestoreArrayF90(da,g,x1,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(da,g,ierr)
      call DADestroy(da,ierr)

      call DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,      &
     &                m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,                 &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr)
      call DAGetGlobalVector(da,g,ierr)
      call DAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,                       &
     &                  xl,yl,PETSC_NULL_INTEGER,ierr)
      call DAVecGetArrayF90(da,g,x2,ierr)
      do i=xs+1,xs+xl
        do j=ys+1,ys+yl
           CHKMEMQ
           x2(i,j) = i + j
           CHKMEMQ
        enddo
      enddo
      call DAVecRestoreArrayF90(da,g,x2,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(da,g,ierr)
      call DADestroy(da,ierr)

      call DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,      &
     &                m,n,p,PETSC_DECIDE,PETSC_DECIDE,                     &
     &                PETSC_DECIDE,dof,s,                                  &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,               &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DAGetGlobalVector(da,g,ierr)
      call DAGetCorners(da,xs,ys,zs,                                       &
     &                  xl,yl,zl,ierr)
      call DAVecGetArrayF90(da,g,x3,ierr)
      do i=xs+1,xs+xl
        do j=ys+1,ys+yl
          do k=zs+1,zs+zl
            CHKMEMQ
            x3(i,j,k) = i + j + k
            CHKMEMQ
          enddo
        enddo
      enddo
      call DAVecRestoreArrayF90(da,g,x3,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(da,g,ierr)
      call DADestroy(da,ierr)

      CALL PetscFinalize(ierr)
      stop
      END PROGRAM 

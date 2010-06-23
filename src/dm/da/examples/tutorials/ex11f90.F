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
      PetscErrorCode ierr
      PetscInt m,n,dof,s,i,j

      m = 5
      n = 6
      dof = 1
      s = 1
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,m,dof,1,             &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DAGetGlobalVector(da,g,ierr)
      call DAVecGetArrayF90(da,g,x1,ierr)
      do i=1,m
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
      call DAVecGetArrayF90(da,g,x2,ierr)
      do i=1,m
        do j=1,n
           CHKMEMQ
           x2(i,j) = i + j
           CHKMEMQ
        enddo
      enddo
      call DAVecRestoreArrayF90(da,g,x2,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(da,g,ierr)
      call DADestroy(da,ierr)
      CALL PetscFinalize(ierr)
      stop
      END PROGRAM 

!-----------------------------------------------------------------------
!
!    Tests DAGetVecGetArray()
!-----------------------------------------------------------------------
!

!#define PETSC_USE_FORTRAN_MODULES 1
#include "finclude/petscsysdef.h"
#include "finclude/petscvecdef.h"
#include "finclude/petscdadef.h"
#if defined(PETSC_USE_FORTRAN_MODULES) || defined(PETSC_USE_FORTRAN_DATATYPES)
      use petsc
#endif
      implicit none
#if !defined(PETSC_USE_FORTRAN_MODULES) && !defined(PETSC_USE_FORTRAN_DATATYPES)
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscvec.h90"
#include "finclude/petscda.h90"
#include "finclude/petscviewer.h"
#endif

#if defined(PETSC_USE_FORTRAN_DATATYPES)
      Type(Vec)  g,l
      Type(DA)   ada
#else
      Vec  g,l
      DA  ada
#endif
      PetscScalar,pointer :: x1(:),x2(:,:)
      PetscScalar,pointer :: x3(:,:,:),x4(:,:,:,:)
      PetscErrorCode ierr
      PetscInt m,n,p,dof,s,i,j,k,xs,xe,xl
      PetscInt ys,yl
      PetscInt zs,zl

      m = 5
      n = 6
      p = 4;
      s = 1
      dof = 1
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,m,dof,1,             &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DAGetGlobalVector(ada,g,ierr)
      call DAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,       &
     &                  xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DAVecGetArrayF90(ada,g,x1,ierr)
      do i=xs,xs+xl-1
         CHKMEMQ
         x1(i) = i
         CHKMEMQ
      enddo
      call DAVecRestoreArrayF90(ada,g,x1,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(ada,g,ierr)
      call DADestroy(ada,ierr)

      call DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,      &
     &                m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,                 &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr)
      call DAGetGlobalVector(ada,g,ierr)
      call DAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,                       &
     &                  xl,yl,PETSC_NULL_INTEGER,ierr)
      call DAVecGetArrayF90(ada,g,x2,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
           CHKMEMQ
           x2(i,j) = i + j
           CHKMEMQ
        enddo
      enddo
      call DAVecRestoreArrayF90(ada,g,x2,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(ada,g,ierr)
      call DADestroy(ada,ierr)

      call DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,      &
     &                m,n,p,PETSC_DECIDE,PETSC_DECIDE,                     &
     &                PETSC_DECIDE,dof,s,                                  &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,               &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DAGetGlobalVector(ada,g,ierr)
      call DAGetCorners(ada,xs,ys,zs,                                       &
     &                  xl,yl,zl,ierr)
      call DAVecGetArrayF90(ada,g,x3,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
          do k=zs,zs+zl-1
            CHKMEMQ
            x3(i,j,k) = i + j + k
            CHKMEMQ
          enddo
        enddo
      enddo
      call DAVecRestoreArrayF90(ada,g,x3,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(ada,g,ierr)
      call DADestroy(ada,ierr)

!
!  Same tests but now with DOF > 1, so dimensions of array are one higher
!
      dof = 2
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,m,dof,1,             &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DAGetGlobalVector(ada,g,ierr)
      call DAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,       &
     &                  xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DAVecGetArrayF90(ada,g,x2,ierr)
      do i=xs,xs+xl-1
         CHKMEMQ
         x2(0,i) = i
         x2(1,i) = -i
         CHKMEMQ
      enddo
      call DAVecRestoreArrayF90(ada,g,x1,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(ada,g,ierr)
      call DADestroy(ada,ierr)

      dof = 2
      call DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,      &
     &                m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,                 &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr)
      call DAGetGlobalVector(ada,g,ierr)
      call DAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,                       &
     &                  xl,yl,PETSC_NULL_INTEGER,ierr)
      call DAVecGetArrayF90(ada,g,x3,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
           CHKMEMQ
           x3(0,i,j) = i + j
           x3(1,i,j) = -(i + j)
           CHKMEMQ
        enddo
      enddo
      call DAVecRestoreArrayF90(ada,g,x3,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(ada,g,ierr)
      call DADestroy(ada,ierr)

      dof = 3
      call DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,      &
     &                m,n,p,PETSC_DECIDE,PETSC_DECIDE,                     &
     &                PETSC_DECIDE,dof,s,                                  &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,               &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DAGetGlobalVector(ada,g,ierr)
      call DAGetCorners(ada,xs,ys,zs,                                       &
     &                  xl,yl,zl,ierr)
      call DAVecGetArrayF90(ada,g,x4,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
          do k=zs,zs+zl-1
            CHKMEMQ
            x4(0,i,j,k) = i + j + k
            x4(1,i,j,k) = -(i + j + k)
            x4(2,i,j,k) = i + j + k
            CHKMEMQ
          enddo
        enddo
      enddo
      call DAVecRestoreArrayF90(ada,g,x4,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DARestoreGlobalVector(ada,g,ierr)
      call DADestroy(ada,ierr)

      CALL PetscFinalize(ierr)
      stop
      END PROGRAM 

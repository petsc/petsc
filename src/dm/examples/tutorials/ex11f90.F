!-----------------------------------------------------------------------
!
!    Tests DMDAGetVecGetArray()
!-----------------------------------------------------------------------
!

!#define PETSC_USE_FORTRAN_MODULES 1
#include <finclude/petscsysdef.h>
#include <finclude/petscvecdef.h>
#include <finclude/petscdmdef.h>
#if defined(PETSC_USE_FORTRAN_MODULES) || defined(PETSC_USE_FORTRAN_DATATYPES)
      use petsc
#endif
      implicit none
#if !defined(PETSC_USE_FORTRAN_MODULES) && !defined(PETSC_USE_FORTRAN_DATATYPES)
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscvec.h90>
#include <finclude/petscdmda.h90>
#include <finclude/petscviewer.h>
#endif

#if defined(PETSC_USE_FORTRAN_DATATYPES)
      Type(Vec)  g
      Type(DM)   ada
#else
      Vec  g
      DM  ada
#endif
      PetscScalar,pointer :: x1(:),x2(:,:)
      PetscScalar,pointer :: x3(:,:,:),x4(:,:,:,:)
      PetscErrorCode ierr
      PetscInt m,n,p,dof,s,i,j,k,xs,xl
      PetscInt ys,yl
      PetscInt zs,zl

      m = 5
      n = 6
      p = 4;
      s = 1
      dof = 1
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,m,dof,1,             &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DMGetGlobalVector(ada,g,ierr)
      call DMDAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,       &
     &                  xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DMDAVecGetArrayF90(ada,g,x1,ierr)
      do i=xs,xs+xl-1
!         CHKMEMQ
         x1(i) = i
!         CHKMEMQ
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x1,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DMRestoreGlobalVector(ada,g,ierr)
      call DMDestroy(ada,ierr)

      call DMDACreate2d(PETSC_COMM_WORLD,                                   &
     &      DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,                      &
     &      DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,                 &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr)
      call DMGetGlobalVector(ada,g,ierr)
      call DMDAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,                       &
     &                  xl,yl,PETSC_NULL_INTEGER,ierr)
      call DMDAVecGetArrayF90(ada,g,x2,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
!           CHKMEMQ
           x2(i,j) = i + j
!           CHKMEMQ
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x2,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DMRestoreGlobalVector(ada,g,ierr)
      call DMDestroy(ada,ierr)

      call DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,                       &
     &     DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,                               &
     &     DMDA_STENCIL_BOX, m,n,p,PETSC_DECIDE,PETSC_DECIDE,                     &
     &                PETSC_DECIDE,dof,s,                                  &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,               &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DMGetGlobalVector(ada,g,ierr)
      call DMDAGetCorners(ada,xs,ys,zs,                                       &
     &                  xl,yl,zl,ierr)
      call DMDAVecGetArrayF90(ada,g,x3,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
          do k=zs,zs+zl-1
!            CHKMEMQ
            x3(i,j,k) = i + j + k
!            CHKMEMQ
          enddo
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x3,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DMRestoreGlobalVector(ada,g,ierr)
      call DMDestroy(ada,ierr)

!
!  Same tests but now with DOF > 1, so dimensions of array are one higher
!
      dof = 2
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,m,dof,1,             &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DMGetGlobalVector(ada,g,ierr)
      call DMDAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,       &
     &                  xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DMDAVecGetArrayF90(ada,g,x2,ierr)
      do i=xs,xs+xl-1
!         CHKMEMQ
         x2(0,i) = i
         x2(1,i) = -i
!         CHKMEMQ
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x1,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DMRestoreGlobalVector(ada,g,ierr)
      call DMDestroy(ada,ierr)

      dof = 2
      call DMDACreate2d(PETSC_COMM_WORLD,                               &
     &     DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,                       &
     &     DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,                 &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr)
      call DMGetGlobalVector(ada,g,ierr)
      call DMDAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,                       &
     &                  xl,yl,PETSC_NULL_INTEGER,ierr)
      call DMDAVecGetArrayF90(ada,g,x3,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
!           CHKMEMQ
           x3(0,i,j) = i + j
           x3(1,i,j) = -(i + j)
!           CHKMEMQ
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x3,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DMRestoreGlobalVector(ada,g,ierr)
      call DMDestroy(ada,ierr)

      dof = 3
      call DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,                 &
     &     DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,                            &
     &         DMDA_STENCIL_BOX,m,n,p,PETSC_DECIDE,PETSC_DECIDE,                     &
     &                PETSC_DECIDE,dof,s,                                  &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,               &
     &                PETSC_NULL_INTEGER,ada,ierr)
      call DMGetGlobalVector(ada,g,ierr)
      call DMDAGetCorners(ada,xs,ys,zs,                                       &
     &                  xl,yl,zl,ierr)
      call DMDAVecGetArrayF90(ada,g,x4,ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
          do k=zs,zs+zl-1
!            CHKMEMQ
            x4(0,i,j,k) = i + j + k
            x4(1,i,j,k) = -(i + j + k)
            x4(2,i,j,k) = i + j + k
!            CHKMEMQ
          enddo
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x4,ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call DMRestoreGlobalVector(ada,g,ierr)
      call DMDestroy(ada,ierr)

      CALL PetscFinalize(ierr)
      stop
      END PROGRAM

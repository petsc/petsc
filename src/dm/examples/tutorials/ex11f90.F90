      program main
!-----------------------------------------------------------------------
!
!    Tests DMDAGetVecGetArray()
!-----------------------------------------------------------------------
!

#include <petsc/finclude/petscdm.h>
      use petsc
      implicit none

      Type(tVec)  g
      Type(tDM)   ada

      PetscScalar,pointer :: x1(:),x2(:,:)
      PetscScalar,pointer :: x3(:,:,:),x4(:,:,:,:)
      PetscErrorCode ierr
      PetscInt m,n,p,dof,s,i,j,k,xs,xl
      PetscInt ys,yl
      PetscInt zs,zl,sw

      m = 5
      n = 6
      p = 4;
      s = 1
      dof = 1
      sw = 1
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m,dof,sw,PETSC_NULL_INTEGER,ada,ierr);CHKERRA(ierr)
      call DMSetUp(ada,ierr);CHKERRA(ierr)
      call DMGetGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
      call DMDAVecGetArrayF90(ada,g,x1,ierr);CHKERRA(ierr)
      do i=xs,xs+xl-1
!         CHKMEMQ
         x1(i) = i
!         CHKMEMQ
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x1,ierr);CHKERRA(ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDestroy(ada,ierr);CHKERRA(ierr)

      call DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,  &
     &     PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr);CHKERRA(ierr)
      call DMSetUp(ada,ierr);CHKERRA(ierr)
      call DMGetGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,xl,yl,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
      call DMDAVecGetArrayF90(ada,g,x2,ierr);CHKERRA(ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
!           CHKMEMQ
           x2(i,j) = i + j
!           CHKMEMQ
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x2,ierr);CHKERRA(ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDestroy(ada,ierr);CHKERRA(ierr)

      call DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX, m,n,p,PETSC_DECIDE,PETSC_DECIDE,                     &
     &                PETSC_DECIDE,dof,s,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr);CHKERRA(ierr)
      call DMSetUp(ada,ierr);CHKERRA(ierr)
      call DMGetGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDAGetCorners(ada,xs,ys,zs,xl,yl,zl,ierr);CHKERRA(ierr)
      call DMDAVecGetArrayF90(ada,g,x3,ierr);CHKERRA(ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
          do k=zs,zs+zl-1
!            CHKMEMQ
            x3(i,j,k) = i + j + k
!            CHKMEMQ
          enddo
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x3,ierr);CHKERRA(ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDestroy(ada,ierr);CHKERRA(ierr)

!
!  Same tests but now with DOF > 1, so dimensions of array are one higher
!
      dof = 2
      call DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m,dof,sw,PETSC_NULL_INTEGER,ada,ierr);CHKERRA(ierr)
      call DMSetUp(ada,ierr);CHKERRA(ierr)
      call DMGetGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
      call DMDAVecGetArrayF90(ada,g,x2,ierr);CHKERRA(ierr)
      do i=xs,xs+xl-1
!         CHKMEMQ
         x2(0,i) = i
         x2(1,i) = -i
!         CHKMEMQ
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x1,ierr);CHKERRA(ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDestroy(ada,ierr);CHKERRA(ierr)

      dof = 2
      call DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,   &
     &     PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr);CHKERRA(ierr)
      call DMSetUp(ada,ierr);CHKERRA(ierr)
      call DMGetGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,xl,yl,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
      call DMDAVecGetArrayF90(ada,g,x3,ierr);CHKERRA(ierr)
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
!           CHKMEMQ
           x3(0,i,j) = i + j
           x3(1,i,j) = -(i + j)
!           CHKMEMQ
        enddo
      enddo
      call DMDAVecRestoreArrayF90(ada,g,x3,ierr);CHKERRA(ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDestroy(ada,ierr);CHKERRA(ierr)

      dof = 3
      call DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,p,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,                &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ada,ierr);CHKERRA(ierr)
      call DMSetUp(ada,ierr);CHKERRA(ierr)
      call DMGetGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDAGetCorners(ada,xs,ys,zs,xl,yl,zl,ierr);CHKERRA(ierr)
      call DMDAVecGetArrayF90(ada,g,x4,ierr);CHKERRA(ierr)
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
      call DMDAVecRestoreArrayF90(ada,g,x4,ierr);CHKERRA(ierr)
      call VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(ada,g,ierr);CHKERRA(ierr)
      call DMDestroy(ada,ierr);CHKERRA(ierr)

      CALL PetscFinalize(ierr)
      END PROGRAM

!
!/*TEST
!
!   build:
!     requires: !complex
!
!   test:
!     filter: Error: grep -v "Vec Object" | grep -v "Warning: ieee_inexact is signaling"
!
!TEST*/

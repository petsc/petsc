!     Tests DMDAGetVecGetArray()

      program main
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscdmda.h>
      use petscdmda
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

      PetscInt nen,nel
      PetscInt, pointer :: elements(:)

      PetscInt nfields
      character(80), pointer :: namefields(:)
      IS, pointer :: isfields(:)
      DM, pointer :: dmfields(:)
      PetscInt zero, one

      m = 5
      n = 6
      p = 4
      s = 1
      dof = 1
      sw = 1
      zero = 0
      one = 1
      PetscCallA(PetscInitialize(ierr))
      PetscCallA(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m,dof,sw,PETSC_NULL_INTEGER_ARRAY,ada,ierr))
      PetscCallA(DMSetUp(ada,ierr))
      PetscCallA(DMGetGlobalVector(ada,g,ierr))
      PetscCallA(DMDAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      PetscCallA(DMDAVecGetArray(ada,g,x1,ierr))
      do i=xs,xs+xl-1
!         CHKMEMQ
         x1(i) = i
!         CHKMEMQ
      enddo
      PetscCallA(DMDAVecRestoreArray(ada,g,x1,ierr))
      PetscCallA(VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(DMRestoreGlobalVector(ada,g,ierr))
      PetscCallA(DMDestroy(ada,ierr))

      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,ada,ierr))
      PetscCallA(DMSetUp(ada,ierr))
      PetscCallA(DMGetGlobalVector(ada,g,ierr))
      PetscCallA(DMDAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,xl,yl,PETSC_NULL_INTEGER,ierr))
      PetscCallA(DMDAVecGetArray(ada,g,x2,ierr))
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
!           CHKMEMQ
           x2(i,j) = i + j
!           CHKMEMQ
        enddo
      enddo
      PetscCallA(DMDAVecRestoreArray(ada,g,x2,ierr))
      PetscCallA(VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(DMRestoreGlobalVector(ada,g,ierr))

      PetscCallA(DMDAGetElements(ada,nen,nel,elements,ierr))
      do i=1,nen*nel
         PetscCheckA(elements(i) .ge. 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting DMDA elements')
      enddo
      PetscCallA(DMDARestoreElements(ada,nen,nel,elements,ierr))
      PetscCallA(DMDestroy(ada,ierr))

      PetscCallA(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX, m,n,p,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,ada,ierr))
      PetscCallA(DMSetUp(ada,ierr))
      PetscCallA(DMGetGlobalVector(ada,g,ierr))
      PetscCallA(DMDAGetCorners(ada,xs,ys,zs,xl,yl,zl,ierr))
      PetscCallA(DMDAVecGetArray(ada,g,x3,ierr))
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
          do k=zs,zs+zl-1
!            CHKMEMQ
            x3(i,j,k) = i + j + k
!            CHKMEMQ
          enddo
        enddo
      enddo
      PetscCallA(DMDAVecRestoreArray(ada,g,x3,ierr))
      PetscCallA(VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(DMRestoreGlobalVector(ada,g,ierr))
      PetscCallA(DMDestroy(ada,ierr))

!
!  Same tests but now with DOF > 1, so dimensions of array are one higher
!
      dof = 2
      PetscCallA(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,m,dof,sw,PETSC_NULL_INTEGER_ARRAY,ada,ierr))
      PetscCallA(DMSetUp(ada,ierr))
      PetscCallA(DMGetGlobalVector(ada,g,ierr))
      PetscCallA(DMDAGetCorners(ada,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xl,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      PetscCallA(DMDAVecGetArray(ada,g,x2,ierr))
      do i=xs,xs+xl-1
!         CHKMEMQ
         x2(0,i) = i
         x2(1,i) = -i
!         CHKMEMQ
      enddo
      PetscCallA(DMDAVecRestoreArray(ada,g,x1,ierr))
      PetscCallA(VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(DMRestoreGlobalVector(ada,g,ierr))

      ! some testing unrelated to the example
      PetscCallA(DMDASetFieldName(ada,zero,'Field 0',ierr))
      PetscCallA(DMDASetFieldName(ada,one,'Field 1',ierr))
      PetscCallA(DMCreateFieldDecomposition(ada, nfields, namefields, PETSC_NULL_IS_POINTER, PETSC_NULL_DM_POINTER, ierr))
      ! print*,nfields,trim(namefields(1)),trim(namefields(2))
      PetscCallA(DMDestroyFieldDecomposition(ada, nfields, namefields, PETSC_NULL_IS_POINTER, PETSC_NULL_DM_POINTER, ierr))
      PetscCallA(DMCreateFieldDecomposition(ada, nfields, namefields, isfields, dmfields, ierr))
      PetscCallA(DMDestroyFieldDecomposition(ada, nfields, namefields, isfields, dmfields, ierr))

      PetscCallA(DMDestroy(ada,ierr))

      dof = 2
      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,ada,ierr))
      PetscCallA(DMSetUp(ada,ierr))
      PetscCallA(DMGetGlobalVector(ada,g,ierr))
      PetscCallA(DMDAGetCorners(ada,xs,ys,PETSC_NULL_INTEGER,xl,yl,PETSC_NULL_INTEGER,ierr))
      PetscCallA(DMDAVecGetArray(ada,g,x3,ierr))
      do i=xs,xs+xl-1
        do j=ys,ys+yl-1
!           CHKMEMQ
           x3(0,i,j) = i + j
           x3(1,i,j) = -(i + j)
!           CHKMEMQ
        enddo
      enddo
      PetscCallA(DMDAVecRestoreArray(ada,g,x3,ierr))
      PetscCallA(VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(DMRestoreGlobalVector(ada,g,ierr))
      PetscCallA(DMDestroy(ada,ierr))

      dof = 3
      PetscCallA(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,p,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,PETSC_NULL_INTEGER_ARRAY,ada,ierr))
      PetscCallA(DMSetUp(ada,ierr))
      PetscCallA(DMGetGlobalVector(ada,g,ierr))
      PetscCallA(DMDAGetCorners(ada,xs,ys,zs,xl,yl,zl,ierr))
      PetscCallA(DMDAVecGetArray(ada,g,x4,ierr))
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
      PetscCallA(DMDAVecRestoreArray(ada,g,x4,ierr))
      PetscCallA(VecView(g,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(DMRestoreGlobalVector(ada,g,ierr))
      PetscCallA(DMDestroy(ada,ierr))

      PetscCallA(PetscFinalize(ierr))
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

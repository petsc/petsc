      program main              !   Solves the linear system  J x = f
#include <petsc/finclude/petscdef.h>
      use petscksp; use petscdm
      implicit none
      Vec x,f
      Mat J
      DM da
      KSP ksp
      PetscErrorCode ierr
      PetscInt eight,one

      eight = 8
      one = 1
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call DMDACreate1d(MPI_COMM_WORLD,DM_BOUNDARY_NONE,eight,one,one,PETSC_NULL_INTEGER,da,ierr);CHKERRQ(ierr)
      call DMSetFromOptions(da,ierr)
      call DMSetUp(da,ierr)
      call DMCreateGlobalVector(da,x,ierr);CHKERRQ(ierr)
      call VecDuplicate(x,f,ierr);CHKERRQ(ierr)
      call DMSetMatType(da,MATAIJ,ierr);CHKERRQ(ierr)
      call DMCreateMatrix(da,J,ierr);CHKERRQ(ierr)

      call ComputeRHS(da,f,ierr);CHKERRQ(ierr)
      call ComputeMatrix(da,J,ierr);CHKERRQ(ierr)

      call KSPCreate(MPI_COMM_WORLD,ksp,ierr);CHKERRQ(ierr)
      call KSPSetOperators(ksp,J,J,ierr);CHKERRQ(ierr)
      call KSPSetFromOptions(ksp,ierr);CHKERRQ(ierr)
      call KSPSolve(ksp,f,x,ierr);CHKERRQ(ierr)

      call MatDestroy(J,ierr);CHKERRQ(ierr)
      call VecDestroy(x,ierr);CHKERRQ(ierr)
      call VecDestroy(f,ierr);CHKERRQ(ierr)
      call KSPDestroy(ksp,ierr);CHKERRQ(ierr)
      call DMDestroy(da,ierr);CHKERRQ(ierr)
      call PetscFinalize(ierr);CHKERRQ(ierr)
      end

      subroutine  ComputeRHS(da,x,ierr)
      use petscdm; use petscdmda
      implicit none
      DM da
      Vec x
      PetscErrorCode ierr
      PetscInt xs,xm,i,mx
      PetscScalar hx
      PetscScalar, pointer :: xx(:)
      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &     PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &     PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
      call DMDAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
      hx     = 1.0_PETSC_REAL_KIND/(mx-1)
      call DMDAVecGetArrayF90(da,x,xx,ierr);CHKERRQ(ierr)
      do i=xs,xs+xm-1
       xx(i) = i*hx
      enddo
      call DMDAVecRestoreArrayF90(da,x,xx,ierr);CHKERRQ(ierr)
      return
      end
      
      subroutine ComputeMatrix(da,J,ierr)
      use petscdm
      implicit none
      Mat J
      DM da
      PetscErrorCode ierr
      PetscInt xs,xm,i,mx
      PetscScalar hx,one

      one = 1.0
      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &  PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,   &
     &  PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
      call DMDAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);CHKERRQ(ierr)
      hx     = 1.0_PETSC_REAL_KIND/(mx-1)
      do i=xs,xs+xm-1
        if ((i .eq. 0) .or. (i .eq. mx-1)) then
          call MatSetValue(J,i,i,one,INSERT_VALUES,ierr);CHKERRQ(ierr)
        else
          call MatSetValue(J,i,i-1,-hx,INSERT_VALUES,ierr);CHKERRQ(ierr)
          call MatSetValue(J,i,i+1,-hx,INSERT_VALUES,ierr);CHKERRQ(ierr)
          call MatSetValue(J,i,i,2*hx,INSERT_VALUES,ierr);CHKERRQ(ierr)
        endif
      enddo
      call MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      call MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      return
      end

      program main   !   Solves the linear system  J x = f
#include "finclude/petscdef.h"
      use petscksp; use petscda
      Vec x,f; Mat J; DA da; KSP ksp; PetscErrorCode ierr
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,8,1,1,PETSC_NULL_INTEGER,da,ierr)
      call DACreateGlobalVector(da,x,ierr); call VecDuplicate(x,f,ierr)
      call DAGetMatrix(da,MATAIJ,J,ierr)

      call ComputeRHS(da,f,ierr)
      call ComputeMatrix(da,J,ierr)

      call KSPCreate(MPI_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,J,J,SAME_NONZERO_PATTERN,ierr)
      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,f,x,ierr)

      call MatDestroy(J,ierr); call VecDestroy(x,ierr); call VecDestroy(f,ierr)
      call KSPDestroy(ksp,ierr); call DADestroy(da,ierr)
      call PetscFinalize(ierr)
      end
      subroutine  ComputeRHS(da,x,ierr)
#include "finclude/petscdef.h"
      use petscda
      DA da; Vec x; PetscErrorCode ierr; PetscInt xs,xm,i,mx; PetscScalar hx; PetscScalar, pointer :: xx(:)
      call DAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      hx     = 1.d0/(mx-1)
      call VecGetArrayF90(x,xx,ierr)
      do i=xs,xs+xm-1
          xx(i) = i*hx
      enddo
      call VecRestoreArrayF90(x,xx,ierr)
      return 
      end
      subroutine ComputeMatrix(da,J,ierr)
#include "finclude/petscdef.h"
      use petscda
      Mat J; DA da; PetscErrorCode ierr; PetscInt xs,xm,i,mx; PetscScalar hx
      call DAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      call DAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
      hx     = 1.d0/(mx-1)
      do i=xs,xs+xm-1
        if ((i .eq. 0) .or. (i .eq. mx-1)) then
          call MatSetValue(J,i,i,1d0,INSERT_VALUES,ierr)
        else 
          call MatSetValue(J,i,i-1,-hx,INSERT_VALUES,ierr)
          call MatSetValue(J,i,i+1,-hx,INSERT_VALUES,ierr)
          call MatSetValue(J,i,i,2*hx,INSERT_VALUES,ierr)
        endif
      enddo
      call MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY,ierr); call MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY,ierr)
      return 
      end

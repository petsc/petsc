      program main              !   Solves the linear system  J x = f
#include <petsc/finclude/petsc.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscksp
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
      call DMDACreate1d(MPI_COMM_WORLD,DM_BOUNDARY_NONE,eight,one,one,PETSC_NULL_INTEGER,da,ierr);CHKERRA(ierr)
      call DMSetFromOptions(da,ierr)
      call DMSetUp(da,ierr)
      call DMCreateGlobalVector(da,x,ierr);CHKERRA(ierr)
      call VecDuplicate(x,f,ierr);CHKERRA(ierr)
      call DMSetMatType(da,MATAIJ,ierr);CHKERRA(ierr)
      call DMCreateMatrix(da,J,ierr);CHKERRA(ierr)

      call ComputeRHS(da,f,ierr);CHKERRA(ierr)
      call ComputeMatrix(da,J,ierr);CHKERRA(ierr)

      call KSPCreate(MPI_COMM_WORLD,ksp,ierr);CHKERRA(ierr)
      call KSPSetOperators(ksp,J,J,ierr);CHKERRA(ierr)
      call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
      call KSPSolve(ksp,f,x,ierr);CHKERRA(ierr)

      call MatDestroy(J,ierr);CHKERRA(ierr)
      call VecDestroy(x,ierr);CHKERRA(ierr)
      call VecDestroy(f,ierr);CHKERRA(ierr)
      call KSPDestroy(ksp,ierr);CHKERRA(ierr)
      call DMDestroy(da,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

! AVX512 crashes without this..
      block data init
      implicit none
      PetscScalar sd
      common /cb/ sd
      data sd /0/
      end
      subroutine knl_workaround(xx)
      implicit none
      PetscScalar xx
      PetscScalar sd
      common /cb/ sd
      sd = sd+xx
      end

      subroutine  ComputeRHS(da,x,ierr)
      use petscdmda
      implicit none
      DM da
      Vec x
      PetscErrorCode ierr
      PetscInt xs,xm,i,mx
      PetscScalar hx
      PetscScalar, pointer :: xx(:)
      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &                 PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &                 PETSC_NULL_INTEGER,ierr);PetscCall(ierr)
      call DMDAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);PetscCall(ierr)
      hx     = 1.0_PETSC_REAL_KIND/(mx-1)
      call DMDAVecGetArrayF90(da,x,xx,ierr);PetscCall(ierr)
      do i=xs,xs+xm-1
        call knl_workaround(xx(i))
        xx(i) = i*hx
      enddo
      call DMDAVecRestoreArrayF90(da,x,xx,ierr);PetscCall(ierr)
      return
      end

      subroutine ComputeMatrix(da,J,ierr)
      use petscdm
      use petscmat
      implicit none
      Mat J
      DM da
      PetscErrorCode ierr
      PetscInt xs,xm,i,mx
      PetscScalar hx,one

      one = 1.0
      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &                 PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,   &
     &                 PETSC_NULL_INTEGER,ierr);PetscCall(ierr)
      call DMDAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr);PetscCall(ierr)
      hx     = 1.0_PETSC_REAL_KIND/(mx-1)
      do i=xs,xs+xm-1
        if ((i .eq. 0) .or. (i .eq. mx-1)) then
          call MatSetValue(J,i,i,one,INSERT_VALUES,ierr);PetscCall(ierr)
        else
          call MatSetValue(J,i,i-1,-hx,INSERT_VALUES,ierr);PetscCall(ierr)
          call MatSetValue(J,i,i+1,-hx,INSERT_VALUES,ierr);PetscCall(ierr)
          call MatSetValue(J,i,i,2*hx,INSERT_VALUES,ierr);PetscCall(ierr)
        endif
      enddo
      call MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY,ierr);PetscCall(ierr)
      call MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY,ierr);PetscCall(ierr)
      return
      end

!/*TEST
!
!   test:
!      args: -ksp_converged_reason
!
!TEST*/

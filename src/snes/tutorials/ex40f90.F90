!
!  Demonstrates use of DMDASNESSetFunctionLocal() from Fortran
!
!    Note: the access to the entries of the local arrays below use the Fortran
!   convention of starting at zero. However calls to MatSetValues()  start at 0.
!   Also note that you will have to map the i,j,k coordinates to the local PETSc ordering
!   before calling MatSetValuesLocal(). Often you will find that using PETSc's default
!   code for computing the Jacobian works fine and you will not need to implement
!   your own FormJacobianLocal().

      program ex40f90

#include <petsc/finclude/petscsnes.h>
#include <petsc/finclude/petscdmda.h>
      use petscsnes
      use petscdmda
      implicit none

      SNES             snes
      PetscErrorCode   ierr
      DM               da
      PetscInt         ten,two,one
      external         FormFunctionLocal

      PetscCallA(PetscInitialize(ierr))
      ten = 10
      one = 1
      two = 2

      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,ten,ten,PETSC_DECIDE,PETSC_DECIDE,two,one,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr))
      PetscCallA(DMSetFromOptions(da,ierr))
      PetscCallA(DMSetUp(da,ierr))

!       Create solver object and associate it with the unknowns (on the grid)

      PetscCallA(SNESCreate(PETSC_COMM_WORLD,snes,ierr))
      PetscCallA(SNESSetDM(snes,da,ierr))

      PetscCallA(DMDASNESSetFunctionLocal(da,INSERT_VALUES,FormFunctionLocal,0,ierr))
      PetscCallA(SNESSetFromOptions(snes,ierr))

!      Solve the nonlinear system
!
      PetscCallA(SNESSolve(snes,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))

      PetscCallA(SNESDestroy(snes,ierr))
      PetscCallA(DMDestroy(da,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

      subroutine FormFunctionLocal(in,x,f,dummy,ierr)
      implicit none
      PetscInt i,j,k,dummy
      DMDALocalInfo in(DMDA_LOCAL_INFO_SIZE)
      PetscScalar x(in(DMDA_LOCAL_INFO_DOF),XG_RANGE,YG_RANGE)
      PetscScalar f(in(DMDA_LOCAL_INFO_DOF),X_RANGE,Y_RANGE)
      PetscErrorCode ierr

      do i=in(DMDA_LOCAL_INFO_XS)+1,in(DMDA_LOCAL_INFO_XS)+in(DMDA_LOCAL_INFO_XM)
         do j=in(DMDA_LOCAL_INFO_YS)+1,in(DMDA_LOCAL_INFO_YS)+in(DMDA_LOCAL_INFO_YM)
            do k=1,in(DMDA_LOCAL_INFO_DOF)
               f(k,i,j) = x(k,i,j)*x(k,i,j) - 2.0
            enddo
         enddo
      enddo

      return
      end

!/*TEST
!
!   test:
!     args: -snes_monitor_short -snes_view -da_refine 1 -pc_type mg -pc_mg_type full -ksp_type fgmres -pc_mg_galerkin pmat
!     requires: !single
!
!TEST*/

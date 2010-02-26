!
!  Demonstrates use of DMMGSetSNESLocal() from Fortran
!
!    Note: the access to the entries of the local arrays below use the Fortran
!   convention of starting at zero. However calls to MatSetValues()  start at 0.
!   Also note that you will have to map the i,j,k coordinates to the local PETSc ordering
!   before calling MatSetValuesLocal(). Often you will find that using PETSc's default
!   code for computing the Jacobian works fine and you will not need to implement
!   your own FormJacobianLocal().

      program ex40f90
      implicit none
#include "finclude/petsc.h"

      DMMG             dmmg
      PetscErrorCode   ierr
      DA               da
      external         FormFunctionLocal


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,         &
     &                -10,-10,PETSC_DECIDE,PETSC_DECIDE,2,1,                  &
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr)


!       Create solver object and associate it with the unknowns (on the grid)

      call DMMGCreate(PETSC_COMM_WORLD,1,PETSC_NULL_OBJECT,dmmg,ierr)
      call DMMGSetDM(dmmg,da,ierr)
      call DADestroy(da,ierr)

      call DMMGSetSNESLocal(dmmg,FormFunctionLocal,                                 &
     &             PETSC_NULL_FUNCTION,0,0,ierr)
      call DMMGSetFromOptions(dmmg,ierr)

!      Solve the nonlinear system
!
      call DMMGSolve(dmmg,ierr)

      call DMMGDestroy(dmmg,ierr)
      call PetscFinalize(ierr)
      end


      subroutine FormFunctionLocal(in,x,f,dmmg,ierr)
      implicit none
      DMMG dmmg
      PetscInt i,j,k
      DALocalInfo in(DA_LOCAL_INFO_SIZE)
      PetscScalar x(in(DA_LOCAL_INFO_DOF),                                          &
     &              XG_RANGE,                                            &
     &              YG_RANGE)  
      PetscScalar f(in(DA_LOCAL_INFO_DOF),                                 &
     &             X_RANGE,                                             &
     &             Y_RANGE)
      PetscErrorCode ierr

      do i=in(DA_LOCAL_INFO_XS)+1,in(DA_LOCAL_INFO_XS)+in(DA_LOCAL_INFO_MX)
      do j=in(DA_LOCAL_INFO_YS)+1,in(DA_LOCAL_INFO_YS)+in(DA_LOCAL_INFO_MY)
      do k=1,in(DA_LOCAL_INFO_DOF)
         f(k,i,j) = x(k,i,j)*x(k,i,j) - 2.0
      CHKMEMQ
      enddo
      enddo
      enddo

      return
      end      


      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscdm.h>
#include <finclude/petscdmda.h>

       PetscInt is,js,iw,jw
       PetscErrorCode ierr
       KSP ksp
       DM dm
       external ComputeRHS,ComputeMatrix,ComputeInitialGuess

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       call KSPCreate(MPI_COMM_WORLD,ksp,ierr)
       call DMDACreate2D(MPI_COMM_WORLD, DMDA_BOUNDARY_NONE,              &
     &    DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, 3,3, PETSC_DECIDE,       &
     &    PETSC_DECIDE,1,1, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER,       &
     &    dm, ierr)
       call DMSetInitialGuess(dm,ComputeInitialGuess,ierr)
       call DMSetFunction(dm,ComputeRHS,ierr)
       call DMSetJacobian(dm,ComputeMatrix,ierr)
       call KSPSetDM(ksp,dm,ierr)
       call DMDAGetCorners(dm,is,js,PETSC_NULL_INTEGER,iw,jw,             &
     &                     PETSC_NULL_INTEGER,ierr)
       call KSPSetFromOptions(ksp,ierr)
       call KSPSetUp(ksp,ierr)
       call KSPSolve(ksp,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
       call KSPDestroy(ksp,ierr)
       call DMDestroy(dm,ierr)
       call PetscFinalize(ierr)
       return
       end


       subroutine ComputeInitialGuess(dm,b,ierr)
       implicit none
       PetscErrorCode  ierr
       DM dm
       Vec b
       PetscScalar  h
       h=0.0
       call VecSet(b,h,ierr)
       end subroutine 

       subroutine ComputeRHS(dm,x,b,ierr)
       implicit none
       PetscErrorCode  ierr
       DM dm
       Vec x,b
       PetscScalar  h
       h=1.0
       call VecSet(b,h,ierr)
       end subroutine 

       subroutine ComputeMatrix(dm,x,A,B,str,ierr)
       implicit none
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
       PetscErrorCode  ierr
       DM dm
       Vec x
       Mat A,B
       MatStructure str
       PetscInt  rstart,rend,i
       PetscScalar h
       h = 1.0
       call MatGetOwnershipRange(B,rstart,rend,ierr)
       do 10, i=rstart,rend-1
         call MatSetValues(B,1,i,1,i,h,INSERT_VALUES,ierr)
 10    continue
       call MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY,ierr)
       call MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY,ierr)
       if ( A .ne. B) then
         call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
         call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
       endif
       end subroutine 

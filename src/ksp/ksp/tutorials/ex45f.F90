      program main
#include <petsc/finclude/petscksp.h>
      use petscdmda
      use petscksp
      implicit none

       PetscInt is,js,iw,jw
       PetscInt one,three
       PetscErrorCode ierr
       KSP ksp
       DM dm
       external ComputeRHS,ComputeMatrix,ComputeInitialGuess

       one = 1
       three = 3

       PetscCallA(PetscInitialize(ierr))
       PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
       PetscCallA(DMDACreate2D(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,three,three,PETSC_DECIDE,PETSC_DECIDE,one,one, PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, dm, ierr))
       PetscCallA(DMSetFromOptions(dm,ierr))
       PetscCallA(DMSetUp(dm,ierr))
       PetscCallA(KSPSetDM(ksp,dm,ierr))
       PetscCallA(KSPSetComputeInitialGuess(ksp,ComputeInitialGuess,0,ierr))
       PetscCallA(KSPSetComputeRHS(ksp,ComputeRHS,0,ierr))
       PetscCallA(KSPSetComputeOperators(ksp,ComputeMatrix,0,ierr))
       PetscCallA(DMDAGetCorners(dm,is,js,PETSC_NULL_INTEGER,iw,jw,PETSC_NULL_INTEGER,ierr))
       PetscCallA(KSPSetFromOptions(ksp,ierr))
       PetscCallA(KSPSetUp(ksp,ierr))
       PetscCallA(KSPSolve(ksp,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))
       PetscCallA(KSPDestroy(ksp,ierr))
       PetscCallA(DMDestroy(dm,ierr))
       PetscCallA(PetscFinalize(ierr))
       end

       subroutine ComputeInitialGuess(ksp,b,ctx,ierr)
       use petscksp
       implicit none
       PetscErrorCode  ierr
       KSP ksp
       PetscInt ctx(*)
       Vec b
       PetscScalar  h

       h=0.0
       PetscCall(VecSet(b,h,ierr))
       end subroutine

       subroutine ComputeRHS(ksp,b,dummy,ierr)
       use petscksp
       implicit none

       PetscErrorCode  ierr
       KSP ksp
       Vec b
       integer dummy(*)
       PetscScalar  h,Hx,Hy
       PetscInt  mx,my
       DM dm

       PetscCall(KSPGetDM(ksp,dm,ierr))
       PetscCall(DMDAGetInfo(dm,PETSC_NULL_INTEGER,mx,my,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))

       Hx = 1.0 / real(mx-1)
       Hy = 1.0 / real(my-1)
       h = Hx*Hy
       PetscCall(VecSet(b,h,ierr))
       end subroutine

      subroutine ComputeMatrix(ksp,A,B,dummy,ierr)
      use petscksp
       implicit none
       PetscErrorCode  ierr
       KSP ksp
       Mat A,B
       integer dummy(*)
       DM dm

      PetscInt    i,j,mx,my,xm
      PetscInt    ym,xs,ys,i1,i5
      PetscScalar  v(5),Hx,Hy
      PetscScalar  HxdHy,HydHx
      MatStencil   row(4),col(4,5)

      i1 = 1
      i5 = 5
      PetscCall(KSPGetDM(ksp,dm,ierr))
      PetscCall(DMDAGetInfo(dm,PETSC_NULL_INTEGER,mx,my,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))

      Hx = 1.0 / real(mx-1)
      Hy = 1.0 / real(my-1)
      HxdHy = Hx/Hy
      HydHx = Hy/Hx
      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      do 10,j=ys,ys+ym-1
        do 20,i=xs,xs+xm-1
          row(MatStencil_i) = i
          row(MatStencil_j) = j
          if (i.eq.0 .or. j.eq.0 .or. i.eq.mx-1 .or. j.eq.my-1) then
            v(1) = 2.0*(HxdHy + HydHx)
            PetscCall(MatSetValuesStencil(B,i1,row,i1,row,v,INSERT_VALUES,ierr))
          else
            v(1) = -HxdHy
            col(MatStencil_i,1) = i
            col(MatStencil_j,1) = j-1
            v(2) = -HydHx
            col(MatStencil_i,2) = i-1
            col(MatStencil_j,2) = j
            v(3) = 2.0*(HxdHy + HydHx)
            col(MatStencil_i,3) = i
            col(MatStencil_j,3) = j
            v(4) = -HydHx
            col(MatStencil_i,4) = i+1
            col(MatStencil_j,4) = j
            v(5) = -HxdHy
            col(MatStencil_i,5) = i
            col(MatStencil_j,5) = j+1
            PetscCall(MatSetValuesStencil(B,i1,row,i5,col,v,INSERT_VALUES,ierr))
            endif
 20      continue
 10   continue
       PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY,ierr))
       PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY,ierr))
       if (A .ne. B) then
         PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
         PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))
       endif
       end subroutine

!/*TEST
!
!   test:
!      nsize: 4
!      args: -ksp_monitor_short -da_refine 5 -pc_type mg -pc_mg_levels 5 -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 2 -mg_levels_pc_type jacobi -ksp_pc_side right
!
!TEST*/

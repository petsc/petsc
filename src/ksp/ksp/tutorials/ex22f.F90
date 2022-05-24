!
!   Laplacian in 3D. Modeled by the partial differential equation
!
!   Laplacian u = 1,0 < x,y,z < 1,
!
!   with boundary conditions
!
!   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
!
!   This uses multigrid to solve the linear system

      program main

#include <petsc/finclude/petscksp.h>
      use petscdmda
      use petscksp
      implicit none

      PetscErrorCode   ierr
      DM               da
      KSP              ksp
      Vec              x
      external         ComputeRHS,ComputeMatrix
      PetscInt i1,i3
      PetscInt         ctx

      PetscCallA(PetscInitialize(ierr))

      i3 = 3
      i1 = 1
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,i3,i3,i3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,i1,i1,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr))
      PetscCallA(DMSetFromOptions(da,ierr))
      PetscCallA(DMSetUp(da,ierr))
      PetscCallA(KSPSetDM(ksp,da,ierr))
      PetscCallA(KSPSetComputeRHS(ksp,ComputeRHS,ctx,ierr))
      PetscCallA(KSPSetComputeOperators(ksp,ComputeMatrix,ctx,ierr))

      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(KSPSolve(ksp,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))
      PetscCallA(KSPGetSolution(ksp,x,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(DMDestroy(da,ierr))
      PetscCallA(PetscFinalize(ierr))

      end

      subroutine ComputeRHS(ksp,b,ctx,ierr)
      use petscksp
      implicit none

      PetscErrorCode  ierr
      PetscInt mx,my,mz
      PetscScalar  h
      Vec          b
      KSP          ksp
      DM           da
      PetscInt     ctx

      PetscCall(KSPGetDM(ksp,da,ierr))
      PetscCall(DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,my,mz,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))
      h    = 1.0/real((mx-1)*(my-1)*(mz-1))

      PetscCall(VecSet(b,h,ierr))
      return
      end

      subroutine ComputeMatrix(ksp,JJ,jac,ctx,ierr)
      use petscksp
      implicit none

      Mat          jac,JJ
      PetscErrorCode    ierr
      KSP          ksp
      DM           da
      PetscInt    i,j,k,mx,my,mz,xm
      PetscInt    ym,zm,xs,ys,zs,i1,i7
      PetscScalar  v(7),Hx,Hy,Hz
      PetscScalar  HxHydHz,HyHzdHx
      PetscScalar  HxHzdHy
      MatStencil   row(4),col(4,7)
      PetscInt     ctx
      i1 = 1
      i7 = 7
      PetscCall(KSPGetDM(ksp,da,ierr))
      PetscCall(DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,my,mz,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr))

      Hx = 1.0 / real(mx-1)
      Hy = 1.0 / real(my-1)
      Hz = 1.0 / real(mz-1)
      HxHydHz = Hx*Hy/Hz
      HxHzdHy = Hx*Hz/Hy
      HyHzdHx = Hy*Hz/Hx
      PetscCall(DMDAGetCorners(da,xs,ys,zs,xm,ym,zm,ierr))

      do 10,k=zs,zs+zm-1
        do 20,j=ys,ys+ym-1
          do 30,i=xs,xs+xm-1
          row(MatStencil_i) = i
          row(MatStencil_j) = j
          row(MatStencil_k) = k
          if (i.eq.0 .or. j.eq.0 .or. k.eq.0 .or. i.eq.mx-1 .or. j.eq.my-1 .or. k.eq.mz-1) then
            v(1) = 2.0*(HxHydHz + HxHzdHy + HyHzdHx)
            PetscCall(MatSetValuesStencil(jac,i1,row,i1,row,v,INSERT_VALUES,ierr))
          else
            v(1) = -HxHydHz
             col(MatStencil_i,1) = i
             col(MatStencil_j,1) = j
             col(MatStencil_k,1) = k-1
            v(2) = -HxHzdHy
             col(MatStencil_i,2) = i
             col(MatStencil_j,2) = j-1
             col(MatStencil_k,2) = k
            v(3) = -HyHzdHx
             col(MatStencil_i,3) = i-1
             col(MatStencil_j,3) = j
             col(MatStencil_k,3) = k
            v(4) = 2.0*(HxHydHz + HxHzdHy + HyHzdHx)
             col(MatStencil_i,4) = i
             col(MatStencil_j,4) = j
             col(MatStencil_k,4) = k
            v(5) = -HyHzdHx
             col(MatStencil_i,5) = i+1
             col(MatStencil_j,5) = j
             col(MatStencil_k,5) = k
            v(6) = -HxHzdHy
             col(MatStencil_i,6) = i
             col(MatStencil_j,6) = j+1
             col(MatStencil_k,6) = k
            v(7) = -HxHydHz
             col(MatStencil_i,7) = i
             col(MatStencil_j,7) = j
             col(MatStencil_k,7) = k+1
      PetscCall(MatSetValuesStencil(jac,i1,row,i7,col,v,INSERT_VALUES,ierr))
          endif
 30       continue
 20     continue
 10   continue

      PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))
      return
      end

!/*TEST
!
!   test:
!      args: -pc_mg_type full -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type preconditioned -pc_type mg -da_refine 2 -ksp_type fgmres
!      requires: !single
!      output_file: output/ex22_1.out
!
!TEST*/

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
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!     petscsys.h  - base PETSc routines      petscvec.h - vectors
!     petscmat.h - matrices
!     petscksp.h    - Krylov subspace methods  petscpc.h  - preconditioners

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscdmda.h>
      PetscErrorCode   ierr
      DM               da
      KSP              ksp
      Vec              x
      external         ComputeRHS,ComputeMatrix
      PetscInt i1,i3

      call  PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      i3 = -3
      i1 = 1
      call KSPCreate(MPI_COMM_WORLD,ksp,ierr)
      call DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,            &
     &    DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,                               &
     &    DMDA_STENCIL_STAR,i3,i3,i3,PETSC_DECIDE,PETSC_DECIDE,                        &
     &    PETSC_DECIDE,i1,i1,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                        &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DMSetFunction(da,ComputeRHS,ierr)
      call DMSetJacobian(da,ComputeMatrix,ierr)
      call KSPSetDM(ksp,da,ierr)

      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call KSPGetSolution(ksp,x,ierr)
!      call VecView(x,PETSC_NULL_OBJECT,ierr)
      call KSPDestroy(ksp,ierr)
      call DMDestroy(da,ierr)
      call PetscFinalize(ierr)

      end

      subroutine ComputeRHS(da,x,b,ierr)
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>

      PetscErrorCode  ierr
      PetscInt mx,my,mz
      PetscScalar  h
      Vec          x,b
      DM           da

      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,my,mz,                        &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                 &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                 &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                 &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                 &
     &               PETSC_NULL_INTEGER,ierr)
      h    = 1.d0/((mx-1)*(my-1)*(mz-1))

      call VecSet(b,h,ierr)
      return
      end


      subroutine ComputeMatrix(da,x,JJ,jac,str,ierr)
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>

      Mat          jac,JJ
      PetscErrorCode    ierr

      DM           da
      PetscInt    i,j,k,mx,my,mz,xm
      PetscInt    ym,zm,xs,ys,zs,i1,i7
      PetscScalar  v(7),Hx,Hy,Hz
      PetscScalar  HxHydHz,HyHzdHx
      PetscScalar  HxHzdHy
      MatStencil   row(4),col(4,7)
      Vec          x
      MatStructure str
      i1 = 1
      i7 = 7
      call DMDAGetInfo(da,PETSC_NULL_INTEGER,mx,my,mz,                       &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                &
     &               PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,                &
     &               PETSC_NULL_INTEGER,ierr)

      Hx = 1.d0 / (mx-1)
      Hy = 1.d0 / (my-1)
      Hz = 1.d0 / (mz-1)
      HxHydHz = Hx*Hy/Hz
      HxHzdHy = Hx*Hz/Hy
      HyHzdHx = Hy*Hz/Hx
      call DMDAGetCorners(da,xs,ys,zs,xm,ym,zm,ierr)

      do 10,k=zs,zs+zm-1
        do 20,j=ys,ys+ym-1
          do 30,i=xs,xs+xm-1
          row(MatStencil_i) = i
          row(MatStencil_j) = j
          row(MatStencil_k) = k
          if (i.eq.0 .or. j.eq.0 .or. k.eq.0 .or. i.eq.mx-1 .or.         &
     &         j.eq.my-1 .or. k.eq.mz-1) then
            v(1) = 2.d0*(HxHydHz + HxHzdHy + HyHzdHx)
            call MatSetValuesStencil(jac,i1,row,i1,row,v,INSERT_VALUES,    &
     &                               ierr)
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
            v(4) = 2.d0*(HxHydHz + HxHzdHy + HyHzdHx)
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
      call MatSetValuesStencil(jac,i1,row,i7,col,v,INSERT_VALUES,               &
     &                               ierr)
          endif
 30       continue
 20     continue
 10   continue

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return
      end

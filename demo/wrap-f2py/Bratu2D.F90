! ---------------------------------------------------------------------
!
!    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
!    the partial differential equation
!  
!            -Laplacian(u) - lambda * exp(u) = 0,  0 < x,y < 1,
!  
!    with boundary conditions
!   
!             u = 0  for  x = 0, x = 1, y = 0, y = 1,
!  
!    A finite difference approximation with the usual 5-point stencil
!    is used to discretize the boundary value problem to obtain a
!    nonlinear system of equations. The problem is solved in a 2D
!    rectangular domain, using distributed arrays (DAs) to partition
!    the parallel grid.
!
! --------------------------------------------------------------------

#include "finclude/petscdef.h"

#undef  CHKERRQ
#define CHKERRQ(n) if ((n) .ne. 0) return;


#include "petscversion.h"
#if PETSC_VERSION_(3,1,0)
#include "finclude/petscvecdef.h"
#include "finclude/petscmatdef.h"
#include "finclude/petscdadef.h"
#else
#define DA                   DM
#define DAGetInfo            DMDAGetInfo
#define DAGetCorners         DMDAGetCorners
#define DAGetGhostCorners    DMDAGetGhostCorners
#define DAVecGetArray        DMDAVecGetArray
#define DAVecRestoreArray    DMDAVecRestoreArray
#define DAGetLocalVector     DMGetLocalVector
#define DARestoreLocalVector DMRestoreLocalVector
#define DAGlobalToLocalBegin DMGlobalToLocalBegin
#define DAGlobalToLocalEnd   DMGlobalToLocalEnd   
#endif

! --------------------------------------------------------------------

module Bratu2D

  use petsc
#if PETSC_VERSION_(3,1,0)
  use petscvec
  use petscmat
  use petscda
#endif

  type gridinfo
     PetscInt mx,xs,xe,xm,gxs,gxe,gxm
     PetscInt my,ys,ye,ym,gys,gye,gym
  end type gridinfo

contains

  subroutine GetGridInfo(da, grd, ierr)
    implicit none
    DA             da
    type(gridinfo) grd
    PetscErrorCode ierr
    !
#if PETSC_VERSION_(3,1,0)
    call DAGetInfo(da, PETSC_NULL_INTEGER, &
         &         grd%mx, grd%my, PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,ierr); CHKERRQ(ierr)
#else
    call DAGetInfo(da, PETSC_NULL_INTEGER, &
         &         grd%mx, grd%my, PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
         &         PETSC_NULL_INTEGER,ierr); CHKERRQ(ierr)
#endif
    call DAGetCorners(da, &
         &            grd%xs,grd%ys,PETSC_NULL_INTEGER, &
         &            grd%xm,grd%ym,PETSC_NULL_INTEGER,ierr); CHKERRQ(ierr)
    call DAGetGhostCorners(da, &
         &                 grd%gxs,grd%gys,PETSC_NULL_INTEGER, &
         &                 grd%gxm,grd%gym,PETSC_NULL_INTEGER,ierr); CHKERRQ(ierr)
    !
    grd%xs  = grd%xs+1
    grd%ys  = grd%ys+1
    grd%gxs = grd%gxs+1
    grd%gys = grd%gys+1
    
    grd%ye  = grd%ys+grd%ym-1
    grd%xe  = grd%xs+grd%xm-1
    grd%gye = grd%gys+grd%gym-1
    grd%gxe = grd%gxs+grd%gxm-1

  end subroutine GetGridInfo

  subroutine InitGuessLocal(grd, x, lambda, ierr)
    implicit none
    type(gridinfo) grd
    PetscScalar    x(grd%xs:grd%xe,grd%ys:grd%ye)
    PetscReal      lambda
    PetscErrorCode ierr
    !
    PetscInt       i, j
    PetscReal      hx,hy,temp,temp1,one
    
    one = 1.0
    hx  = one/(dble(grd%mx-1))
    hy  = one/(dble(grd%my-1))
    temp1 = lambda/(lambda+one)
    
    do j=grd%ys,grd%ye
       temp = dble(min(j-1,grd%my-j))*hy
       do i=grd%xs,grd%xe
          if (i==1 .or. j==1 .or. i==grd%mx .or. j==grd%my) then
             ! boundary points
             x(i,j) = 0.0
          else
             ! interior grid points
             x(i,j) = temp1*sqrt(min(dble(min(i-1,grd%mx-i)*hx),dble(temp)))
          end if
       end do
    end do
    ierr = 0

  end subroutine InitGuessLocal

  subroutine FunctionLocal(grd, x, f, lambda, ierr)
    implicit none
    type(gridinfo) grd
    PetscScalar    x(grd%gxs:grd%gxe,grd%gys:grd%gye)
    PetscScalar    f(grd%xs:grd%xe,grd%ys:grd%ye)
    PetscReal      lambda
    PetscErrorCode ierr
    !
    PetscInt       i,j
    PetscReal      hx,hy,hxdhy,hydhx,sc,one,two
    PetscScalar    u,uxx,uyy

    one    = 1.0
    two    = 2.0
    hx     = one/dble(grd%mx-1)
    hy     = one/dble(grd%my-1)
    sc     = hx*hy
    hxdhy  = hx/hy
    hydhx  = hy/hx

    do j=grd%ys,grd%ye
       do i=grd%xs,grd%xe
          if (i==1 .or. j==1 .or. i==grd%mx .or. j==grd%my) then
             ! boundary points
             f(i,j) = x(i,j) - 0.0
          else
             ! interior grid points
             u = x(i,j) 
             uxx =  (two*u - x(i-1,j) - x(i+1,j)) * hydhx
             uyy =  (two*u - x(i,j-1) - x(i,j+1)) * hxdhy
             f(i,j) = uxx + uyy - lambda*exp(u)*sc
          end if
       end do
    end do
    ierr = 0

  end subroutine FunctionLocal

  subroutine JacobianLocal(grd, x, Jac, lambda, ierr)
    implicit none
    type(gridinfo) grd
    PetscScalar    x(grd%gxs:grd%gxe,grd%gys:grd%gye)
    Mat            Jac
    PetscReal      lambda
    PetscErrorCode ierr
    !
    PetscInt      i,j,row(1),col(5)
    PetscInt      ione,ifive
    PetscReal     hx,hy,hxdhy,hydhx,sc,v(5),one,two

    ione   = 1
    ifive  = 5
    one    = 1.0
    two    = 2.0
    hx     = one/dble(grd%mx-1)
    hy     = one/dble(grd%my-1)
    sc     = hx*hy
    hxdhy  = hx/hy
    hydhx  = hy/hx

    do j=grd%ys,grd%ye
       row = (j - grd%gys)*grd%gxm + grd%xs - grd%gxs - 1
       do i=grd%xs,grd%xe
          row = row + 1
          if (i==1 .or. j==1 .or. i==grd%mx .or. j==grd%my) then
             ! boundary points
             col(1) = row(1)
             v(1)   = one
             call MatSetValuesLocal(Jac,ione,row,ione,col,v,INSERT_VALUES,ierr); CHKERRQ(ierr)
          else
             ! interior grid points
             v(1) = -hxdhy
             v(2) = -hydhx
             v(3) = two*(hydhx + hxdhy) - lambda*exp(x(i,j))*sc
             v(4) = -hydhx
             v(5) = -hxdhy
             col(1) = row(1) - grd%gxm
             col(2) = row(1) - 1
             col(3) = row(1)
             col(4) = row(1) + 1
             col(5) = row(1) + grd%gxm
             call MatSetValuesLocal(Jac,ione,row,ifive,col,v,INSERT_VALUES,ierr); CHKERRQ(ierr)
          end if
       end do
    end do
    
  end subroutine JacobianLocal

end module Bratu2D

! --------------------------------------------------------------------

subroutine FormInitGuess(da, X, lambda, ierr)
  use Bratu2D
  implicit none
  DA  da
  Vec X
  PetscReal lambda
  PetscErrorCode ierr
  !
  type(gridinfo)      :: grd
  PetscScalar,pointer :: xx(:)
  
  call VecGetArrayF90(X,xx,ierr); CHKERRQ(ierr)

  call GetGridInfo(da,grd,ierr); CHKERRQ(ierr)
  call InitGuessLocal(grd,xx,lambda,ierr); CHKERRQ(ierr)

  call VecRestoreArrayF90(X,xx,ierr); CHKERRQ(ierr)

end subroutine FormInitGuess


subroutine FormFunction(da, X, F, lambda, ierr)
  use Bratu2D
  implicit none
  DA  da
  Vec X
  Vec F
  PetscReal lambda
  PetscErrorCode ierr
  !
  type(gridinfo)      :: grd
  Vec                 :: localX
  PetscScalar,pointer :: xx(:)
  PetscScalar,pointer :: ff(:)
  
  call DAGetLocalVector(da,localX,ierr); CHKERRQ(ierr)
  call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr); CHKERRQ(ierr)
  call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr); CHKERRQ(ierr)

  call VecGetArrayF90(localX,xx,ierr); CHKERRQ(ierr)
  call VecGetArrayF90(F,ff,ierr); CHKERRQ(ierr)
  
  call GetGridInfo(da,grd,ierr); CHKERRQ(ierr)
  call FunctionLocal(grd,xx,ff,lambda,ierr); CHKERRQ(ierr)

  call VecRestoreArrayF90(F,ff,ierr); CHKERRQ(ierr)
  call VecRestoreArrayF90(localX,xx,ierr); CHKERRQ(ierr)

  call DARestoreLocalVector(da,localX,ierr); CHKERRQ(ierr)
  
end subroutine FormFunction


subroutine FormJacobian(da, X, J, lambda, ierr)
  use Bratu2D
  implicit none
  DA  da
  Vec X
  Mat J
  PetscReal lambda
  PetscErrorCode ierr
  !
  type(gridinfo)      :: grd
  Vec                 :: localX
  PetscScalar,pointer :: xx(:)

  call DAGetLocalVector(da,localX,ierr); CHKERRQ(ierr)
  call DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX,ierr); CHKERRQ(ierr)
  call DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX,ierr); CHKERRQ(ierr)
  call VecGetArrayF90(localX,xx,ierr); CHKERRQ(ierr)

  call GetGridInfo(da,grd,ierr); CHKERRQ(ierr)
  call JacobianLocal(grd,xx,J,lambda,ierr); CHKERRQ(ierr)

  call VecRestoreArrayF90(localX,xx,ierr); CHKERRQ(ierr)
  call DARestoreLocalVector(da,localX,ierr); CHKERRQ(ierr)

  call MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY,ierr); CHKERRQ(ierr)
  call MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY,ierr); CHKERRQ(ierr)

end subroutine FormJacobian

! --------------------------------------------------------------------


! Local Variables:
! mode: f90
! End:


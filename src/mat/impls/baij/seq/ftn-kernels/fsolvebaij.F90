!
!
!    Fortran kernel for sparse triangular solve in the BAIJ matrix format
! This ONLY works for factorizations in the NATURAL ORDERING, i.e.
! with MatSolve_SeqBAIJ_4_NaturalOrdering()
!
#include <petsc/finclude/petscsys.h>
!

pure subroutine FortranSolveBAIJ4Unroll(n,x,ai,aj,adiag,a,b)
  implicit none (type, external)
  MatScalar, intent(in) :: a(0:*)
  PetscScalar, intent(inout) :: x(0:*)
  PetscScalar, intent(in) :: b(0:*)
  PetscInt, intent(in) :: n
  PetscInt, intent(in) :: ai(0:*), aj(0:*), adiag(0:*)

  PetscInt :: i,j,jstart,jend
  PetscInt :: idx,ax,jdx
  PetscScalar :: s1,s2,s3,s4
  PetscScalar :: x1,x2,x3,x4

  PETSC_AssertAlignx(16,a(1))
  PETSC_AssertAlignx(16,x(1))
  PETSC_AssertAlignx(16,b(1))
  PETSC_AssertAlignx(16,ai(1))
  PETSC_AssertAlignx(16,aj(1))
  PETSC_AssertAlignx(16,adiag(1))

  !
  ! Forward Solve
  !
  x(0:3) = b(0:3)
  idx  = 0
  do i=1,n-1
    jstart = ai(i)
    jend   = adiag(i) - 1
    ax     = 16*jstart
    idx    = idx + 4
    s1     = b(idx+0)
    s2     = b(idx+1)
    s3     = b(idx+2)
    s4     = b(idx+3)
    do j=jstart,jend
      jdx = 4*aj(j)

      x1  = x(jdx+0)
      x2  = x(jdx+1)
      x3  = x(jdx+2)
      x4  = x(jdx+3)
      s1  = s1-(a(ax+0)*x1+a(ax+4)*x2+a(ax+ 8)*x3+a(ax+12)*x4)
      s2  = s2-(a(ax+1)*x1+a(ax+5)*x2+a(ax+9)*x3 +a(ax+13)*x4)
      s3  = s3-(a(ax+2)*x1+a(ax+6)*x2+a(ax+10)*x3+a(ax+14)*x4)
      s4  = s4-(a(ax+3)*x1+a(ax+7)*x2+a(ax+11)*x3+a(ax+15)*x4)
      ax  = ax + 16
    end do
    x(idx+0) = s1
    x(idx+1) = s2
    x(idx+2) = s3
    x(idx+3) = s4
  end do

  !
  ! Backward solve the upper triangular
  !
  do i=n-1,0,-1
    jstart = adiag(i) + 1
    jend   = ai(i+1) - 1
    ax     = 16*jstart
    s1     = x(idx+0)
    s2     = x(idx+1)
    s3     = x(idx+2)
    s4     = x(idx+3)
    do j=jstart,jend
      jdx   = 4*aj(j)
      x1    = x(jdx+0)
      x2    = x(jdx+1)
      x3    = x(jdx+2)
      x4    = x(jdx+3)
      s1 = s1-(a(ax+0)*x1+a(ax+4)*x2+a(ax+ 8)*x3+a(ax+12)*x4)
      s2 = s2-(a(ax+1)*x1+a(ax+5)*x2+a(ax+9)*x3 +a(ax+13)*x4)
      s3 = s3-(a(ax+2)*x1+a(ax+6)*x2+a(ax+10)*x3+a(ax+14)*x4)
      s4 = s4-(a(ax+3)*x1+a(ax+7)*x2+a(ax+11)*x3+a(ax+15)*x4)
      ax = ax + 16
    end do
    ax      = 16*adiag(i)
    x(idx+0) = a(ax+0)*s1+a(ax+4)*s2+a(ax+ 8)*s3+a(ax+12)*s4
    x(idx+1) = a(ax+1)*s1+a(ax+5)*s2+a(ax+9)*s3 +a(ax+13)*s4
    x(idx+2) = a(ax+2)*s1+a(ax+6)*s2+a(ax+10)*s3+a(ax+14)*s4
    x(idx+3) = a(ax+3)*s1+a(ax+7)*s2+a(ax+11)*s3+a(ax+15)*s4
    idx      = idx - 4
  end do
end subroutine FortranSolveBAIJ4Unroll

!   version that does not call BLAS 2 operation for each row block
!
subroutine FortranSolveBAIJ4(n,x,ai,aj,adiag,a,b,w)
  implicit none
  MatScalar, intent(in) :: a(0:*)
  PetscScalar, intent(inout) :: x(0:*),w(0:*)
  PetscScalar, intent(in) :: b(0:*)
  PetscInt, intent(in) :: n
  PetscInt, intent(in) :: ai(0:*), aj(0:*), adiag(0:*)

  PetscInt :: ii,jj,i,j
  PetscInt :: jstart,jend,idx,ax,jdx,kdx,nn
  PetscScalar :: s(0:3)

  PETSC_AssertAlignx(16,a(1))
  PETSC_AssertAlignx(16,w(1))
  PETSC_AssertAlignx(16,x(1))
  PETSC_AssertAlignx(16,b(1))
  PETSC_AssertAlignx(16,ai(1))
  PETSC_AssertAlignx(16,aj(1))
  PETSC_AssertAlignx(16,adiag(1))
  !
  !     Forward Solve
  !
  x(0:3) = b(0:3)
  idx  = 0
  do i=1,n-1
    !
    ! Pack required part of vector into work array
    !
    kdx    = 0
    jstart = ai(i)
    jend   = adiag(i) - 1

    if (jend - jstart >= 500) write(6,*) 'Overflowing vector FortranSolveBAIJ4()'

    do j=jstart,jend

      jdx       = 4*aj(j)
      w(kdx:kdx+3) = x(jdx:jdx+3)
      kdx       = kdx + 4
    end do

    ax       = 16*jstart
    idx      = idx + 4
    s(0:3) = b(idx:idx+3)
    !
    !    s = s - a(ax:)*w
    !
    nn = 4*(jend - jstart + 1) - 1
    do ii=0,3
      do jj=0,nn
        s(ii) = s(ii) - a(ax+4*jj+ii)*w(jj)
      end do
    end do

    x(idx:idx+3) = s(0:3)
  end do
  !
  ! Backward solve the upper triangular
  !
  do i=n-1,0,-1
     jstart    = adiag(i) + 1
     jend      = ai(i+1) - 1
     ax        = 16*jstart
     s(0:3) = x(idx:idx+3)
     !
     !   Pack each chunk of vector needed
     !
     kdx = 0
     if (jend - jstart >= 500) write(6,*) 'Overflowing vector FortranSolveBAIJ4()'

     do j=jstart,jend
       jdx      = 4*aj(j)
       w(kdx:kdx+3) = x(jdx:jdx+3)
       kdx      = kdx + 4
     end do
     nn = 4*(jend - jstart + 1) - 1
     do ii=0,3
       do jj=0,nn
         s(ii) = s(ii) - a(ax+4*jj+ii)*w(jj)
       end do
     end do

     ax      = 16*adiag(i)
     x(idx)  = a(ax+0)*s(0)+a(ax+4)*s(1)+a(ax+ 8)*s(2)+a(ax+12)*s(3)
     x(idx+1)= a(ax+1)*s(0)+a(ax+5)*s(1)+a(ax+9)*s(2) +a(ax+13)*s(3)
     x(idx+2)= a(ax+2)*s(0)+a(ax+6)*s(1)+a(ax+10)*s(2)+a(ax+14)*s(3)
     x(idx+3)= a(ax+3)*s(0)+a(ax+7)*s(1)+a(ax+11)*s(2)+a(ax+15)*s(3)
     idx     = idx - 4
  end do
end subroutine FortranSolveBAIJ4

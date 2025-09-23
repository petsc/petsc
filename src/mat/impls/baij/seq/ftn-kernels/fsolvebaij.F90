!
!
!    Fortran kernel for sparse triangular solve in the BAIJ matrix format
! This ONLY works for factorizations in the NATURAL ORDERING, i.e.
! with MatSolve_SeqBAIJ_4_NaturalOrdering()
!
#include <petsc/finclude/petscsys.h>
!

pure subroutine FortranSolveBAIJ4Unroll(n, x, ai, aj, adiag, a, b)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  MatScalar, intent(in) :: a(0:*)
  PetscScalar, intent(inout) :: x(0:*)
  PetscScalar, intent(in) :: b(0:*)
  PetscInt, intent(in) :: n
  PetscInt, intent(in) :: ai(0:*), aj(0:*), adiag(0:*)

  PetscInt :: i, j, jstart, jend
  PetscInt :: idx, ax, jdx
  PetscScalar :: s(0:3)

  PETSC_AssertAlignx(16, a(1))
  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, b(1))
  PETSC_AssertAlignx(16, ai(1))
  PETSC_AssertAlignx(16, aj(1))
  PETSC_AssertAlignx(16, adiag(1))

  !
  ! Forward Solve
  !
  x(0:3) = b(0:3)
  idx = 0
  do i = 1, n - 1
    jstart = ai(i)
    jend = adiag(i) - 1
    ax = 16*jstart
    idx = idx + 4
    s(0:3) = b(idx + 0:idx + 3)
    do j = jstart, jend
      jdx = 4*aj(j)

      s(0) = s(0) - (a(ax + 0)*x(jdx + 0) + a(ax + 4)*x(jdx + 1) + a(ax + 8)*x(jdx + 2) + a(ax + 12)*x(jdx + 3))
      s(1) = s(1) - (a(ax + 1)*x(jdx + 0) + a(ax + 5)*x(jdx + 1) + a(ax + 9)*x(jdx + 2) + a(ax + 13)*x(jdx + 3))
      s(2) = s(2) - (a(ax + 2)*x(jdx + 0) + a(ax + 6)*x(jdx + 1) + a(ax + 10)*x(jdx + 2) + a(ax + 14)*x(jdx + 3))
      s(3) = s(3) - (a(ax + 3)*x(jdx + 0) + a(ax + 7)*x(jdx + 1) + a(ax + 11)*x(jdx + 2) + a(ax + 15)*x(jdx + 3))
      ax = ax + 16
    end do
    x(idx + 0:idx + 3) = s(0:3)
  end do

  !
  ! Backward solve the upper triangular
  !
  do i = n - 1, 0, -1
    jstart = adiag(i) + 1
    jend = ai(i + 1) - 1
    ax = 16*jstart
    s(0:3) = x(idx + 0:idx + 3)
    do j = jstart, jend
      jdx = 4*aj(j)
      s(0) = s(0) - (a(ax + 0)*x(jdx + 0) + a(ax + 4)*x(jdx + 1) + a(ax + 8)*x(jdx + 2) + a(ax + 12)*x(jdx + 3))
      s(1) = s(1) - (a(ax + 1)*x(jdx + 0) + a(ax + 5)*x(jdx + 1) + a(ax + 9)*x(jdx + 2) + a(ax + 13)*x(jdx + 3))
      s(2) = s(2) - (a(ax + 2)*x(jdx + 0) + a(ax + 6)*x(jdx + 1) + a(ax + 10)*x(jdx + 2) + a(ax + 14)*x(jdx + 3))
      s(3) = s(3) - (a(ax + 3)*x(jdx + 0) + a(ax + 7)*x(jdx + 1) + a(ax + 11)*x(jdx + 2) + a(ax + 15)*x(jdx + 3))
      ax = ax + 16
    end do
    ax = 16*adiag(i)
    x(idx + 0) = a(ax + 0)*s(0) + a(ax + 4)*s(1) + a(ax + 8)*s(2) + a(ax + 12)*s(3)
    x(idx + 1) = a(ax + 1)*s(0) + a(ax + 5)*s(1) + a(ax + 9)*s(2) + a(ax + 13)*s(3)
    x(idx + 2) = a(ax + 2)*s(0) + a(ax + 6)*s(1) + a(ax + 10)*s(2) + a(ax + 14)*s(3)
    x(idx + 3) = a(ax + 3)*s(0) + a(ax + 7)*s(1) + a(ax + 11)*s(2) + a(ax + 15)*s(3)
    idx = idx - 4
  end do
end subroutine FortranSolveBAIJ4Unroll

!   version that does not call BLAS 2 operation for each row block
!
pure subroutine FortranSolveBAIJ4(n, x, ai, aj, adiag, a, b, w)
  use, intrinsic :: ISO_C_binding
  implicit none
  MatScalar, intent(in) :: a(0:*)
  PetscScalar, intent(inout) :: x(0:*), w(0:*)
  PetscScalar, intent(in) :: b(0:*)
  PetscInt, intent(in) :: n
  PetscInt, intent(in) :: ai(0:*), aj(0:*), adiag(0:*)

  PetscInt :: ii, jj, i, j
  PetscInt :: jstart, jend, idx, ax, jdx, kdx, nn
  PetscScalar :: s(0:3)

  PETSC_AssertAlignx(16, a(1))
  PETSC_AssertAlignx(16, w(1))
  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, b(1))
  PETSC_AssertAlignx(16, ai(1))
  PETSC_AssertAlignx(16, aj(1))
  PETSC_AssertAlignx(16, adiag(1))
  !
  !     Forward Solve
  !
  x(0:3) = b(0:3)
  idx = 0
  do i = 1, n - 1
    !
    ! Pack required part of vector into work array
    !
    kdx = 0
    jstart = ai(i)
    jend = adiag(i) - 1

    if (jend - jstart >= 500) error stop 'Overflowing vector FortranSolveBAIJ4()'

    do j = jstart, jend
      jdx = 4*aj(j)
      w(kdx:kdx + 3) = x(jdx:jdx + 3)
      kdx = kdx + 4
    end do

    ax = 16*jstart
    idx = idx + 4
    s(0:3) = b(idx:idx + 3)
    !
    !    s = s - a(ax:)*w
    !
    nn = 4*(jend - jstart + 1) - 1
    do ii = 0, 3
      do jj = 0, nn
        s(ii) = s(ii) - a(ax + 4*jj + ii)*w(jj)
      end do
    end do

    x(idx:idx + 3) = s(0:3)
  end do
  !
  ! Backward solve the upper triangular
  !
  do i = n - 1, 0, -1
    jstart = adiag(i) + 1
    jend = ai(i + 1) - 1
    ax = 16*jstart
    s(0:3) = x(idx:idx + 3)
    !
    !   Pack each chunk of vector needed
    !
    kdx = 0
    if (jend - jstart >= 500) error stop 'Overflowing vector FortranSolveBAIJ4()'

    do j = jstart, jend
      jdx = 4*aj(j)
      w(kdx:kdx + 3) = x(jdx:jdx + 3)
      kdx = kdx + 4
    end do
    nn = 4*(jend - jstart + 1) - 1
    do ii = 0, 3
      do jj = 0, nn
        s(ii) = s(ii) - a(ax + 4*jj + ii)*w(jj)
      end do
    end do

    ax = 16*adiag(i)
    x(idx) = a(ax + 0)*s(0) + a(ax + 4)*s(1) + a(ax + 8)*s(2) + a(ax + 12)*s(3)
    x(idx + 1) = a(ax + 1)*s(0) + a(ax + 5)*s(1) + a(ax + 9)*s(2) + a(ax + 13)*s(3)
    x(idx + 2) = a(ax + 2)*s(0) + a(ax + 6)*s(1) + a(ax + 10)*s(2) + a(ax + 14)*s(3)
    x(idx + 3) = a(ax + 3)*s(0) + a(ax + 7)*s(1) + a(ax + 11)*s(2) + a(ax + 15)*s(3)
    idx = idx - 4
  end do
end subroutine FortranSolveBAIJ4

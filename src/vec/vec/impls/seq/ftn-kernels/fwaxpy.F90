!
!
!    Fortran kernel for the WAXPY() vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranWAXPY(n,a,x,y,w)
  implicit none (type, external)
  PetscScalar, intent(in) :: a
  PetscScalar, intent(in) :: x(*),y(*)
  PetscScalar, intent(inout) :: w(*)
  PetscInt, intent(in) :: n

  PetscInt i

  PETSC_AssertAlignx(16,x(1))
  PETSC_AssertAlignx(16,y(1))
  PETSC_AssertAlignx(16,w(1))

  do i=1,n
    w(i) = a*x(i) + y(i)
  end do
end subroutine FortranWAXPY

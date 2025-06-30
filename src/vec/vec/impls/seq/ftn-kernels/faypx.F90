!
!
!    Fortran kernel for the AYPX() vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranAYPX(n,a,x,y)
  implicit none (type, external)
  PetscScalar, intent(in) :: a
  PetscScalar, intent(in) :: x(*)
  PetscScalar, intent(inout) :: y(*)
  PetscInt, intent(in) :: n

  PetscInt i

  do i=1,n
    y(i) = x(i) + a*y(i)
  end do
end subroutine FortranAYPX

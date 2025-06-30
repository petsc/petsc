!
!
!    Fortran kernel for the AYPX() vector routine
!
#include <petsc/finclude/petscsys.h>
!
subroutine FortranAYPX(n,a,x,y)
  implicit none
  PetscScalar  a
  PetscScalar  x(*),y(*)
  PetscInt n

  PetscInt i

  do i=1,n
    y(i) = x(i) + a*y(i)
  end do

end subroutine FortranAYPX

!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ/CRL format
!
#include <petsc/finclude/petscsys.h>
!
subroutine FortranMultCRL(m,rmax,x,y,icols,acols)
  implicit none
  PetscInt m,rmax,icols(m,rmax)
  PetscScalar x(0:m-1),y(m)
  PetscScalar acols(m,rmax)

  PetscInt    i,j

  do j=1,m
    y(j) = acols(j,1)*x(icols(j,1))
  end do

  do i=2,rmax
    do j=1,m
      y(j) = y(j) + acols(j,i)*x(icols(j,i))
    end do
  end do

end subroutine FortranMultCRL

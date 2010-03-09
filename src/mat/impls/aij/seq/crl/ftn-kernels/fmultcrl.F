!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ/CRL format
!
#include "finclude/petscsysdef.h"
!
      subroutine FortranMultCRL(m,rmax,x,y,icols,acols)
      implicit none
      PetscInt m,rmax,icols(m,rmax)
      PetscScalar x(0:m-1),y(m)
      PetscScalar acols(m,rmax)

      PetscInt    i,j

      do 5 j=1,m
          y(j) = acols(j,1)*x(icols(j,1))
 5    continue

      do 10,i=2,rmax
        do 20 j=1,m
          y(j) = y(j) + acols(j,i)*x(icols(j,i))
 20     continue
 10   continue

      return 
      end


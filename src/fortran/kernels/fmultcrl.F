!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ/CRL format
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranMultCRL(m,rmax,x,y,icols,acols)
      implicit none
      PetscScalar x(0:m-1),acols(m,rmax),y(m)
      PetscInt    m,rmax,icols(m,rmax)

      PetscInt    i,j

      do 10,i=1,rmax
        do 20 j=1,m
          y(j) = y(j) + acols(j,i)*x(icols(j,i))
 20     continue
 10   continue

      return 
      end


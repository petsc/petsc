!
#include "finclude/petscdef.h"
!
!    Fortran kernel for the z = x * y
!
!
      subroutine Fortranxtimesy(x,y,z,n)
      implicit none
      PetscScalar  x(*),y(*),z(*)
      PetscInt n

      PetscInt i

      do 10,i=1,n
        z(i) = x(i) * y(i)
 10   continue

      return 
      end

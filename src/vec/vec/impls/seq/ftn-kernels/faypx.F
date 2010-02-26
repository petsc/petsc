!
!
!    Fortran kernel for the AYPX() vector routine
!
#include "finclude/petscsysdef.h"
!
      subroutine FortranAYPX(n,a,x,y)
      implicit none
      PetscScalar  a
      PetscScalar  x(*),y(*)
      PetscInt n

      PetscInt i

      do 10,i=1,n
        y(i) = x(i) + a*y(i)
 10   continue

      return 
      end

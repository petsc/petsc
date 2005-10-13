!
!
!    Fortran kernel for the MAXPY() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranMAXPY4(x,a1,a2,a3,a4,y1,y2,y3,y4,n)
      implicit none
      PetscScalar a1,a2,a3,a4
      PetscScalar x(*),y1(*),y2(*),y3(*),y4(*)
      PetscInt n

      PetscInt i

      do 10,i=1,n
        x(i) = x(i) + a1*y1(i) + a2*y2(i) + a3*y3(i) + a4*y4(i)
 10   continue

      return 
      end

      subroutine FortranMAXPY3(x,a1,a2,a3,y1,y2,y3,n)
      implicit none
      PetscScalar a1,a2,a3,x(*),y1(*),y2(*)
      PetscScalar y3(*)
      PetscInt n

      PetscInt i

      do 10,i=1,n
        x(i) = x(i) + a1*y1(i) + a2*y2(i) + a3*y3(i)
 10   continue

      return 
      end



      subroutine FortranMAXPY2(x,a1,a2,y1,y2,n)
      implicit none
      PetscScalar  a1,a2,x(*),y1(*),y2(*)
      PetscInt n

      PetscInt i

      do 10,i=1,n
        x(i) = x(i) + a1*y1(i) + a2*y2(i)
 10   continue

      return 
      end
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

!
!
!    Fortran kernel for the MDot() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranMDot4_BGL(x,y1,y2,y3,y4,n,sum1,sum2,sum3,sum4)
      implicit none
      PetscScalar  sum1,sum2,sum3,sum4
      PetscScalar  x(*),y1(*),y2(*),y3(*),y4(*)
      PetscInt n
      PetscInt i

      call ALIGNX(16,x(1))
      call ALIGNX(16,y1(1))
      call ALIGNX(16,y2(1))
      call ALIGNX(16,y3(1))
      call ALIGNX(16,y4(1))
      
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
        sum2 = sum2 + x(i)*PetscConj(y2(i))
        sum3 = sum3 + x(i)*PetscConj(y3(i))
        sum4 = sum4 + x(i)*PetscConj(y4(i))
 10   continue

      return 
      end

      subroutine FortranMDot3_BGL(x,y1,y2,y3,n,sum1,sum2,sum3)
      implicit none
      PetscScalar  sum1,sum2,sum3
      PetscScalar  x(*),y1(*),y2(*),y3(*)
      PetscInt n
      PetscInt i

      call ALIGNX(16,x(1))
      call ALIGNX(16,y1(1))
      call ALIGNX(16,y2(1))
      call ALIGNX(16,y3(1))
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
        sum2 = sum2 + x(i)*PetscConj(y2(i))
        sum3 = sum3 + x(i)*PetscConj(y3(i))
 10   continue

      return 
      end

      subroutine FortranMDot2_BGL(x,y1,y2,n,sum1,sum2)
      implicit none
      PetscScalar  sum1,sum2,x(*),y1(*),y2(*)
      PetscInt n
      PetscInt i

      call ALIGNX(16,x(1))
      call ALIGNX(16,y1(1))
      call ALIGNX(16,y2(1))
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
        sum2 = sum2 + x(i)*PetscConj(y2(i))
 10   continue

      return 
      end


      subroutine FortranMDot1_BGL(x,y1,n,sum1)
      implicit none
      PetscScalar  sum1,x(*),y1(*)
      PetscInt n
      PetscInt i

      call ALIGNX(16,x(1))
      call ALIGNX(16,y1(1))
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
 10   continue

      return 
      end

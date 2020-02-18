!
!
!    Fortran kernel for the MDot() vector routine
!
#include <petsc/finclude/petscsys.h>
!
      subroutine FortranMDot4(x,y1,y2,y3,y4,n,sum1,sum2,sum3,sum4)
      implicit none
      PetscScalar  sum1,sum2,sum3,sum4
      PetscScalar  x(*),y1(*),y2(*),y3(*),y4(*)
      PetscInt n
      PetscInt i

      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y1(1))
      PETSC_AssertAlignx(16,y2(1))
      PETSC_AssertAlignx(16,y3(1))
      PETSC_AssertAlignx(16,y4(1))

      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
        sum2 = sum2 + x(i)*PetscConj(y2(i))
        sum3 = sum3 + x(i)*PetscConj(y3(i))
        sum4 = sum4 + x(i)*PetscConj(y4(i))
 10   continue

      return
      end

      subroutine FortranMDot3(x,y1,y2,y3,n,sum1,sum2,sum3)
      implicit none
      PetscScalar  sum1,sum2,sum3
      PetscScalar  x(*),y1(*),y2(*),y3(*)
      PetscInt n
      PetscInt i

      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y1(1))
      PETSC_AssertAlignx(16,y2(1))
      PETSC_AssertAlignx(16,y3(1))
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
        sum2 = sum2 + x(i)*PetscConj(y2(i))
        sum3 = sum3 + x(i)*PetscConj(y3(i))
 10   continue

      return
      end

      subroutine FortranMDot2(x,y1,y2,n,sum1,sum2)
      implicit none
      PetscScalar  sum1,sum2,x(*),y1(*),y2(*)
      PetscInt n
      PetscInt i

      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y1(1))
      PETSC_AssertAlignx(16,y2(1))
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
        sum2 = sum2 + x(i)*PetscConj(y2(i))
 10   continue

      return
      end


      subroutine FortranMDot1(x,y1,n,sum1)
      implicit none
      PetscScalar  sum1,x(*),y1(*)
      PetscInt n
      PetscInt i

      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y1(1))
      do 10,i=1,n
        sum1 = sum1 + x(i)*PetscConj(y1(i))
 10   continue

      return
      end

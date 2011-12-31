!
!
!    Fortran kernel for the Norm() vector routine
!
#include <finclude/petscsysdef.h>
!
      subroutine FortranNormSqr(x,n,sum1)
      implicit none
      PetscScalar x(*)
      PetscReal   sum1
      PetscInt n

      PetscInt i

      PETSC_AssertAlignx(16,x(1))

      do 10,i=1,n
        sum1 = sum1 + PetscRealPart(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end


      subroutine FortranNormSqrUnroll(x,n,sum1)
      implicit none
      PetscScalar x(*)
      PetscReal   sum1
      PetscInt n

      PetscInt i

      PETSC_AssertAlignx(16,x(1))

      do 10,i=1,n,4
        sum1 = sum1 + PetscRealPart(x(i)*PetscConj(x(i)))                                         &
     &              + PetscRealPart(x(i+1)*PetscConj(x(i+1)))                                     &
     &              + PetscRealPart(x(i+2)*PetscConj(x(i+2)))                                     &
     &              + PetscRealPart(x(i+3)*PetscConj(x(i+3))) 
 10   continue

      return 
      end


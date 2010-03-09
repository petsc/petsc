!
!
!    Fortran kernel for the copy vector routine
!
#include "finclude/petscsysdef.h"
!
      subroutine FortranCopy(n,x,y)
      implicit none
      PetscScalar  x(*),y(*)
      PetscInt n
      PetscInt i
      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y(1))
      do 10,i=1,n
        y(i) = x(i)
 10   continue
      return 
      end


      subroutine FortranZero(n,x)
      implicit none
      PetscScalar  x(*)
      PetscInt n
      PetscInt i
      PETSC_AssertAlignx(16,x(1))
      do 10,i=1,n
        x(i) = 0.0
 10   continue
      return 
      end


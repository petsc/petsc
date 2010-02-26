!
!
!    Fortran kernel for the MAXPY() vector routine
!
#include "finclude/petscsysdef.h"
!

      Subroutine FortranMAXPY4(x, a0, a1, a2, a3, y0, y1, y2, y3, n)
      implicit none
      PetscScalar  a0,a1,a2,a3
      PetscScalar  x(*),y0(*)
      PetscScalar y1(*),y2(*),y3(*)
      PetscInt n, i

      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y0(1))
      PETSC_AssertAlignx(16,y1(1))
      PETSC_AssertAlignx(16,Y2(1))
      PETSC_AssertAlignx(16,Y3(1))
      do i=1,n
          x(i)  = x(i) + a0*y0(i) + a1*y1(i) + a2*y2(i) + a3*y3(i)
      enddo
      return
      end

      subroutine FortranMAXPY3(x,a0,a1,a2,y0,y1,y2,n)
      implicit none
      PetscScalar  a0,a1,a2,x(*)
      PetscScalar y0(*),y1(*),y2(*)
      PetscInt n
      PetscInt i
      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y0(1))
      PETSC_AssertAlignx(16,y1(1))
      PETSC_AssertAlignx(16,y2(1))
      do 10,i=1,n
         x(i) = x(i) + a0*y0(i) + a1*y1(i) + a2*y2(i)
  10  continue
      return
      end

      Subroutine FortranMAXPY2(x, a0, a1, y0, y1, n)
      implicit none
      PetscScalar  a0,a1,x(*)
      PetscScalar  y0(*),y1(*)
      PetscInt n, i
      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y0(1))
      PETSC_AssertAlignx(16,y1(1))
      do i=1,n
          x(i)  = x(i) + a0*y0(i) + a1*y1(i) 
      enddo
      return
      end

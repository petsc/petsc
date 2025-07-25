!
!  Test Fortran binding of sort routines
!
module ex49fmodule
  use petsc
#include "petsc/finclude/petsc.h"
  implicit none
  type uctx
     PetscInt myint
  end type uctx
contains
  subroutine CompareIntegers(a,b,ctx,res)
    implicit none

    PetscInt :: a,b
    type(uctx) :: ctx
    integer  :: res

    if (a < b) then
       res = -1
    else if (a == b) then
       res = 0
    else
       res = 1
    end if
  end subroutine CompareIntegers
end module ex49fmodule

program main

  use ex49fmodule
  implicit none

  PetscErrorCode          ierr
  PetscCount,parameter::  iN = 3
  PetscInt, parameter ::  N = 3
  PetscInt                x(N),x1(N),y(N),z(N)
  PetscMPIInt             mx(N),my(N)
  PetscScalar             s(N)
  PetscReal               r(N)
  PetscMPIInt,parameter:: two=2, five=5, seven=7
  type(uctx)::            ctx
  PetscInt                i
  PetscSizeT              sizeofentry

  PetscCallA(PetscInitialize(ierr))

  x  = [3, 2, 1]
  x1 = [3, 2, 1]
  y  = [6, 5, 4]
  z  = [3, 5, 2]
  mx = [five, seven, two]
  my = [five, seven, two]
  s  = [1.0, 2.0, 3.0]
  r  = [1.0, 2.0, 3.0]
#if defined(PETSC_USE_64BIT_INDICES)
  sizeofentry = 8
#else
  sizeofentry = 4
#endif
  ctx%myint = 1
  PetscCallA(PetscSortInt(iN,x,ierr))
  PetscCallA(PetscTimSort(N,x1,sizeofentry,CompareIntegers,ctx,ierr))
  do i = 1,N
    PetscCheckA(x1(i) .eq. x(i),PETSC_COMM_SELF,PETSC_ERR_PLIB,'PetscTimSort and PetscSortInt arrays did not match')
  end do
  PetscCallA(PetscSortIntWithArray(iN,y,x,ierr))
  PetscCallA(PetscSortIntWithArrayPair(iN,x,y,z,ierr))

  PetscCallA(PetscSortMPIInt(iN,mx,ierr))
  PetscCallA(PetscSortMPIIntWithArray(iN,mx,my,ierr))
  PetscCallA(PetscSortMPIIntWithIntArray(iN,mx,y,ierr))

  PetscCallA(PetscSortIntWithScalarArray(iN,x,s,ierr))

  PetscCallA(PetscSortReal(iN,r,ierr))
  PetscCallA(PetscSortRealWithArrayInt(iN,r,x,ierr))

  PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!      output_file: output/empty.out
!
!TEST*/

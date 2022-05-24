!
!  Test Fortran binding of sort routines
!
module UserContext
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
    return
  end subroutine CompareIntegers
end module UserContext

program main

  use UserContext
  implicit none

  PetscErrorCode          ierr
  PetscInt,parameter::    N=3
  PetscMPIInt,parameter:: mN=3
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
  sizeofentry = 8;
#else
  sizeofentry = 4;
#endif
  ctx%myint = 1
  PetscCallA(PetscSortInt(N,x,ierr))
  PetscCallA(PetscTimSort(N,x1,sizeofentry,CompareIntegers,ctx,ierr))
  do i = 1,N
     if (x1(i) .ne. x(i)) then
        SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscTimSort and PetscSortInt arrays did not match")
     end if
  end do
  PetscCallA(PetscSortIntWithArray(N,y,x,ierr))
  PetscCallA(PetscSortIntWithArrayPair(N,x,y,z,ierr))

  PetscCallA(PetscSortMPIInt(N,mx,ierr))
  PetscCallA(PetscSortMPIIntWithArray(mN,mx,my,ierr))
  PetscCallA(PetscSortMPIIntWithIntArray(mN,mx,y,ierr))

  PetscCallA(PetscSortIntWithScalarArray(N,x,s,ierr))

  PetscCallA(PetscSortReal(N,r,ierr))
  PetscCallA(PetscSortRealWithArrayInt(N,r,x,ierr))

  PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!
!TEST*/

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
  PetscMPIInt             mx(N),my(N),mz(N)
  PetscScalar             s(N)
  PetscReal               r(N)
  PetscMPIInt,parameter:: two=2, five=5, seven=7
  type(uctx)::            ctx
  PetscInt                i
  PetscSizeT              sizeofentry

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  x  = [3, 2, 1]
  x1 = [3, 2, 1]
  y  = [6, 5, 4]
  z  = [3, 5, 2]
  mx = [five, seven, two]
  my = [five, seven, two]
  mz = [five, seven, two]
  s  = [1.0, 2.0, 3.0]
  r  = [1.0, 2.0, 3.0]
  sizeofentry = sizeof(dummyint)
  ctx%myint = 1
  call PetscSortInt(N,x,ierr)
  call PetscTimSort(N,x1,sizeofentry,CompareIntegers,ctx,ierr)
  do i = 1,N
     if (x1(i) .ne. x(i)) then
        SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscTimSort and PetscSortInt arrays did not match")
     end if
  end do
  call PetscSortIntWithArray(N,y,x,ierr)
  call PetscSortIntWithArrayPair(N,x,y,z,ierr)

  call PetscSortMPIInt(N,mx,ierr)
  call PetscSortMPIIntWithArray(mN,mx,my,ierr)
  call PetscSortMPIIntWithIntArray(mN,mx,y,ierr)

  call PetscSortIntWithScalarArray(N,x,s,ierr)

  call PetscSortReal(N,r,ierr)
  call PetscSortRealWithArrayInt(N,r,x,ierr)

  call PetscFinalize(ierr)
end program main

!/*TEST
!
!   test:
!
!TEST*/

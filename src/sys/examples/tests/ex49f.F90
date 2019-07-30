!
!  Test Fortran binding of sort routines
!
program main
#include "petsc/finclude/petsc.h"

  use petsc
  implicit none

  PetscErrorCode          ierr
  PetscInt,parameter::    N=3
  PetscMPIInt,parameter:: mN=3
  PetscInt                x(N),y(N),z(N)
  PetscMPIInt             mx(N),my(N),mz(N)
  PetscScalar             s(N)
  PetscReal               r(N)

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  x  = [3, 2, 1]
  y  = [6, 5, 4]
  z  = [3, 5, 2]
  mx = [5, 7, 2]
  my = [5, 7, 2]
  mz = [5, 7, 2]
  s  = [1.0, 2.0, 3.0]
  r  = [1.0, 2.0, 3.0]

  call PetscSortInt(N,x,ierr)
  call PetscSortIntWithArray(N,y,x,ierr)
  call PetscSortIntWithArrayPair(N,x,y,z,ierr)

  call PetscSortMPIInt(N,mx,ierr)
  call PetscSortMPIIntWithArray(mN,mx,my,ierr)
  call PetscSortMPIIntWithIntArray(mN,mx,y,ierr)

  call PetscSortIntWithScalarArray(N,x,s,ierr)

  call PetscSortReal(N,r,ierr)
  call PetscSortRealWithArrayInt(N,r,x,ierr)

  call PetscFinalize(ierr)
end program

!/*TEST
!
!   test:
!
!TEST*/

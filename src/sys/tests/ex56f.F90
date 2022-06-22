!
!  Tests PetscHasExternalPackage().
!
program main

#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      character(len=256)      pkg, outputString
      PetscBool               has,flg
      PetscErrorCode          ierr

      PetscCallA(PetscInitialize(ierr))
      pkg = "hdf5"
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-pkg",pkg,flg,ierr))
      PetscCallA(PetscHasExternalPackage(pkg,has,ierr))
      write (outputString,*) 'PETSc has '//trim(pkg)//'?',has,'\n'
      PetscCallA(PetscPrintf(PETSC_COMM_WORLD,outputString,ierr))
      PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!      suffix: blaslapack
!      args: -pkg blaslapack
!   test:
!      suffix: hdf5
!      requires: hdf5
!      args: -pkg hdf5
!   test:
!      suffix: no-hdf5
!      requires: !hdf5
!      args: -pkg hdf5
!   test:
!      suffix: parmetis
!      requires: parmetis
!      args: -pkg parmetis
!   test:
!      suffix: no-parmetis
!      requires: !parmetis
!      args: -pkg parmetis
!   test:
!      suffix: yaml
!      requires: yaml
!      args: -pkg yaml
!   test:
!      suffix: no-yaml
!      requires: !yaml
!      args: -pkg yaml
!
!TEST*/

!
!  Description: Demonstrates error handling with incorrect Fortran objects
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
      PetscErrorCode ierr
      PetscInt       test

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,ierr))

      test = 1
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-test', test, PETSC_NULL_BOOL, ierr))
      if (test == 1) then
        PetscCallA(KSPSolve(PETSC_NULL_KSP,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))
      else if (test == 3) then
        PetscCallA(KSPDestroy(PETSC_NULL_KSP,ierr))
      endif

!     These should error but do not
!     PetscCallA(KSPCreate(PETSC_COMM_WORLD,PETSC_NULL_KSP,ierr))
!       when ksp has not been created
!     PetscCallA(KSPSolve(ksp,PETSC_NULL_VEC,PETSC_NULL_VEC,ierr))
!     PetscCallA(KSPDestroy(ksp,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      args: -petsc_ci_portable_error_output -error_output_stdout -test 1
!      filter: grep -E "(PETSC ERROR)"
!
!   test:
!      suffix: 3
!      args: -petsc_ci_portable_error_output -error_output_stdout -test 3
!      filter: grep -E "(PETSC ERROR)"
!
!TEST*/

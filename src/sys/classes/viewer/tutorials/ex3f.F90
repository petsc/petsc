#include <petsc/finclude/petscsys.h>
program ex1f90
  use petscsys
  implicit none
  integer4 unit

  PetscErrorCode ierr
  PetscCallA(PetscInitialize(ierr))

  unit = 22
  call PetscViewerASCIIStdoutSetFileUnit(unit, ierr)
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!       requires: defined(PETSC_USE_LOG)
!       args: -log_view
!       output_file: output/empty.out
!
!TEST*/

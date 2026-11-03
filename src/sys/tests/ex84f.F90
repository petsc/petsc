#include <petsc/finclude/petscsys.h>
! Ensure matching C and Fortran constants (Fortran variant)
program ex84
  use PETScSys
  implicit none

  interface
    subroutine get_constants(pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity) bind(c)
      use PETScSys
      implicit none(type, external)
      PetscReal, intent(out) :: pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity

    end subroutine get_constants
  end interface

  PetscReal :: pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity
  PetscErrorCode :: ierr

  PetscCallA(PetscInitialize(ierr))
  call get_constants(pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity)

  PetscCheckA(pi == PETSC_PI, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_PI: value differs between C and Fortran')
  PetscCheckA(phi == PETSC_PHI, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_PHI: value differs between C and Fortran')
  PetscCheckA(sqrt2 == PETSC_SQRT2, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_SQRT2: value differs between C and Fortran')
  PetscCheckA(e == PETSC_E, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_E: value differs between C and Fortran')
  PetscCheckA(max_real == PETSC_MAX_REAL, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_MAX_REAL: value differs between C and Fortran')
  PetscCheckA(min_real == PETSC_MIN_REAL, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_MIN_REAL: value differs between C and Fortran')
  PetscCheckA(machine_epsilon == PETSC_MACHINE_EPSILON, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_MACHINE_EPSILON: value differs between C and Fortran')
  PetscCheckA(sqrt_machine_epsilon == PETSC_SQRT_MACHINE_EPSILON, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_SQRT_MACHINE_EPSILON: value differs between C and Fortran')
  PetscCheckA(small == PETSC_SMALL, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_SMALL: value differs between C and Fortran')
  PetscCheckA(infinity == PETSC_INFINITY, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_INFINITY: value differs between C and Fortran')
  PetscCheckA(ninfinity == PETSC_NINFINITY, MPI_COMM_WORLD, PETSC_ERR_PLIB, 'PETSC_NINFINITY: value differs between C and Fortran')
  PetscCallA(PetscFinalize(ierr))
end program ex84

!/*TEST
!
!   build:
!     depends: ex84.c
!
!   test:
!     output_file: output/empty.out
!
!TEST*/

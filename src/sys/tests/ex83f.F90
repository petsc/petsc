#include <petsc/finclude/petscsys.h>
module ex83f_module
  use PETScSys
  implicit none

contains

  subroutine get_constants(pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity) bind(c)
    PetscReal, intent(out) :: pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity

    pi = PETSC_PI
    phi = PETSC_PHI
    sqrt2 = PETSC_SQRT2
    e = PETSC_E
    max_real = PETSC_MAX_REAL
    min_real = PETSC_MIN_REAL
    machine_epsilon = PETSC_MACHINE_EPSILON
    sqrt_machine_epsilon = PETSC_SQRT_MACHINE_EPSILON
    small = PETSC_SMALL
    infinity = PETSC_INFINITY
    ninfinity = PETSC_NINFINITY
  end subroutine get_constants

end module ex83f_module

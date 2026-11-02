#include <petscsys.h>

PETSC_INTERN void get_constants(PetscReal *pi, PetscReal *phi, PetscReal *sqrt2, PetscReal *e, PetscReal *max_real, PetscReal *min_real, PetscReal *machine_epsilon, PetscReal *sqrt_machine_epsilon, PetscReal *small, PetscReal *infinity, PetscReal *ninfinity)
{
  *pi                   = PETSC_PI;
  *phi                  = PETSC_PHI;
  *sqrt2                = PETSC_SQRT2;
  *e                    = PETSC_E;
  *max_real             = PETSC_MAX_REAL;
  *min_real             = PETSC_MIN_REAL;
  *machine_epsilon      = PETSC_MACHINE_EPSILON;
  *sqrt_machine_epsilon = PETSC_SQRT_MACHINE_EPSILON;
  *small                = PETSC_SMALL;
  *infinity             = PETSC_INFINITY;
  *ninfinity            = PETSC_NINFINITY;
}

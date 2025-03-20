#include <petscvec.h>
#include <petscsection.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecgetvaluessection_     VECGETVALUESSECTION
  #define vecrestorevaluessection_ VECRESTOREVALUESSECTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecgetvaluessection_     vecgetvaluessection
  #define vecrestorevaluessection_ vecrestorevaluessection
#endif

PETSC_EXTERN void vecgetvaluessection_(Vec *v, PetscSection *section, PetscInt *point, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     len;

  *__ierr = VecGetValuesSection(*v, *section, *point, &fa);
  if (*__ierr) return;
  *__ierr = PetscSectionGetDof(*section, *point, &len);
  if (*__ierr) return;
  *__ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestorevaluessection_(Vec *v, PetscSection *section, PetscInt *point, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;

  *__ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*__ierr) return;
}

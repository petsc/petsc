#include <petscsection.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsectiongetconstraintindicesf90_           PETSCSECTIONGETCONSTRAINTINDICESF90
#define petscsectionrestoreconstraintindicesf90_       PETSCSECTIONRESTORECONSTRAINTINDICESF90
#define petscsectionsetconstraintindicesf90_           PETSCSECTIONSETCONSTRAINTINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectiongetconstraintindicesf90_           petscsectiongetconstraintindicesf90
#define petscsectionrestoreconstraintindicesf90_       petscsectionrestoreconstraintindicesf90
#define petscsectionsetconstraintindicesf90_           petscsectionsetconstraintindicesf90
#endif

PETSC_EXTERN void PETSC_STDCALL petscsectiongetconstraintindicesf90_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;

  *ierr = PetscSectionGetConstraintIndices(*s, *point, &idx); if (*ierr) return;
  *ierr = PetscSectionGetConstraintDof(*s, *point, &n); if (*ierr) return;
  *ierr = F90Array1dCreate((void *) idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL petscsectionrestoreconstraintindicesf90_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(indices, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL petscsectionsetconstraintindicesf90_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;

  *ierr = F90Array1dAccess(indices, MPIU_INT, (void **) &idx PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = PetscSectionSetConstraintIndices(*s, *point, idx); if (*ierr) return;
}

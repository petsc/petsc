#include <petscsection.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsectiongetconstraintindicesf90_           PETSCSECTIONGETCONSTRAINTINDICESF90
#define petscsectionrestoreconstraintindicesf90_       PETSCSECTIONRESTORECONSTRAINTINDICESF90
#define petscsectionsetconstraintindicesf90_           PETSCSECTIONSETCONSTRAINTINDICESF90
#define petscsectiongetfieldconstraintindicesf90_      PETSCSECTIONGETFIELDINDICESF90
#define petscsectionrestorefieldconstraintindicesf90_  PETSCSECTIONRESTOREFIELDCONSTRAINTINDICESF90
#define petscsectionsetfieldconstraintindicesf90_      PETSCSECTIONSETFIELDCONSTRAINTINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectiongetconstraintindicesf90_           petscsectiongetconstraintindicesf90
#define petscsectionrestoreconstraintindicesf90_       petscsectionrestoreconstraintindicesf90
#define petscsectionsetconstraintindicesf90_           petscsectionsetconstraintindicesf90
#define petscsectiongetfieldconstraintindicesf90_      petscsectiongetfieldconstraintindicesf90
#define petscsectionrestorefieldconstraintindicesf90_  petscsectionrestorefieldconstraintindicesf90
#define petscsectionsetfieldconstraintindicesf90_      petscsectionsetfieldconstraintindicesf90
#endif

PETSC_EXTERN void petscsectiongetconstraintindicesf90_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;

  *ierr = PetscSectionGetConstraintIndices(*s, *point, &idx); if (*ierr) return;
  *ierr = PetscSectionGetConstraintDof(*s, *point, &n); if (*ierr) return;
  *ierr = F90Array1dCreate((void *) idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscsectionrestoreconstraintindicesf90_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(indices, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void petscsectionsetconstraintindicesf90_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;

  *ierr = F90Array1dAccess(indices, MPIU_INT, (void **) &idx PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = PetscSectionSetConstraintIndices(*s, *point, idx); if (*ierr) return;
}

PETSC_EXTERN void petscsectiongetfieldconstraintindicesf90_(PetscSection *s, PetscInt *point, PetscInt *field, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;

  *ierr = PetscSectionGetFieldConstraintIndices(*s, *point, *field, &idx); if (*ierr) return;
  *ierr = PetscSectionGetFieldConstraintDof(*s, *point, *field, &n); if (*ierr) return;
  *ierr = F90Array1dCreate((void *) idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscsectionrestorefieldconstraintindicesf90_(PetscSection *s, PetscInt *point, PetscInt *field, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(indices, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void petscsectionsetfieldconstraintindicesf90_(PetscSection *s, PetscInt *point, PetscInt *field, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;

  *ierr = F90Array1dAccess(indices, MPIU_INT, (void **) &idx PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = PetscSectionSetFieldConstraintIndices(*s, *point, *field, idx); if (*ierr) return;
}

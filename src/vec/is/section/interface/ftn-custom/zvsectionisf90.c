#include <petscsection.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsectiongetconstraintindices_          PETSCSECTIONGETCONSTRAINTINDICES
  #define petscsectionrestoreconstraintindices_      PETSCSECTIONRESTORECONSTRAINTINDICES
  #define petscsectiongetfieldconstraintindices_     PETSCSECTIONGETFIELDCONSTRAINTINDICES
  #define petscsectionrestorefieldconstraintindices_ PETSCSECTIONRESTOREFIELDCONSTRAINTINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsectiongetconstraintindices_          petscsectiongetconstraintindices
  #define petscsectionrestoreconstraintindices_      petscsectionrestoreconstraintindices
  #define petscsectiongetfieldconstraintindices_     petscsectiongetfieldconstraintindices
  #define petscsectionrestorefieldconstraintindices_ petscsectionrestorefieldconstraintindices
#endif

PETSC_EXTERN void petscsectiongetconstraintindices_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;

  *ierr = PetscSectionGetConstraintIndices(*s, *point, &idx);
  if (*ierr) return;
  *ierr = PetscSectionGetConstraintDof(*s, *point, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscsectionrestoreconstraintindices_(PetscSection *s, PetscInt *point, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(indices, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscsectiongetfieldconstraintindices_(PetscSection *s, PetscInt *point, PetscInt *field, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;

  *ierr = PetscSectionGetFieldConstraintIndices(*s, *point, *field, &idx);
  if (*ierr) return;
  *ierr = PetscSectionGetFieldConstraintDof(*s, *point, *field, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscsectionrestorefieldconstraintindices_(PetscSection *s, PetscInt *point, PetscInt *field, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(indices, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

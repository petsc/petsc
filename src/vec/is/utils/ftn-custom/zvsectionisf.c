#include <petsc/private/ftnimpl.h>
#include <petscis.h>
#include <petscsection.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsectiongetpointsyms_          PETSCSECTIONGETPOINTSYMS
  #define petscsectionrestorepointsyms_      PETSCSECTIONRESTOREPOINTSYMS
  #define petscsectiongetfieldpointsyms_     PETSCSECTIONGETFIELDPOINTSYMS
  #define petscsectionrestorefieldpointsyms_ PETSCSECTIONRESTOREFIELDPOINTSYMS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsectiongetpointsyms_          petscsectiongetpointsyms
  #define petscsectionrestorepointsyms_      petscsectionrestorepointsyms
  #define petscsectiongetfieldpointsyms_     petscsectiongetfieldpointsyms
  #define petscsectionrestorefieldpointsyms_ petscsectionrestorefieldpointsyms
#endif

PETSC_EXTERN void petscsectiongetpointsyms_(PetscSection section, PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr)
{
  *__ierr = PetscSectionGetPointSyms(section, *numPoints, points, (const PetscInt ***)perms, (const PetscScalar ***)rots);
}
PETSC_EXTERN void petscsectionrestorepointsyms_(PetscSection section, PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr)
{
  *__ierr = PetscSectionRestorePointSyms(section, *numPoints, points, (const PetscInt ***)perms, (const PetscScalar ***)rots);
}
PETSC_EXTERN void petscsectiongetfieldpointsyms_(PetscSection section, PetscInt *field, PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr)
{
  *__ierr = PetscSectionGetFieldPointSyms(section, *field, *numPoints, points, (const PetscInt ***)perms, (const PetscScalar ***)rots);
}
PETSC_EXTERN void petscsectionrestorefieldpointsyms_(PetscSection section, PetscInt *field, PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr)
{
  *__ierr = PetscSectionRestoreFieldPointSyms(section, *field, *numPoints, points, (const PetscInt ***)perms, (const PetscScalar ***)rots);
}

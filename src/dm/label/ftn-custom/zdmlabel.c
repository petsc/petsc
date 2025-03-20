#include <petsc/private/ftnimpl.h>
#include <petscdmlabel.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsectionsymlabelsetstratum_ PETSCSECTIONSYMLABELSETSTRATUM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsectionsymlabelsetstratum_ petscsectionsymlabelsetstratum
#endif

PETSC_EXTERN void petscsectionsymlabelsetstratum_(PetscSectionSym *sym, PetscInt *stratum, PetscInt *size, PetscInt *minOrient, PetscInt *maxOrient, PetscCopyMode *mode, PetscInt **perms, PetscScalar **rots, int *__ierr)
{
  *__ierr = PetscSectionSymLabelSetStratum(*sym, *stratum, *size, *minOrient, *maxOrient, *mode, (const PetscInt **)perms, (const PetscScalar **)rots);
}

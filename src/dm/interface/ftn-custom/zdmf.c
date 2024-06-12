#include <petsc/private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmcreatesuperdm_ DMCREATESUPERDM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmcreatesuperdm_ dmreatesuperdm
#endif

PETSC_EXTERN void dmcreatesuperdm_(DM dms[], PetscInt *len, IS ***is, DM *superdm, int *ierr)
{
  *ierr = DMCreateSuperDM(dms, *len, *is, superdm);
}

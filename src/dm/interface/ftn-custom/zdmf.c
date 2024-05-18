#include <petsc/private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmcreatesuperdm_ DMCREATESUPERDM
  #define dmcreatesubdm_   DMCREATESUBDM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmcreatesuperdm_ dmreatesuperdm
  #define dmcreatesubdm_   dmreatesubdm
#endif

PETSC_EXTERN void dmcreatesuperdm_(DM dms[], PetscInt *len, IS ***is, DM *superdm, int *ierr)
{
  *ierr = DMCreateSuperDM(dms, *len, *is, superdm);
}

PETSC_EXTERN void dmcreatesubdm_(DM *dm, PetscInt *numFields, PetscInt fields[], IS *is, DM *subdm, int *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  *ierr = DMCreateSubDM(*dm, *numFields, fields, is, subdm);
}

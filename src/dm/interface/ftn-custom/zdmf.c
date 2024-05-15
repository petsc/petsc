#include <petsc/private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmcreatesuperdm_ DMCREATESUPERDM
  #define dmcreatesubdm_   DMCREATESUBDM
  #define dmdestroy_       DMDESTROY
  #define dmload_          DMLOAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmcreatesuperdm_ dmreatesuperdm
  #define dmcreatesubdm_   dmreatesubdm
  #define dmdestroy_       dmdestroy
  #define dmload_          dmload
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

PETSC_EXTERN void dmdestroy_(DM *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = DMDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}

#include <petsc/private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmcreateinterpolation_ DMCREATEINTERPOLATION
  #define dmview_                DMVIEW
  #define dmlabelview_           DMLABELVIEW
  #define dmviewfromoptions_     DMVIEWFROMOPTIONS
  #define dmcreatesuperdm_       DMCREATESUPERDM
  #define dmcreatesubdm_         DMCREATESUBDM
  #define dmdestroy_             DMDESTROY
  #define dmload_                DMLOAD
  #define dmsetfield_            DMSETFIELD
  #define dmaddfield_            DMADDFIELD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmcreateinterpolation_ dmcreateinterpolation
  #define dmview_                dmview
  #define dmlabelview_           dmlabelview
  #define dmviewfromoptions_     dmviewfromoptions
  #define dmcreatesuperdm_       dmreatesuperdm
  #define dmcreatesubdm_         dmreatesubdm
  #define dmdestroy_             dmdestroy
  #define dmload_                dmload
  #define dmsetfield_            dmsetfield
  #define dmaddfield_            dmaddfield
#endif

PETSC_EXTERN void dmsetfield_(DM *dm, PetscInt *f, DMLabel label, PetscObject *disc, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(label);
  *ierr = DMSetField(*dm, *f, label, *disc);
}

PETSC_EXTERN void dmaddfield_(DM *dm, DMLabel label, PetscObject *disc, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(label);
  *ierr = DMAddField(*dm, label, *disc);
}

PETSC_EXTERN void dmload_(DM *dm, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = DMLoad(*dm, v);
}

PETSC_EXTERN void dmview_(DM *da, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = DMView(*da, v);
}

PETSC_EXTERN void dmviewfromoptions_(DM *dm, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = DMViewFromOptions(*dm, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void dmcreateinterpolation_(DM *dmc, DM *dmf, Mat *mat, Vec *vec, int *ierr)
{
  CHKFORTRANNULLOBJECT(vec);
  *ierr = DMCreateInterpolation(*dmc, *dmf, mat, vec);
}

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

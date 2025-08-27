#include <petsc/private/ftnimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmcreatesuperdm_                       DMCREATESUPERDM
  #define dmcreatefielddecompositiongetname_     DMCREATEFIELDDECOMPOSITIONGETNAME
  #define dmcreatefielddecompositiongetisdm_     DMCREATEFIELDDECOMPOSITIONGETISDM
  #define dmcreatefielddecompositionrestoreisdm_ DMCREATEFIELDDECOMPOSITIONRESTOREISDM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmcreatesuperdm_                       dmreatesuperdm
  #define dmcreatefielddecompositiongetname_     dmcreatefielddecompositiongetname
  #define dmcreatefielddecompositiongetisdm_     dmcreatefielddecompositiongetisdm
  #define dmcreatefielddecompositionrestoreisdm_ dmcreatefielddecompositionrestoreisdm
#endif

PETSC_EXTERN void dmcreatesuperdm_(DM dms[], PetscInt *len, IS ***is, DM *superdm, PetscErrorCode *ierr)
{
  *ierr = DMCreateSuperDM(dms, *len, *is, superdm);
}

PETSC_EXTERN void dmcreatefielddecompositiongetname_(DM *dm, PetscInt *i, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T l_b)
{
  PetscInt n;
  char   **names;
  *ierr = DMCreateFieldDecomposition(*dm, &n, &names, NULL, NULL);
  if (*ierr) return;
  *ierr = PetscStrncpy((char *)name, names[*i - 1], l_b);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, l_b);
  for (PetscInt j = 0; j < n; j++) *ierr = PetscFree(names[j]);
  *ierr = PetscFree(names);
}

PETSC_EXTERN void dmcreatefielddecompositiongetisdm_(DM *dm, F90Array1d *iss, F90Array1d *dms, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscInt n;
  IS      *tis;
  DM      *tdm;

  if (iss && dms) {
    *ierr = DMCreateFieldDecomposition(*dm, &n, NULL, &tis, &tdm);
  } else if (iss) {
    *ierr = DMCreateFieldDecomposition(*dm, &n, NULL, &tis, NULL);
  } else if (dms) {
    *ierr = DMCreateFieldDecomposition(*dm, &n, NULL, NULL, &tdm);
  }
  if (*ierr) return;
  if (iss) *ierr = F90Array1dCreate(tis, MPIU_FORTRANADDR, 1, n, iss PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  if (dms) *ierr = F90Array1dCreate(tdm, MPIU_FORTRANADDR, 1, n, dms PETSC_F90_2PTR_PARAM(ptrd2));
}

PETSC_EXTERN void dmcreatefielddecompositionrestoreisdm_(DM *dm, F90Array1d *iss, F90Array1d *dms, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscInt n;

  *ierr = DMGetNumFields(*dm, &n);
  if (*ierr) return;
  if (iss) {
    IS *tis;
    *ierr = F90Array1dAccess(iss, MPIU_FORTRANADDR, (void **)&tis PETSC_F90_2PTR_PARAM(ptrd1));
    if (*ierr) return;
    *ierr = F90Array1dDestroy(iss, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd1));
    if (*ierr) return;
    for (PetscInt i = 0; i < n; i++) *ierr = ISDestroy(&tis[i]);
    *ierr = PetscFree(tis);
    if (*ierr) return;
  }
  if (dms) {
    DM *tdm;
    *ierr = F90Array1dAccess(dms, MPIU_FORTRANADDR, (void **)&tdm PETSC_F90_2PTR_PARAM(ptrd2));
    if (*ierr) return;
    *ierr = F90Array1dDestroy(dms, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd2));
    if (*ierr) return;
    for (PetscInt i = 0; i < n; i++) *ierr = DMDestroy(&tdm[i]);
    *ierr = PetscFree(tdm);
    if (*ierr) return;
  }
}

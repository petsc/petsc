#include <petsc/private/fortranimpl.h>
#include <petscdmlabel.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmlabelview_                    DMLABELVIEW
#define petscsectionsymlabelsetstratum_ PETSCSECTIONSYMLABELSETSTRATUM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmlabelview_                    dmlabelview
#define petscsectionsymlabelsetstratum_ petscsectionsymlabelsetstratum
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmlabelview_(DMLabel *label, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = DMLabelView(*label, v);
}

PETSC_EXTERN void  petscsectionsymlabelsetstratum_(PetscSectionSym *sym,PetscInt *stratum,PetscInt *size,PetscInt *minOrient,PetscInt *maxOrient,PetscCopyMode *mode, PetscInt **perms, PetscScalar **rots, int *__ierr){
*__ierr = PetscSectionSymLabelSetStratum(*sym,*stratum,*size,*minOrient,*maxOrient,*mode,(const PetscInt **)perms,(const PetscScalar **)rots);
}

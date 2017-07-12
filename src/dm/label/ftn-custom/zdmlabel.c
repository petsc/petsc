#include <petsc/private/fortranimpl.h>
#include <petscdmlabel.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmlabelview_                    DMLABELVIEW
#define petscsectionsymlabelsetstratum_ PETSCSECTIONSYMLABELSETSTRATUM
#define dmlabelgetname_                 DMLABELGETNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmlabelview_                    dmlabelview
#define petscsectionsymlabelsetstratum_ petscsectionsymlabelsetstratum
#define dmlabelgetname_                 dmlabelgetname
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmlabelview_(DMLabel *label, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = DMLabelView(*label, v);
}

PETSC_EXTERN void PETSC_STDCALL  petscsectionsymlabelsetstratum_(PetscSectionSym *sym,PetscInt *stratum,PetscInt *size,PetscInt *minOrient,PetscInt *maxOrient,PetscCopyMode *mode, PetscInt **perms, PetscScalar **rots, int *__ierr ){
*__ierr = PetscSectionSymLabelSetStratum(*sym,*stratum,*size,*minOrient,*maxOrient,*mode,(const PetscInt **)perms,(const PetscScalar **)rots);
}

PETSC_EXTERN void PETSC_STDCALL dmlabelgetname_(DMLabel *label,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tmp;
  *ierr = DMLabelGetName(*label,&tmp);
  *ierr = PetscStrncpy(name,tmp,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

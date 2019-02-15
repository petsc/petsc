#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcfieldsplitgetsubksp_        PCFIELDSPLITGETSUBKSP
#define pcfieldsplitschurgetsubksp_   PCFIELDSPLITSCHURGETSUBKSP
#define pcfieldsplitsetis_            PCFIELDSPLITSETIS
#define pcfieldsplitgetis_            PCFIELDSPLITGETIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcfieldsplitgetsubksp_        pcfieldsplitgetsubksp
#define pcfieldsplitschurgetsubksp_   pcfieldsplitschurgetsubksp
#define pcfieldsplitsetis_            pcfieldsplitsetis
#define pcfieldsplitgetis_            pcfieldsplitgetis
#endif

PETSC_EXTERN void PETSC_STDCALL pcfieldsplitschurgetsubksp_(PC *pc,PetscInt *n_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP      *tksp;
  PetscInt i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  *ierr = PCFieldSplitSchurGetSubKSP(*pc,&nloc,&tksp); if (*ierr) return;
  if (n_local) *n_local = nloc;
  CHKFORTRANNULLOBJECT(ksp);
  if (ksp) {
    for (i=0; i<nloc; i++) ksp[i] = tksp[i];
  }
  *ierr = PetscFree(tksp);
}

PETSC_EXTERN void PETSC_STDCALL pcfieldsplitgetsubksp_(PC *pc,PetscInt *n_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP      *tksp;
  PetscInt i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  *ierr = PCFieldSplitGetSubKSP(*pc,&nloc,&tksp); if (*ierr) return;
  if (n_local) *n_local = nloc;
  CHKFORTRANNULLOBJECT(ksp);
  if (ksp) {
    for (i=0; i<nloc; i++) ksp[i] = tksp[i];
  }
  *ierr = PetscFree(tksp);
}

PETSC_EXTERN void PETSC_STDCALL  pcfieldsplitsetis_(PC *pc, char* splitname PETSC_MIXED_LEN(len),IS *is, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(splitname,len,t);
  *ierr = PCFieldSplitSetIS(*pc,t,*is);if (*ierr) return;
  FREECHAR(splitname,t);
}


PETSC_EXTERN void PETSC_STDCALL  pcfieldsplitgetis_(PC *pc, char* splitname PETSC_MIXED_LEN(len),IS *is, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(splitname,len,t);
  *ierr = PCFieldSplitGetIS(*pc,t,is);if (*ierr) return;
  FREECHAR(splitname,t);
}




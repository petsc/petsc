#include <petsc-private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcfieldsplitgetsubksp_        PCFIELDSPLITGETSUBKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcfieldsplitgetsubksp_        pcfieldsplitgetsubksp
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcfieldsplitgetsubksp_(PC *pc,PetscInt *n_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP *tksp;
  PetscInt  i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  *ierr = PCFieldSplitGetSubKSP(*pc,&nloc,&tksp); if (*ierr) return;
  if (n_local) *n_local = nloc;
  CHKFORTRANNULLOBJECT(ksp);
  if (ksp) {
    for (i=0; i<nloc; i++){
      ksp[i] = tksp[i];
    }
  }
}

EXTERN_C_END

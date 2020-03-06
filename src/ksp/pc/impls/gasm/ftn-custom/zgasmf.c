#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcgasmgetsubksp_            PCGASMGETSUBKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcgasmgetsubksp_            pcgasmgetsubksp
#endif

PETSC_EXTERN void pcgasmgetsubksp_(PC *pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP      *tksp;
  PetscInt i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  CHKFORTRANNULLOBJECT(ksp);
  *ierr = PCGASMGetSubKSP(*pc,&nloc,first_local,&tksp);
  if (n_local) *n_local = nloc;
  if (ksp) {
    for (i=0; i<nloc; i++) ksp[i] = tksp[i];
  }
}



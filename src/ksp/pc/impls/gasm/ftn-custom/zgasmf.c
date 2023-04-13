#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcgasmgetsubksp1_ PCGASMGETSUBKSP1
  #define pcgasmgetsubksp2_ PCGASMGETSUBKSP2
  #define pcgasmgetsubksp3_ PCGASMGETSUBKSP3
  #define pcgasmgetsubksp4_ PCGASMGETSUBKSP4
  #define pcgasmgetsubksp5_ PCGASMGETSUBKSP5
  #define pcgasmgetsubksp6_ PCGASMGETSUBKSP6
  #define pcgasmgetsubksp7_ PCGASMGETSUBKSP7
  #define pcgasmgetsubksp8_ PCGASMGETSUBKSP8
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcgasmgetsubksp1_ pcgasmgetsubksp1
  #define pcgasmgetsubksp2_ pcgasmgetsubksp2
  #define pcgasmgetsubksp3_ pcgasmgetsubksp3
  #define pcgasmgetsubksp4_ pcgasmgetsubksp4
  #define pcgasmgetsubksp5_ pcgasmgetsubksp5
  #define pcgasmgetsubksp6_ pcgasmgetsubksp6
  #define pcgasmgetsubksp7_ pcgasmgetsubksp7
  #define pcgasmgetsubksp8_ pcgasmgetsubksp8
#endif

PETSC_EXTERN void pcgasmgetsubksp1_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  KSP     *tksp;
  PetscInt i, nloc;
  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  CHKFORTRANNULLOBJECT(ksp);
  *ierr = PCGASMGetSubKSP(*pc, &nloc, first_local, &tksp);
  if (n_local) *n_local = nloc;
  if (ksp) {
    for (i = 0; i < nloc; i++) ksp[i] = tksp[i];
  }
}

PETSC_EXTERN void pcgasmgetsubksp2_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

PETSC_EXTERN void pcgasmgetsubksp3_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

PETSC_EXTERN void pcgasmgetsubksp4_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

PETSC_EXTERN void pcgasmgetsubksp5_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

PETSC_EXTERN void pcgasmgetsubksp6_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

PETSC_EXTERN void pcgasmgetsubksp7_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

PETSC_EXTERN void pcgasmgetsubksp8_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcgasmgetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

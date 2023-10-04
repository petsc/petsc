#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcbjacobigetsubksp1_ PCBJACOBIGETSUBKSP1
  #define pcbjacobigetsubksp2_ PCBJACOBIGETSUBKSP2
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcbjacobigetsubksp1_ pcbjacobigetsubksp1
  #define pcbjacobigetsubksp2_ pcbjacobigetsubksp2
#endif

PETSC_EXTERN void pcbjacobigetsubksp1_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  KSP     *tksp;
  PetscInt i, nloc;
  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCBJacobiGetSubKSP(*pc, &nloc, first_local, &tksp);
  if (*ierr) return;
  if (n_local) *n_local = nloc;
  CHKFORTRANNULLOBJECT(ksp);
  if (ksp) {
    for (i = 0; i < nloc; i++) { ksp[i] = tksp[i]; }
  }
}
PETSC_EXTERN void pcbjacobigetsubksp2_(PC *pc, PetscInt *n_local, PetscInt *first_local, KSP *ksp, PetscErrorCode *ierr)
{
  pcbjacobigetsubksp1_(pc, n_local, first_local, ksp, ierr);
}

#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcgasmsetsubdomains_      PCGASMSETSUBDOMAINS
  #define pcgasmdestroysubdomains_  PCGASMDESTROYSUBDOMAINS
  #define pcgasmgetsubksp1_         PCGASMGETSUBKSP1
  #define pcgasmgetsubksp2_         PCGASMGETSUBKSP2
  #define pcgasmgetsubksp3_         PCGASMGETSUBKSP3
  #define pcgasmgetsubksp4_         PCGASMGETSUBKSP4
  #define pcgasmgetsubksp5_         PCGASMGETSUBKSP5
  #define pcgasmgetsubksp6_         PCGASMGETSUBKSP6
  #define pcgasmgetsubksp7_         PCGASMGETSUBKSP7
  #define pcgasmgetsubksp8_         PCGASMGETSUBKSP8
  #define pcgasmcreatesubdomains2d_ PCGASMCREATESUBDOMAINS2D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcgasmsetsubdomains_      pcgasmsetsubdomains
  #define pcgasmdestroysubdomains_  pcgasmdestroysubdomains
  #define pcgasmgetsubksp2_         pcgasmgetsubksp2
  #define pcgasmgetsubksp3_         pcgasmgetsubksp3
  #define pcgasmgetsubksp4_         pcgasmgetsubksp4
  #define pcgasmgetsubksp5_         pcgasmgetsubksp5
  #define pcgasmgetsubksp6_         pcgasmgetsubksp6
  #define pcgasmgetsubksp7_         pcgasmgetsubksp7
  #define pcgasmgetsubksp8_         pcgasmgetsubksp8
  #define pcgasmcreatesubdomains2d_ pcgasmcreatesubdomains2d
#endif

PETSC_EXTERN void pcgasmsetsubdomains_(PC *pc, PetscInt *n, IS *is, IS *isl, int *ierr)
{
  *ierr = PCGASMSetSubdomains(*pc, *n, is, isl);
}

PETSC_EXTERN void pcgasmdestroysubdomains_(PetscInt *n, IS *is, IS *isl, int *ierr)
{
  IS *iis, *iisl;
  *ierr = PetscMalloc1(*n, &iis);
  if (*ierr) return;
  *ierr = PetscArraycpy(iis, is, *n);
  if (*ierr) return;
  *ierr = PetscMalloc1(*n, &iisl);
  if (*ierr) return;
  *ierr = PetscArraycpy(iisl, isl, *n);
  *ierr = PCGASMDestroySubdomains(*n, &iis, &iisl);
}

PETSC_EXTERN void pcgasmcreatesubdomains2d_(PC *pc, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N, PetscInt *dof, PetscInt *overlap, PetscInt *Nsub, IS *is, IS *isl, int *ierr)
{
  IS *iis, *iisl;
  *ierr = PCGASMCreateSubdomains2D(*pc, *m, *n, *M, *N, *dof, *overlap, Nsub, &iis, &iisl);
  if (*ierr) return;
  *ierr = PetscArraycpy(is, iis, *Nsub);
  if (*ierr) return;
  *ierr = PetscArraycpy(isl, iisl, *Nsub);
  if (*ierr) return;
  *ierr = PetscFree(iis);
  if (*ierr) return;
  *ierr = PetscFree(iisl);
}

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

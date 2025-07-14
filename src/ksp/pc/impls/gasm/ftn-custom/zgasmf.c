#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcgasmdestroysubdomains_  PCGASMDESTROYSUBDOMAINS
  #define pcgasmgetsubksp_          PCGASMGETSUBKSP
  #define pcgasmrestoresubksp_      PCGASMRESTORESUBKSP
  #define pcgasmcreatesubdomains2d_ PCGASMCREATESUBDOMAINS2D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcgasmdestroysubdomains_  pcgasmdestroysubdomains
  #define pcgasmgetsubksp_          pcgasmgetsubksp
  #define pcgasmrestoresubksp_      pcgasmrestoresubksp
  #define pcgasmcreatesubdomains2d_ pcgasmcreatesubdomains2d
#endif

PETSC_EXTERN void pcgasmdestroysubdomains_(PetscInt *n, F90Array1d *is1, F90Array1d *is2, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  IS *isa, *isb;

  *ierr = F90Array1dAccess(is1, MPIU_FORTRANADDR, (void **)&isa PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dAccess(is2, MPIU_FORTRANADDR, (void **)&isb PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(is1, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(is2, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
  *ierr = PCGASMDestroySubdomains(*n, &isa, &isb);
}

PETSC_EXTERN void pcgasmcreatesubdomains2d_(PC *pc, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N, PetscInt *dof, PetscInt *overlap, PetscInt *Nsub, F90Array1d *is1, F90Array1d *is2, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  IS *iis, *iisl;
  *ierr = PCGASMCreateSubdomains2D(*pc, *m, *n, *M, *N, *dof, *overlap, Nsub, &iis, &iisl);
  if (*ierr) return;
  *ierr = F90Array1dCreate(iis, MPIU_FORTRANADDR, 1, *Nsub, is1 PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dCreate(iisl, MPIU_FORTRANADDR, 1, *Nsub, is2 PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
}

PETSC_EXTERN void pcgasmgetsubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc, flocal;

  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCGASMGetSubKSP(*pc, &nloc, &flocal, &tksp);
  if (n_local) *n_local = nloc;
  if (first_local) *first_local = flocal;
  *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcgasmrestoresubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ksp, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}

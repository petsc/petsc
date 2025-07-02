#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcasmgetsubksp_           PCASMGETSUBKSP
  #define pcasmrestoresubksp_       PCASMRESTORESUBKSP
  #define pcasmgetlocalsubmatrices_ PCASMGETLOCALSUBMATRICES
  #define pcasmgetlocalsubdomains_  PCASMGETLOCALSUBDOMAINS
  #define pcasmcreatesubdomains_    PCASMCREATESUBDOMAINS
  #define pcasmdestroysubdomains_   PCASMDESTROYSUBDOMAINS
  #define pcasmcreatesubdomains2d_  PCASMCREATESUBDOMAINS2D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcasmgetsubksp_           pcasmgetsubksp
  #define pcasmrestoresubksp_       pcasmrestoresubksp
  #define pcasmgetlocalsubmatrices_ pcasmgetlocalsubmatrices
  #define pcasmgetlocalsubdomains_  pcasmgetlocalsubdomains
  #define pcasmcreatesubdomains_    pcasmcreatesubdomains
  #define pcasmdestroysubdomains_   pcasmdestroysubdomains
  #define pcasmcreatesubdomains2d_  pcasmcreatesubdomains2d
#endif

PETSC_EXTERN void pcasmcreatesubdomains_(Mat *mat, PetscInt n, F90Array1d *is, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1))
{
  IS *insubs;

  CHKFORTRANNULLOBJECT(is);
  *ierr = PCASMCreateSubdomains(*mat, n, &insubs);
  if (*ierr) return;
  if (insubs) *ierr = F90Array1dCreate(insubs, MPIU_FORTRANADDR, 1, n, is PETSC_F90_2PTR_PARAM(ptrd1));
}

PETSC_EXTERN void pcasmgetlocalsubmatrices_(PC *pc, PetscInt *n, F90Array1d *mat, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt nloc;
  Mat     *tmat;

  CHKFORTRANNULLOBJECT(mat);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubmatrices(*pc, &nloc, &tmat);
  if (n) *n = nloc;
  if (mat) *ierr = F90Array1dCreate(tmat, MPIU_FORTRANADDR, 1, nloc, mat PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcasmgetlocalsubdomains_(PC *pc, PetscInt *n, F90Array1d *is, F90Array1d *is_local, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscInt nloc;
  IS      *tis, *tis_local;

  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLOBJECT(is_local);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubdomains(*pc, &nloc, &tis, &tis_local);
  if (*ierr) return;
  if (n) *n = nloc;
  if (is) *ierr = F90Array1dCreate(tis, MPIU_FORTRANADDR, 1, nloc, is PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  if (is_local) *ierr = F90Array1dCreate(tis_local, MPIU_FORTRANADDR, 1, nloc, is_local PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
}

PETSC_EXTERN void pcasmdestroysubdomains_(PetscInt *n, F90Array1d *is1, F90Array1d *is2, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  IS *isa, *isb;

  *ierr = F90Array1dAccess(is1, MPIU_FORTRANADDR, (void **)&isa PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dAccess(is2, MPIU_FORTRANADDR, (void **)&isb PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
  *ierr = PCASMDestroySubdomains(*n, &isa, &isb);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(is1, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(is2, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
}

PETSC_EXTERN void pcasmcreatesubdomains2d_(PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N, PetscInt *dof, PetscInt *overlap, PetscInt *Nsub, F90Array1d *is1, F90Array1d *is2, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  IS *iis, *iisl;

  *ierr = PCASMCreateSubdomains2D(*m, *n, *M, *N, *dof, *overlap, Nsub, &iis, &iisl);
  if (*ierr) return;
  *ierr = F90Array1dCreate(iis, MPIU_FORTRANADDR, 1, *Nsub, is1 PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dCreate(iisl, MPIU_FORTRANADDR, 1, *Nsub, is2 PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
}

PETSC_EXTERN void pcasmgetsubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc, flocal;

  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCASMGetSubKSP(*pc, &nloc, &flocal, &tksp);
  if (n_local) *n_local = nloc;
  if (first_local) *first_local = flocal;
  *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcasmrestoresubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ksp, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}

#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define pcasmgetsubksp_                 PCASMGETSUBKSP
  #define pcasmrestoresubksp_             PCASMRESTORESUBKSP
  #define pcasmgetlocalsubmatrices_       PCASMGETLOCALSUBMATRICES
  #define pcasmgetlocalsubdomains_        PCASMGETLOCALSUBDOMAINS
  #define pcasmweightedgetscaling_        PCASMWEIGHTEDGETSCALING
  #define pcasmweightedsetcomputescaling_ PCASMWEIGHTEDSETCOMPUTESCALING
  #define pcasmcreatesubdomains_          PCASMCREATESUBDOMAINS
  #define pcasmdestroysubdomains_         PCASMDESTROYSUBDOMAINS
  #define pcasmcreatesubdomains2d_        PCASMCREATESUBDOMAINS2D
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define pcasmgetsubksp_                 pcasmgetsubksp
  #define pcasmrestoresubksp_             pcasmrestoresubksp
  #define pcasmgetlocalsubmatrices_       pcasmgetlocalsubmatrices
  #define pcasmgetlocalsubdomains_        pcasmgetlocalsubdomains
  #define pcasmweightedgetscaling_        pcasmweightedgetscaling
  #define pcasmweightedsetcomputescaling_ pcasmweightedsetcomputescaling
  #define pcasmcreatesubdomains_          pcasmcreatesubdomains
  #define pcasmdestroysubdomains_         pcasmdestroysubdomains
  #define pcasmcreatesubdomains2d_        pcasmcreatesubdomains2d
#endif

PETSC_EXTERN void pcasmcreatesubdomains_(Mat *A, PetscInt *n, F90Array1d *outis, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1))
{
  IS *insubs;

  if (FORTRANNULLISPOINTER(outis)) {
    *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_NULL, PETSC_ERROR_INITIAL, "PCASMCreateSubdomains() requires an output array; do not use PETSC_NULL_IS_POINTER");
    return;
  }
  *ierr = PCASMCreateSubdomains(*A, *n, &insubs);
  if (*ierr) return;
  *ierr = F90Array1dCreate(insubs, MPIU_FORTRANADDR, 1, *n, outis PETSC_F90_2PTR_PARAM(ptrd1));
}

PETSC_EXTERN void pcasmgetlocalsubmatrices_(PC *pc, PetscInt *n, F90Array1d *mat, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt nloc;
  Mat     *tmat;

  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubmatrices(*pc, &nloc, &tmat);
  if (*ierr) return;
  if (n) *n = nloc;
  if (FORTRANNULLMATPOINTER(mat)) return;
  if (tmat) *ierr = F90Array1dCreate(tmat, MPIU_FORTRANADDR, 1, nloc, mat PETSC_F90_2PTR_PARAM(ptrd));
  else f90array1ddestroyfortranaddr_(mat PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcasmweightedgetscaling_(PC *pc, PetscInt *n, F90Array1d *scaling, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt nloc;
  Vec     *tscaling;

  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMWeightedGetScaling(*pc, &nloc, &tscaling);
  if (*ierr) return;
  if (n) *n = nloc;
  if (tscaling) *ierr = F90Array1dCreate(tscaling, MPIU_FORTRANADDR, 1, nloc, scaling PETSC_F90_2PTR_PARAM(ptrd));
  else f90array1ddestroyfortranaddr_(scaling PETSC_F90_2PTR_PARAM(ptrd));
}

static struct {
  PetscFortranCallbackId computescaling;
} _cb;

static PetscErrorCode ourcomputescaling(PC pc, PetscInt local, Vec scaling, PETSC_UNUSED PetscCtx ctx)
{
  PetscObjectUseFortranCallbackSubType(pc, _cb.computescaling, (PC *, PetscInt *, Vec *, void *, PetscErrorCode *), (&pc, &local, &scaling, _ctx, &ierr));
}

PETSC_EXTERN void pcasmweightedsetcomputescaling_(PC *pc, void (*fn)(PC *, PetscInt *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(fn);
  if (!fn) {
    *ierr = PCASMWeightedSetComputeScaling(*pc, NULL, NULL);
    return;
  }
  *ierr = PetscObjectSetFortranCallback((PetscObject)*pc, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.computescaling, (PetscFortranCallbackFn *)fn, ctx);
  if (*ierr) return;
  *ierr = PCASMWeightedSetComputeScaling(*pc, ourcomputescaling, NULL);
}

PETSC_EXTERN void pcasmgetlocalsubdomains_(PC *pc, PetscInt *n, F90Array1d *is, F90Array1d *is_local, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscInt nloc;
  IS      *tis, *tis_local;

  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubdomains(*pc, &nloc, &tis, &tis_local);
  if (*ierr) return;
  if (n) *n = nloc;
  if (!FORTRANNULLISPOINTER(is)) {
    if (tis) *ierr = F90Array1dCreate(tis, MPIU_FORTRANADDR, 1, nloc, is PETSC_F90_2PTR_PARAM(ptrd1));
    else f90array1ddestroyfortranaddr_(is PETSC_F90_2PTR_PARAM(ptrd1));
  }
  if (*ierr) return;
  if (!FORTRANNULLISPOINTER(is_local)) {
    if (tis_local) *ierr = F90Array1dCreate(tis_local, MPIU_FORTRANADDR, 1, nloc, is_local PETSC_F90_2PTR_PARAM(ptrd2));
    else f90array1ddestroyfortranaddr_(is_local PETSC_F90_2PTR_PARAM(ptrd2));
  }
}

PETSC_EXTERN void pcasmdestroysubdomains_(PetscInt *n, F90Array1d *is, F90Array1d *is_local, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  IS       *isa, *isb = NULL;
  PetscBool has_local = PetscNot(FORTRANNULLISPOINTER(is_local));

  if (FORTRANNULLISPOINTER(is)) {
    *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_NULL, PETSC_ERROR_INITIAL, "PCASMDestroySubdomains() requires the subdomain array; do not use PETSC_NULL_IS_POINTER");
    return;
  }
  *ierr = F90Array1dAccess(is, MPIU_FORTRANADDR, (void **)&isa PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  if (has_local) {
    *ierr = F90Array1dAccess(is_local, MPIU_FORTRANADDR, (void **)&isb PETSC_F90_2PTR_PARAM(ptrd2));
    if (*ierr) return;
  }
  *ierr = PCASMDestroySubdomains(*n, &isa, has_local ? &isb : NULL);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(is, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  if (has_local) *ierr = F90Array1dDestroy(is_local, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd2));
}

PETSC_EXTERN void pcasmcreatesubdomains2d_(PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N, PetscInt *dof, PetscInt *overlap, PetscInt *Nsub, F90Array1d *is, F90Array1d *is_local, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  IS *iis, *iisl;

  if (FORTRANNULLISPOINTER(is) || FORTRANNULLISPOINTER(is_local)) {
    *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_NULL, PETSC_ERROR_INITIAL, "PCASMCreateSubdomains2D() requires both output arrays; do not use PETSC_NULL_IS_POINTER");
    return;
  }
  *ierr = PCASMCreateSubdomains2D(*m, *n, *M, *N, *dof, *overlap, Nsub, &iis, &iisl);
  if (*ierr) return;
  *ierr = F90Array1dCreate(iis, MPIU_FORTRANADDR, 1, *Nsub, is PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dCreate(iisl, MPIU_FORTRANADDR, 1, *Nsub, is_local PETSC_F90_2PTR_PARAM(ptrd2));
  if (*ierr) return;
}

PETSC_EXTERN void pcasmgetsubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc, flocal;

  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCASMGetSubKSP(*pc, &nloc, first_local ? &flocal : NULL, &tksp);
  if (*ierr) return;
  if (n_local) *n_local = nloc;
  if (first_local) *first_local = flocal;
  if (FORTRANNULLKSPPOINTER(ksp)) return;
  *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcasmrestoresubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = PETSC_SUCCESS;
  if (FORTRANNULLKSPPOINTER(ksp)) return;
  *ierr = F90Array1dDestroy(ksp, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}

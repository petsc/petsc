/*
     The KSP orthogonalization routines, used in GMRES and other solvers.
*/
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/*@
  KSPOrthogonalizationSet - Sets the orthogonalization routine used by `KSPGMRES` and other solvers.

  Logically Collective

  Input Parameters:
+ ksp    - the Krylov space solver context
- orthog - orthogonalization function; see `KSPOrthogonalizationFn` for the calling sequence

  Options Database Key:
. -ksp_orthogonalization (cgs|mgs) - choose between classical (default) or modified Gram-Schmidt for orthogonalization

  Level: intermediate

  Notes:
  This is used by solvers that explicitly orthogonalize a set of vectors, such as the `KSPGMRES` variants
  and `KSPIDR`; it has no effect on methods such as `KSPBCGS` that never call it.

  Two orthogonalization routines are predefined, `KSPOrthogonalizationModifiedGramSchmidt()` and the default
  `KSPOrthogonalizationClassicalGramSchmidt()`.

  Use `KSPOrthogonalizationSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](ch_ksp), `KSPOrthogonalizationFn`, `KSPOrthogonalizationGet()`, `KSPOrthogonalizationSetCGSRefinementType()`, `KSPOrthogonalizationClassicalGramSchmidt()`, `KSPOrthogonalizationModifiedGramSchmidt()`
@*/
PetscErrorCode KSPOrthogonalizationSet(KSP ksp, KSPOrthogonalizationFn *orthog)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidFunction(orthog, 2);
  ksp->orthog = orthog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPOrthogonalizationGet - Gets the orthogonalization routine used by `KSPGMRES` and other solvers.

  Not Collective

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameter:
. orthog - orthogonalization function; see `KSPOrthogonalizationFn` for the calling sequence

  Level: intermediate

  Notes:
  Two orthogonalization routines are predefined, `KSPOrthogonalizationModifiedGramSchmidt()` and the default
  `KSPOrthogonalizationClassicalGramSchmidt()`.

  Use `KSPOrthogonalizationSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](ch_ksp), `KSPOrthogonalizationFn`, `KSPOrthogonalizationSetCGSRefinementType()`, `KSPOrthogonalizationClassicalGramSchmidt()`, `KSPOrthogonalizationModifiedGramSchmidt()`
@*/
PetscErrorCode KSPOrthogonalizationGet(KSP ksp, KSPOrthogonalizationFn **orthog)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(orthog, 2);
  *orthog = ksp->orthog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPOrthogonalizationSetCGSRefinementType - Sets the type of iterative refinement to use in the classical Gram-Schmidt
  orthogonalization used by `KSPGMRES` and other solvers.

  Logically Collective

  Input Parameters:
+ ksp  - the Krylov space solver context
- type - the type of refinement

  Options Database Key:
. -ksp_orthogonalization_cgs_refinement_type (refine_never|refine_ifneeded|refine_always) - refinement type

  Level: intermediate

  Notes:
  This option applies only if the orthogonalization method is classical Gram-Schmidt (CGS), see `KSPOrthogonalizationSet()`.

  The default refinement type is `KSP_ORTHOGONALIZATION_CGS_REFINE_NEVER`.

  For a very small set of problems, not using refinement, that is `KSP_ORTHOGONALIZATION_CGS_REFINE_NEVER`, may be unstable, thus causing `KSPSolve()`
  to not converge.

.seealso: [](ch_ksp), `KSPOrthogonalizationSet()`, `KSPOrthogonalizationCGSRefinementType`, `KSPOrthogonalizationClassicalGramSchmidt()`, `KSPOrthogonalizationGetCGSRefinementType()`,
          `KSPOrthogonalizationGet()`
@*/
PetscErrorCode KSPOrthogonalizationSetCGSRefinementType(KSP ksp, KSPOrthogonalizationCGSRefinementType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, type, 2);
  ksp->cgstype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPOrthogonalizationGetCGSRefinementType - Gets the type of iterative refinement to use in the classical Gram-Schmidt
  orthogonalization used by `KSPGMRES` and other solvers.

  Not Collective

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameter:
. type - the type of refinement

  Level: intermediate

.seealso: [](ch_ksp), `KSPOrthogonalizationSet()`, `KSPOrthogonalizationCGSRefinementType`, `KSPOrthogonalizationClassicalGramSchmidt()`, `KSPOrthogonalizationSetCGSRefinementType()`,
          `KSPOrthogonalizationGet()`
@*/
PetscErrorCode KSPOrthogonalizationGetCGSRefinementType(KSP ksp, KSPOrthogonalizationCGSRefinementType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ksp->cgstype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPOrthogonalizationModifiedGramSchmidt -  This is the basic orthogonalization routine
  using modified Gram-Schmidt.

  Collective

  Input Parameters:
+ ksp - the Krylov space solver context
. V   - array of previously computed orthonormal vectors
. n   - number of vectors
- x   - vector to be orthogonalized, modified on output (may be `NULL`)

  Output Parameter:
. h - computed orthogonalization coefficients

  Options Database Key:
. -ksp_orthogonalization mgs - choose modified Gram-Schmidt (MGS) for orthogonalization

  Level: intermediate

  Notes:
  If no `x` is given, then the vector to be orthogonalized is assumed to be located at `V[n]`.
  The input vectors `V` must be orthogonal and with unit two-norm. The output vector `x` is
  not normalized.

  In general this is much slower than `KSPOrthogonalizationClassicalGramSchmidt()` but has better stability properties.

.seealso: [](ch_ksp), `KSPOrthogonalizationSet()`, `KSPOrthogonalizationClassicalGramSchmidt()`, `KSPOrthogonalizationGet()`
@*/
PetscErrorCode KSPOrthogonalizationModifiedGramSchmidt(KSP ksp, Vec V[], PetscInt n, Vec x, PetscScalar h[])
{
  PetscInt     j;
  PetscScalar *hh = h;
  Vec          z  = x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(V, 2);
  PetscValidLogicalCollectiveInt(ksp, n, 3);
  if (x) PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscAssertPointer(h, 5);
  PetscCall(PetscLogEventBegin(KSP_Orthogonalization, ksp, 0, 0, 0));
  if (!z) z = V[n];
  for (j = 0; j < n; j++) {
    /* (z, v(j)) */
    PetscCall(VecDot(z, V[j], hh));
    KSPCheckDot(ksp, *hh);
    if (ksp->reason) break;
    /* z <- z - hh[j] v(j) */
    PetscCall(VecAXPY(z, -(*hh++), V[j]));
  }
  PetscCall(PetscLogEventEnd(KSP_Orthogonalization, ksp, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPOrthogonalizationClassicalGramSchmidt -  This is the basic orthogonalization routine
  using classical Gram-Schmidt with possible iterative refinement to improve the stability.

  Collective

  Input Parameters:
+ ksp - the Krylov space solver context
. V   - array of previously computed orthonormal vectors
. n   - number of vectors
- x   - vector to be orthogonalized, modified on output (may be `NULL`)

  Output Parameter:
. h - computed orthogonalization coefficients

  Options Database Keys:
+ -ksp_orthogonalization cgs                                                              - choose classical Gram-Schmidt (CGS) for orthogonalization
- -ksp_orthogonalization_cgs_refinement_type (refine_never|refine_ifneeded|refine_always) - determine if iterative refinement is used to increase the stability of the
                                                                                            classical Gram-Schmidt orthogonalization

  Level: intermediate

  Notes:
  If no `x` is given, then the vector to be orthogonalized is assumed to be located at `V[n]`.
  The input vectors `V` must be orthogonal and with unit two-norm. The output vector `x` is
  not normalized.

  Use `KSPOrthogonalizationSetCGSRefinementType()` to determine if iterative refinement is to be used.
  This is much faster than `KSPOrthogonalizationModifiedGramSchmidt()` but has the small possibility of stability issues
  that can usually be handled by using a single step of iterative refinement with `KSPOrthogonalizationSetCGSRefinementType()`.

.seealso: [](ch_ksp), `KSPOrthogonalizationCGSRefinementType`, `KSPOrthogonalizationSet()`, `KSPOrthogonalizationSetCGSRefinementType()`,
           `KSPOrthogonalizationGetCGSRefinementType()`, `KSPOrthogonalizationGet()`, `KSPOrthogonalizationModifiedGramSchmidt()`
@*/
PetscErrorCode KSPOrthogonalizationClassicalGramSchmidt(KSP ksp, Vec V[], PetscInt n, Vec x, PetscScalar h[])
{
  PetscInt     j;
  PetscScalar *hh = h, *lhh;
  Vec          z  = x;
  PetscReal    hnrm, wnrm;
  PetscBool    refine = (PetscBool)(ksp->cgstype == KSP_ORTHOGONALIZATION_CGS_REFINE_ALWAYS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(V, 2);
  PetscValidLogicalCollectiveInt(ksp, n, 3);
  if (x) PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscAssertPointer(h, 5);
  PetscCall(PetscLogEventBegin(KSP_Orthogonalization, ksp, 0, 0, 0));
  if (!z) z = V[n];
  if (ksp->lorthogwork < n) {
    PetscCall(PetscFree(ksp->orthogwork));
    ksp->lorthogwork = PetscMax(30, PetscMax(2 * ksp->lorthogwork, n));
    PetscCall(PetscMalloc1(ksp->lorthogwork, &ksp->orthogwork));
  }
  lhh = ksp->orthogwork;

  /* Clear hh since we will accumulate values into them */
  for (j = 0; j < n; j++) hh[j] = 0.0;

  /*
     This is really a matrix-vector product, with the matrix stored
     as pointer to rows
  */
  PetscCall(VecMDot(z, n, V, lhh)); /* <v,z> */
  for (j = 0; j < n; j++) {
    KSPCheckDot(ksp, lhh[j]);
    if (ksp->reason) goto done;
    lhh[j] = -lhh[j];
  }

  /*
         This is really a matrix-vector product:
         [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from z.
  */
  PetscCall(VecMAXPY(z, n, lhh, V));
  /* note lhh[j] is -<v,z> , hence the subtraction */
  for (j = 0; j < n; j++) {
    hh[j] -= lhh[j]; /* hh += <v,z> */
  }

  /*
     the second step classical Gram-Schmidt is only necessary
     when a simple test criteria is not passed
  */
  if (ksp->cgstype == KSP_ORTHOGONALIZATION_CGS_REFINE_IFNEEDED) {
    hnrm = 0.0;
    for (j = 0; j < n; j++) hnrm += PetscRealPart(lhh[j] * PetscConj(lhh[j]));

    hnrm = PetscSqrtReal(hnrm);
    PetscCall(VecNorm(z, NORM_2, &wnrm));
    KSPCheckNorm(ksp, wnrm);
    if (ksp->reason) goto done;
    if (wnrm < hnrm) {
      refine = PETSC_TRUE;
      PetscCall(PetscInfo(ksp, "Performing iterative refinement wnorm %g hnorm %g\n", (double)wnrm, (double)hnrm));
    }
  }

  if (refine) {
    PetscCall(VecMDot(z, n, V, lhh)); /* <v,z> */
    for (j = 0; j < n; j++) {
      KSPCheckDot(ksp, lhh[j]);
      if (ksp->reason) goto done;
      lhh[j] = -lhh[j];
    }
    PetscCall(VecMAXPY(z, n, lhh, V));
    /* note lhh[j] is -<v,z> , hence the subtraction */
    for (j = 0; j < n; j++) {
      hh[j] -= lhh[j]; /* hh += <v,z> */
    }
  }
done:
  PetscCall(PetscLogEventEnd(KSP_Orthogonalization, ksp, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

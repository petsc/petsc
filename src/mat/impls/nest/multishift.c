#include <../src/mat/impls/nest/matnestimpl.h> /*I "petscmat.h" I*/

static PetscErrorCode MatMultiShiftDestroy(PetscCtxRt ctx)
{
  Mat_MultiShift mctx = *(Mat_MultiShift *)ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&mctx->K));
  PetscCall(MatDestroy(&mctx->M));
  PetscCall(PetscFree(mctx->sigma));
  PetscCall(PetscFree(mctx->cmplx));
  PetscCall(PetscFree(mctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build G = K + sigma M, explicitly or not */
PetscErrorCode MatMultiShiftBuildShiftedMatrix_Internal(Mat K, PetscScalar sigma, Mat M, MatStructure str, PetscBool explicitmat, Mat *G)
{
  PetscScalar scal[] = {1.0, 1.0};

  PetscFunctionBegin;
  if (explicitmat) {
    PetscCall(MatDuplicate(K, MAT_COPY_VALUES, G));
    if (M) PetscCall(MatAXPY(*G, sigma, M, str));
    else PetscCall(MatShift(*G, sigma));
  } else {
    PetscInt Mk, Nk, mk, nk;

    PetscCall(MatGetSize(K, &Mk, &Nk));
    PetscCall(MatGetLocalSize(K, &mk, &nk));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)K), G));
    PetscCall(MatSetSizes(*G, mk, nk, Mk, Nk));
    PetscCall(MatSetType(*G, MATCOMPOSITE));
    PetscCall(MatCompositeAddMat(*G, K));
    if (M) PetscCall(MatCompositeAddMat(*G, M));
    PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));
    if (M) {
      scal[1] = sigma;
      PetscCall(MatCompositeSetScalings(*G, scal));
    } else PetscCall(MatShift(*G, sigma));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !PetscDefined(USE_COMPLEX)
/* Build G = sigma M, explicitly or not, where M may be I; K is used for dimensions only */
static PetscErrorCode BuildScaledMatrix(Mat K, PetscScalar sigma, Mat M, PetscBool explicitmat, Mat *G)
{
  PetscFunctionBegin;
  if (explicitmat) {
    if (M) {
      PetscCall(MatDuplicate(M, MAT_COPY_VALUES, G));
      PetscCall(MatScale(*G, sigma));
    } else { // M=I
      PetscCall(MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, G));
      PetscCall(MatZeroEntries(*G));
      PetscCall(MatShift(*G, sigma));
    }
  } else {
    PetscInt Mk, Nk, mk, nk;

    PetscCall(MatGetSize(K, &Mk, &Nk));
    PetscCall(MatGetLocalSize(K, &mk, &nk));
    if (M) {
      PetscCall(MatCreate(PetscObjectComm((PetscObject)K), G));
      PetscCall(MatSetSizes(*G, mk, nk, Mk, Nk));
      PetscCall(MatSetType(*G, MATCOMPOSITE));
      PetscCall(MatCompositeAddMat(*G, M));
      PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));
      PetscCall(MatCompositeSetScalings(*G, &sigma));
    } else PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)K), mk, nk, Mk, Nk, sigma, G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
  MatCreateNestFromMultipleShifts - Creates a `MATNEST` matrix that represents a family of shifted matrices
  $K + \sigma_i M$ for a number of shifts $\sigma_i$.

  Collective

  Input Parameters:
+ K               - the first `Mat` (stiffness) forming the shifted matrices
. nshift          - number of shifts
. sigma           - array of shifts $\sigma_i$, its length is `nshift`
. sigma_imaginary - imaginary parts of the shifts $\sigma_i$ in case of complex-conjugate pairs (can be `NULL`); only used when `PetscScalar` is `PetscReal`
. M               - the second `Mat` (mass) forming the shifted matrices (if `NULL` the identity matrix is assumed)
. explicitmat     - whether the shifted matrices should be built explicitly or not
- str             - `MatStructure` flag

  Output Parameter:
. A - the resulting matrix

  Level: intermediate

  Notes:
  This is intended for solving a family of shifted linear systems, $(K + \sigma_i M) x_i = b$,
  where in some applications $M = I$. This function returns a `MATNEST` `A` whose `nshift`
  diagonal blocks are the matrices $K + \sigma_i M$, either built explicitly or not, depending
  on the `explicitmat` argument. If not explicit, the diagonal blocks are created as `MATCOMPOSITE`.
  If `explicitmat` is true, and $M$ is not the identity, then $K + \sigma_i M$ is built with a
  call to `MatAXPY()`, where the flag `str` is used to indicate the relation between the sparsity
  patterns of $K$ and $M$. The flag is also stored in `A` and reused for the same purpose by
  solvers that build a shifted matrix explicitly from `A`, such as `KSPEKSM`, so it must describe
  the relation between the two patterns even when `explicitmat` is `PETSC_FALSE`.

  To solve all the shifted linear systems simultaneously, pass this matrix to a
  `KSP` solver such as `KSPEKSM`, along with compatible solution and right-hand side vectors. Since
  the right-hand side $b$ is the same for all the shifted linear systems, one can call
  `MatCreateVecNestFromMultipleShifts()` to easily create a nested `Vec` containing `nshift` references to $b$.

  When `PetscScalar` is `PetscReal` it is possible to provide complex conjugate pairs of shifts by
  passing a nonzero imaginary part in `sigma_imaginary[i]`. In that case `A` is no longer block
  diagonal but contains 2x2 diagonal blocks for each complex-conjugate pair.

.seealso: [](ch_matrices), `MatCreateVecNestFromMultipleShifts()`, `MatAXPY()`, `MATCOMPOSITE`, `KSP`, `KSPEKSM`
@*/
PetscErrorCode MatCreateNestFromMultipleShifts(Mat K, PetscInt nshift, const PetscScalar sigma[], const PetscScalar sigma_imaginary[], Mat M, PetscBool explicitmat, MatStructure str, Mat *A)
{
  PetscInt       i;
  Mat           *mats;
  Mat_MultiShift mctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(K, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(K, nshift, 2);
  PetscAssertPointer(sigma, 3);
  if (sigma_imaginary) PetscAssertPointer(sigma_imaginary, 4);
  if (M) PetscValidHeaderSpecific(M, MAT_CLASSID, 5);
  PetscValidLogicalCollectiveBool(K, explicitmat, 6);
  PetscValidLogicalCollectiveEnum(K, str, 7);
  PetscAssertPointer(A, 8);

  PetscCheck(nshift > 0, PetscObjectComm((PetscObject)K), PETSC_ERR_ARG_OUTOFRANGE, "The value nshift must be > 0");

  /* build context and array of shifted matrices */
  PetscCall(PetscNew(&mctx));
  PetscCall(PetscObjectReference((PetscObject)K));
  mctx->K = K;
  PetscCall(PetscObjectReference((PetscObject)M));
  mctx->M      = M;
  mctx->nshift = nshift;
  mctx->str    = str;
  PetscCall(PetscMalloc1(nshift, &mctx->sigma));
  if (!PetscDefined(USE_COMPLEX)) PetscCall(PetscCalloc1(nshift, &mctx->cmplx));
  PetscCall(PetscCalloc1(nshift * nshift, &mats));

  for (i = 0; i < nshift; i++) {
#if PetscDefined(USE_COMPLEX)
    mctx->sigma[i] = sigma[i];
    PetscCall(MatMultiShiftBuildShiftedMatrix_Internal(K, sigma[i], M, str, explicitmat, mats + i + i * nshift));
#else
    mctx->sigma[i] = sigma[i];
    PetscCall(MatMultiShiftBuildShiftedMatrix_Internal(K, mctx->sigma[i], M, str, explicitmat, mats + i + i * nshift));
    if (sigma_imaginary && sigma_imaginary[i] != 0.0) {
      PetscCheck(i < nshift - 1, PetscObjectComm((PetscObject)K), PETSC_ERR_ARG_WRONG, "The last shift of the array is complex; shifts must come in complex-conjugate pairs");
      PetscCheck(sigma[i + 1] == sigma[i] && sigma_imaginary[i + 1] == -sigma_imaginary[i], PetscObjectComm((PetscObject)K), PETSC_ERR_ARG_WRONG, "The shifts must either be real or form a (consecutive) complex-conjugate pair");
      mctx->sigma[i + 1] = sigma_imaginary[i];
      mctx->cmplx[i]     = PETSC_TRUE;
      // 2x2 block in the nested matrix
      PetscCall(PetscObjectReference((PetscObject)mats[i + i * nshift]));
      mats[i + 1 + (i + 1) * nshift] = mats[i + i * nshift];
      PetscCall(BuildScaledMatrix(K, sigma_imaginary[i], M, explicitmat, mats + i + (i + 1) * nshift));
      PetscCall(BuildScaledMatrix(K, -sigma_imaginary[i], M, explicitmat, mats + i + 1 + i * nshift));
      i++; // skip next shift
    }
#endif
  }

  /* build MATNEST */
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)K), nshift, NULL, nshift, NULL, mats, A));
  for (i = 0; i < nshift * nshift; i++) PetscCall(MatDestroy(&mats[i]));
  PetscCall(PetscFree(mats));
  PetscCall(MatNestSetVecType(*A, VECNEST));

  /* compose context */
  PetscCall(MatGetState(*A, &mctx->state)); // used to detect later changes such as MatScale()
  PetscCall(PetscObjectContainerCompose((PetscObject)*A, "MatMultiShift", mctx, MatMultiShiftDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateVecNestFromMultipleShifts - Creates a `VECNEST` that is compatible with a matrix
  created with `MatCreateNestFromMultipleShifts()`.

  Collective

  Input Parameters:
+ A - a `Mat` created with `MatCreateNestFromMultipleShifts()`
- v - an optional vector (set to `NULL` if not needed)

  Output Parameter:
. vout - the resulting vector

  Level: intermediate

  Notes:
  The result is a `VECNEST` compatible with `A`, so that it can, e.g., be multiplied against.
  If the input vector `v` is passed, `vout` will contain `nshift` references to `v`, where
  `nshift` is the number of subvectors of `vout`. Hence, it is intended as a read-only right-hand
  side, not a solution vector.

  In real scalars, in the case of a complex-conjugate pair only the first block of the pair is
  set while the second is left zero. The reason is that in this case the second block represents
  the imaginary part, which is zero since `v` is real.

.seealso: [](ch_matrices), `MatCreateNestFromMultipleShifts()`
@*/
PetscErrorCode MatCreateVecNestFromMultipleShifts(Mat A, Vec v, Vec *vout)
{
  PetscInt       i;
  Mat_MultiShift mctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (v) PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscAssertPointer(vout, 3);
  MatCheckMultiShift(A, &mctx);

  PetscCall(MatCreateVecs(A, NULL, vout));
  if (v)
    for (i = 0; i < mctx->nshift; i++) {
      PetscCall(VecNestSetSubVec(*vout, i, v));
      if (!PetscDefined(USE_COMPLEX) && mctx->cmplx[i]) i++; // complex shift, leave a zero block since b is real
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

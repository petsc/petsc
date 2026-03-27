#include <petscda.h>
#include <petsc/private/daimpl.h>
#include <petsc/private/daensembleimpl.h>

typedef struct {
  PetscDA_Ensemble en;
  Vec              mean;
  Vec              y_mean;
  Vec              delta_scaled;
  Vec              w;
  Vec              r_inv_sqrt;
  Mat              Z;
  Mat              S;
  Mat              T_sqrt;
  Mat              w_ones;
} PetscDA_ETKF;

/*
  BroadcastWeightVector - Creates matrix with weight vector replicated across all columns

  Input Parameters:
+ w - weight vector of size m (analysis weights from ETKF update)
- m - ensemble size (number of columns to replicate, must equal vector size)

  Output Parameter:
. w_ones - m x m dense matrix where each column is a copy of w (i.e., w * 1^T)

  Notes:
  This function constructs the broadcast matrix w * 1^T, where w is the m-dimensional
  weight vector and 1 is an m-dimensional vector of ones. This matrix is a fundamental
  component in the ETKF transform: G = w * 1^T + sqrt(m-1) * T^{1/2} * U.

  The implementation uses direct array access for performance, avoiding the overhead of
  repeated vector wrapping and copying. This is particularly efficient for dense matrices
  where memory is contiguous column-wise.

  Complexity: O(m^2) time and memory.

  Level: developer

*/
static PetscErrorCode BroadcastWeightVector(Vec w, PetscInt m, Mat w_ones)
{
  const PetscScalar *w_array;
  PetscScalar       *mat_array;
  PetscInt           w_size, w_size_local, mat_rows_local, mat_cols_local;
  PetscInt           i, lda;

  PetscFunctionBegin;
  PetscCheck(m > 0, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive for broadcasting, got %" PetscInt_FMT, m);
  /* Check for potential overflow in matrix size calculation */
  PetscCheck(m <= PETSC_MAX_INT / m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m = %" PetscInt_FMT " too large", m);

  /* Verify dimensions */
  PetscCall(VecGetSize(w, &w_size));
  PetscCall(VecGetLocalSize(w, &w_size_local));
  PetscCheck(w_size == m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_INCOMP, "Weight vector global size (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ")", w_size, m);

  /* Verify consistent parallel layout between vector and matrix */
  PetscCall(MatGetLocalSize(w_ones, &mat_rows_local, &mat_cols_local));
  PetscCheck(mat_rows_local == w_size_local, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix row distribution (%" PetscInt_FMT ") inconsistent with vector distribution (%" PetscInt_FMT ")", mat_rows_local, w_size_local);
  PetscCheck(mat_cols_local == m, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix local columns (%" PetscInt_FMT ") must equal global columns m (%" PetscInt_FMT ") for MPIDense", mat_cols_local, m);

  /* Access raw arrays for efficient broadcasting */
  PetscCall(VecGetArrayRead(w, &w_array));
  PetscCall(MatDenseGetArrayWrite(w_ones, &mat_array));
  PetscCall(MatDenseGetLDA(w_ones, &lda));

  /* Copy w to each column of w_ones */
  /* Note: MatDense uses column-major storage. We copy the vector w into each column. */
  for (i = 0; i < m; i++) PetscCall(PetscArraycpy(mat_array + i * lda, w_array, w_size_local));

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayWrite(w_ones, &mat_array));
  PetscCall(VecRestoreArrayRead(w, &w_array));

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(w_ones, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  UpdateEnsembleWithTransform - Updates ensemble via ETKF transform: E = mean * 1' + X * G [Alg 6.4 line 9]

  Input Parameters:
+ mean     - ensemble mean vector (size state_size), must be initialized
. X        - scaled anomaly matrix (state_size x ensemble_size), X = (E - mean*1')/sqrt(m-1)
. G        - ETKF transform matrix (ensemble_size x ensemble_size), G = w*1' + sqrt(m-1)*T^{1/2}*U
. m        - ensemble size (number of columns in ensemble), must be > 0
- ensemble - ensemble matrix to update in-place (state_size x ensemble_size)

  Notes:
  This function performs the final step (Step 10) of the ETKF analysis algorithm from
  Asch, M., Bocquet, M., and Nodet, M., transforming the forecast ensemble into the analysis ensemble.
  The operation E^a = mean + X * G is computed using matrix-matrix multiplication followed
  by column-wise addition to efficiently handle large state spaces.

  Error Handling:
  - Validates all input dimensions for consistency
  - Checks for positive ensemble size
  - Ensures proper matrix/vector initialization
  - Handles parallel assembly correctly

  Performance Considerations:
  - Memory: Creates one temporary matrix X_G of size (state_size x m)
  - Time complexity: O(state_size * m^2) for matrix multiply + O(state_size * m) for additions
  - Optimization: Uses direct array access for dense matrices to avoid Vec overhead
  - Parallel: Fully parallelizable across both matrix multiply and column updates

  Level: developer

*/
static PetscErrorCode UpdateEnsembleWithTransform(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  Mat                X_G;
  const PetscScalar *xg_array, *mean_array;
  PetscScalar       *ens_array;
  PetscInt           x_rows, x_cols, g_rows, g_cols, ens_rows, ens_cols;
  PetscInt           n_local_ens, n_local_xg, mean_local_size;
  PetscInt           lda_ens, lda_xg;
  PetscInt           mean_size, i, j;

  PetscFunctionBegin;
  /* Validate input parameters for correct types and null pointers */
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 3);
  PetscValidLogicalCollectiveInt(X, m, 4);
  PetscValidHeaderSpecific(ensemble, MAT_CLASSID, 5);

  /* Retrieve and validate matrix dimensions for compatibility */
  PetscCall(MatGetSize(X, &x_rows, &x_cols));
  PetscCall(MatGetSize(G, &g_rows, &g_cols));
  PetscCall(MatGetSize(ensemble, &ens_rows, &ens_cols));
  PetscCall(VecGetSize(mean, &mean_size));

  /* Verify dimension consistency across all inputs */
  PetscCheck(x_cols == m, PetscObjectComm((PetscObject)X), PETSC_ERR_ARG_INCOMP, "Anomaly matrix X columns (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", x_cols, m);
  PetscCheck(g_rows == m, PetscObjectComm((PetscObject)G), PETSC_ERR_ARG_INCOMP, "Transform matrix G rows (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", g_rows, m);
  PetscCheck(g_cols == m, PetscObjectComm((PetscObject)G), PETSC_ERR_ARG_INCOMP, "Transform matrix G must be square, got %" PetscInt_FMT " x %" PetscInt_FMT, g_rows, g_cols);
  PetscCheck(ens_rows == x_rows, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble rows (%" PetscInt_FMT ") must match anomaly matrix X rows (%" PetscInt_FMT ")", ens_rows, x_rows);
  PetscCheck(ens_cols == m, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble columns (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", ens_cols, m);
  PetscCheck(mean_size == x_rows, PetscObjectComm((PetscObject)mean), PETSC_ERR_ARG_INCOMP, "Mean vector size (%" PetscInt_FMT ") must match state size (%" PetscInt_FMT ")", mean_size, x_rows);

  /* Compute transformed anomaly matrix: X_G = X * G (state_size x m) */
  PetscCall(MatMatMult(X, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X_G));

  /* Access underlying data arrays for direct performance access
     This avoids creating/destroying m Vec objects and calling VecWAXPY m times. */
  PetscCall(MatDenseGetArrayRead(X_G, &xg_array));
  PetscCall(MatDenseGetArrayWrite(ensemble, &ens_array));
  PetscCall(VecGetArrayRead(mean, &mean_array));

  /* Get local dimensions and strides for array traversal */
  PetscCall(MatGetLocalSize(ensemble, &n_local_ens, NULL));
  PetscCall(MatGetLocalSize(X_G, &n_local_xg, NULL));
  PetscCall(VecGetLocalSize(mean, &mean_local_size));
  PetscCall(MatDenseGetLDA(ensemble, &lda_ens));
  PetscCall(MatDenseGetLDA(X_G, &lda_xg));

  /* Verify local dimensions match before direct array access */
  PetscCheck(n_local_ens == n_local_xg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local row size mismatch: ensemble (%" PetscInt_FMT ") vs X_G (%" PetscInt_FMT ")", n_local_ens, n_local_xg);
  PetscCheck(n_local_ens == mean_local_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local row size mismatch: ensemble (%" PetscInt_FMT ") vs mean (%" PetscInt_FMT ")", n_local_ens, mean_local_size);

  /* Update each ensemble member: E_ij = (XG)_ij + mean_i
     Loop over columns (j) and rows (i) of the local data block */
  for (j = 0; j < m; j++) {
    const PetscScalar *xg_col  = xg_array + j * lda_xg;
    PetscScalar       *ens_col = ens_array + j * lda_ens;
    for (i = 0; i < n_local_ens; i++) ens_col[i] = xg_col[i] + mean_array[i];
  }

  /* Restore arrays and finalize assembly */
  PetscCall(VecRestoreArrayRead(mean, &mean_array));
  PetscCall(MatDenseRestoreArrayWrite(ensemble, &ens_array));
  PetscCall(MatDenseRestoreArrayRead(X_G, &xg_array));

  PetscCall(MatAssemblyBegin(ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ensemble, MAT_FINAL_ASSEMBLY));

  /* Clean up temporary transformed anomaly matrix */
  PetscCall(MatDestroy(&X_G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDADestroy_ETKF(PetscDA da)
{
  PetscDA_ETKF *impl = (PetscDA_ETKF *)da->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&impl->mean));
  PetscCall(VecDestroy(&impl->y_mean));
  PetscCall(VecDestroy(&impl->delta_scaled));
  PetscCall(VecDestroy(&impl->w));
  PetscCall(VecDestroy(&impl->r_inv_sqrt));
  PetscCall(MatDestroy(&impl->Z));
  PetscCall(MatDestroy(&impl->S));
  PetscCall(MatDestroy(&impl->T_sqrt));
  PetscCall(MatDestroy(&impl->w_ones));
  PetscCall(PetscDADestroy_Ensemble(da));
  PetscCall(PetscFree(da->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAEnsembleAnalysis_ETKF(PetscDA da, Vec observation, Mat H)
{
  PetscDA_ETKF *impl = (PetscDA_ETKF *)da->data;
  Mat           X;
  PetscInt      m = impl->en.size;
  PetscScalar   scale, sqrt_m_minus_1;
  PetscBool     reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  PetscCall(PetscInfo(da, "square root type %s, %" PetscInt_FMT " ensembles\n", (impl->en.sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky", m));

  /* Check for reallocation needs */
  if (impl->mean) {
    PetscInt mean_size;
    PetscCall(VecGetSize(impl->mean, &mean_size));
    if (mean_size != da->state_size) reallocate = PETSC_TRUE;
  }
  if (impl->Z) {
    PetscInt z_rows, z_cols;
    PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
    if (z_rows != da->obs_size || z_cols != impl->en.size) reallocate = PETSC_TRUE;
  }
  if (impl->w) {
    PetscInt w_size;
    PetscCall(VecGetSize(impl->w, &w_size));
    if (w_size != impl->en.size) reallocate = PETSC_TRUE;
  }

  /* Initialize or reallocate persistent work objects */
  if (!impl->mean || reallocate) {
    PetscCall(VecDestroy(&impl->mean));
    PetscCall(VecDestroy(&impl->y_mean));
    PetscCall(VecDestroy(&impl->delta_scaled));
    PetscCall(VecDestroy(&impl->w));
    PetscCall(VecDestroy(&impl->r_inv_sqrt));
    PetscCall(MatDestroy(&impl->Z));
    PetscCall(MatDestroy(&impl->S));
    PetscCall(MatDestroy(&impl->T_sqrt));
    PetscCall(MatDestroy(&impl->w_ones));

    /* Create mean vector from ensemble matrix (right vector = state space) */
    PetscCall(MatCreateVecs(impl->en.ensemble, NULL, &impl->mean));

    /* Create Z matrix (obs_size x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)impl->en.ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &impl->Z));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->Z, "dense_"));
    PetscCall(MatSetFromOptions(impl->Z));
    PetscCall(MatSetUp(impl->Z));

    /* Create observation space vectors from Z matrix (left vector = observation space) */
    PetscCall(MatCreateVecs(impl->Z, NULL, &impl->y_mean));
    PetscCall(VecDuplicate(impl->y_mean, &impl->delta_scaled));
    PetscCall(VecDuplicate(da->obs_error_var, &impl->r_inv_sqrt));

    /* Create w vector (size m) for analysis weights */
    PetscCall(MatCreateVecs(impl->Z, &impl->w, NULL));

    /* Create S matrix (same layout as Z) */
    PetscCall(MatDuplicate(impl->Z, MAT_DO_NOT_COPY_VALUES, &impl->S));

    /* Create T_sqrt matrix (m x m) - usually small */
    /* T_sqrt will hold the result of applying T^{-1/2} to identity matrix */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)impl->en.ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->T_sqrt));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->T_sqrt, "dense_"));
    PetscCall(MatSetFromOptions(impl->T_sqrt));
    PetscCall(MatSetUp(impl->T_sqrt));

    /* Create w_ones matrix (m x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)impl->en.ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->w_ones));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->w_ones, "dense_"));
    PetscCall(MatSetFromOptions(impl->w_ones));
    PetscCall(MatSetUp(impl->w_ones));
  }

  /* Alg 6.4 line 1-2: Compute ensemble mean and scaled anomalies */
  PetscCall(PetscDAEnsembleComputeMean(da, impl->mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  /* Note: PetscDAComputeAnomalies creates a NEW matrix X every time.
     We should probably optimize this too in the future, but for now we follow the API. */
  PetscCall(PetscDAEnsembleComputeAnomalies(da, impl->mean, &X));

  /* Alg 6.4 line 3-4: Compute observation ensemble Z = H * E */
  {
    MatReuse scall = MAT_INITIAL_MATRIX;
    if (impl->Z) {
      PetscInt z_rows, z_cols;
      PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
      if (z_rows == da->obs_size && z_cols == impl->en.size) scall = MAT_REUSE_MATRIX;
      else {
        PetscCall(MatDestroy(&impl->Z));
        scall = MAT_INITIAL_MATRIX;
      }
    }
    PetscCall(MatMatMult(H, impl->en.ensemble, scall, PETSC_DEFAULT, &impl->Z));
  }

  /* Compute observation mean y_mean = H * x_mean */
  PetscCall(MatMult(H, impl->mean, impl->y_mean));

  /* Alg 6.4 line 5-6: Build normalized innovation statistics */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  /* S = R^{-1/2} * (Z - y_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDAEnsembleComputeNormalizedInnovationMatrix(impl->Z, impl->y_mean, impl->r_inv_sqrt, m, scale, impl->S));

  /* delta_scaled = R^{-1/2} * (y^o - y_mean) [Alg 6.4 line 6] */
  PetscCall(VecWAXPY(impl->delta_scaled, -1.0, impl->y_mean, observation));
  PetscCall(VecPointwiseMult(impl->delta_scaled, impl->delta_scaled, impl->r_inv_sqrt));

  /* Alg 6.4 line 7: Factor T = (I + S^T S) and store factorization */
  /* Note: Inflation is handled inside PetscDAEnsembleTFactor by shifting the diagonal of T */
  PetscCall(PetscDAEnsembleTFactor(da, impl->S));

  /* Alg 6.4 line 8: Compute analysis weights w = T^{-1} * S^T * delta_scaled */
  {
    Vec s_transpose_delta;
    /* Create temporary vector for S^T * delta_scaled */
    PetscCall(MatCreateVecs(impl->Z, &s_transpose_delta, NULL));
    PetscCall(MatMultTranspose(impl->S, impl->delta_scaled, s_transpose_delta));

    PetscCall(PetscDAEnsembleApplyTInverse(da, s_transpose_delta, impl->w));
    PetscCall(VecDestroy(&s_transpose_delta));
  }

  /* Alg 6.4 line 9: Compute square-root transform T^{-1/2} */
  PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, impl->T_sqrt));

  /* Alg 6.4 line 9: Form transform G = w * 1' + sqrt(m - 1) * T^{1/2} * U */
  {
    Mat T_sqrt_scaled;
    PetscCall(MatDuplicate(impl->T_sqrt, MAT_COPY_VALUES, &T_sqrt_scaled));
    PetscCall(MatScale(T_sqrt_scaled, sqrt_m_minus_1));

    /* w_ones = w * 1' (broadcast weight vector to all columns) */
    PetscCall(BroadcastWeightVector(impl->w, m, impl->w_ones));

    /* G = w_ones + sqrt(m-1)*T_sqrt
     Accumulate the scaled T_sqrt into w_ones to form the transform matrix G */
    PetscCall(MatAXPY(impl->w_ones, 1.0, T_sqrt_scaled, SAME_NONZERO_PATTERN));

    PetscCall(MatDestroy(&T_sqrt_scaled));
  }

  /* Alg 6.4 line 9: Update ensemble E = x_mean * 1' + X * G */
  PetscCall(UpdateEnsembleWithTransform(impl->mean, X, impl->w_ones, m, impl->en.ensemble));

  /* Cleanup temporary X matrix */
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDAEnsembleForecast_Ensemble(PetscDA da, PetscErrorCode (*model)(Vec, Vec, PetscCtx), PetscCtx ctx)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Vec               col_in, col_out, temp;
  PetscInt          i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  /* Create temp vector from ensemble matrix (right vector = state space) */
  PetscCall(MatCreateVecs(en->ensemble, NULL, &temp));

  for (i = 0; i < en->size; i++) {
    PetscCall(MatDenseGetColumnVecRead(en->ensemble, i, &col_in));
    PetscCall(model(col_in, temp, ctx));
    PetscCall(MatDenseRestoreColumnVecRead(en->ensemble, i, &col_in));

    PetscCall(MatDenseGetColumnVecWrite(en->ensemble, i, &col_out));
    PetscCall(VecCopy(temp, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(en->ensemble, i, &col_out));
  }

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCDAETKF - Ensemble transform Kalman filter data assimilation using a deterministic square-root update that avoids stochastic perturbations.

   Options Database Keys:
+  -petscda_type etkf                            - set the `PetscDAType` to `PETSCDAETKF`
.  -petscda_ensemble_size <size>                 - number of ensemble members
-  -petscda_ensemble_sqrt_type <cholesky, eigen> - the square root of the matrix to use

   Level: beginner

   Note:
   The ETKF algorithm is based on Algorithm 6.4 in {cite}`da2016`

.seealso: [](ch_da), `PetscDA`, `PetscDACreate()`, `PETSCDALETKF`, `PetscDAEnsembleSetSize()`, `PetscDASetSizes()`, `PetscDAEnsembleSetSqrtType()`,
          `PetscDAEnsembleSetInflation()`, `PetscDAType`,
          `PetscDAEnsembleComputeMean()`, `PetscDAEnsembleComputeAnomalies()`, `PetscDAEnsembleAnalysis()`, `PetscDAEnsembleForecast()`
M*/
PETSC_INTERN PetscErrorCode PetscDACreate_ETKF(PetscDA da)
{
  PetscDA_ETKF *impl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&impl));
  da->data = impl;
  PetscCall(PetscDACreate_Ensemble(da));
  da->ops->setup          = PetscDASetUp_Ensemble;
  da->ops->destroy        = PetscDADestroy_ETKF;
  da->ops->view           = PetscDAView_Ensemble;
  da->ops->setfromoptions = PetscDASetFromOptions_Ensemble;
  impl->en.analysis       = PetscDAEnsembleAnalysis_ETKF;
  impl->en.forecast       = PetscDAEnsembleForecast_Ensemble;
  PetscFunctionReturn(PETSC_SUCCESS);
}

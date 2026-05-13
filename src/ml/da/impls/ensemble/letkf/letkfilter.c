#include <petscda.h>
#include <petsc/private/daimpl.h>
#include <petsc/private/daensembleimpl.h>
#include <petscblaslapack.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>

static PetscErrorCode PetscDALETKFInstallQ(PetscDA, Mat, PetscInt, PetscInt, Mat);
static PetscErrorCode PetscDALETKFResetLocalization_LETKF(PetscDA);

/* Names must match the PetscDALETKFLocalizationType enum order in include/petscda.h. */
const char *const PetscDALETKFLocalizationTypes[] = {"none", "gaspari_cohn", "gaussian", "boxcar", "PetscDALETKFLocalizationType", "PETSCDA_LETKF_LOC_", NULL};

/* The Kokkos analysis paths key off the type of the obs-error covariance Mat (R), since R is
   created via MatSetType + MatSetFromOptions and inherits whatever -mat_type the user requested.
   Returns PETSC_FALSE when R is not yet built or when Kokkos kernels are unavailable. */
static PetscErrorCode PetscDALETKFUseKokkosBackend(PetscDA da, PetscBool *use_kokkos)
{
  PetscFunctionBegin;
  *use_kokkos = PETSC_FALSE;
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  if (da->R) PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, use_kokkos, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Free cached coordinate inputs (used only for built-in kernels). */
static PetscErrorCode PetscDALETKFClearCoordinates(PetscDA_LETKF *impl)
{
  PetscFunctionBegin;
  for (PetscInt d = 0; d < 3; d++) {
    PetscCall(VecDestroy(&impl->coord_xyz[d]));
    impl->coord_bd[d] = 0.0;
  }
  PetscCall(MatDestroy(&impl->coord_H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFReplicateWeightVector - replicate weight vector w across all columns of w_ones (m x m dense).
  Used only by the LOC_NONE fast path. w lives on PETSC_COMM_SELF (size m); w_ones is a SELF SeqDense m x m.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFReplicateWeightVector(Vec w, PetscInt m, Mat w_ones)
{
  const PetscScalar *w_array;
  PetscScalar       *mat_array;
  PetscInt           w_size, lda, wo_rows, wo_cols;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(w, &w_size));
  PetscCheck(w_size == m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_INCOMP, "w size %" PetscInt_FMT " != m %" PetscInt_FMT, w_size, m);
  PetscCall(MatGetSize(w_ones, &wo_rows, &wo_cols));
  PetscCheck(wo_rows == m && wo_cols == m, PetscObjectComm((PetscObject)w_ones), PETSC_ERR_ARG_INCOMP, "w_ones must be %" PetscInt_FMT " x %" PetscInt_FMT ", got %" PetscInt_FMT " x %" PetscInt_FMT, m, m, wo_rows, wo_cols);
  PetscCall(VecGetArrayRead(w, &w_array));
  PetscCall(MatDenseGetArrayWrite(w_ones, &mat_array));
  PetscCall(MatDenseGetLDA(w_ones, &lda));
  for (PetscInt i = 0; i < m; i++) PetscCall(PetscArraycpy(mat_array + i * lda, w_array, m));
  PetscCall(MatDenseRestoreArrayWrite(w_ones, &mat_array));
  PetscCall(VecRestoreArrayRead(w, &w_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFEnsureGlobalScratch - Lazily allocate the per-rank-replicated m-sized SELF scratch
  used by the LOC_NONE fast path (impl->w, impl->s_transpose_delta, impl->T_sqrt, impl->w_ones).
  Both the CPU (PetscDALETKFGlobalAnalysis) and Kokkos (PetscDALETKFGlobalAnalysis_Kokkos) backends
  call this on entry; the per-vertex paths skip it.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFEnsureGlobalScratch(PetscDA_LETKF *impl, PetscInt m)
{
  PetscFunctionBegin;
  if (!impl->w) PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &impl->w));
  if (!impl->s_transpose_delta) PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &impl->s_transpose_delta));
  if (!impl->T_sqrt) PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &impl->T_sqrt));
  if (!impl->w_ones) PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &impl->w_ones));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFUpdateEnsembleWithTransform - E = mean*1' + X*G.
  Used only by the LOC_NONE fast path. G is replicated on PETSC_COMM_SELF (every rank holds the
  same m x m), X and the ensemble share the same row distribution; the local rows of E are
  X_local * G + mean_local broadcast across columns. Computed via a per-rank BLASgemm for X*G
  plus a column-broadcast add of mean.
*/
static PetscErrorCode PetscDALETKFUpdateEnsembleWithTransform(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  const PetscScalar *x_array, *g_array, *mean_array;
  PetscScalar       *xg_buf, *ens_array;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscBLASInt       n_local_b, m_b, lda_x_b, lda_g_b;
  PetscInt           n_local_ens, n_local_x, n_g_rows, n_g_cols, lda_x, lda_g, lda_ens;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(ensemble, &n_local_ens, NULL));
  PetscCall(MatGetLocalSize(X, &n_local_x, NULL));
  PetscCheck(n_local_x == n_local_ens, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "X local rows (%" PetscInt_FMT ") must match ensemble local rows (%" PetscInt_FMT ")", n_local_x, n_local_ens);
  PetscCall(MatGetSize(G, &n_g_rows, &n_g_cols));
  PetscCheck(n_g_rows == m && n_g_cols == m, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "G must be %" PetscInt_FMT " x %" PetscInt_FMT ", got %" PetscInt_FMT " x %" PetscInt_FMT, m, m, n_g_rows, n_g_cols);
  PetscCall(MatDenseGetArrayRead(X, &x_array));
  PetscCall(MatDenseGetLDA(X, &lda_x));
  PetscCall(MatDenseGetArrayRead(G, &g_array));
  PetscCall(MatDenseGetLDA(G, &lda_g));
  PetscCall(MatDenseGetArrayWrite(ensemble, &ens_array));
  PetscCall(MatDenseGetLDA(ensemble, &lda_ens));
  PetscCall(VecGetArrayRead(mean, &mean_array));
  PetscCall(PetscMalloc1((size_t)n_local_ens * m, &xg_buf));
  PetscCall(PetscBLASIntCast(n_local_ens, &n_local_b));
  PetscCall(PetscBLASIntCast(m, &m_b));
  PetscCall(PetscBLASIntCast(lda_x, &lda_x_b));
  PetscCall(PetscBLASIntCast(lda_g, &lda_g_b));
  if (n_local_ens > 0) PetscCallBLAS("BLASgemm", BLASgemm_("N", "N", &n_local_b, &m_b, &m_b, &one, x_array, &lda_x_b, g_array, &lda_g_b, &zero, xg_buf, &n_local_b));
  for (PetscInt j = 0; j < m; j++)
    for (PetscInt i = 0; i < n_local_ens; i++) ens_array[i + j * lda_ens] = mean_array[i] + xg_buf[i + j * n_local_ens];
  PetscCall(PetscFree(xg_buf));
  PetscCall(VecRestoreArrayRead(mean, &mean_array));
  PetscCall(MatDenseRestoreArrayWrite(ensemble, &ens_array));
  PetscCall(MatDenseRestoreArrayRead(G, &g_array));
  PetscCall(MatDenseRestoreArrayRead(X, &x_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDADestroy_LETKF(PetscDA da)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&impl->mean));
  PetscCall(VecDestroy(&impl->y_mean));
  PetscCall(VecDestroy(&impl->delta_scaled));
  PetscCall(VecDestroy(&impl->w));
  PetscCall(VecDestroy(&impl->s_transpose_delta));
  PetscCall(VecDestroy(&impl->r_inv_sqrt));
  PetscCall(VecDestroy(&impl->H_temp_in));
  PetscCall(VecDestroy(&impl->H_temp_out));
  PetscCall(PetscFree(impl->H_vec_type));
  PetscCall(MatDestroy(&impl->Z));
  PetscCall(MatDestroy(&impl->S));
  PetscCall(MatDestroy(&impl->T_sqrt));
  PetscCall(MatDestroy(&impl->w_ones));
  PetscCall(MatDestroy(&impl->Q));
  PetscCall(PetscDALETKFClearCoordinates(impl));
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscDALETKFDestroyLocalization_Kokkos(impl));
#endif
  PetscCall(PetscDALETKFDestroyObsScatter(impl));
  PetscCall(PetscDADestroy_Ensemble(da));
  PetscCall(PetscFree(da->data));

  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalizationRadius_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFGetLocalizationRadius_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalizationType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFGetLocalizationType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalizationCoordinates_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFResetLocalization_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ExtractLocalObservations - Extracts local observations for a vertex using localization matrix Q (CPU version)

  Input Parameters:
+ Q          - localization matrix (state_size/ndof x obs_size), variable nnz per row
. vertex_idx - index of the vertex (row of Q)
. Z_global   - global observation ensemble matrix (obs_size x m) OR local work matrix
. y_global   - global observation vector (size obs_size) OR local work vector
. y_mean_global - global observation mean (size obs_size) OR local work vector
. r_inv_sqrt_global - global R^{-1/2} (size obs_size) OR local work vector
. obs_g2l    - map from global observation index to local index (if using local work vectors)
. m          - ensemble size

  Output Parameters:
. Z_local    - local observation ensemble (p_local x m), pre-allocated
. y_local    - local observation vector (size p_local), pre-allocated
. y_mean_local - local observation mean (size p_local), pre-allocated
- r_inv_sqrt_local - local R^{-1/2} (size p_local), pre-allocated
*/
static PetscErrorCode ExtractLocalObservations(Mat Q, PetscInt vertex_idx, Mat Z_global, Vec y_global, Vec y_mean_global, Vec r_inv_sqrt_global, PetscHMapI obs_g2l, PetscInt m, Mat Z_local, Vec y_local, Vec y_mean_local, Vec r_inv_sqrt_local)
{
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscInt           ncols, k, j, p_local;
  const PetscScalar *z_global_array, *y_global_array, *y_mean_global_array, *r_inv_sqrt_global_array;
  PetscScalar       *z_local_array, *y_local_array, *y_mean_local_array, *r_inv_sqrt_local_array;
  PetscInt           lda_z_global, lda_z_local;

  PetscFunctionBegin;
  /* Get the row of Q corresponding to this vertex */
  PetscCall(MatGetRow(Q, vertex_idx, &ncols, &cols, &vals));

  /* Get array access to global data */
  PetscCall(MatDenseGetArrayRead(Z_global, &z_global_array));
  PetscCall(VecGetArrayRead(y_global, &y_global_array));
  PetscCall(VecGetArrayRead(y_mean_global, &y_mean_global_array));
  PetscCall(VecGetArrayRead(r_inv_sqrt_global, &r_inv_sqrt_global_array));

  /* Get array access to local data */
  PetscCall(MatDenseGetArrayWrite(Z_local, &z_local_array));
  PetscCall(VecGetArray(y_local, &y_local_array));
  PetscCall(VecGetArray(y_mean_local, &y_mean_local_array));
  PetscCall(VecGetArray(r_inv_sqrt_local, &r_inv_sqrt_local_array));

  /* Get leading dimensions */
  PetscCall(MatDenseGetLDA(Z_global, &lda_z_global));
  PetscCall(MatDenseGetLDA(Z_local, &lda_z_local));
  PetscCall(VecGetLocalSize(y_local, &p_local));

  /* Extract local observations and weight R^{-1/2} */
  for (k = 0; k < ncols; k++) {
    PetscInt    obs_idx   = cols[k];
    PetscScalar weight    = vals[k];
    PetscInt    local_idx = obs_idx;

    /* If using local work vectors, map global index to local index */
    if (obs_g2l) {
      PetscCall(PetscHMapIGet(obs_g2l, obs_idx, &local_idx));
      PetscCheck(local_idx >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Observation index %" PetscInt_FMT " not found in local map", obs_idx);
    }

    y_local_array[k]          = y_global_array[local_idx];
    y_mean_local_array[k]     = y_mean_global_array[local_idx];
    r_inv_sqrt_local_array[k] = r_inv_sqrt_global_array[local_idx] * PetscSqrtScalar(weight);

    /* Extract Z matrix row (column-major layout) */
    for (j = 0; j < m; j++) z_local_array[k + j * lda_z_local] = z_global_array[local_idx + j * lda_z_global];
  }

  /* Zero the unused tail [ncols, p_local) so a shorter row does not leak the previous row's
     trailing values into the downstream normalized-innovation computation. Caller does not need
     to MatZeroEntries/VecZeroEntries the workspace between iterations. */
  for (k = ncols; k < p_local; k++) {
    y_local_array[k]          = 0.0;
    y_mean_local_array[k]     = 0.0;
    r_inv_sqrt_local_array[k] = 0.0;
    for (j = 0; j < m; j++) z_local_array[k + j * lda_z_local] = 0.0;
  }

  /* Restore arrays */
  PetscCall(VecRestoreArray(r_inv_sqrt_local, &r_inv_sqrt_local_array));
  PetscCall(VecRestoreArray(y_mean_local, &y_mean_local_array));
  PetscCall(VecRestoreArray(y_local, &y_local_array));
  PetscCall(MatDenseRestoreArrayWrite(Z_local, &z_local_array));
  PetscCall(VecRestoreArrayRead(r_inv_sqrt_global, &r_inv_sqrt_global_array));
  PetscCall(VecRestoreArrayRead(y_mean_global, &y_mean_global_array));
  PetscCall(VecRestoreArrayRead(y_global, &y_global_array));
  PetscCall(MatDenseRestoreArrayRead(Z_global, &z_global_array));

  /* Restore Q row */
  PetscCall(MatRestoreRow(Q, vertex_idx, &ncols, &cols, &vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFLocalAnalysis - Performs local LETKF analysis for all grid points (CPU version)

  Input Parameters:
+ da             - the PetscDA context
. impl           - LETKF implementation data
. m              - ensemble size
. n_vertices     - number of grid points
. X              - global anomaly matrix (state_size x m)
. observation    - observation vector
. Z_global       - global observation ensemble (obs_size x m)
. y_mean_global  - global observation mean
- r_inv_sqrt_global - global R^{-1/2}

  Output:
. da->ensemble - updated with analysis ensemble

  Notes:
  This function performs the local analysis loop for LETKF, processing each grid point
  independently using its local observations defined by the localization matrix Q.
  This is the CPU version that does not use Kokkos acceleration.

  All local analysis workspace objects (Z_local, S_local, T_sqrt_local, G_local, y_local,
  y_mean_local, delta_scaled_local, r_inv_sqrt_local, w_local, s_transpose_delta_local) are
  created with PETSC_COMM_SELF because the analysis at each vertex is serial and independent.
*/
PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA da, PetscDA_LETKF *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  PetscDA_Ensemble  *en = &impl->en;
  Mat                Z_local, S_local, T_sqrt_local, G_local;
  Mat                X_rows, E_analysis_rows;
  Vec                y_local, y_mean_local, delta_scaled_local, r_inv_sqrt_local;
  Vec                w_local, s_transpose_delta_local;
  const PetscScalar *w_array, *x_array, *g_array, *mean_array;
  PetscScalar       *g_array_w, *e_array, *x_rows_array, *ea_rows_array;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscBLASInt       ndof_b, m_b, lda_xrows_b, lda_g_b, lda_ea_b;
  PetscInt           ndof, max_nnz, rstart;
  PetscInt           lda_x, lda_e, lda_xrows, lda_g, lda_ea;
  PetscReal          sqrt_m_minus_1, scale;

  PetscFunctionBegin;
  ndof           = da->ndof;
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  scale          = 1.0 / sqrt_m_minus_1;
  max_nnz        = impl->max_nnz_per_row;

  /* X and ensemble are accessed at row offsets up to (n_vertices-1)*ndof + (ndof-1).
     Mirror the precondition the Kokkos path enforces so a bad LDA fails fast on either backend. */
  PetscCall(MatDenseGetLDA(X, &lda_x));
  PetscCall(MatDenseGetLDA(en->ensemble, &lda_e));
  PetscCheck(lda_x >= n_vertices * ndof, PetscObjectComm((PetscObject)X), PETSC_ERR_ARG_INCOMP, "X leading dimension %" PetscInt_FMT " < n_vertices*ndof %" PetscInt_FMT, lda_x, n_vertices * ndof);
  PetscCheck(lda_e >= n_vertices * ndof, PetscObjectComm((PetscObject)en->ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble leading dimension %" PetscInt_FMT " < n_vertices*ndof %" PetscInt_FMT, lda_e, n_vertices * ndof);

  /* Create local analysis workspace (max_nnz x m matrices and vectors) */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, max_nnz, m, NULL, &Z_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)Z_local, "dense_"));
  PetscCall(MatSetFromOptions(Z_local));
  PetscCall(MatSetUp(Z_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, max_nnz, m, NULL, &S_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)S_local, "dense_"));
  PetscCall(MatSetFromOptions(S_local));
  PetscCall(MatSetUp(S_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &T_sqrt_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)T_sqrt_local, "dense_"));
  PetscCall(MatSetFromOptions(T_sqrt_local));
  PetscCall(MatSetUp(T_sqrt_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &G_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)G_local, "dense_"));
  PetscCall(MatSetFromOptions(G_local));
  PetscCall(MatSetUp(G_local));

  /* Create vectors using MatCreateVecs() from Z_local (max_nnz x m) */
  PetscCall(MatCreateVecs(Z_local, &w_local, &y_local));
  PetscCall(VecDuplicate(y_local, &y_mean_local));
  PetscCall(VecDuplicate(y_local, &delta_scaled_local));
  PetscCall(VecDuplicate(y_local, &r_inv_sqrt_local));
  PetscCall(VecDuplicate(w_local, &s_transpose_delta_local));

  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, ndof, m, NULL, &X_rows));
  PetscCall(MatDuplicate(X_rows, MAT_DO_NOT_COPY_VALUES, &E_analysis_rows));

  /* X_rows, G_local, E_analysis_rows are loop-invariant; their LDAs and the BLAS-int
     casts of the gemm shape never change inside the n_vertices loop. Hoist to spare
     the dispatch overhead at every vertex. */
  PetscCall(MatDenseGetLDA(G_local, &lda_g));
  PetscCall(MatDenseGetLDA(X_rows, &lda_xrows));
  PetscCall(MatDenseGetLDA(E_analysis_rows, &lda_ea));
  PetscCall(PetscBLASIntCast(ndof, &ndof_b));
  PetscCall(PetscBLASIntCast(m, &m_b));
  PetscCall(PetscBLASIntCast(lda_xrows, &lda_xrows_b));
  PetscCall(PetscBLASIntCast(lda_g, &lda_g_b));
  PetscCall(PetscBLASIntCast(lda_ea, &lda_ea_b));

  /* LETKF: Loop over all grid points and perform local analysis */
  PetscCall(MatGetOwnershipRange(impl->Q, &rstart, NULL));

  /* X, impl->mean, and en->ensemble are loop-invariant; their array views are read or
     written at offsets that change per iteration but the underlying storage does not.
     Hoisting the Get/Restore pairs out of the n_vertices loop avoids repeated lock and
     validation overhead inside the hot path. Safe to hold across the loop because the
     inner MatDenseGetArray{,Read,Write} calls on X_rows, E_analysis_rows, and G_local
     operate on disjoint SELF matrices and never alias X/mean/en->ensemble. */
  PetscCall(MatDenseGetArrayRead(X, &x_array));
  PetscCall(VecGetArrayRead(impl->mean, &mean_array));
  PetscCall(MatDenseGetArrayWrite(en->ensemble, &e_array));

  for (PetscInt i_grid_point = 0; i_grid_point < n_vertices; i_grid_point++) {
    /* Extract local observations for this grid point using Q[i_grid_point,:].
       ExtractLocalObservations() zeros the unwritten [ncols, max_nnz) tail of each
       workspace, so we do not need to MatZeroEntries/VecZeroEntries every iteration. */
    /* Note: i_grid_point is local index, but MatGetRow needs global index */
    PetscCall(ExtractLocalObservations(impl->Q, rstart + i_grid_point, Z_global, observation, y_mean_global, r_inv_sqrt_global, impl->obs_g2l, m, Z_local, y_local, y_mean_local, r_inv_sqrt_local));

    /* Compute local normalized innovation matrix: S_local = R_local^{-1/2} * (Z_local - y_mean_local * 1') / sqrt(m - 1) */
    PetscCall(PetscDAEnsembleComputeNormalizedInnovationMatrix(Z_local, y_mean_local, r_inv_sqrt_local, m, scale, S_local));

    /* Compute local delta_scaled = R_local^{-1/2} * (y_local - y_mean_local) */
    PetscCall(VecWAXPY(delta_scaled_local, -1.0, y_mean_local, y_local));
    PetscCall(VecPointwiseMult(delta_scaled_local, delta_scaled_local, r_inv_sqrt_local));

    /* Factor local T = (I + S_local^T * S_local) */
    PetscCall(PetscDAEnsembleTFactor(da, S_local));

    /* Compute local analysis weights: w_local = T_local^{-1} * S_local^T * delta_scaled_local */
    PetscCall(MatMultTranspose(S_local, delta_scaled_local, s_transpose_delta_local));
    PetscCall(PetscDAEnsembleApplyTInverse(da, s_transpose_delta_local, w_local));

    /* Compute local square-root transform: T_sqrt_local = T_local^{-1/2} (U is identity, so pass NULL) */
    PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, T_sqrt_local));

    /* Form local transform G_local = w_local * 1' + sqrt(m - 1) * T_sqrt_local * U
       Instead of creating w_ones_local = w_local * 1', we add w_local to each column of G_local */
    PetscCall(MatCopy(T_sqrt_local, G_local, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(G_local, sqrt_m_minus_1));
    PetscCall(VecGetArrayRead(w_local, &w_array));
    PetscCall(MatDenseGetArray(G_local, &g_array_w));
    for (PetscInt j = 0; j < m; j++)
      for (PetscInt k = 0; k < m; k++) g_array_w[k + j * lda_g] += w_array[k];
    PetscCall(MatDenseRestoreArray(G_local, &g_array_w));
    PetscCall(VecRestoreArrayRead(w_local, &w_array));

    /* LETKF Algorithm 2, Line 13: Update ensemble at grid point i_grid_point
       E_a[i,:] = x_bar_f[i] + X_f[i,:] * G_local

       Where:
       - x_bar_f[i] is the forecast mean at grid point i_grid_point (ndof values from global mean vector)
       - X_f[i,:] is the forecast anomaly rows at grid point i_grid_point (ndof rows from global anomaly matrix X)
       - G_local = w_local * 1' + sqrt(m-1) * T_local^{1/2} * U (computed above in G_local)
     */
    /* Extract ndof rows starting at (i_grid_point * ndof) from X: X_f[i_grid_point*ndof:(i_grid_point+1)*ndof, :]
       Hold X_rows / E_analysis_rows with a single read/write GetArray each so the fill, gemm,
       mean-add, and copy-out share one Get/Restore pair per vertex. */
    PetscCall(MatDenseGetArray(X_rows, &x_rows_array));
    for (PetscInt j = 0; j < m; j++)
      for (PetscInt k = 0; k < ndof; k++) x_rows_array[k + j * lda_xrows] = x_array[(i_grid_point * ndof + k) + j * lda_x];

    /* Apply local transform via direct BLASgemm: E_analysis_rows = X_rows * G_local.
       Replaces a per-vertex MatMatMult; ndof and m are typically small (1-100), so the
       MatProduct dispatch overhead dominated. */
    PetscCall(MatDenseGetArrayRead(G_local, &g_array));
    PetscCall(MatDenseGetArray(E_analysis_rows, &ea_rows_array));
    PetscCallBLAS("BLASgemm", BLASgemm_("N", "N", &ndof_b, &m_b, &m_b, &one, x_rows_array, &lda_xrows_b, g_array, &lda_g_b, &zero, ea_rows_array, &lda_ea_b));
    PetscCall(MatDenseRestoreArrayRead(G_local, &g_array));
    PetscCall(MatDenseRestoreArray(X_rows, &x_rows_array));

    /* Add local mean and store result back in ensemble at row offset i_grid_point*ndof. */
    for (PetscInt j = 0; j < m; j++) {
      for (PetscInt k = 0; k < ndof; k++) {
        ea_rows_array[k + j * lda_ea] += mean_array[i_grid_point * ndof + k];
        e_array[(i_grid_point * ndof + k) + j * lda_e] = ea_rows_array[k + j * lda_ea];
      }
    }
    PetscCall(MatDenseRestoreArray(E_analysis_rows, &ea_rows_array));
  }
  PetscCall(MatDenseRestoreArrayWrite(en->ensemble, &e_array));
  PetscCall(VecRestoreArrayRead(impl->mean, &mean_array));
  PetscCall(MatDenseRestoreArrayRead(X, &x_array));
  PetscCall(MatDestroy(&E_analysis_rows));
  PetscCall(MatDestroy(&X_rows));
  PetscCall(VecDestroy(&s_transpose_delta_local));
  PetscCall(VecDestroy(&w_local));
  PetscCall(VecDestroy(&r_inv_sqrt_local));
  PetscCall(VecDestroy(&delta_scaled_local));
  PetscCall(VecDestroy(&y_mean_local));
  PetscCall(VecDestroy(&y_local));
  PetscCall(MatDestroy(&G_local));
  PetscCall(MatDestroy(&T_sqrt_local));
  PetscCall(MatDestroy(&S_local));
  PetscCall(MatDestroy(&Z_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFGlobalAnalysis - LOC_NONE fast path: a single global ETKF analysis with no
  per-vertex localization. The m x m T factor and weight vector live on PETSC_COMM_SELF so
  every rank does the identical eigendecomp; only the gram S^T*S and the projection S^T*delta
  need an MPI reduction. Dispatches to the Kokkos backend when R is a Kokkos matrix.
*/
static PetscErrorCode PetscDALETKFGlobalAnalysis(PetscDA da, PetscDA_LETKF *impl, PetscInt m, Mat X, Vec observation)
{
  const PetscScalar *s_array, *d_array;
  PetscScalar       *gram, *buf;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscBLASInt       m_b, n_obs_local_b, s_lda_b, ione = 1;
  PetscMPIInt        m_squared_mpi, m_mpi;
  PetscInt           n_obs_local, s_lda;
  PetscReal          sqrt_m_minus_1, scale;
  PetscBool          use_kokkos;

  PetscFunctionBegin;
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  scale          = 1.0 / sqrt_m_minus_1;
  PetscCall(PetscDALETKFUseKokkosBackend(da, &use_kokkos));
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  if (use_kokkos) {
    PetscCall(PetscDALETKFGlobalAnalysis_Kokkos(da, impl, m, X, observation));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  PetscCheck(!use_kokkos, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "Kokkos backend selected but PetscDALETKFGlobalAnalysis_Kokkos is unavailable in this build");

  PetscCall(PetscDALETKFEnsureGlobalScratch(impl, m));

  /* S = R^{-1/2} * (Z - y_mean*1') / sqrt(m-1) */
  PetscCall(PetscDAEnsembleComputeNormalizedInnovationMatrix(impl->Z, impl->y_mean, impl->r_inv_sqrt, m, scale, impl->S));

  /* delta_scaled = R^{-1/2} * (y^o - y_mean) */
  PetscCall(VecWAXPY(impl->delta_scaled, -1.0, impl->y_mean, observation));
  PetscCall(VecPointwiseMult(impl->delta_scaled, impl->delta_scaled, impl->r_inv_sqrt));

  /* Factor T = (1/rho)I + S^T*S replicated on every rank. */
  PetscCall(PetscCalloc1((size_t)m * m, &gram));
  PetscCall(MatGetLocalSize(impl->S, &n_obs_local, NULL));
  PetscCall(MatDenseGetArrayRead(impl->S, &s_array));
  PetscCall(MatDenseGetLDA(impl->S, &s_lda));
  PetscCall(PetscBLASIntCast(m, &m_b));
  PetscCall(PetscBLASIntCast(n_obs_local, &n_obs_local_b));
  PetscCall(PetscBLASIntCast(s_lda, &s_lda_b));
  if (n_obs_local > 0) PetscCallBLAS("BLASgemm", BLASgemm_("T", "N", &m_b, &m_b, &n_obs_local_b, &one, s_array, &s_lda_b, s_array, &s_lda_b, &zero, gram, &m_b));
  PetscCall(PetscMPIIntCast((PetscInt64)m * m, &m_squared_mpi));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, gram, m_squared_mpi, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)da)));
  PetscCall(PetscDAEnsembleTFactorFromGram(da, m, gram));
  PetscCall(PetscFree(gram));

  /* w = T^{-1} * (S^T * delta_scaled), with the projection reduced across ranks. Hold the
     buffer with VecGetArray across both the local gemv and the in-place allreduce so the
     reduction sees this rank's contribution (VecGetArrayWrite would make the post-gemv data
     undefined after restore). Ranks with no local obs zero the buffer directly because gemv
     (which would overwrite via beta = 0) is skipped on n_obs_local == 0. */
  PetscCall(VecGetArrayRead(impl->delta_scaled, &d_array));
  PetscCall(VecGetArray(impl->s_transpose_delta, &buf));
  if (n_obs_local > 0) PetscCallBLAS("BLASgemv", BLASgemv_("T", &n_obs_local_b, &m_b, &one, s_array, &s_lda_b, d_array, &ione, &zero, buf, &ione));
  else PetscCall(PetscArrayzero(buf, m));
  PetscCall(PetscMPIIntCast(m, &m_mpi));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, buf, m_mpi, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)da)));
  PetscCall(VecRestoreArray(impl->s_transpose_delta, &buf));
  PetscCall(VecRestoreArrayRead(impl->delta_scaled, &d_array));
  PetscCall(MatDenseRestoreArrayRead(impl->S, &s_array));

  PetscCall(PetscDAEnsembleApplyTInverse(da, impl->s_transpose_delta, impl->w));

  /* T_sqrt = T^{-1/2} */
  PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, impl->T_sqrt));

  /* G = w*1' + sqrt(m-1) * T_sqrt (in impl->w_ones, all on PETSC_COMM_SELF). */
  PetscCall(PetscDALETKFReplicateWeightVector(impl->w, m, impl->w_ones));
  PetscCall(MatAXPY(impl->w_ones, sqrt_m_minus_1, impl->T_sqrt, SAME_NONZERO_PATTERN));

  /* E = mean*1' + X * G */
  PetscCall(PetscDALETKFUpdateEnsembleWithTransform(impl->mean, X, impl->w_ones, m, impl->en.ensemble));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFRebuildHTemps - ensure the cached H-compatible work vecs match H's current
  layout and vec type, rebuilding them (and any caches that depend on H's backend) when H has
  drifted since the last analysis. The two `if` blocks (invalidate, then allocate) must remain
  separate rather than an if/else: when the first block fires it nulls `impl->H_temp_in` via
  VecDestroy, and the second `if (!impl->H_temp_in)` must then re-create the temps in the same
  call. Collapsing to if/else would leave a freshly-destroyed cache uninitialized until the next
  analysis, breaking the post-condition that on return the H_temp_* and H_vec_type fields are
  populated and consistent with the current H.
*/
static PetscErrorCode PetscDALETKFRebuildHTemps(PetscDA da, PetscDA_LETKF *impl, Mat H)
{
  PetscInt  cur_in_local, cur_out_local, want_in_local, want_out_local;
  VecType   want_type;
  PetscBool type_match;

  PetscFunctionBegin;
  PetscCall(MatGetVecType(H, &want_type));
  if (impl->H_temp_in) {
    PetscCall(VecGetLocalSize(impl->H_temp_in, &cur_in_local));
    PetscCall(VecGetLocalSize(impl->H_temp_out, &cur_out_local));
    PetscCall(MatGetLocalSize(H, &want_out_local, &want_in_local));
    PetscCall(PetscStrcmp(impl->H_vec_type, want_type, &type_match));
    if (!type_match || cur_in_local != want_in_local || cur_out_local != want_out_local) {
      PetscCall(VecDestroy(&impl->H_temp_in));
      PetscCall(VecDestroy(&impl->H_temp_out));
      PetscCall(PetscFree(impl->H_vec_type));
      /* The obs-scatter source layout is templated off H, and Q's device mirrors live in the
         backend matching the old H vec type (Kokkos vs host); reset the full localization
         cache so the next analysis rebuilds Q and its mirrors against the new H. */
      PetscCall(PetscDALETKFResetLocalization_LETKF(da));
    }
  }
  if (!impl->H_temp_in) {
    PetscCall(MatCreateVecs(H, &impl->H_temp_in, &impl->H_temp_out));
    PetscCall(PetscStrallocpy(want_type, &impl->H_vec_type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAEnsembleAnalysis_LETKF(PetscDA da, Vec observation, Mat H)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;
  Mat            X;
  PetscInt       m;
  PetscBool      reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  m = impl->en.size;
  PetscCheck(m >= 2, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be >= 2 for LETKF; got %" PetscInt_FMT, m);

  /* Check for reallocation needs */
  if (impl->mean) {
    PetscInt mean_size;
    PetscCall(VecGetSize(impl->mean, &mean_size));
    if (mean_size != da->state_size) reallocate = PETSC_TRUE;
  }
  if (impl->Z) {
    PetscInt z_rows, z_cols;
    PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
    if (z_rows != da->obs_size || z_cols != m) reallocate = PETSC_TRUE;
  }
  /* impl->T_sqrt is owned only by the LOC_NONE fast path (PetscDALETKFEnsureGlobalScratch());
     the per-vertex path uses its own SELF T_sqrt_local. This check fires on a NONE -> per-vertex
     transition with a changed m, where the stale T_sqrt would mis-size the next NONE cycle. */
  if (impl->T_sqrt) {
    PetscInt t_rows, t_cols;
    PetscCall(MatGetSize(impl->T_sqrt, &t_rows, &t_cols));
    if (t_rows != m || t_cols != m) reallocate = PETSC_TRUE;
  }

  /* Initialize or reallocate persistent work objects */
  if (!impl->mean || reallocate) {
    /* On reallocation the cached Q (and obs scatter / Kokkos device buffers) describe a
       prior state_size/obs_size and must be torn down so the next analysis rebuilds them
       against the new layout. Skip on first-time init (nothing to reset yet). */
    if (reallocate) PetscCall(PetscDALETKFResetLocalization_LETKF(da));
    PetscCall(VecDestroy(&impl->mean));
    PetscCall(VecDestroy(&impl->y_mean));
    PetscCall(VecDestroy(&impl->delta_scaled));
    PetscCall(VecDestroy(&impl->w));
    PetscCall(VecDestroy(&impl->s_transpose_delta));
    PetscCall(VecDestroy(&impl->r_inv_sqrt));
    PetscCall(VecDestroy(&impl->H_temp_in));
    PetscCall(VecDestroy(&impl->H_temp_out));
    PetscCall(PetscFree(impl->H_vec_type));
    PetscCall(MatDestroy(&impl->Z));
    PetscCall(MatDestroy(&impl->S));
    PetscCall(MatDestroy(&impl->T_sqrt));
    PetscCall(MatDestroy(&impl->w_ones));

    /* Create mean vector from ensemble matrix (left vector = state space) */
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

    /* Create S matrix (same layout as Z) */
    PetscCall(MatDuplicate(impl->Z, MAT_DO_NOT_COPY_VALUES, &impl->S));

    /* T_sqrt and w_ones are m x m and used only by the LOC_NONE fast path; allocate them
       lazily in PetscDALETKFGlobalAnalysis() so the per-vertex paths do not pay for them. */
  }

  /* Alg 6.4 line 1-2: Compute ensemble mean and scaled anomalies */
  PetscCall(PetscDAEnsembleComputeMean(da, impl->mean));

  /* Create anomaly matrix X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDAEnsembleComputeAnomalies(da, impl->mean, &X));

  /* Alg 6.4 line 3-4: Compute GLOBAL observation ensemble Z = H * E column-by-column,
     staged through H-compatible cached work vecs because impl->Z (MATDENSE) and H
     (possibly MATAIJKOKKOS) cannot share a MatMatMult product type. */
  /* Lazily allocate / rebuild the cached H-compatible work vecs (and reset Q if H's vec-type
     backend changed). */
  PetscCall(PetscDALETKFRebuildHTemps(da, impl, H));

  /* Lazily build Q for built-in distance-based kernels using cached coordinates. The dispatcher
     selects host vs Kokkos backend from the type of the cached observation operator. Setters
     destroy Q via PetscDALETKFResetLocalization() when their inputs change, so a non-NULL Q is
     guaranteed to match the current (type, radius, coord_*) tuple. Built after PetscDALETKFRebuildHTemps()
     because that call may reset Q via PetscDALETKFResetLocalization_LETKF() when H's vec-type
     backend changed since the last analysis. */
  if (impl->type != PETSCDA_LETKF_LOC_NONE && !impl->Q) {
    Mat       Q_new = NULL;
    PetscInt  max_nnz_local, n_nnz_local;
    PetscBool use_kokkos;

    PetscCheck(impl->coord_H, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Coordinates not set; call PetscDALETKFSetLocalizationCoordinates() before analysis");
    PetscCheck(impl->localization_radius > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Localization radius not set; call PetscDALETKFSetLocalizationRadius() before analysis");
    /* Q's backend must match the analysis-time backend, which keys off da->R; otherwise a CPU
       analysis would walk a Kokkos Q (or vice versa) and pay backend-mismatch transfer overhead. */
    PetscCall(PetscDALETKFUseKokkosBackend(da, &use_kokkos));
    /* Backends compute the per-rank max-nnz/row and total local nnz from row_counts[] in hand;
       passing them through avoids both a MatGetRow walk and a MatGetInfo call in
       PetscDALETKFInstallQ() (both of which force a device->host sync on AIJKOKKOS Q).
       The analysis-time `H` is threaded into InstallQ as the obs-scatter source template so the
       scatter rows match the vectors actually being scattered, even if `H`'s row partition or
       vec type differs from the `coord_H` cached at SetLocalizationCoordinates time. */
    PetscCall(PetscDALETKFCreateLocalizationMat(impl->type, impl->localization_radius, impl->coord_xyz, impl->coord_bd, impl->coord_H, use_kokkos, &Q_new, &max_nnz_local, &n_nnz_local));
    PetscCall(PetscDALETKFInstallQ(da, Q_new, max_nnz_local, n_nnz_local, H));
    PetscCall(MatDestroy(&Q_new));
  }

  /* Compute Z = H * E column by column to avoid Kokkos vector type issues */
  for (PetscInt j = 0; j < m; j++) {
    Vec col_in, col_out;
    PetscCall(MatDenseGetColumnVecRead(impl->en.ensemble, j, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(impl->Z, j, &col_out));
    PetscCall(VecCopy(col_in, impl->H_temp_in));
    PetscCall(MatMult(H, impl->H_temp_in, impl->H_temp_out));
    PetscCall(VecCopy(impl->H_temp_out, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(impl->Z, j, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(impl->en.ensemble, j, &col_in));
  }

  /* Compute GLOBAL observation mean y_mean = H * x_mean using the same cached temps. */
  PetscCall(VecCopy(impl->mean, impl->H_temp_in));
  PetscCall(MatMult(H, impl->H_temp_in, impl->H_temp_out));
  PetscCall(VecCopy(impl->H_temp_out, impl->y_mean));

  /* Compute GLOBAL R^{-1/2} (assumes diagonal R) */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  if (impl->type == PETSCDA_LETKF_LOC_NONE) PetscCall(PetscDALETKFGlobalAnalysis(da, impl, m, X, observation));
  else {
    PetscInt  n_local, n_obs_local, rows_old, cols_old;
    PetscBool use_kokkos;

    PetscCall(PetscDALETKFUseKokkosBackend(da, &use_kokkos));

    /* Per-vertex local analysis path. PetscDALETKFInstallQ() above already templated the
       obs-scatter on the live `H`, and PetscDALETKFRebuildHTemps() invalidates Q (and therefore
       triggers a re-InstallQ here) whenever H's row layout or vec type drifts, so impl->obs_scat
       is guaranteed non-NULL and compatible with the live observation vectors. */
    PetscCall(VecScatterBegin(impl->obs_scat, observation, impl->obs_work, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(impl->obs_scat, observation, impl->obs_work, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecScatterBegin(impl->obs_scat, impl->y_mean, impl->y_mean_work, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(impl->obs_scat, impl->y_mean, impl->y_mean_work, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecScatterBegin(impl->obs_scat, impl->r_inv_sqrt, impl->r_inv_sqrt_work, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(impl->obs_scat, impl->r_inv_sqrt, impl->r_inv_sqrt_work, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetLocalSize(impl->obs_work, &n_obs_local));
    if (impl->Z_work) {
      PetscCall(MatGetSize(impl->Z_work, &rows_old, &cols_old));
      if (rows_old != n_obs_local || cols_old != m) PetscCall(MatDestroy(&impl->Z_work));
    }
    if (!impl->Z_work) PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n_obs_local, m, NULL, &impl->Z_work));
    for (PetscInt i = 0; i < m; i++) {
      Vec z_col_global, z_col_local;
      PetscCall(MatDenseGetColumnVecRead(impl->Z, i, &z_col_global));
      PetscCall(MatDenseGetColumnVecWrite(impl->Z_work, i, &z_col_local));
      PetscCall(VecScatterBegin(impl->obs_scat, z_col_global, z_col_local, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(impl->obs_scat, z_col_global, z_col_local, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(MatDenseRestoreColumnVecRead(impl->Z, i, &z_col_global));
      PetscCall(MatDenseRestoreColumnVecWrite(impl->Z_work, i, &z_col_local));
    }

    PetscCall(MatGetLocalSize(impl->Q, &n_local, NULL));
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
    if (use_kokkos) PetscCall(PetscDALETKFLocalAnalysis_Kokkos(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
    else PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
#else
    PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
#endif
  }

  PetscCall(MatDestroy(&X));
  /* Emit -petscda_view once per localization configuration, after the first analysis has built Q
     so the viewer can report the localization-matrix state. The flag is cleared whenever
     PetscDALETKFResetLocalization_LETKF() drops Q (radius/type/coords change), so reconfiguring
     fires a fresh view. Matches KSPSolve()/SNESSolve() which self-call ViewFromOptions at the tail. */
  if (!impl->view_emitted) {
    PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));
    impl->view_emitted = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFResetLocalization_LETKF - destroy the cached Q matrix, obs scatter, and any device
  buffers tied to Q. Coordinates/type/radius are preserved so the next analysis rebuilds Q from
  the current inputs. Called by the setters that mutate Q-determining inputs.
*/
static PetscErrorCode PetscDALETKFResetLocalization_LETKF(PetscDA da)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  /* Drop only the Q device mirrors; the persistent cusolver/rocblas/SYCL handle and the
     eigensolver workspace are reused across Q rebuilds. */
  if (impl->Q) PetscCall(PetscDALETKFDestroyQDeviceMirrors_Kokkos(impl));
#endif
  PetscCall(PetscDALETKFDestroyObsScatter(impl));
  PetscCall(MatDestroy(&impl->Q));
  impl->max_nnz_per_row = 0;
  impl->view_emitted    = PETSC_FALSE;
  impl->n_nnz_local     = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalizationRadius_LETKF(PetscDA da, PetscReal radius)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  PetscCheck(radius > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Localization radius must be positive, got %g", (double)radius);
  if (impl->localization_radius != radius) {
    impl->localization_radius = radius;
    PetscCall(PetscDALETKFResetLocalization_LETKF(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalizationType_LETKF(PetscDA da, PetscDALETKFLocalizationType type)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  PetscCheck(type >= 0 && type < PETSCDA_LETKF_LOC_NUM_TYPES, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid localization type %d; must be in [0,%d)", (int)type, (int)PETSCDA_LETKF_LOC_NUM_TYPES);
  if (impl->type != type) {
    impl->type = type;
    PetscCall(PetscDALETKFResetLocalization_LETKF(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFGetLocalizationType_LETKF(PetscDA da, PetscDALETKFLocalizationType *type)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  *type = impl->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalizationCoordinates_LETKF(PetscDA da, const Vec xyz[3], const PetscReal bd[3], Mat H)
{
  PetscDA_LETKF *impl    = (PetscDA_LETKF *)da->data;
  PetscBool      changed = PETSC_FALSE;
  PetscInt       H_rows, H_cols, vert_global, vert_local;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  PetscAssertPointer(xyz, 2);
  /* bd[d] > 0 selects periodic handling for dimension d; bd[d] == 0 means non-periodic. Reject
     negative values so a stray sign flip cannot silently re-interpret as non-periodic. */
  if (bd)
    for (PetscInt d = 0; d < 3; d++) PetscCheck(bd[d] >= 0.0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Periodic-domain extent bd[%" PetscInt_FMT "] must be non-negative (use 0 for non-periodic), got %g", d, (double)bd[d]);
  /* Validate H and xyz[0] against the PetscDA's recorded sizes at the API boundary so a
     structurally mismatched H or coordinate vector is rejected here, where the caller can fix
     it, rather than after the previous Q/obs-scatter has already been torn down inside the
     lazy-build path. */
  PetscCall(MatGetSize(H, &H_rows, &H_cols));
  PetscCheck(H_rows == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H has %" PetscInt_FMT " rows; PetscDA obs_size is %" PetscInt_FMT, H_rows, da->obs_size);
  PetscCheck(da->ndof > 0 && da->state_size % da->ndof == 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "state_size (%" PetscInt_FMT ") must be a positive multiple of ndof (%" PetscInt_FMT ")", da->state_size, da->ndof);
  PetscCall(VecGetSize(xyz[0], &vert_global));
  PetscCall(VecGetLocalSize(xyz[0], &vert_local));
  PetscCheck(vert_global == da->state_size / da->ndof, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "xyz[0] global size %" PetscInt_FMT " != vertex count state_size/ndof (%" PetscInt_FMT ")", vert_global, da->state_size / da->ndof);
  /* H's columns must match xyz[0]'s rows so PetscDALETKFComputeObsCoords() can MatMult(H, xyz[d]).
     The multi-DOF observation operator (cols == state_size) is a frequent mistake here; reject it
     before PetscDALETKFResetLocalization_LETKF() tears down the previous Q. */
  PetscCheck(H_cols == vert_global, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H has %" PetscInt_FMT " columns; expected %" PetscInt_FMT " (vertex count, == xyz[0] global size). Did you pass the multi-DOF observation operator instead of the per-vertex one?", H_cols, vert_global);
  /* Per-dim slots beyond xyz[0] must share xyz[0]'s global size and local partition; otherwise
     PetscDALETKFGatherObsBbox() would walk mismatched coordinate arrays and read past valid memory. */
  for (PetscInt d = 1; d < 3; d++) {
    PetscInt other_global, other_local;

    if (!xyz[d]) continue;
    PetscCall(VecGetSize(xyz[d], &other_global));
    PetscCall(VecGetLocalSize(xyz[d], &other_local));
    PetscCheck(other_global == vert_global, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "xyz[%" PetscInt_FMT "] global size %" PetscInt_FMT " != xyz[0] global size %" PetscInt_FMT, d, other_global, vert_global);
    PetscCheck(other_local == vert_local, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "xyz[%" PetscInt_FMT "] local size %" PetscInt_FMT " != xyz[0] local size %" PetscInt_FMT, d, other_local, vert_local);
  }
  /* Compare against the cached (xyz, bd, H) tuple by pointer/value so that re-supplying the same
     geometry (a common pattern when the tutorial reapplies the same observation operator each
     analysis cycle) does not invalidate Q and force the obs-scatter and device buffers to be
     rebuilt. The contract requires the user to call this again after mutating any of these objects. */
  for (PetscInt d = 0; d < 3; d++) {
    if (impl->coord_xyz[d] != xyz[d]) changed = PETSC_TRUE;
    if (impl->coord_bd[d] != (bd ? bd[d] : 0.0)) changed = PETSC_TRUE;
  }
  if (impl->coord_H != H) changed = PETSC_TRUE;
  if (!changed) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscDALETKFClearCoordinates(impl));
  for (PetscInt d = 0; d < 3; d++) {
    if (xyz[d]) {
      PetscCall(PetscObjectReference((PetscObject)xyz[d]));
      impl->coord_xyz[d] = xyz[d];
    }
    impl->coord_bd[d] = bd ? bd[d] : 0.0;
  }
  PetscCall(PetscObjectReference((PetscObject)H));
  impl->coord_H = H;
  PetscCall(PetscDALETKFResetLocalization_LETKF(da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFGetLocalizationRadius_LETKF(PetscDA da, PetscReal *radius)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  *radius = impl->localization_radius;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Install a freshly built localization matrix Q (validate sizes, cache row nnz bounds, wire Kokkos
  device buffers when applicable). Only called from the lazy-build path. The Kokkos device-side
  setup runs only when PETSc was built with Kokkos kernels; the bare CPU analysis path (used by
  serial non-Kokkos builds) needs only the size validation and nnz bookkeeping below.

  scatter_H templates the obs-scatter source layout. The caller passes the analysis-time `H` so
  that the scatter source matches the vectors the per-vertex path is about to scatter, even when
  the user's analysis-time `H` has a different row partition or vec type than the `coord_H` that
  was cached at SetLocalizationCoordinates time. Q's column footprint (the unique global obs
  indices each rank touches) is independent of scatter_H's row layout, so this is well-defined as
  long as scatter_H and Q agree on the global obs-space size - `PetscDALETKFSetupObsScatter()`
  PetscCheck's that.
*/
static PetscErrorCode PetscDALETKFInstallQ(PetscDA da, Mat Q, PetscInt max_nnz_local, PetscInt n_nnz_local, Mat scatter_H)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;
  PetscInt       nrows, ncols;
  PetscBool      use_kokkos;

  PetscFunctionBegin;
  PetscCheck(da->ndof > 0 && da->state_size % da->ndof == 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "state_size (%" PetscInt_FMT ") must be a positive multiple of ndof (%" PetscInt_FMT ")", da->state_size, da->ndof);
  PetscCall(MatGetSize(Q, &nrows, &ncols));
  PetscCheck(nrows == da->state_size / da->ndof, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix rows (%" PetscInt_FMT ") must equal vertex count state_size/ndof (%" PetscInt_FMT ")", nrows, da->state_size / da->ndof);
  PetscCheck(ncols == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix columns (%" PetscInt_FMT ") must match observation size (%" PetscInt_FMT ")", ncols, da->obs_size);
  PetscCheck(max_nnz_local >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "max_nnz_local must be >= 0, got %" PetscInt_FMT, max_nnz_local);
  PetscCheck(n_nnz_local >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "n_nnz_local must be >= 0, got %" PetscInt_FMT, n_nnz_local);
  /* The obs-scatter is needed by both the CPU and Kokkos per-vertex paths whenever Q exists.
     Validate before tearing down the previous Q/obs-scatter so a missing prerequisite leaves
     the impl in its prior usable state instead of a half-installed one. */
  PetscValidHeaderSpecific(scatter_H, MAT_CLASSID, 5);
  PetscCall(PetscDALETKFUseKokkosBackend(da, &use_kokkos));

#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  /* Drop only the Q device mirrors; the eigensolver workspace and solver handle persist. */
  if (impl->Q) PetscCall(PetscDALETKFDestroyQDeviceMirrors_Kokkos(impl));
#endif
  /* Destroy the previous obs-scatter so SetupObsScatter() can rebuild it for the new Q footprint. */
  PetscCall(PetscDALETKFDestroyObsScatter(impl));

  PetscCall(PetscObjectReference((PetscObject)Q));
  PetscCall(MatDestroy(&impl->Q));
  impl->Q = Q;

  /* The CSR backends already walked row_counts[] to size their allocations; reuse the per-rank
     max (and total local nnz) computed there instead of MatGetRow/MatGetInfo walks, which would
     force a device->host sync on AIJKOKKOS Q. Allreduce the max so impl->max_nnz_per_row holds
     the global max all ranks need for sizing; n_nnz_local stays per-rank. */
  impl->max_nnz_per_row = max_nnz_local;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &impl->max_nnz_per_row, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)da)));
  impl->n_nnz_local = n_nnz_local;

  PetscCall(PetscDALETKFSetupObsScatter(impl, scatter_H));
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  /* Gate the device-mirror setup on the analysis-time backend: a Kokkos-capable build with a
     non-Kokkos da->R runs the CPU per-vertex path, which never reads Q_device_*. Skipping the
     setup avoids a redundant MatGetRow walk over every local row of Q and the three device-resident
     Kokkos Views it would allocate. */
  if (use_kokkos) PetscCall(PetscDALETKFSetupLocalization_Kokkos(impl));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAView_LETKF(PetscDA da, PetscViewer viewer)
{
  PetscBool      iascii = PETSC_FALSE, is_kokkos = PETSC_FALSE;
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCall(PetscDAView_Ensemble(da, viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscDALETKFUseKokkosBackend(da, &is_kokkos));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: %s\n", is_kokkos ? "Kokkos" : "CPU"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization type: %s\n", PetscDALETKFLocalizationTypes[impl->type]));
    if (impl->type != PETSCDA_LETKF_LOC_NONE) {
      if (impl->localization_radius > 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization radius: %g\n", (double)impl->localization_radius));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization radius: (unset)\n"));
      if (is_kokkos) {
        if (impl->batch_size > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "  GPU batch size: %" PetscInt_FMT "\n", impl->batch_size));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "  GPU batch size: auto\n"));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_LETKF(PetscDA da, PetscOptionItems *PetscOptionsObjectPtr)
{
  PetscDA_LETKF   *impl               = (PetscDA_LETKF *)da->data;
  PetscOptionItems PetscOptionsObject = *PetscOptionsObjectPtr;
  PetscReal        radius;
  PetscInt         type_idx, batch_size;
  PetscBool        type_set = PETSC_FALSE, radius_set = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscDASetFromOptions_Ensemble(da, PetscOptionsObjectPtr));
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscDA LETKF Options");
  batch_size = impl->batch_size;
  PetscCall(PetscOptionsInt("-petscda_letkf_batch_size", "Batch size for GPU processing (0 = auto)", "PETSCDALETKF", batch_size, &batch_size, NULL));
  PetscCheck(batch_size >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "batch_size must be >= 0, got %" PetscInt_FMT, batch_size);
  impl->batch_size = batch_size;
  radius           = impl->localization_radius;
  PetscCall(PetscOptionsReal("-petscda_letkf_localization_radius", "Localization cutoff radius for built-in kernels", "PetscDALETKFSetLocalizationRadius", radius, &radius, &radius_set));
  if (radius_set) PetscCall(PetscDALETKFSetLocalizationRadius(da, radius));
  type_idx = (PetscInt)impl->type;
  PetscCall(PetscOptionsEList("-petscda_letkf_localization_type", "Localization kernel type", "PetscDALETKFSetLocalizationType", PetscDALETKFLocalizationTypes, PETSCDA_LETKF_LOC_NUM_TYPES, PetscDALETKFLocalizationTypes[type_idx], &type_idx, &type_set));
  if (type_set) PetscCall(PetscDALETKFSetLocalizationType(da, (PetscDALETKFLocalizationType)type_idx));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCDALETKF - The Local ETKF performs the analysis update locally around each grid point, enabling scalable assimilation on large
   domains by avoiding the global ensemble covariance matrix.

   Options Database Keys:
+ -petscda_type letkf                                                  - set the `PetscDAType` to `PETSCDALETKF`
. -petscda_ensemble_size size                                          - number of ensemble members
. -petscda_ensemble_inflation factor                                   - multiplicative inflation factor applied to anomalies
. -petscda_letkf_batch_size batch_size                                 - set the batch size for GPU processing
. -petscda_letkf_localization_radius radius                            - localization cutoff radius for the built-in kernels (must be positive)
- -petscda_letkf_localization_type (none|gaspari_cohn|gaussian|boxcar) - select the localization kernel

   Level: beginner

   Notes:
   The default localization kernel is `PETSCDA_LETKF_LOC_GASPARI_COHN`, which requires the user to
   call `PetscDALETKFSetLocalizationRadius()` and `PetscDALETKFSetLocalizationCoordinates()` before
   the first analysis. To skip localization entirely use `PetscDALETKFSetLocalizationType(da, PETSCDA_LETKF_LOC_NONE)`
   (or `-petscda_letkf_localization_type none`).
   Both the CPU and Kokkos analysis paths support multi-rank runs; the Kokkos backend is selected
   when the covariance matrix `da->R` is a Kokkos AIJ type, otherwise the CPU per-vertex (or LOC_NONE
   replicated) path is used.

.seealso: [](ch_da), `PetscDA`, `PetscDACreate()`, `PetscDALETKFSetLocalizationRadius()`, `PetscDALETKFGetLocalizationRadius()`,
          `PetscDALETKFSetLocalizationType()`, `PetscDALETKFGetLocalizationType()`, `PetscDALETKFSetLocalizationCoordinates()`,
          `PetscDALETKFResetLocalization()`, `PetscDAEnsembleSetSize()`, `PetscDASetSizes()`, `PetscDAEnsembleSetInflation()`,
          `PetscDAEnsembleComputeMean()`, `PetscDAEnsembleComputeAnomalies()`, `PetscDAEnsembleAnalysis()`, `PetscDAEnsembleForecast()`
M*/

PETSC_INTERN PetscErrorCode PetscDACreate_LETKF(PetscDA da)
{
  PetscDA_LETKF *impl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&impl));
  da->data = impl;
  PetscCall(PetscDACreate_Ensemble(da));
  da->ops->setup          = PetscDASetUp_Ensemble;
  da->ops->destroy        = PetscDADestroy_LETKF;
  da->ops->view           = PetscDAView_LETKF;
  da->ops->setfromoptions = PetscDASetFromOptions_LETKF;
  impl->en.analysis       = PetscDAEnsembleAnalysis_LETKF;
  impl->en.forecast       = PetscDAEnsembleForecast_Ensemble;

  impl->localization_radius = 0.0;
  impl->Q                   = NULL;
  impl->batch_size          = 0;
  impl->type                = PETSCDA_LETKF_LOC_GASPARI_COHN;
  impl->view_emitted        = PETSC_FALSE;
  for (PetscInt d = 0; d < 3; d++) {
    impl->coord_xyz[d] = NULL;
    impl->coord_bd[d]  = 0.0;
  }
  impl->coord_H = NULL;

  /* Register the method for setting localization */
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalizationRadius_C", PetscDALETKFSetLocalizationRadius_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFGetLocalizationRadius_C", PetscDALETKFGetLocalizationRadius_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalizationType_C", PetscDALETKFSetLocalizationType_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFGetLocalizationType_C", PetscDALETKFGetLocalizationType_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalizationCoordinates_C", PetscDALETKFSetLocalizationCoordinates_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFResetLocalization_C", PetscDALETKFResetLocalization_LETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFSetLocalizationRadius - Sets the localization cutoff radius used by LETKF's built-in distance-based kernels.

  Logically Collective

  Input Parameters:
+ da     - the `PetscDA` context
- radius - the localization cutoff radius (must be positive; use a large value for effectively no localization)

  Level: advanced

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetLocalizationCoordinates()`, `PetscDALETKFGetLocalizationRadius()`
@*/
PetscErrorCode PetscDALETKFSetLocalizationRadius(PetscDA da, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveReal(da, radius, 2);
  PetscTryMethod(da, "PetscDALETKFSetLocalizationRadius_C", (PetscDA, PetscReal), (da, radius));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFGetLocalizationRadius - Gets the localization cutoff radius used by LETKF's built-in distance-based kernels.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. radius - the localization cutoff radius

  Level: advanced

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetLocalizationRadius()`
@*/
PetscErrorCode PetscDALETKFGetLocalizationRadius(PetscDA da, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(radius, 2);
  PetscUseMethod(da, "PetscDALETKFGetLocalizationRadius_C", (PetscDA, PetscReal *), (da, radius));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFSetLocalizationType - Selects the localization kernel used by `PETSCDALETKF`.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` context
- type - the kernel type (see `PetscDALETKFLocalizationType`)

  Level: intermediate

  Notes:
  Use `PETSCDA_LETKF_LOC_NONE` to bypass localization entirely; the analysis is then mathematically
  equivalent to the global ETKF and dispatches through a single global eigensolve plus a dense
  `BLASgemm` weight transform reduced across ranks, instead of the per-vertex local loop.

  For the built-in distance-based kernels (`PETSCDA_LETKF_LOC_GASPARI_COHN`, `PETSCDA_LETKF_LOC_GAUSSIAN`,
  `PETSCDA_LETKF_LOC_BOXCAR`) you must also call `PetscDALETKFSetLocalizationRadius()` and
  `PetscDALETKFSetLocalizationCoordinates()`. The localization matrix is then constructed
  lazily before the first analysis.
  All three built-in kernels are 1 at distance 0; `radius` selects the effective support but the
  cutoff distance and continuity at the cutoff differ.
  `PETSCDA_LETKF_LOC_GASPARI_COHN` is compactly supported with cutoff at distance `2*radius`, and
  is C^1 continuous everywhere (it tapers smoothly to zero at the cutoff).
  `PETSCDA_LETKF_LOC_GAUSSIAN` is `exp(-d^2 / (2*radius^2))` truncated at distance `2*radius`; the
  truncation introduces a discontinuity of `exp(-2)` (~0.135) at the cutoff, so prefer
  `PETSCDA_LETKF_LOC_GASPARI_COHN` if a smooth taper at the cutoff matters.
  `PETSCDA_LETKF_LOC_BOXCAR` is 1 inside `radius` and 0 outside; the discontinuity is by design.

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFLocalizationType`, `PetscDALETKFGetLocalizationType()`,
          `PetscDALETKFSetLocalizationRadius()`, `PetscDALETKFSetLocalizationCoordinates()`
@*/
PetscErrorCode PetscDALETKFSetLocalizationType(PetscDA da, PetscDALETKFLocalizationType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(da, type, 2);
  PetscCheck(type >= PETSCDA_LETKF_LOC_NONE && type < PETSCDA_LETKF_LOC_NUM_TYPES, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Localization type %d out of range [0, %d)", (int)type, (int)PETSCDA_LETKF_LOC_NUM_TYPES);
  PetscTryMethod(da, "PetscDALETKFSetLocalizationType_C", (PetscDA, PetscDALETKFLocalizationType), (da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFGetLocalizationType - Returns the localization kernel currently used by `PETSCDALETKF`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. type - the kernel type

  Level: intermediate

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFLocalizationType`, `PetscDALETKFSetLocalizationType()`
@*/
PetscErrorCode PetscDALETKFGetLocalizationType(PetscDA da, PetscDALETKFLocalizationType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscUseMethod(da, "PetscDALETKFGetLocalizationType_C", (PetscDA, PetscDALETKFLocalizationType *), (da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFSetLocalizationCoordinates - Provides the geometry used to lazily build the
  localization matrix `Q` for a built-in `PETSCDALETKF` kernel.

  Collective

  Input Parameters:
+ da  - the `PetscDA` context
. xyz - length-3 array of coordinate vectors, one per spatial dimension; set unused trailing
        slots to `NULL` (the spatial dimension is taken to be the index of the first `NULL`,
        so `{x, y, NULL}` is 2D and `{x, NULL, NULL}` is 1D)
. bd  - length-3 array of periodic-domain extents (use 0 for non-periodic dimensions); pass
        `NULL` to mean fully non-periodic
- H   - the observation operator (used to map state-space coordinates to observation locations)

  Level: intermediate

  Notes:
  The `xyz` array must always have three slots even in 1D or 2D; trailing slots are set to `NULL`.
  This matches the internal cached layout `coord_xyz[3]` and the layout used by both Q backends.

  The localization matrix `Q` is built on first analysis (or whenever the type, radius or
  coordinates change) using the kernel selected by `PetscDALETKFSetLocalizationType()`. If the
  current type is `PETSCDA_LETKF_LOC_NONE`, the coordinates are cached but the analysis continues
  to run the NONE fast path; switch to a distance-based kernel via
  `PetscDALETKFSetLocalizationType()` for the cached coordinates to take effect.
  The reference counts on `xyz` and `H` are increased; the caller may destroy them afterwards.

  The cached coordinate `Vec`s are referenced, not deep-copied. If the caller mutates the contents
  of any element of `xyz` after this call (for example, after a remesh or recoordinate step), the
  cached `Q` will not be rebuilt automatically; call `PetscDALETKFResetLocalization()` to invalidate
  `Q` and force a rebuild on the next analysis.

  The columns of `Q` are global indices into the observation vector, derived from the row
  ownership and sparsity of the `H` cached here. The `H` passed to `PetscDAEnsembleAnalysis()`
  must therefore use the same global row indexing (same observation ordering and the same global
  obs-space size) as the `H` cached here. The analysis-time `H` may differ from the cached `H`
  in MPI row partition or vec type - the obs-scatter is templated on the analysis-time `H`, so
  those differences are absorbed automatically. A structurally different `H` (rows referring to
  different physical observations) will produce wrong analyses without raising an error; in that
  case, call `PetscDALETKFSetLocalizationCoordinates()` again with the new `H` to rebuild `Q`.

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetLocalizationType()`,
          `PetscDALETKFSetLocalizationRadius()`
@*/
PetscErrorCode PetscDALETKFSetLocalizationCoordinates(PetscDA da, const Vec xyz[3], const PetscReal bd[3], Mat H)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  /* xyz is required; bd is optional. Use an always-on PetscCheck rather than the debug-only
     PetscAssertPointer() so a NULL xyz argument is rejected cleanly in optimized builds before
     the xyz[0] dereference below. */
  PetscCheck(xyz, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_NULL, "xyz must be a non-NULL length-3 array of Vec");
  PetscCheck(xyz[0], PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "xyz[0] must be a valid Vec; the spatial dimension is taken to be the index of the first NULL slot in xyz[3]");
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  if (bd)
    for (PetscInt d = 0; d < 3; d++) PetscValidLogicalCollectiveReal(da, bd[d], 3);
  PetscTryMethod(da, "PetscDALETKFSetLocalizationCoordinates_C", (PetscDA, const Vec[3], const PetscReal[3], Mat), (da, xyz, bd, H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFResetLocalization - Discards the cached localization matrix `Q` so the next analysis
  rebuilds it from the current type, radius, and coordinates.

  Collective

  Input Parameter:
. da - the `PetscDA` context

  Level: advanced

  Notes:
  The setters `PetscDALETKFSetLocalizationType()`, `PetscDALETKFSetLocalizationRadius()`, and
  `PetscDALETKFSetLocalizationCoordinates()` already invalidate `Q` when their inputs actually
  change, so most users never need to call this directly. Use it when an input was mutated outside
  of the setters (for example, the entries of a cached coordinate `Vec` were edited in place, or
  the cached observation operator `H` was reassembled with different sparsity).

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetLocalizationType()`,
          `PetscDALETKFSetLocalizationRadius()`, `PetscDALETKFSetLocalizationCoordinates()`
@*/
PetscErrorCode PetscDALETKFResetLocalization(PetscDA da)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscTryMethod(da, "PetscDALETKFResetLocalization_C", (PetscDA), (da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

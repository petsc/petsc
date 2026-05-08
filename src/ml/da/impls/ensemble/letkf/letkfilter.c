#include <petscda.h>
#include <petsc/private/daimpl.h>
#include <petsc/private/daensembleimpl.h>
#include <petscblaslapack.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>

static PetscErrorCode PetscDALETKFInstallQ(PetscDA, Mat);

/* Names must match the PetscDALETKFLocalizationType enum order in include/petscda.h. */
const char *const PetscDALETKFLocalizationTypes[] = {"none", "gaspari_cohn", "gaussian", "boxcar", "PetscDALETKFLocalizationType", "PETSCDA_LETKF_LOC_", NULL};

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
  PetscDALETKFBroadcastWeightVector - replicate weight vector w across all columns of w_ones (m x m dense).
  Used only by the LOC_NONE fast path. w lives on PETSC_COMM_SELF (size m); w_ones is a SELF SeqDense m x m.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFBroadcastWeightVector(Vec w, PetscInt m, Mat w_ones)
{
  const PetscScalar *w_array;
  PetscScalar       *mat_array;
  PetscInt           w_size_local, lda;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(w, &w_size_local));
  PetscCall(VecGetArrayRead(w, &w_array));
  PetscCall(MatDenseGetArrayWrite(w_ones, &mat_array));
  PetscCall(MatDenseGetLDA(w_ones, &lda));
  for (PetscInt i = 0; i < m; i++) PetscCall(PetscArraycpy(mat_array + i * lda, w_array, w_size_local));
  PetscCall(MatDenseRestoreArrayWrite(w_ones, &mat_array));
  PetscCall(VecRestoreArrayRead(w, &w_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  UpdateEnsembleWithTransform_LETKF - E = mean*1' + X*G.
  Used only by the LOC_NONE fast path. G is replicated on PETSC_COMM_SELF (every rank holds the
  same m x m), X and the ensemble share the same row distribution; the local rows of E are
  X_local * G + mean_local broadcast across columns. Computed via a per-rank BLASgemm for X*G
  plus a column-broadcast add of mean.
*/
static PetscErrorCode UpdateEnsembleWithTransform_LETKF(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  const PetscScalar *x_array, *g_array, *mean_array;
  PetscScalar       *xg_buf, *ens_array;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscBLASInt       n_local_b, m_b, lda_x_b, lda_g_b;
  PetscInt           n_local_ens, lda_x, lda_g, lda_ens;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(ensemble, &n_local_ens, NULL));
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
  PetscCall(VecDestroy(&impl->r_inv_sqrt));
  PetscCall(MatDestroy(&impl->Z));
  PetscCall(MatDestroy(&impl->S));
  PetscCall(MatDestroy(&impl->T_sqrt));
  PetscCall(MatDestroy(&impl->w_ones));
  PetscCall(MatDestroy(&impl->Q));
  PetscCall(PetscDALETKFClearCoordinates(impl));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
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
  y_mean_local, delta_scaled_local, r_inv_sqrt_local, w_local, s_transpose_delta) are
  created with PETSC_COMM_SELF because the analysis at each vertex is serial and independent.
*/
PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA da, PetscDA_LETKF *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Mat               Z_local, S_local, T_sqrt_local, G_local;
  Vec               y_local, y_mean_local, delta_scaled_local, r_inv_sqrt_local;
  Vec               w_local, s_transpose_delta;
  PetscInt          i_grid_point;
  PetscInt          ndof, max_nnz;
  PetscReal         sqrt_m_minus_1, scale;
  PetscInt          rstart;
  Mat               X_rows, E_analysis_rows;

  PetscFunctionBegin;
  ndof           = da->ndof;
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  scale          = 1.0 / sqrt_m_minus_1;
  max_nnz        = impl->max_nnz_per_row;

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
  PetscCall(VecDuplicate(w_local, &s_transpose_delta));

  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, ndof, m, NULL, &X_rows));
  PetscCall(MatDuplicate(X_rows, MAT_DO_NOT_COPY_VALUES, &E_analysis_rows));

  /* LETKF: Loop over all grid points and perform local analysis */
  PetscCall(MatGetOwnershipRange(impl->Q, &rstart, NULL));

  for (i_grid_point = 0; i_grid_point < n_vertices; i_grid_point++) {
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
    PetscCall(MatMultTranspose(S_local, delta_scaled_local, s_transpose_delta));
    PetscCall(PetscDAEnsembleApplyTInverse(da, s_transpose_delta, w_local));

    /* Compute local square-root transform: T_sqrt_local = T_local^{-1/2} (U is identity, so pass NULL) */
    PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, T_sqrt_local));

    /* Form local transform G_local = w_local * 1' + sqrt(m - 1) * T_sqrt_local * U
       Instead of creating w_ones_local = w_local * 1', we add w_local to each column of G_local */
    PetscCall(MatCopy(T_sqrt_local, G_local, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(G_local, sqrt_m_minus_1));
    {
      const PetscScalar *w_array;
      PetscScalar       *g_array;
      PetscInt           j, k, lda_g;

      PetscCall(VecGetArrayRead(w_local, &w_array));
      PetscCall(MatDenseGetArrayWrite(G_local, &g_array));
      PetscCall(MatDenseGetLDA(G_local, &lda_g));
      for (j = 0; j < m; j++)
        for (k = 0; k < m; k++) g_array[k + j * lda_g] += w_array[k];
      PetscCall(MatDenseRestoreArrayWrite(G_local, &g_array));
      PetscCall(VecRestoreArrayRead(w_local, &w_array));
    }

    /* LETKF Algorithm 2, Line 13: Update ensemble at grid point i_grid_point
       E_a[i,:] = x_bar_f[i] + X_f[i,:] * G_local

       Where:
       - x_bar_f[i] is the forecast mean at grid point i_grid_point (ndof values from global mean vector)
       - X_f[i,:] is the forecast anomaly rows at grid point i_grid_point (ndof rows from global anomaly matrix X)
       - G_local = w_local * 1' + sqrt(m-1) * T_local^{1/2} * U (computed above in G_local)
     */
    {
      const PetscScalar *x_array, *mean_array;
      PetscScalar       *e_array, *x_rows_array, *ea_rows_array;
      PetscInt           j, k, lda_x, lda_e;

      /* Extract ndof rows starting at (i_grid_point * ndof) from X: X_f[i_grid_point*ndof:(i_grid_point+1)*ndof, :] */
      PetscCall(MatDenseGetArrayRead(X, &x_array));
      PetscCall(MatDenseGetArray(X_rows, &x_rows_array));
      PetscCall(MatDenseGetLDA(X, &lda_x));
      for (j = 0; j < m; j++) {
        for (k = 0; k < ndof; k++) x_rows_array[k + j * ndof] = x_array[(i_grid_point * ndof + k) + j * lda_x];
      }
      PetscCall(MatDenseRestoreArray(X_rows, &x_rows_array));
      PetscCall(MatDenseRestoreArrayRead(X, &x_array));

      /* Apply local transform: E_analysis_rows = X_rows * G_local^T */
      PetscCall(MatMatMult(X_rows, G_local, MAT_REUSE_MATRIX, PETSC_DEFAULT, &E_analysis_rows));

      /* Add local mean: E_a[i_grid_point*ndof:(i_grid_point+1)*ndof, :] = x_bar_f[i_grid_point*ndof:(i_grid_point+1)*ndof] + X_f[...] * G_local */
      PetscCall(VecGetArrayRead(impl->mean, &mean_array));
      PetscCall(MatDenseGetArray(E_analysis_rows, &ea_rows_array));
      for (j = 0; j < m; j++) {
        for (k = 0; k < ndof; k++) ea_rows_array[k + j * ndof] += mean_array[i_grid_point * ndof + k];
      }
      PetscCall(MatDenseRestoreArray(E_analysis_rows, &ea_rows_array));
      PetscCall(VecRestoreArrayRead(impl->mean, &mean_array));

      /* Store result back in ensemble[i_grid_point*ndof:(i_grid_point+1)*ndof, :] */
      PetscCall(MatDenseGetArrayWrite(en->ensemble, &e_array));
      PetscCall(MatDenseGetLDA(en->ensemble, &lda_e));
      PetscCall(MatDenseGetArrayRead(E_analysis_rows, (const PetscScalar **)&ea_rows_array));
      for (j = 0; j < m; j++) {
        for (k = 0; k < ndof; k++) e_array[(i_grid_point * ndof + k) + j * lda_e] = ea_rows_array[k + j * ndof];
      }
      PetscCall(MatDenseRestoreArrayRead(E_analysis_rows, (const PetscScalar **)&ea_rows_array));
      PetscCall(MatDenseRestoreArrayWrite(en->ensemble, &e_array));
    }
  }
  PetscCall(MatDestroy(&E_analysis_rows));
  PetscCall(MatDestroy(&X_rows));
  PetscCall(VecDestroy(&s_transpose_delta));
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

static PetscErrorCode PetscDAEnsembleAnalysis_LETKF(PetscDA da, Vec observation, Mat H)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;
  Mat            X;
  PetscInt       m;
  PetscBool      reallocate = PETSC_FALSE;
  PetscReal      sqrt_m_minus_1, scale;

  PetscFunctionBegin;
  m              = impl->en.size;
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  scale          = 1.0 / sqrt_m_minus_1;

  /* Lazily build Q for built-in distance-based kernels using cached coordinates. The dispatcher
     selects host vs Kokkos backend from the type of the cached observation operator. */
  if (impl->type != PETSCDA_LETKF_LOC_NONE && (!impl->Q || impl->Q_dirty)) {
    Mat Q_new = NULL;

    PetscCheck(impl->coord_H, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Coordinates not set; call PetscDALETKFSetLocalizationCoordinates() before analysis.");
    PetscCheck(impl->localization_radius > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Localization radius not set; call PetscDALETKFSetLocalizationRadius() before analysis.");
    PetscCall(PetscDALETKFCreateLocalizationMat(impl->type, impl->localization_radius, impl->coord_xyz, impl->coord_bd, impl->coord_H, &Q_new));
    PetscCall(PetscDALETKFInstallQ(da, Q_new));
    PetscCall(MatDestroy(&Q_new));
    impl->Q_dirty = PETSC_FALSE;
  }

  /* The eigendecomposition of T = I + S^T*S (m x m) requires each vertex to see at least
     m local observations or T is rank-deficient. */
  PetscCheck(impl->type == PETSCDA_LETKF_LOC_NONE || impl->Q, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Localization matrix Q not set. Call PetscDALETKFSetLocalizationCoordinates() first.");
  PetscCheck(impl->type == PETSCDA_LETKF_LOC_NONE || m <= impl->min_nnz_per_row, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Ensemble size (%" PetscInt_FMT ") must be <= minimum local observations per vertex (%" PetscInt_FMT "). Increase localization radius or decrease ensemble size", m,
             impl->min_nnz_per_row);

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

    /* Create S matrix (same layout as Z) */
    PetscCall(MatDuplicate(impl->Z, MAT_DO_NOT_COPY_VALUES, &impl->S));

    /* T_sqrt and w_ones are m x m and used only by the LOC_NONE fast path; allocate them on
       PETSC_COMM_SELF so the eigendecomp and AXPY run replicated on every rank. */
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &impl->T_sqrt));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &impl->w_ones));
  }

  /* Alg 6.4 line 1-2: Compute ensemble mean and scaled anomalies */
  PetscCall(PetscDAEnsembleComputeMean(da, impl->mean));

  /* Create anomaly matrix X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDAEnsembleComputeAnomalies(da, impl->mean, &X));

  /* Alg 6.4 line 3-4: Compute GLOBAL observation ensemble Z = H * E.
     We multiply column-by-column rather than via a single MatMatMult because impl->Z is
     created with the user-controllable `dense_` options prefix (typically MATDENSE), while
     H may be MATAIJKOKKOS. PETSc does not currently support that mixed-type product, so
     each column is staged through (temp_in, temp_out) pairs created from H. The temp pair
     is allocated outside the loop, so the per-column overhead is just two VecCopy. A
     single-call fast path would require either coercing impl->Z to match H's vec type at
     setup or extending Mat product registrations.
     TODO: take the fast path when MatProductSetType succeeds for the (H, ensemble) pair. */
  {
    Vec col_in, col_out, temp_in, temp_out;

    /* Create temporary vectors compatible with H's type */
    PetscCall(MatCreateVecs(H, &temp_in, &temp_out));

    /* Compute Z = H * E column by column to avoid Kokkos vector type issues */
    for (PetscInt j = 0; j < m; j++) {
      PetscCall(MatDenseGetColumnVecRead(impl->en.ensemble, j, &col_in));
      PetscCall(MatDenseGetColumnVecWrite(impl->Z, j, &col_out));

      /* Copy to temp vector, multiply, then copy back */
      PetscCall(VecCopy(col_in, temp_in));
      PetscCall(MatMult(H, temp_in, temp_out));
      PetscCall(VecCopy(temp_out, col_out));

      PetscCall(MatDenseRestoreColumnVecWrite(impl->Z, j, &col_out));
      PetscCall(MatDenseRestoreColumnVecRead(impl->en.ensemble, j, &col_in));
    }

    PetscCall(VecDestroy(&temp_out));
    PetscCall(VecDestroy(&temp_in));
  }

  /* Compute GLOBAL observation mean y_mean = H * x_mean */
  /* Use temporary vector compatible with H's type */
  {
    Vec temp_mean, temp_y_mean;
    PetscCall(MatCreateVecs(H, &temp_mean, &temp_y_mean));
    PetscCall(VecCopy(impl->mean, temp_mean));
    PetscCall(MatMult(H, temp_mean, temp_y_mean));
    PetscCall(VecCopy(temp_y_mean, impl->y_mean));
    PetscCall(VecDestroy(&temp_y_mean));
    PetscCall(VecDestroy(&temp_mean));
  }

  /* Compute GLOBAL R^{-1/2} (assumes diagonal R) */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  /* NONE fast-path: no localization -> single global ETKF analysis.
     The m x m T factor and weight vector live on PETSC_COMM_SELF so every rank does the
     identical eigendecomp; only the gram S^T*S and the projection S^T*delta need an MPI
     reduction. */
  if (impl->type == PETSCDA_LETKF_LOC_NONE) {
    Vec                s_transpose_delta = NULL;
    const PetscScalar *s_array, *d_array;
    PetscScalar       *gram = NULL;
    PetscScalar       *buf;
    PetscScalar        one = 1.0, zero = 0.0;
    PetscBLASInt       mB, n_obs_localB, s_ldaB, ione = 1;
    PetscMPIInt        mmB, mMPI;
    PetscInt           n_obs_local, s_lda;
    PetscBool          use_kokkos = PETSC_FALSE;

#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
    if (da->R) PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &use_kokkos, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
    if (use_kokkos) {
      if (!impl->w) PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &impl->w));
      PetscCall(PetscDALETKFGlobalAnalysis_Kokkos(da, impl, m, X, observation));
      goto cleanup;
    }
#else
    (void)use_kokkos;
#endif

    /* w (size m) is allocated lazily because the per-vertex path doesn't need it. */
    if (!impl->w) PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &impl->w));

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
    PetscCall(PetscBLASIntCast(m, &mB));
    PetscCall(PetscBLASIntCast(n_obs_local, &n_obs_localB));
    PetscCall(PetscBLASIntCast(s_lda, &s_ldaB));
    if (n_obs_local > 0) PetscCallBLAS("BLASgemm", BLASgemm_("T", "N", &mB, &mB, &n_obs_localB, &one, s_array, &s_ldaB, s_array, &s_ldaB, &zero, gram, &mB));
    PetscCall(PetscMPIIntCast((PetscInt64)m * m, &mmB));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, gram, mmB, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)da)));
    PetscCall(PetscDAEnsembleTFactorFromGram(da, m, gram));
    PetscCall(PetscFree(gram));

    /* w = T^{-1} * (S^T * delta_scaled), with the projection reduced across ranks. */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &s_transpose_delta));
    PetscCall(VecGetArrayRead(impl->delta_scaled, &d_array));
    PetscCall(VecGetArrayWrite(s_transpose_delta, &buf));
    PetscCall(PetscArrayzero(buf, m));
    if (n_obs_local > 0) PetscCallBLAS("BLASgemv", BLASgemv_("T", &n_obs_localB, &mB, &one, s_array, &s_ldaB, d_array, &ione, &zero, buf, &ione));
    PetscCall(VecRestoreArrayWrite(s_transpose_delta, &buf));
    PetscCall(VecRestoreArrayRead(impl->delta_scaled, &d_array));
    PetscCall(MatDenseRestoreArrayRead(impl->S, &s_array));

    PetscCall(VecGetArray(s_transpose_delta, &buf));
    PetscCall(PetscMPIIntCast(m, &mMPI));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, buf, mMPI, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)da)));
    PetscCall(VecRestoreArray(s_transpose_delta, &buf));

    PetscCall(PetscDAEnsembleApplyTInverse(da, s_transpose_delta, impl->w));
    PetscCall(VecDestroy(&s_transpose_delta));

    /* T_sqrt = T^{-1/2} */
    PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, impl->T_sqrt));

    /* G = w*1' + sqrt(m-1) * T_sqrt (in impl->w_ones, all on PETSC_COMM_SELF). */
    PetscCall(PetscDALETKFBroadcastWeightVector(impl->w, m, impl->w_ones));
    PetscCall(MatAXPY(impl->w_ones, sqrt_m_minus_1, impl->T_sqrt, SAME_NONZERO_PATTERN));

    /* E = mean*1' + X * G */
    PetscCall(UpdateEnsembleWithTransform_LETKF(impl->mean, X, impl->w_ones, m, impl->en.ensemble));
  } else {
    PetscInt  n_local, n_obs_local;
    PetscBool use_kokkos = PETSC_FALSE;

    /* Per-vertex local analysis path.
       PetscDALETKFInstallQ() always builds the obs-scatter, so impl->obs_scat is non-NULL whenever
       impl->Q exists. Backend-agnostic: scatter global obs-space data into per-rank work vectors. */
    PetscCall(VecScatterBegin(impl->obs_scat, observation, impl->obs_work, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(impl->obs_scat, observation, impl->obs_work, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecScatterBegin(impl->obs_scat, impl->y_mean, impl->y_mean_work, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(impl->obs_scat, impl->y_mean, impl->y_mean_work, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecScatterBegin(impl->obs_scat, impl->r_inv_sqrt, impl->r_inv_sqrt_work, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(impl->obs_scat, impl->r_inv_sqrt, impl->r_inv_sqrt_work, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetLocalSize(impl->obs_work, &n_obs_local));
    if (!impl->Z_work) {
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n_obs_local, m, NULL, &impl->Z_work));
    } else {
      PetscInt m_old, n_old;
      PetscCall(MatGetSize(impl->Z_work, &n_old, &m_old));
      if (m_old != m || n_old != n_obs_local) {
        PetscCall(MatDestroy(&impl->Z_work));
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n_obs_local, m, NULL, &impl->Z_work));
      }
    }
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
    if (da->R) PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &use_kokkos, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
    if (use_kokkos) PetscCall(PetscDALETKFLocalAnalysis_Kokkos(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
    else PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
#else
    (void)use_kokkos;
    PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
#endif
  }

#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
cleanup:
#endif
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalizationRadius_LETKF(PetscDA da, PetscReal radius)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  PetscCheck(radius > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Localization radius must be positive, got %g", (double)radius);
  impl->localization_radius = radius;
  impl->Q_dirty             = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalizationType_LETKF(PetscDA da, PetscDALETKFLocalizationType type)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  PetscCheck(type >= 0 && type < PETSCDA_LETKF_LOC_NUM_TYPES, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid localization type %d; must be in [0,%d)", (int)type, (int)PETSCDA_LETKF_LOC_NUM_TYPES);
  if (impl->type != type) {
    impl->type    = type;
    impl->Q_dirty = PETSC_TRUE;
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
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  PetscAssertPointer(xyz, 2);
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
  impl->Q_dirty = PETSC_TRUE;
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
*/
static PetscErrorCode PetscDALETKFInstallQ(PetscDA da, Mat Q)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;
  PetscInt       nrows, ncols, rstart, rend, nnz, mm[2];

  PetscFunctionBegin;
  PetscCall(MatGetSize(Q, &nrows, &ncols));
  PetscCheck(nrows == da->state_size / da->ndof, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix rows (%" PetscInt_FMT ") must equal vertex count state_size/ndof (%" PetscInt_FMT ")", nrows, da->state_size / da->ndof);
  PetscCheck(ncols == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix columns (%" PetscInt_FMT ") must match observation size (%" PetscInt_FMT ")", ncols, da->obs_size);
  /* The obs-scatter is needed by both the CPU and Kokkos per-vertex paths whenever Q exists.
     Validate before tearing down the previous Q/obs-scatter so a missing prerequisite leaves
     the impl in its prior usable state instead of a half-installed one. */
  PetscCheck(impl->coord_H, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDALETKFSetLocalizationCoordinates() must be called before the first analysis so the obs-scatter has H available");

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  if (impl->Q) PetscCall(PetscDALETKFDestroyLocalization_Kokkos(impl));
#endif
  /* Destroy the previous obs-scatter so SetupObsScatter() can rebuild it for the new Q footprint. */
  PetscCall(PetscDALETKFDestroyObsScatter(impl));

  PetscCall(MatDestroy(&impl->Q));
  PetscCall(PetscObjectReference((PetscObject)Q));
  impl->Q = Q;

  impl->max_nnz_per_row = 0;
  impl->min_nnz_per_row = PETSC_INT_MAX;
  PetscCall(MatGetOwnershipRange(Q, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(Q, i, &nnz, NULL, NULL));
    if (nnz > impl->max_nnz_per_row) impl->max_nnz_per_row = nnz;
    if (nnz < impl->min_nnz_per_row) impl->min_nnz_per_row = nnz;
    PetscCall(MatRestoreRow(Q, i, &nnz, NULL, NULL));
  }
  /* Coalesce max and min into a single MAX reduction by negating min; cuts the allreduce
     latency in half. The sentinel PETSC_INT_MAX round-trips correctly through negation. */
  mm[0] = impl->max_nnz_per_row;
  mm[1] = -impl->min_nnz_per_row;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, mm, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)da)));
  impl->max_nnz_per_row = mm[0];
  impl->min_nnz_per_row = -mm[1];
  /* If every rank owned zero rows the MIN reduction returns the sentinel; clamp to 0 so the
     value displayed by the viewer and consumed by downstream checks is meaningful. */
  if (impl->min_nnz_per_row == PETSC_INT_MAX) impl->min_nnz_per_row = 0;

  PetscCall(PetscDALETKFSetupObsScatter(impl, impl->coord_H));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscDALETKFSetupLocalization_Kokkos(impl));
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
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (da->R) PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &is_kokkos, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: %s\n", is_kokkos ? "Kokkos" : "CPU"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization type: %s\n", PetscDALETKFLocalizationTypes[impl->type]));
    if (impl->type != PETSCDA_LETKF_LOC_NONE) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization radius: %g\n", (double)impl->localization_radius));
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
  PetscInt         type_idx;
  PetscBool        type_set = PETSC_FALSE, radius_set = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscDASetFromOptions_Ensemble(da, PetscOptionsObjectPtr));
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscDA LETKF Options");
  PetscCall(PetscOptionsInt("-petscda_letkf_batch_size", "Batch size for GPU processing", "", impl->batch_size, &impl->batch_size, NULL));
  radius = impl->localization_radius;
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
+  -petscda_type letkf                                                  - set the `PetscDAType` to `PETSCDALETKF`
.  -petscda_ensemble_size size                                          - number of ensemble members
.  -petscda_letkf_batch_size batch_size                                 - set the batch size for GPU processing
.  -petscda_letkf_localization_radius radius                            - localization cutoff radius for the built-in kernels (must be positive)
-  -petscda_letkf_localization_type (none|gaspari_cohn|gaussian|boxcar) - select the localization kernel

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
          `PetscDALETKFSetLocalizationCoordinates()`, `PetscDAEnsembleSetSize()`, `PetscDASetSizes()`, `PetscDAEnsembleSetInflation()`,
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
  impl->Q_dirty             = PETSC_FALSE;
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

  Level: intermediate

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
  cached `Q` will not be rebuilt automatically; call `PetscDALETKFSetLocalizationCoordinates()`
  again to invalidate `Q` and force a rebuild on the next analysis.

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetLocalizationType()`,
          `PetscDALETKFSetLocalizationRadius()`
@*/
PetscErrorCode PetscDALETKFSetLocalizationCoordinates(PetscDA da, const Vec xyz[3], const PetscReal bd[3], Mat H)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(xyz, 2);
  PetscCheck(xyz[0], PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "xyz[0] must be a valid Vec; the spatial dimension is taken to be the index of the first NULL slot in xyz[3]");
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscTryMethod(da, "PetscDALETKFSetLocalizationCoordinates_C", (PetscDA, const Vec[3], const PetscReal[3], Mat), (da, xyz, bd, H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

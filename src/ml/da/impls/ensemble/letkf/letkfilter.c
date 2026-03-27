#include <petscda.h>
#include <petsc/private/daimpl.h>
#include <petsc/private/daensembleimpl.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>

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
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscDALETKFDestroyLocalization_Kokkos(impl));
#endif
  PetscCall(ISDestroy(&impl->obs_is_local));
  PetscCall(VecScatterDestroy(&impl->obs_scat));
  PetscCall(VecDestroy(&impl->obs_work));
  PetscCall(VecDestroy(&impl->y_mean_work));
  PetscCall(VecDestroy(&impl->r_inv_sqrt_work));
  PetscCall(MatDestroy(&impl->Z_work));
  PetscCall(PetscDADestroy_Ensemble(da));
  PetscCall(PetscFree(da->data));

  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalization_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetObsPerVertex_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFGetObsPerVertex_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ExtractLocalObservations - Extracts local observations for a vertex using localization matrix Q (CPU version)

  Input Parameters:
+ Q          - localization matrix (state_size/ndof x obs_size), each row has constant non-zeros
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
  PetscInt           ncols, k, j;
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

  /* Assemble local matrices/vectors */
  PetscCall(MatAssemblyBegin(Z_local, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Z_local, MAT_FINAL_ASSEMBLY));
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
  PetscInt          ndof;
  PetscReal         sqrt_m_minus_1, scale;
  PetscInt          rstart;
  Mat               X_rows, E_analysis_rows;

  PetscFunctionBegin;
  ndof           = da->ndof;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  /* Create local analysis workspace (n_obs_vertex x m matrices and vectors) */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->n_obs_vertex, m, NULL, &Z_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)Z_local, "dense_"));
  PetscCall(MatSetFromOptions(Z_local));
  PetscCall(MatSetUp(Z_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->n_obs_vertex, m, NULL, &S_local));
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

  /* Create vectors using MatCreateVecs from Z_local (n_obs_vertex x m) */
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
    /* Extract local observations for this grid point using Q[i_grid_point,:] */
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

  PetscFunctionBegin;
  m = impl->en.size;

  /* Check if localization matrix Q is set */
  PetscCheck(impl->Q, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Localization matrix Q not set. Call PetscDALETKFSetLocalization() first.");

  /* Warn if Cholesky sqrt type is used with LETKF - it produces an asymmetric
     T^{-1/2} = L^{-T} which is incorrect for the local perturbation update.
     LETKF requires the symmetric square root T^{-1/2} = V * D^{-1/2} * V^T. */
  PetscCheck(impl->en.sqrt_type != PETSCDA_SQRT_CHOLESKY, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cholesky sqrt type produces asymmetric T^{-1/2}, which is incorrect for LETKF. Use -petscda_ensemble_sqrt_type eigen or PetscDAEnsembleSetSqrtType(da, PETSCDA_SQRT_EIGEN) instead.");

  /* Check that ensemble size <= number of local observations per vertex.
     The eigen decomposition of T = I + S^T*S (m x m) requires that the
     local observation count p >= m; otherwise T is rank-deficient and the
     decomposition is ill-posed. */
  PetscCheck(m <= impl->n_obs_vertex, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Ensemble size (%" PetscInt_FMT ") must be <= number of local observations per vertex (%" PetscInt_FMT ") for LETKF eigen decomposition to be well-posed", m,
             impl->n_obs_vertex);

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

  /* Create anomaly matrix X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDAEnsembleComputeAnomalies(da, impl->mean, &X));

  /* Alg 6.4 line 3-4: Compute GLOBAL observation ensemble Z = H * E */
  /* Note: When H is a Kokkos matrix type (e.g., aijkokkos), MatMatMult may fail
     with non-Kokkos dense matrices. Use column-by-column multiplication with
     temporary vectors that are compatible with H's type. */
  {
    Vec      col_in, col_out, temp_in, temp_out;
    PetscInt j;

    /* Create temporary vectors compatible with H's type */
    PetscCall(MatCreateVecs(H, &temp_in, &temp_out));

    /* Compute Z = H * E column by column to avoid Kokkos vector type issues */
    for (j = 0; j < m; j++) {
      PetscCall(MatDenseGetColumnVecRead(impl->en.ensemble, j, &col_in));
      PetscCall(MatDenseGetColumnVecWrite(impl->Z, j, &col_out));

      /* Copy to temp vector, multiply, then copy back */
      PetscCall(VecCopy(col_in, temp_in));
      PetscCall(MatMult(H, temp_in, temp_out));
      PetscCall(VecCopy(temp_out, col_out));

      PetscCall(MatDenseRestoreColumnVecWrite(impl->Z, j, &col_out));
      PetscCall(MatDenseRestoreColumnVecRead(impl->en.ensemble, j, &col_in));
    }
    PetscCall(MatAssemblyBegin(impl->Z, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(impl->Z, MAT_FINAL_ASSEMBLY));

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
  /* Perform local analysis for all vertices */

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  /* Use GPU version only if:
     1. sqrt_type is eigen (GPU version only implements eigen/SVD, not cholesky)
     2. H matrix is a Kokkos type (aijkokkos) */
  {
    PetscBool use_gpu = PETSC_FALSE;
    if (impl->en.sqrt_type == PETSCDA_SQRT_EIGEN) {
  #if !defined(PETSC_USE_COMPLEX)
      /* Check if H matrix is a Kokkos type */
      PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &use_gpu, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
  #endif
    }

    /* Scatter global vectors to local work vectors if available */
    if (impl->obs_scat) {
      PetscCall(VecScatterBegin(impl->obs_scat, observation, impl->obs_work, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(impl->obs_scat, observation, impl->obs_work, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecScatterBegin(impl->obs_scat, impl->y_mean, impl->y_mean_work, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(impl->obs_scat, impl->y_mean, impl->y_mean_work, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecScatterBegin(impl->obs_scat, impl->r_inv_sqrt, impl->r_inv_sqrt_work, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(impl->obs_scat, impl->r_inv_sqrt, impl->r_inv_sqrt_work, INSERT_VALUES, SCATTER_FORWARD));

      /* Handle Z matrix (scatter columns) */
      {
        PetscInt n_obs_local;
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
      }
    }

    if (use_gpu) {
      PetscInt n_local;
      PetscCall(MatGetLocalSize(impl->Q, &n_local, NULL));
      /* Use local work vectors for GPU analysis */
      PetscCall(PetscDALETKFLocalAnalysis_GPU(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
    } else {
      PetscInt n_local;
      PetscCall(MatGetLocalSize(impl->Q, &n_local, NULL));
      if (impl->obs_scat) {
        PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, impl->obs_work, impl->Z_work, impl->y_mean_work, impl->r_inv_sqrt_work));
      } else {
        PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, observation, impl->Z, impl->y_mean, impl->r_inv_sqrt));
      }
    }
  }
#else
  /* Without Kokkos, use CPU version */
  {
    PetscInt n_local;
    PetscCall(MatGetLocalSize(impl->Q, &n_local, NULL));
    PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_local, X, observation, impl->Z, impl->y_mean, impl->r_inv_sqrt));
  }
#endif
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetObsPerVertex_LETKF(PetscDA da, PetscInt n_obs_vertex)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  impl->n_obs_vertex = n_obs_vertex;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFGetObsPerVertex_LETKF(PetscDA da, PetscInt *n_obs_vertex)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");
  *n_obs_vertex = impl->n_obs_vertex;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalization_LETKF(PetscDA da, Mat Q, Mat H)
{
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;
  PetscInt       i, nrows, ncols, nnz, rstart, rend;

  PetscFunctionBegin;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");

  /* Get matrix dimensions */
  PetscCall(MatGetSize(Q, &nrows, &ncols));

  /* Validate matrix dimensions */
  PetscCheck(nrows == da->state_size / da->ndof, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix rows (%" PetscInt_FMT ") must match state size (%" PetscInt_FMT ")", nrows, da->state_size);
  PetscCheck(ncols == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix columns (%" PetscInt_FMT ") must match observation size (%" PetscInt_FMT ")", ncols, da->obs_size);

  /* Validate that each row has const non-zero entries */
  PetscCall(MatGetOwnershipRange(Q, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(Q, i, &nnz, &cols, &vals));
    PetscCheck(nnz == impl->n_obs_vertex, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Row %" PetscInt_FMT " has %" PetscInt_FMT " non-zeros, expected %" PetscInt_FMT, i, nnz, (PetscInt)impl->n_obs_vertex);
    PetscCall(MatRestoreRow(Q, i, &nnz, &cols, &vals));
  }

  /* Store the localization matrix */
  PetscCall(MatDestroy(&impl->Q));
  PetscCall(PetscObjectReference((PetscObject)Q));
  impl->Q = Q;
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscDALETKFSetupLocalization_Kokkos(impl, H));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAView_LETKF(PetscDA da, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscDA_LETKF *impl = (PetscDA_LETKF *)da->data;

  PetscFunctionBegin;
  PetscCall(PetscDAView_Ensemble(da, viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (impl->en.sqrt_type == PETSCDA_SQRT_CHOLESKY) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU\n"));
    } else {
      /* Check if R matrix is Kokkos type to determine if GPU will be used */
      if (da->R) {
        PetscBool is_kokkos = PETSC_FALSE;
        PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &is_kokkos, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
        if (is_kokkos) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: Kokkos\n"));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU\n"));
        }
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU or Kokkos (depending on covariance matrix type)\n"));
      }
    }
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU\n"));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local observations per vertex: %" PetscInt_FMT "\n", impl->n_obs_vertex));
    if (impl->batch_size > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  GPU batch size: %" PetscInt_FMT "\n", impl->batch_size));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  GPU batch size: auto\n"));
    }
    if (impl->Q) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: set\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: not set\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_LETKF(PetscDA da, PetscOptionItems *PetscOptionsObjectPtr)
{
  PetscDA_LETKF   *impl               = (PetscDA_LETKF *)da->data;
  PetscOptionItems PetscOptionsObject = *PetscOptionsObjectPtr;

  PetscFunctionBegin;
  PetscCall(PetscDASetFromOptions_Ensemble(da, PetscOptionsObjectPtr));
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscDA LETKF Options");
  PetscCall(PetscOptionsInt("-petscda_letkf_batch_size", "Batch size for GPU processing", "", impl->batch_size, &impl->batch_size, NULL));
  PetscCall(PetscOptionsInt("-petscda_letkf_obs_per_vertex", "Number of local observations per vertex", "", impl->n_obs_vertex, &impl->n_obs_vertex, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCDALETKF - The Local ETKF performs the analysis update locally around each grid point, enabling scalable assimilation on large
   domains by avoiding the global ensemble covariance matrix.

   Options Database Keys:
+  -petscda_type letkf                           - set the `PetscDAType` to `PETSCDALETKF`
.  -petscda_ensemble_size <size>                 - number of ensemble members
.  -petscda_ensemble_sqrt_type <cholesky, eigen> - the square root of the matrix to use
.  -petscda_letkf_batch_size <batch_size>        - set the batch size for GPU processing
-  -petscda_letkf_obs_per_vertex <n_obs_vertex>  - number of observations per vertex

   Level: beginner

.seealso: [](ch_da), `PetscDA`, `PetscDACreate()`, `PETSCDAETKF`, `PetscDALETKFSetObsPerVertex()`, `PetscDALETKFGetObsPerVertex()`,
          `PetscDALETKFSetLocalization()`, `PetscDAEnsembleSetSize()`, `PetscDASetSizes()`, `PetscDAEnsembleSetSqrtType()`, `PetscDAEnsembleSetInflation()`,
          `PetscDAEnsembleComputeMean()`, `PetscDAEnsembleComputeAnomalies()`, `PetscDAEnsembleAnalysis()`, `PetscDAEnsembleForecast()`
M*/

PETSC_INTERN PetscErrorCode PetscDACreate_LETKF(PetscDA da)
{
  PetscDA_LETKF *impl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&impl));
  da->data = impl;
  PetscCall(PetscDACreate_Ensemble(da));
  da->ops->destroy        = PetscDADestroy_LETKF;
  da->ops->view           = PetscDAView_LETKF;
  da->ops->setfromoptions = PetscDASetFromOptions_LETKF;
  impl->en.analysis       = PetscDAEnsembleAnalysis_LETKF;
  impl->en.forecast       = PetscDAEnsembleForecast_Ensemble;

  impl->n_obs_vertex = 9;
  impl->Q            = NULL;
  impl->batch_size   = 0;

  /* Register the method for setting localization */
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalization_C", PetscDALETKFSetLocalization_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetObsPerVertex_C", PetscDALETKFSetObsPerVertex_LETKF));
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFGetObsPerVertex_C", PetscDALETKFGetObsPerVertex_LETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFSetObsPerVertex - Sets the number of local observations per vertex for the LETKF algorithm.

  Logically Collective

  Input Parameters:
+ da           - the `PetscDA` context
- n_obs_vertex - number of observations per vertex

  Level: advanced

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetLocalization()`
@*/
PetscErrorCode PetscDALETKFSetObsPerVertex(PetscDA da, PetscInt n_obs_vertex)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, n_obs_vertex, 2);
  PetscTryMethod(da, "PetscDALETKFSetObsPerVertex_C", (PetscDA, PetscInt), (da, n_obs_vertex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFGetObsPerVertex - Gets the number of local observations per vertex for the LETKF algorithm.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. n_obs_vertex - number of observations per vertex

  Level: advanced

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`, `PetscDALETKFSetObsPerVertex()`
@*/
PetscErrorCode PetscDALETKFGetObsPerVertex(PetscDA da, PetscInt *n_obs_vertex)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(n_obs_vertex, 2);
  PetscUseMethod(da, "PetscDALETKFGetObsPerVertex_C", (PetscDA, PetscInt *), (da, n_obs_vertex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFSetLocalization - Sets the localization matrix for the LETKF algorithm.

  Collective

  Input Parameters:
+ da - the `PetscDA` context
. Q  - the localization matrix (N x P)
- H  - the observation operator matrix (P x N)

  Level: advanced

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDA`
@*/
PetscErrorCode PetscDALETKFSetLocalization(PetscDA da, Mat Q, Mat H)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(Q, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 3);
  PetscTryMethod(da, "PetscDALETKFSetLocalization_C", (PetscDA, Mat, Mat), (da, Q, H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

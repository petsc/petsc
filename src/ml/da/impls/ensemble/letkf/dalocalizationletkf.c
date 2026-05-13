#include <petsc.h>
#include <petscmat.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf_kernels.h>

/* Bbox layout is `[min_0..min_{dim-1}, max_0..max_{dim-1}]`, i.e. all mins then all maxes. */
static inline PetscBool PetscDALETKFCoordInBbox(PetscInt dim, const PetscReal *coord, const PetscReal *bbox)
{
  for (PetscInt d = 0; d < dim; ++d)
    if (coord[d] < bbox[d] || coord[d] > bbox[dim + d]) return PETSC_FALSE;
  return PETSC_TRUE;
}

/*
  PetscDALETKFGatherObsBbox - Bounding-box-pruned redistribution of observation coordinates.

  Each rank obtains the global indices and coordinates of just those observations whose location can
  fall within `cutoff` of any vertex it owns. Replaces an earlier all-gather-to-all of the full obs
  coordinate set, which scaled as O(n_obs_global) per rank in both memory and bandwidth.

  For each non-periodic dimension d, the local-vertex bbox is `[vmin_d - cutoff, vmax_d + cutoff]`.
  Bboxes are exchanged with `MPI_Allgather`; each rank then scans its locally-owned obs (in `H`'s row
  distribution) and forwards every obs into each rank whose padded bbox contains it. Periodic
  dimensions (`bd[d] > 0`) are passed unfiltered because wrap-around defeats a scalar bbox test;
  the exact distance gate inside the Q-construction kernels handles those dims.

  Output buffers `*obs_idx_out` (global obs indices, length `*n_obs_filt`) and `*obs_coords_out`
  (flattened `[*n_obs_filt][dim]` row-major) are allocated with `PetscMalloc1()` and must be freed
  by the caller. The order of records is whatever Alltoallv produces - column ordering inside Q is
  re-sorted by `MatSetValues()`, so callers may use it directly.

  The per-row distance check inside the AIJ and Kokkos Q-construction kernels still gates on weight
  > 0, so the resulting Q is identical to the all-gather variant - this routine only trims the
  candidate set passed in.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFGatherObsBbox(PetscInt dim, Vec xyz[], PetscReal bd[], PetscReal cutoff, Mat H, Vec obs_vecs[], PetscInt *n_obs_filt, PetscInt **obs_idx_out, PetscReal **obs_coords_out)
{
  MPI_Comm           comm;
  PetscMPIInt        size, two_dim_mpi;
  PetscInt           n_vert_local, obs_rstart, obs_rend, n_obs_local;
  PetscInt           n_periodic = 0;
  PetscInt           total_send = 0, total_recv = 0;
  PetscReal          local_bbox[2 * 3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; /* layout: [vmin_0..vmin_{d-1}, vmax_0..vmax_{d-1}] */
  PetscReal         *all_bboxes        = NULL;
  PetscInt          *send_counts = NULL, *send_displs = NULL, *recv_counts = NULL, *recv_displs = NULL;
  PetscInt          *send_idx = NULL, *recv_idx = NULL, *pos = NULL;
  PetscReal         *send_crd = NULL, *recv_crd = NULL;
  PetscMPIInt       *send_counts_mpi = NULL, *send_displs_mpi = NULL, *recv_counts_mpi = NULL, *recv_displs_mpi = NULL;
  PetscMPIInt       *send_counts_crd = NULL, *send_displs_crd = NULL, *recv_counts_crd = NULL, *recv_displs_crd = NULL;
  const PetscScalar *obs_arr[3] = {NULL, NULL, NULL};

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCheck(dim >= 1 && dim <= 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "Spatial dimension must be in [1, 3]; got %" PetscInt_FMT " (local_bbox is sized for at most 3 dims)", dim);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscMPIIntCast(2 * dim, &two_dim_mpi));
  PetscCall(VecGetLocalSize(xyz[0], &n_vert_local));
  PetscCall(MatGetOwnershipRange(H, &obs_rstart, &obs_rend));
  n_obs_local = obs_rend - obs_rstart;

  /* Single-rank fast path: no exchange needed, and MPI-uni does not implement Alltoallv. */
  if (size == 1) {
    PetscInt  *idx_out;
    PetscReal *crd_out;

    PetscCall(PetscMalloc1(n_obs_local, &idx_out));
    PetscCall(PetscMalloc1((size_t)n_obs_local * dim, &crd_out));
    for (PetscInt d = 0; d < dim; ++d) PetscCall(VecGetArrayRead(obs_vecs[d], &obs_arr[d]));
    for (PetscInt k = 0; k < n_obs_local; ++k) {
      idx_out[k] = obs_rstart + k;
      for (PetscInt d = 0; d < dim; ++d) crd_out[(size_t)k * dim + d] = PetscRealPart(obs_arr[d][k]);
    }
    for (PetscInt d = 0; d < dim; ++d) PetscCall(VecRestoreArrayRead(obs_vecs[d], &obs_arr[d]));
    *n_obs_filt     = n_obs_local;
    *obs_idx_out    = idx_out;
    *obs_coords_out = crd_out;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Step 1: per-dim local-vertex bbox padded by cutoff. Periodic dims pass through (-/+ MAX_REAL).
     A rank with no vertices uses (+MAX, -MAX) so its bbox excludes every obs. When every dim is
     periodic the prune is a no-op and the gather degenerates to all-to-all; emit a PetscInfo
     under `-info` so this is visible without surprising serial users. Fires on every Q rebuild
     (Q is built lazily and re-built when the setters invalidate it). */
  for (PetscInt d = 0; d < dim; ++d)
    if (bd[d] > 0.0) n_periodic++;
  /* size == 1 already returned above, so the prune-is-a-no-op notice always applies to a real multi-rank gather. */
  if (n_periodic == dim) PetscCall(PetscInfo((PetscObject)H, "All %" PetscInt_FMT " dim(s) periodic; bbox prune passes through, obs gather degenerates to all-to-all\n", dim));
  for (PetscInt d = 0; d < dim; ++d) {
    if (bd[d] > 0.0) {
      local_bbox[d]       = -PETSC_MAX_REAL;
      local_bbox[dim + d] = PETSC_MAX_REAL;
    } else if (n_vert_local == 0) {
      local_bbox[d]       = PETSC_MAX_REAL;
      local_bbox[dim + d] = -PETSC_MAX_REAL;
    } else {
      const PetscScalar *arr;
      PetscReal          vmin = PETSC_MAX_REAL, vmax = -PETSC_MAX_REAL;

      PetscCall(VecGetArrayRead(xyz[d], &arr));
      for (PetscInt i = 0; i < n_vert_local; ++i) {
        PetscReal x = PetscRealPart(arr[i]);
        if (x < vmin) vmin = x;
        if (x > vmax) vmax = x;
      }
      PetscCall(VecRestoreArrayRead(xyz[d], &arr));
      local_bbox[d]       = vmin - cutoff;
      local_bbox[dim + d] = vmax + cutoff;
    }
  }

  /* Step 2: Allgather bboxes. Layout per rank: [vmin_0..vmin_{d-1}, vmax_0..vmax_{d-1}]. */
  PetscCall(PetscMalloc1((size_t)size * 2 * dim, &all_bboxes));
  PetscCallMPI(MPI_Allgather(local_bbox, two_dim_mpi, MPIU_REAL, all_bboxes, two_dim_mpi, MPIU_REAL, comm));

  /* Step 3: Count records per dest rank. The Get/Restore pair is opened only over the count loop so
     a malloc failure later cannot leave obs_vecs locked. */
  PetscCall(PetscMalloc2(size, &send_counts, size + 1, &send_displs));
  PetscCall(PetscArrayzero(send_counts, size));
  for (PetscInt d = 0; d < dim; ++d) PetscCall(VecGetArrayRead(obs_vecs[d], &obs_arr[d]));
  for (PetscInt k = 0; k < n_obs_local; ++k) {
    PetscReal coord_k[3] = {0.0, 0.0, 0.0};

    for (PetscInt d = 0; d < dim; ++d) coord_k[d] = PetscRealPart(obs_arr[d][k]);
    for (PetscMPIInt r = 0; r < size; ++r) {
      if (PetscDALETKFCoordInBbox(dim, coord_k, &all_bboxes[(size_t)r * 2 * dim])) send_counts[r]++;
    }
  }
  for (PetscInt d = 0; d < dim; ++d) PetscCall(VecRestoreArrayRead(obs_vecs[d], &obs_arr[d]));

  send_displs[0] = 0;
  for (PetscMPIInt r = 0; r < size; ++r) send_displs[r + 1] = send_displs[r] + send_counts[r];
  total_send = send_displs[size];

  /* Step 4: Pack send buffers (re-acquire obs_vecs only for the duration of the pack). */
  PetscCall(PetscMalloc1(total_send, &send_idx));
  PetscCall(PetscMalloc1((size_t)total_send * dim, &send_crd));
  PetscCall(PetscMalloc1(size, &pos));
  for (PetscMPIInt r = 0; r < size; ++r) pos[r] = send_displs[r];
  for (PetscInt d = 0; d < dim; ++d) PetscCall(VecGetArrayRead(obs_vecs[d], &obs_arr[d]));
  for (PetscInt k = 0; k < n_obs_local; ++k) {
    PetscReal coord_k[3] = {0.0, 0.0, 0.0};

    for (PetscInt d = 0; d < dim; ++d) coord_k[d] = PetscRealPart(obs_arr[d][k]);
    for (PetscMPIInt r = 0; r < size; ++r) {
      if (PetscDALETKFCoordInBbox(dim, coord_k, &all_bboxes[(size_t)r * 2 * dim])) {
        PetscInt p  = pos[r]++;
        send_idx[p] = obs_rstart + k;
        for (PetscInt d = 0; d < dim; ++d) send_crd[(size_t)p * dim + d] = coord_k[d];
      }
    }
  }
  for (PetscInt d = 0; d < dim; ++d) PetscCall(VecRestoreArrayRead(obs_vecs[d], &obs_arr[d]));
  PetscCall(PetscFree(pos));

  /* Step 5: Exchange counts, then index and coord payloads. */
  PetscCall(PetscMalloc2(size, &recv_counts, size + 1, &recv_displs));
  PetscCall(PetscMalloc4(size, &send_counts_mpi, size, &send_displs_mpi, size, &recv_counts_mpi, size, &recv_displs_mpi));
  for (PetscMPIInt r = 0; r < size; ++r) {
    PetscCall(PetscMPIIntCast(send_counts[r], &send_counts_mpi[r]));
    PetscCall(PetscMPIIntCast(send_displs[r], &send_displs_mpi[r]));
  }
  PetscCallMPI(MPI_Alltoall(send_counts_mpi, 1, MPI_INT, recv_counts_mpi, 1, MPI_INT, comm));
  recv_displs_mpi[0] = 0;
  for (PetscMPIInt r = 1; r < size; ++r) PetscCall(PetscMPIIntCast((PetscInt64)recv_displs_mpi[r - 1] + (PetscInt64)recv_counts_mpi[r - 1], &recv_displs_mpi[r]));
  recv_displs[0] = 0;
  for (PetscMPIInt r = 0; r < size; ++r) {
    recv_counts[r]     = recv_counts_mpi[r];
    recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
  }
  total_recv = recv_displs[size];

  PetscCall(PetscMalloc1(total_recv, &recv_idx));
  PetscCall(PetscMalloc1((size_t)total_recv * dim, &recv_crd));

  PetscCallMPI(MPI_Alltoallv(send_idx, send_counts_mpi, send_displs_mpi, MPIU_INT, recv_idx, recv_counts_mpi, recv_displs_mpi, MPIU_INT, comm));

  PetscCall(PetscMalloc4(size, &send_counts_crd, size, &send_displs_crd, size, &recv_counts_crd, size, &recv_displs_crd));
  for (PetscMPIInt r = 0; r < size; ++r) {
    PetscCall(PetscMPIIntCast((PetscInt64)send_counts_mpi[r] * dim, &send_counts_crd[r]));
    PetscCall(PetscMPIIntCast((PetscInt64)send_displs_mpi[r] * dim, &send_displs_crd[r]));
    PetscCall(PetscMPIIntCast((PetscInt64)recv_counts_mpi[r] * dim, &recv_counts_crd[r]));
    PetscCall(PetscMPIIntCast((PetscInt64)recv_displs_mpi[r] * dim, &recv_displs_crd[r]));
  }
  PetscCallMPI(MPI_Alltoallv(send_crd, send_counts_crd, send_displs_crd, MPIU_REAL, recv_crd, recv_counts_crd, recv_displs_crd, MPIU_REAL, comm));

  PetscCall(PetscFree4(send_counts_mpi, send_displs_mpi, recv_counts_mpi, recv_displs_mpi));
  PetscCall(PetscFree4(send_counts_crd, send_displs_crd, recv_counts_crd, recv_displs_crd));
  PetscCall(PetscFree2(send_counts, send_displs));
  PetscCall(PetscFree2(recv_counts, recv_displs));
  PetscCall(PetscFree(send_idx));
  PetscCall(PetscFree(send_crd));
  PetscCall(PetscFree(all_bboxes));

  *n_obs_filt     = total_recv;
  *obs_idx_out    = recv_idx;
  *obs_coords_out = recv_crd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFCoalesceNnzMinMax - In-place MAX-reduction of per-row nnz min/max across `comm`.

  Coalesces (max, min) into a single MAX allreduce by negating the min, which halves the latency
  versus two separate reductions. The sentinel `PETSC_INT_MAX` round-trips through negation, so a
  rank with zero local rows does not pollute the global min; on return the sentinel is clamped to 0
  to keep the value meaningful for viewers and downstream checks.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFCoalesceNnzMinMax(MPI_Comm comm, PetscInt *min_inout, PetscInt *max_inout)
{
  PetscInt mm[2];

  PetscFunctionBegin;
  mm[0] = *max_inout;
  mm[1] = -(*min_inout);
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, mm, 2, MPIU_INT, MPI_MAX, comm));
  *max_inout = mm[0];
  *min_inout = -mm[1];
  if (*min_inout == PETSC_INT_MAX) *min_inout = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFComputeObsCoords - Allocate and fill obs_locs[d] = H * xyz[d] for d in [0, dim).

  Determines `dim` from the contiguous non-NULL prefix of `xyz` (with a contiguity check) and
  returns a freshly allocated `Vec[dim]` whose entries are H-image vectors carrying coordinate
  values at observation locations. Caller frees with `PetscDALETKFDestroyObsCoords()`.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFComputeObsCoords(Mat H, Vec xyz[], PetscInt *dim_out, Vec **obs_vecs_out)
{
  MPI_Comm comm;
  PetscInt dim = 0;
  Vec     *obs_vecs;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  for (PetscInt d = 0; d < 3; ++d) {
    if (xyz[d]) {
      PetscCheck(d == dim, comm, PETSC_ERR_ARG_WRONG, "Coordinate slots must be contiguous from xyz[0]; got NULL before xyz[%" PetscInt_FMT "]", d);
      dim++;
    }
  }
  PetscCheck(dim >= 1, comm, PETSC_ERR_ARG_WRONG, "At least one coordinate vector required in xyz[0]");
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  for (PetscInt d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, xyz[d], obs_vecs[d]));
  }
  *dim_out      = dim;
  *obs_vecs_out = obs_vecs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFDestroyObsCoords - Tear down the obs_vecs array allocated by PetscDALETKFComputeObsCoords().
*/
PETSC_INTERN PetscErrorCode PetscDALETKFDestroyObsCoords(PetscInt dim, Vec **obs_vecs)
{
  PetscFunctionBegin;
  if (!*obs_vecs) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt d = 0; d < dim; ++d) PetscCall(VecDestroy(&(*obs_vecs)[d]));
  PetscCall(PetscFree(*obs_vecs));
  *obs_vecs = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFAssembleQFromCSR - Materialize the localization Mat `Q` from a per-rank CSR triple.

  Shared by the AIJ and Kokkos backends after each has produced (`row_counts`, `row_offsets`,
  `col_indices`, `values`) describing every local row of `Q`. `mat_type` selects `MATAIJ` or
  `MATAIJKOKKOS`. `H` is consulted only for its row ownership range, which determines the
  diagonal-vs-off-diagonal split used for MPI preallocation. The output Mat is fully assembled.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFAssembleQFromCSR(Mat H, PetscInt n_vert_local, PetscInt n_obs_local, PetscInt n_obs_global, MatType mat_type, const PetscInt row_counts[], const PetscInt row_offsets[], const PetscInt col_indices[], const PetscScalar values[], Mat *Q)
{
  MPI_Comm  comm;
  PetscInt  rstart, cstart, cend;
  PetscInt *d_nnz, *o_nnz;
  PetscBool is_aij, is_aijkok, is_seqaij, is_mpiaij;

  PetscFunctionBegin;
  /* The seq/MPI AIJ preallocation calls below are no-ops for non-AIJ types, so an unsupported
     mat_type would silently fall through to slow MatSetValues with no preallocation. Reject it
     up front instead. */
  PetscCall(PetscStrcmp(mat_type, MATAIJ, &is_aij));
  PetscCall(PetscStrcmp(mat_type, MATAIJKOKKOS, &is_aijkok));
  PetscCall(PetscStrcmp(mat_type, MATSEQAIJ, &is_seqaij));
  PetscCall(PetscStrcmp(mat_type, MATMPIAIJ, &is_mpiaij));
  PetscCheck(is_aij || is_aijkok || is_seqaij || is_mpiaij, PetscObjectComm((PetscObject)H), PETSC_ERR_SUP, "Unsupported Q mat_type \"%s\"; expected MATAIJ or MATAIJKOKKOS", mat_type);
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCall(MatGetOwnershipRange(H, &cstart, &cend));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, mat_type));
  PetscCall(MatGetOwnershipRange(*Q, &rstart, NULL));
  PetscCall(PetscCalloc1(n_vert_local, &d_nnz));
  PetscCall(PetscCalloc1(n_vert_local, &o_nnz));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt nnz = row_counts[i];
    PetscInt off = row_offsets[i];
    for (PetscInt k = 0; k < nnz; ++k) {
      PetscInt col = col_indices[off + k];
      if (col >= cstart && col < cend) d_nnz[i]++;
      else o_nnz[i]++;
    }
  }
  /* MatXAIJSetPreallocation dispatches to the right Seq/MPI variant for AIJ matrices. In serial
     d_nnz already holds the full per-row count (cstart=0, cend=n_obs_global covers every column),
     so the same dnnz/onnz pair is correct in both layouts. */
  PetscCall(MatXAIJSetPreallocation(*Q, 1, d_nnz, o_nnz, NULL, NULL));
  PetscCall(PetscFree(d_nnz));
  PetscCall(PetscFree(o_nnz));

  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt global_row = rstart + i;
    PetscInt nnz        = row_counts[i];
    PetscInt off        = row_offsets[i];
    PetscCall(MatSetValues(*Q, 1, &global_row, nnz, &col_indices[off], &values[off], INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFLogQStats - PetscInfo one-liner summarizing min/max nnz per row of `Q`.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFLogQStats(Mat Q, PetscDALETKFLocalizationType type, PetscReal radius, PetscInt n_vert_local, PetscInt n_obs_global, const PetscInt row_counts[])
{
  MPI_Comm comm;
  PetscInt local_min = PETSC_INT_MAX, local_max = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Q, &comm));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    if (row_counts[i] < local_min) local_min = row_counts[i];
    if (row_counts[i] > local_max) local_max = row_counts[i];
  }
  PetscCall(PetscDALETKFCoalesceNnzMinMax(comm, &local_min, &local_max));
  PetscCall(PetscInfo((PetscObject)Q, "LETKF localization (type=%s, radius=%g): %" PetscInt_FMT " vertices, %" PetscInt_FMT " obs, nnz/row min=%" PetscInt_FMT " max=%" PetscInt_FMT "\n", PetscDALETKFLocalizationTypes[type], (double)radius, n_vert_local, n_obs_global, local_min, local_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFCreateLocalizationMat_AIJ - host (`MATAIJ`) implementation of the localization-weight Mat `Q`.

  Counterpart to `PetscDALETKFCreateLocalizationMat_Kokkos()`; same two-pass count/fill structure but plain C
  with no Kokkos dependency. Selected by the `PetscDALETKFCreateLocalizationMat()` dispatcher when `H` is not
  a Kokkos matrix type.
*/
static PetscErrorCode PetscDALETKFCreateLocalizationMat_AIJ(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, Mat *Q, PetscInt *max_nnz_local, PetscInt *n_nnz_local)
{
  PetscInt     dim, n_vert_local, n_obs_global, n_obs_local, n_obs_cand;
  PetscInt     total_nnz   = 0;
  PetscInt64   total_nnz64 = 0;
  PetscInt     row_max     = 0;
  PetscInt    *row_counts, *row_offsets, *col_indices, *obs_global_idx;
  PetscScalar *values;
  PetscReal    cutoff, cutoff2;
  PetscReal   *vertex_coords, *obs_coords;
  Vec         *obs_vecs;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(H, &n_obs_local, NULL));
  PetscCall(MatGetSize(H, &n_obs_global, NULL));
  PetscCall(PetscDALETKFComputeObsCoords(H, xyz, &dim, &obs_vecs));
  PetscCall(VecGetLocalSize(xyz[0], &n_vert_local));

  /* Vertex coordinates flattened as [vert][dim] (row-major). */
  PetscCall(PetscMalloc1((size_t)n_vert_local * dim, &vertex_coords));
  for (PetscInt d = 0; d < dim; ++d) {
    const PetscScalar *local_coords_array;
    PetscCall(VecGetArrayRead(xyz[d], &local_coords_array));
    for (PetscInt i = 0; i < n_vert_local; ++i) vertex_coords[i * dim + d] = PetscRealPart(local_coords_array[i]);
    PetscCall(VecRestoreArrayRead(xyz[d], &local_coords_array));
  }

  /* Single source of truth for the cutoff policy lives in letkf_kernels.h; LETKFCutoff() returns
     the un-squared bound directly to avoid sqrt(r*r) round-trip FP error in the bbox prune
     (matters for the BOXCAR kernel's strict (distance < radius) test on boundary obs). */
  cutoff  = LETKFCutoff(type, radius);
  cutoff2 = cutoff * cutoff;

  /* Bbox-pruned obs gather: replaces the all-to-all materialization of the global obs coord set. */
  PetscCall(PetscDALETKFGatherObsBbox(dim, xyz, bd, cutoff, H, obs_vecs, &n_obs_cand, &obs_global_idx, &obs_coords));

  /* Pass 1: Count nnz per row (only entries with positive kernel weight).
     We pay the kernel evaluation twice (here and in Pass 2) to keep CSR allocations
     sized exactly. The alternative -- allocate at the cutoff-bbox upper bound, fill,
     then compact -- would peak at ~2x the memory and still touch every candidate.
     Both passes go through LETKFRowWeight() (in letkf_kernels.h, shared with the
     Kokkos backend) so the CPU and device kernels stay in lockstep. */
  PetscCall(PetscMalloc1(n_vert_local, &row_counts));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt count = 0;
    for (PetscInt j = 0; j < n_obs_cand; ++j) {
      if (LETKFRowWeight(type, radius, cutoff2, dim, &vertex_coords[i * dim], &obs_coords[j * dim], bd) > 0.0) count++;
    }
    row_counts[i] = count;
  }

  /* Prefix sum for CSR row offsets, total nnz, and per-rank max nnz. Accumulate the running
     total in 64-bit and cast so we trip a clear error instead of silently wrapping when
     localization radius * obs density overflows PetscInt; mirrors the Kokkos backend
     (kokkos/dalocalizationletkf.kokkos.cxx). The per-rank max feeds PetscDALETKFInstallQ()
     without a downstream MatGetRow walk. */
  PetscCall(PetscMalloc1(n_vert_local + 1, &row_offsets));
  row_offsets[0] = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    total_nnz64 += (PetscInt64)row_counts[i];
    row_offsets[i + 1] = row_offsets[i] + row_counts[i];
    if (row_counts[i] > row_max) row_max = row_counts[i];
  }
  PetscCall(PetscIntCast(total_nnz64, &total_nnz));

  /* Pass 2: Fill column indices and weights. The (w > 0.0) gate must match Pass 1 exactly
     (same LETKFRowWeight() call) so each row writes precisely row_counts[i] entries. */
  PetscCall(PetscMalloc1(total_nnz, &col_indices));
  PetscCall(PetscMalloc1(total_nnz, &values));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt offset = row_offsets[i];
    PetscInt pos    = 0;
    for (PetscInt j = 0; j < n_obs_cand; ++j) {
      PetscReal w = LETKFRowWeight(type, radius, cutoff2, dim, &vertex_coords[i * dim], &obs_coords[j * dim], bd);
      if (w > 0.0) {
        col_indices[offset + pos] = obs_global_idx[j];
        values[offset + pos]      = w;
        pos++;
      }
    }
    PetscAssert(pos == row_counts[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "LETKF localization Pass 1/2 mismatch on row %" PetscInt_FMT ": pass1=%" PetscInt_FMT " pass2=%" PetscInt_FMT, i, row_counts[i], pos);
  }

  PetscCall(PetscDALETKFAssembleQFromCSR(H, n_vert_local, n_obs_local, n_obs_global, MATAIJ, row_counts, row_offsets, col_indices, values, Q));
  PetscCall(PetscDALETKFLogQStats(*Q, type, radius, n_vert_local, n_obs_global, row_counts));

  *max_nnz_local = row_max;
  *n_nnz_local   = total_nnz;
  PetscCall(PetscDALETKFDestroyObsCoords(dim, &obs_vecs));
  PetscCall(PetscFree(vertex_coords));
  PetscCall(PetscFree(obs_coords));
  PetscCall(PetscFree(obs_global_idx));
  PetscCall(PetscFree(row_counts));
  PetscCall(PetscFree(row_offsets));
  PetscCall(PetscFree(col_indices));
  PetscCall(PetscFree(values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFCreateLocalizationMat - construct the LETKF localization-weight Mat `Q` for a built-in
  distance kernel. Validates the common arguments and dispatches to either the Kokkos backend or the
  host `MATAIJ` backend based on `use_kokkos`. The caller chooses the backend so this matches the
  analysis-time backend selection (which keys off the obs-error covariance Mat `da->R`); using `H`'s
  type here would dispatch inconsistently when the user has a Kokkos `H` but a CPU `R` (or vice versa).

  Output `Q` has rows indexed by local grid vertices and columns indexed by global observations; the
  caller owns the returned Mat.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFCreateLocalizationMat(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, PetscBool use_kokkos, Mat *Q, PetscInt *max_nnz_local, PetscInt *n_nnz_local)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscAssertPointer(xyz, 3);
  PetscAssertPointer(bd, 4);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 5);
  PetscAssertPointer(Q, 7);
  PetscAssertPointer(max_nnz_local, 8);
  PetscAssertPointer(n_nnz_local, 9);
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCheck(type == PETSCDA_LETKF_LOC_GASPARI_COHN || type == PETSCDA_LETKF_LOC_GAUSSIAN || type == PETSCDA_LETKF_LOC_BOXCAR, comm, PETSC_ERR_ARG_WRONG, "Built-in kernel required, got localization type %d", (int)type);
  PetscCheck(radius > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Localization radius must be positive, got %g", (double)radius);
  /* xyz[] contiguity and dim>=1 are enforced by PetscDALETKFComputeObsCoords() inside the backends. */
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  if (use_kokkos) {
    PetscCall(PetscDALETKFCreateLocalizationMat_Kokkos(type, radius, xyz, bd, H, Q, max_nnz_local, n_nnz_local));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  PetscCall(PetscDALETKFCreateLocalizationMat_AIJ(type, radius, xyz, bd, H, Q, max_nnz_local, n_nnz_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

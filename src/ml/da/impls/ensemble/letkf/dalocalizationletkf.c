#include <petsc.h>
#include <petscmat.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf_kernels.h>

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
      const PetscReal *bbox = &all_bboxes[(size_t)r * 2 * dim];
      PetscBool        in   = PETSC_TRUE;

      for (PetscInt d = 0; d < dim; ++d) {
        if (coord_k[d] < bbox[d] || coord_k[d] > bbox[dim + d]) {
          in = PETSC_FALSE;
          break;
        }
      }
      if (in) send_counts[r]++;
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
      const PetscReal *bbox = &all_bboxes[(size_t)r * 2 * dim];
      PetscBool        in   = PETSC_TRUE;

      for (PetscInt d = 0; d < dim; ++d) {
        if (coord_k[d] < bbox[d] || coord_k[d] > bbox[dim + d]) {
          in = PETSC_FALSE;
          break;
        }
      }
      if (in) {
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
  PetscDALETKFCreateLocalizationMat_AIJ - host (`MATAIJ`) implementation of the localization-weight Mat `Q`.

  Counterpart to `PetscDALETKFCreateLocalizationMat_Kokkos()`; same two-pass count/fill structure but plain C
  with no Kokkos dependency. Selected by the `PetscDALETKFCreateLocalizationMat()` dispatcher when `H` is not
  a Kokkos matrix type.
*/
static PetscErrorCode PetscDALETKFCreateLocalizationMat_AIJ(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, Mat *Q)
{
  PetscInt     dim = 0, n_vert_local, d, n_obs_global, n_obs_local, n_obs_cand;
  PetscInt     rstart, cstart, cend;
  PetscInt     total_nnz   = 0;
  PetscInt64   total_nnz64 = 0;
  PetscInt     local_min, local_max;
  PetscInt    *d_nnz, *o_nnz, *seq_nnz;
  PetscInt    *row_counts, *row_offsets, *col_indices;
  PetscInt    *obs_global_idx;
  PetscScalar *values;
  PetscReal    cutoff2, cutoff;
  PetscReal   *vertex_coords, *obs_coords;
  Vec         *obs_vecs;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCall(MatGetLocalSize(H, &n_obs_local, NULL));
  PetscCall(MatGetSize(H, &n_obs_global, NULL));
  for (d = 0; d < 3; ++d) {
    if (xyz[d]) {
      PetscCheck(d == dim, comm, PETSC_ERR_ARG_WRONG, "Coordinate slots must be contiguous from xyz[0]; got NULL before xyz[%" PetscInt_FMT "]", d);
      dim++;
    }
  }
  PetscCall(VecGetLocalSize(xyz[0], &n_vert_local));

  /* Compute observation coordinates: obs_locs[d] = H * xyz[d] */
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  for (d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, xyz[d], obs_vecs[d]));
  }

  /* Vertex coordinates flattened as [vert][dim] (row-major). */
  PetscCall(PetscMalloc1((size_t)n_vert_local * dim, &vertex_coords));
  for (d = 0; d < dim; ++d) {
    const PetscScalar *local_coords_array;
    PetscCall(VecGetArrayRead(xyz[d], &local_coords_array));
    for (PetscInt i = 0; i < n_vert_local; ++i) vertex_coords[i * dim + d] = PetscRealPart(local_coords_array[i]);
    PetscCall(VecRestoreArrayRead(xyz[d], &local_coords_array));
  }

  /* Compute cutoff and cutoff^2 directly per type to avoid sqrt(r*r) round-trip FP error in the
     bbox prune; the BOXCAR kernel uses a strict (distance < radius) test, so a 1-ulp shrink of
     cutoff could otherwise drop a boundary observation in the GatherObsBbox stage. */
  cutoff  = (type == PETSCDA_LETKF_LOC_BOXCAR) ? radius : 2.0 * radius;
  cutoff2 = cutoff * cutoff;

  /* Bbox-pruned obs gather: replaces the all-to-all materialization of the global obs coord set. */
  PetscCall(PetscDALETKFGatherObsBbox(dim, xyz, bd, cutoff, H, obs_vecs, &n_obs_cand, &obs_global_idx, &obs_coords));

  /* Pass 1: Count nnz per row (only entries with positive kernel weight).
     We pay the kernel evaluation twice (here and in Pass 2) to keep CSR allocations
     sized exactly. The alternative -- allocate at the cutoff-bbox upper bound, fill,
     then compact -- would peak at ~2x the memory and still touch every candidate.
     Always dispatch through LETKFKernelEval() so the CPU and Kokkos kernels stay in
     lockstep; a future kernel change cannot silently diverge for boxcar. */
  PetscCall(PetscMalloc1(n_vert_local, &row_counts));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt count = 0;
    for (PetscInt j = 0; j < n_obs_cand; ++j) {
      PetscReal dist2 = 0.0;
      for (PetscInt dd = 0; dd < dim; ++dd) {
        PetscReal diff = vertex_coords[i * dim + dd] - obs_coords[j * dim + dd];
        if (bd[dd] > 0.0) {
          PetscReal domain_size = bd[dd];
          if (diff > 0.5 * domain_size) diff -= domain_size;
          else if (diff < -0.5 * domain_size) diff += domain_size;
        }
        dist2 += diff * diff;
      }
      if (dist2 >= cutoff2) continue;
      if (LETKFKernelEval(type, PetscSqrtReal(dist2), radius) > 0.0) count++;
    }
    row_counts[i] = count;
  }

  /* Prefix sum for CSR row offsets, total nnz. Accumulate the running total in 64-bit and cast
     so we trip a clear error instead of silently wrapping when localization radius * obs density
     overflows PetscInt; mirrors the Kokkos backend (kokkos/dalocalizationletkf.kokkos.cxx). */
  PetscCall(PetscMalloc1(n_vert_local + 1, &row_offsets));
  row_offsets[0] = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    total_nnz64 += (PetscInt64)row_counts[i];
    row_offsets[i + 1] = row_offsets[i] + row_counts[i];
  }
  PetscCall(PetscIntCast(total_nnz64, &total_nnz));

  /* Pass 2: Fill column indices and weights. The (dist2 < cutoff2) && (w > 0.0) gate must
     match Pass 1 exactly so each row writes precisely row_counts[i] entries. */
  PetscCall(PetscMalloc1(total_nnz, &col_indices));
  PetscCall(PetscMalloc1(total_nnz, &values));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt offset = row_offsets[i];
    PetscInt pos    = 0;
    for (PetscInt j = 0; j < n_obs_cand; ++j) {
      PetscReal dist2 = 0.0;
      for (PetscInt dd = 0; dd < dim; ++dd) {
        PetscReal diff = vertex_coords[i * dim + dd] - obs_coords[j * dim + dd];
        if (bd[dd] > 0.0) {
          PetscReal domain_size = bd[dd];
          if (diff > 0.5 * domain_size) diff -= domain_size;
          else if (diff < -0.5 * domain_size) diff += domain_size;
        }
        dist2 += diff * diff;
      }
      if (dist2 < cutoff2) {
        PetscReal w = LETKFKernelEval(type, PetscSqrtReal(dist2), radius);
        if (w > 0.0) {
          col_indices[offset + pos] = obs_global_idx[j];
          values[offset + pos]      = w;
          pos++;
        }
      }
    }
    PetscAssert(pos == row_counts[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "LETKF localization Pass 1/2 mismatch on row %" PetscInt_FMT ": pass1=%" PetscInt_FMT " pass2=%" PetscInt_FMT, i, row_counts[i], pos);
  }

  /* Determine column ownership range for diagonal/off-diagonal preallocation split.
     Q's row layout matches xyz[0] (one row per local vertex); columns are indexed by global
     observations matching H's row layout. */
  PetscCall(MatGetOwnershipRange(H, &cstart, &cend));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, MATAIJ));
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
  /* Seq variant takes total per-row nnz (all entries are "diagonal" in the serial layout); MPI variant
     splits diagonal vs off-diagonal blocks. Compute the seq totals into a temporary so the MPI side
     keeps its split intact. */
  PetscCall(PetscMalloc1(n_vert_local, &seq_nnz));
  for (PetscInt i = 0; i < n_vert_local; ++i) seq_nnz[i] = d_nnz[i] + o_nnz[i];
  PetscCall(MatSeqAIJSetPreallocation(*Q, 0, seq_nnz));
  PetscCall(PetscFree(seq_nnz));
  PetscCall(MatMPIAIJSetPreallocation(*Q, 0, d_nnz, 0, o_nnz));
  PetscCall(PetscFree(d_nnz));
  PetscCall(PetscFree(o_nnz));

  /* Fill matrix row by row. */
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt globalRow = rstart + i;
    PetscInt nnz       = row_counts[i];
    PetscInt off       = row_offsets[i];
    PetscCall(MatSetValues(*Q, 1, &globalRow, nnz, &col_indices[off], &values[off], INSERT_VALUES));
  }

  local_min = PETSC_INT_MAX;
  local_max = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    if (row_counts[i] < local_min) local_min = row_counts[i];
    if (row_counts[i] > local_max) local_max = row_counts[i];
  }
  PetscCall(PetscDALETKFCoalesceNnzMinMax(comm, &local_min, &local_max));
  PetscCall(PetscInfo((PetscObject)*Q, "LETKF localization (type=%s, radius=%g): %" PetscInt_FMT " vertices, %" PetscInt_FMT " obs, nnz/row min=%" PetscInt_FMT " max=%" PetscInt_FMT "\n", PetscDALETKFLocalizationTypes[type], (double)radius, n_vert_local, n_obs_global, local_min, local_max));

  for (d = 0; d < dim; ++d) PetscCall(VecDestroy(&obs_vecs[d]));
  PetscCall(PetscFree(obs_vecs));
  PetscCall(PetscFree(vertex_coords));
  PetscCall(PetscFree(obs_coords));
  PetscCall(PetscFree(obs_global_idx));
  PetscCall(PetscFree(row_counts));
  PetscCall(PetscFree(row_offsets));
  PetscCall(PetscFree(col_indices));
  PetscCall(PetscFree(values));

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
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
PETSC_INTERN PetscErrorCode PetscDALETKFCreateLocalizationMat(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, PetscBool use_kokkos, Mat *Q)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscAssertPointer(xyz, 3);
  PetscAssertPointer(bd, 4);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 5);
  PetscAssertPointer(Q, 7);
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCheck(type == PETSCDA_LETKF_LOC_GASPARI_COHN || type == PETSCDA_LETKF_LOC_GAUSSIAN || type == PETSCDA_LETKF_LOC_BOXCAR, comm, PETSC_ERR_ARG_WRONG, "Built-in kernel required, got localization type %d", (int)type);
  PetscCheck(radius > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Localization radius must be positive, got %g", (double)radius);
  /* xyz[] contiguity and dim>=1 are enforced by PetscDALETKFComputeObsCoords() inside the backends. */
#if defined(PETSC_HAVE_KOKKOS_KERNELS) && !defined(PETSC_USE_COMPLEX)
  if (use_kokkos) {
    PetscCall(PetscDALETKFCreateLocalizationMat_Kokkos(type, radius, xyz, bd, H, Q));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  PetscCall(PetscDALETKFCreateLocalizationMat_AIJ(type, radius, xyz, bd, H, Q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc.h>
#include <petscmat.h>
#include <petsc_kokkos.hpp>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <../src/ml/da/impls/ensemble/letkf/letkf_kernels.h>

/*
  PetscDALETKFBuildLocalizationMatrix_Kokkos - Kokkos-backed build of the localization weight matrix `Q`.

  Used by the lazy Q construction inside `PetscDAEnsembleAnalysis_LETKF()` when the observation operator is a
  Kokkos matrix type. The CPU counterpart `PetscDALETKFBuildLocalizationMatrix()` produces a numerically
  identical Q for the polynomial kernels (`PETSCDA_LETKF_LOC_GASPARI_COHN`, `PETSCDA_LETKF_LOC_BOXCAR`) and a
  bit-comparable Q for `PETSCDA_LETKF_LOC_GAUSSIAN` modulo `exp()` rounding.

  `type` must be one of `PETSCDA_LETKF_LOC_GASPARI_COHN`, `PETSCDA_LETKF_LOC_GAUSSIAN`, `PETSCDA_LETKF_LOC_BOXCAR`.
  `PETSCDA_LETKF_LOC_NONE` is a caller error.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFBuildLocalizationMatrix_Kokkos(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, Mat *Q)
{
  PetscInt                           dim = 0, n_vert_local, d, n_obs_global, n_obs_local;
  PetscInt                           rstart, cstart, cend;
  PetscInt                           total_nnz = 0, local_min, local_max;
  PetscInt                          *d_nnz, *o_nnz;
  Vec                               *obs_vecs;
  MPI_Comm                           comm;
  PetscLayout                        cmap;
  const PetscReal                    cutoff2   = LETKFCutoffSquared(type, radius);
  const PetscDALETKFLocalizationType kern_type = type;
  const PetscReal                    kern_r    = radius;

  PetscFunctionBegin;
  PetscAssertPointer(xyz, 3);
  PetscAssertPointer(bd, 4);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 5);
  PetscAssertPointer(Q, 6);

  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCheck(type == PETSCDA_LETKF_LOC_GASPARI_COHN || type == PETSCDA_LETKF_LOC_GAUSSIAN || type == PETSCDA_LETKF_LOC_BOXCAR, comm, PETSC_ERR_ARG_WRONG, "Built-in kernel required, got localization type %d.", (int)type);
  PetscCheck(radius > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Localization radius must be positive, got %g.", (double)radius);
  PetscCall(MatGetLocalSize(H, &n_obs_local, NULL));
  PetscCall(MatGetSize(H, &n_obs_global, NULL));
  for (d = 0; d < 3; ++d) {
    if (xyz[d]) dim++;
    else break;
  }
  PetscCheck(dim > 0, comm, PETSC_ERR_ARG_WRONG, "At least one coordinate vector (xyz[0]) must be non-NULL; got dim=%" PetscInt_FMT, dim);
  PetscCall(VecGetLocalSize(xyz[0], &n_vert_local));

  /* Compute observation coordinates: obs_locs[d] = H * xyz[d] */
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  for (d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, xyz[d], obs_vecs[d]));
  }

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;

  /* Copy vertex coordinates to device */
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace> vertex_coords_dev("vertex_coords", n_vert_local, dim);
  {
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> vertex_coords_host("vertex_coords_host", n_vert_local, dim);
    for (d = 0; d < dim; ++d) {
      const PetscScalar *local_coords_array;
      PetscCall(VecGetArrayRead(xyz[d], &local_coords_array));
      for (PetscInt i = 0; i < n_vert_local; ++i) vertex_coords_host(i, d) = local_coords_array[i];
      PetscCall(VecRestoreArrayRead(xyz[d], &local_coords_array));
    }
    Kokkos::deep_copy(vertex_coords_dev, vertex_coords_host);
  }

  /* Copy observation coordinates to device (gather to all ranks) */
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> obs_coords_dev("obs_coords", n_obs_global, dim);
  {
    Kokkos::View<PetscReal **, Kokkos::LayoutRight, Kokkos::HostSpace> obs_coords_host("obs_coords_host", n_obs_global, dim);
    for (d = 0; d < dim; ++d) {
      VecScatter         ctx;
      Vec                seq_vec;
      const PetscScalar *array;

      PetscCall(VecScatterCreateToAll(obs_vecs[d], &ctx, &seq_vec));
      PetscCall(VecScatterBegin(ctx, obs_vecs[d], seq_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ctx, obs_vecs[d], seq_vec, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecGetArrayRead(seq_vec, &array));
      for (PetscInt j = 0; j < n_obs_global; ++j) obs_coords_host(j, d) = PetscRealPart(array[j]);
      PetscCall(VecRestoreArrayRead(seq_vec, &array));
      PetscCall(VecScatterDestroy(&ctx));
      PetscCall(VecDestroy(&seq_vec));
    }
    Kokkos::deep_copy(obs_coords_dev, obs_coords_host);
  }

  /* Copy boundary data to device */
  Kokkos::View<PetscReal *, MemSpace> bd_dev("bd_dev", dim);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> bd_host("bd_host", dim);
    for (d = 0; d < dim; ++d) bd_host(d) = bd[d];
    Kokkos::deep_copy(bd_dev, bd_host);
  }

  /* Pass 1: Count nnz per row (only entries with positive kernel weight).
     Same trade-off as the host backend: re-evaluate the kernel in Pass 2 rather than
     allocate and compact at the cutoff-bbox upper bound. The arithmetic is cheap on the
     device; the global memory writes saved by exact sizing dominate. */
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace> row_counts_dev("row_counts", n_vert_local);

  Kokkos::parallel_for(
    "CountNnzPerRow", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt dd = 0; dd < dim; ++dd) v_coords[dd] = PetscRealPart(vertex_coords_dev(i, dd));

      PetscInt count = 0;
      for (PetscInt j = 0; j < n_obs_global; ++j) {
        PetscReal dist2 = 0.0;
        for (PetscInt dd = 0; dd < dim; ++dd) {
          PetscReal diff = v_coords[dd] - obs_coords_dev(j, dd);
          if (bd_dev(dd) > 0.0) {
            PetscReal domain_size = bd_dev(dd);
            if (diff > 0.5 * domain_size) diff -= domain_size;
            else if (diff < -0.5 * domain_size) diff += domain_size;
          }
          dist2 += diff * diff;
        }
        if (dist2 < cutoff2 && LETKFKernelEval(kern_type, Kokkos::sqrt(dist2), kern_r) > 0.0) count++;
      }
      row_counts_dev(i) = count;
    });
  Kokkos::fence();

  /* Copy row counts to host for preallocation */
  auto row_counts_host = Kokkos::create_mirror_view(row_counts_dev);
  Kokkos::deep_copy(row_counts_host, row_counts_dev);

  /* Compute prefix sum on host for CSR row offsets */
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>          row_offsets_dev("row_offsets", n_vert_local + 1);
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace> row_offsets_host("row_offsets_host", n_vert_local + 1);
  row_offsets_host(0) = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) row_offsets_host(i + 1) = row_offsets_host(i) + row_counts_host(i);
  Kokkos::deep_copy(row_offsets_dev, row_offsets_host);

  /* Total nnz for output arrays */
  for (PetscInt i = 0; i < n_vert_local; ++i) total_nnz += row_counts_host(i);

  /* Pass 2: Fill column indices and weights */
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>    col_indices_dev("col_indices", total_nnz);
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, MemSpace> values_dev("values", total_nnz);

  Kokkos::parallel_for(
    "FillLocalizationMatrix", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscInt offset = row_offsets_dev(i);

      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt dd = 0; dd < dim; ++dd) v_coords[dd] = PetscRealPart(vertex_coords_dev(i, dd));

      PetscInt pos = 0;
      for (PetscInt j = 0; j < n_obs_global; ++j) {
        PetscReal dist2 = 0.0;
        for (PetscInt dd = 0; dd < dim; ++dd) {
          PetscReal diff = v_coords[dd] - obs_coords_dev(j, dd);
          if (bd_dev(dd) > 0.0) {
            PetscReal domain_size = bd_dev(dd);
            if (diff > 0.5 * domain_size) diff -= domain_size;
            else if (diff < -0.5 * domain_size) diff += domain_size;
          }
          dist2 += diff * diff;
        }
        if (dist2 < cutoff2) {
          PetscReal w = LETKFKernelEval(kern_type, Kokkos::sqrt(dist2), kern_r);
          if (w > 0.0) {
            col_indices_dev(offset + pos) = j;
            values_dev(offset + pos)      = w;
            pos++;
          }
        }
      }
    });
  Kokkos::fence();

  /* Copy results to host */
  auto col_indices_host = Kokkos::create_mirror_view(col_indices_dev);
  auto values_host      = Kokkos::create_mirror_view(values_dev);
  Kokkos::deep_copy(col_indices_host, col_indices_dev);
  Kokkos::deep_copy(values_host, values_dev);

  /* Create Q matrix with variable nnz preallocation */
  PetscCall(VecGetOwnershipRange(xyz[0], &rstart, NULL));

  /* Determine column ownership range for diagonal/off-diagonal preallocation split */
  PetscCall(PetscLayoutCreate(comm, &cmap));
  PetscCall(PetscLayoutSetLocalSize(cmap, n_obs_local));
  PetscCall(PetscLayoutSetSize(cmap, n_obs_global));
  PetscCall(PetscLayoutSetUp(cmap));
  PetscCall(PetscLayoutGetRange(cmap, &cstart, &cend));
  PetscCall(PetscLayoutDestroy(&cmap));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, MATAIJKOKKOS));
  PetscCall(PetscCalloc1(n_vert_local, &d_nnz));
  PetscCall(PetscCalloc1(n_vert_local, &o_nnz));
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt nnz = row_counts_host(i);
    PetscInt off = row_offsets_host(i);
    for (PetscInt k = 0; k < nnz; ++k) {
      PetscInt col = col_indices_host(off + k);
      if (col >= cstart && col < cend) d_nnz[i]++;
      else o_nnz[i]++;
    }
  }
  /* Seq variant takes total per-row nnz (all entries are "diagonal" in the serial layout); MPI variant
     splits diagonal vs off-diagonal blocks. Compute the seq totals into a temporary so the MPI side
     keeps its split intact. */
  {
    PetscInt *seq_nnz;
    PetscCall(PetscMalloc1(n_vert_local, &seq_nnz));
    for (PetscInt i = 0; i < n_vert_local; ++i) seq_nnz[i] = d_nnz[i] + o_nnz[i];
    PetscCall(MatSeqAIJSetPreallocation(*Q, 0, seq_nnz));
    PetscCall(PetscFree(seq_nnz));
  }
  PetscCall(MatMPIAIJSetPreallocation(*Q, 0, d_nnz, 0, o_nnz));
  PetscCall(PetscFree(d_nnz));
  PetscCall(PetscFree(o_nnz));

  /* Fill matrix row by row */
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt globalRow = rstart + i;
    PetscInt nnz       = row_counts_host(i);
    PetscInt off       = row_offsets_host(i);
    PetscCall(MatSetValues(*Q, 1, &globalRow, nnz, &col_indices_host(off), &values_host(off), INSERT_VALUES));
  }

  /* Log statistics */
  local_min = n_obs_global;
  local_max = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    if (row_counts_host(i) < local_min) local_min = row_counts_host(i);
    if (row_counts_host(i) > local_max) local_max = row_counts_host(i);
  }
  PetscCall(PetscInfo((PetscObject)*Q, "LETKF localization (type=%d, radius=%g): %" PetscInt_FMT " vertices, %" PetscInt_FMT " obs, nnz/row min=%" PetscInt_FMT " max=%" PetscInt_FMT "\n", (int)type, (double)radius, n_vert_local, n_obs_global, local_min, local_max));

  /* Cleanup */
  for (d = 0; d < dim; ++d) PetscCall(VecDestroy(&obs_vecs[d]));
  PetscCall(PetscFree(obs_vecs));

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

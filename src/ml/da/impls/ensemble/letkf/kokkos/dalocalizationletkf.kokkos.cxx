#include <petsc.h>
#include <petscmat.h>
#include <petsc_kokkos.hpp>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf_kernels.h>

/*
  PetscDALETKFCreateLocalizationMat_Kokkos - Kokkos (`MATAIJKOKKOS`) implementation of the localization
  weight Mat `Q`.

  Selected by the `PetscDALETKFCreateLocalizationMat()` dispatcher when `H` is a Kokkos matrix type. The
  host counterpart `PetscDALETKFCreateLocalizationMat_AIJ()` produces a numerically identical Q for the
  polynomial kernels (`PETSCDA_LETKF_LOC_GASPARI_COHN`, `PETSCDA_LETKF_LOC_BOXCAR`) and a bit-comparable
  Q for `PETSCDA_LETKF_LOC_GAUSSIAN` modulo `exp()` rounding.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFCreateLocalizationMat_Kokkos(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, Mat *Q)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;

  PetscInt                                                           dim = 0, n_vert_local, d, n_obs_global, n_obs_local, n_obs_cand;
  PetscInt                                                           rstart, cstart, cend;
  PetscInt                                                           total_nnz = 0, local_min, local_max;
  PetscInt                                                          *d_nnz, *o_nnz, *seq_nnz = NULL;
  PetscInt                                                          *obs_global_idx_host;
  PetscReal                                                         *obs_coords_host_buf;
  Vec                                                               *obs_vecs;
  MPI_Comm                                                           comm;
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace>         vertex_coords_dev;
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace>          obs_coords_dev;
  Kokkos::View<PetscInt *, MemSpace>                                 obs_global_idx_dev;
  Kokkos::View<PetscReal *, MemSpace>                                bd_dev;
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>             row_counts_dev, row_offsets_dev, col_indices_dev;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, MemSpace>          values_dev;
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>    row_counts_host, row_offsets_host, col_indices_host;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace> values_host;
  const PetscReal                                                    cutoff2   = LETKFCutoffSquared(type, radius);
  const PetscReal                                                    cutoff    = PetscSqrtReal(cutoff2);
  const PetscDALETKFLocalizationType                                 kern_type = type;
  const PetscReal                                                    kern_r    = radius;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCall(MatGetLocalSize(H, &n_obs_local, NULL));
  PetscCall(MatGetSize(H, &n_obs_global, NULL));
  for (d = 0; d < 3; ++d) {
    if (xyz[d]) dim++;
    else break;
  }
  PetscCall(VecGetLocalSize(xyz[0], &n_vert_local));

  /* Compute observation coordinates: obs_locs[d] = H * xyz[d] */
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  for (d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, xyz[d], obs_vecs[d]));
  }

  /* Copy vertex coordinates to device */
  vertex_coords_dev = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace>("vertex_coords", n_vert_local, dim);
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

  /* Bbox-pruned obs gather: each rank receives only obs whose location can fall within the kernel
     cutoff of any vertex it owns. Keeps obs_coords device memory bounded by the local working set
     instead of the global obs count. Output buffers are PetscMalloc1'd; we wrap them in Kokkos
     unmanaged host views and deep_copy onto the device. */
  PetscCall(PetscDALETKFGatherObsBbox(dim, xyz, bd, cutoff, H, obs_vecs, &n_obs_cand, &obs_global_idx_host, &obs_coords_host_buf));

  obs_coords_dev     = Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace>("obs_coords", n_obs_cand, dim);
  obs_global_idx_dev = Kokkos::View<PetscInt *, MemSpace>("obs_global_idx", n_obs_cand);
  {
    Kokkos::View<const PetscReal **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> obs_coords_host_view(obs_coords_host_buf, n_obs_cand, dim);
    Kokkos::View<const PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>                        obs_idx_host_view(obs_global_idx_host, n_obs_cand);
    Kokkos::deep_copy(obs_coords_dev, obs_coords_host_view);
    Kokkos::deep_copy(obs_global_idx_dev, obs_idx_host_view);
  }

  /* Copy boundary data to device */
  bd_dev = Kokkos::View<PetscReal *, MemSpace>("bd_dev", dim);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> bd_host("bd_host", dim);
    for (d = 0; d < dim; ++d) bd_host(d) = bd[d];
    Kokkos::deep_copy(bd_dev, bd_host);
  }

  /* Pass 1: Count nnz per row (only entries with positive kernel weight).
     Same trade-off as the host backend: re-evaluate the kernel in Pass 2 rather than
     allocate and compact at the cutoff-bbox upper bound. The arithmetic is cheap on the
     device; the global memory writes saved by exact sizing dominate. */
  row_counts_dev = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("row_counts", n_vert_local);

  Kokkos::parallel_for(
    "CountNnzPerRow", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt dd = 0; dd < dim; ++dd) v_coords[dd] = PetscRealPart(vertex_coords_dev(i, dd));

      PetscInt count = 0;
      for (PetscInt j = 0; j < n_obs_cand; ++j) {
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
  row_counts_host = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("row_counts_host", n_vert_local);
  Kokkos::deep_copy(row_counts_host, row_counts_dev);

  /* Compute prefix sum on host for CSR row offsets */
  row_offsets_dev     = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("row_offsets", n_vert_local + 1);
  row_offsets_host    = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("row_offsets_host", n_vert_local + 1);
  row_offsets_host(0) = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) row_offsets_host(i + 1) = row_offsets_host(i) + row_counts_host(i);
  Kokkos::deep_copy(row_offsets_dev, row_offsets_host);

  /* Total nnz for output arrays. Accumulate in 64-bit and cast so we trip a clear error
     instead of silently wrapping when localization radius * obs density overflows PetscInt. */
  {
    PetscInt64 total_nnz64 = 0;
    for (PetscInt i = 0; i < n_vert_local; ++i) total_nnz64 += (PetscInt64)row_counts_host(i);
    PetscCall(PetscIntCast(total_nnz64, &total_nnz));
  }

  /* Pass 2: Fill column indices and weights */
  col_indices_dev = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("col_indices", total_nnz);
  values_dev      = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, MemSpace>("values", total_nnz);

  Kokkos::parallel_for(
    "FillLocalizationMatrix", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscInt offset = row_offsets_dev(i);

      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt dd = 0; dd < dim; ++dd) v_coords[dd] = PetscRealPart(vertex_coords_dev(i, dd));

      PetscInt pos = 0;
      for (PetscInt j = 0; j < n_obs_cand; ++j) {
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
            col_indices_dev(offset + pos) = obs_global_idx_dev(j);
            values_dev(offset + pos)      = w;
            pos++;
          }
        }
      }
    });
  Kokkos::fence();

  /* Copy results to host */
  col_indices_host = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("col_indices_host", total_nnz);
  values_host      = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace>("values_host", total_nnz);
  Kokkos::deep_copy(col_indices_host, col_indices_dev);
  Kokkos::deep_copy(values_host, values_dev);

  /* Create Q matrix with variable nnz preallocation. Q's row layout matches xyz[0] (one row
     per local vertex); columns index global observations matching H's row layout. */
  PetscCall(MatGetOwnershipRange(H, &cstart, &cend));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, MATAIJKOKKOS));
  PetscCall(MatGetOwnershipRange(*Q, &rstart, NULL));
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
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &local_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscInfo((PetscObject)*Q, "LETKF localization (type=%s, radius=%g): %" PetscInt_FMT " vertices, %" PetscInt_FMT " obs, nnz/row min=%" PetscInt_FMT " max=%" PetscInt_FMT "\n", PetscDALETKFLocalizationTypes[type], (double)radius, n_vert_local, n_obs_global, local_min, local_max));

  /* Cleanup */
  for (d = 0; d < dim; ++d) PetscCall(VecDestroy(&obs_vecs[d]));
  PetscCall(PetscFree(obs_vecs));
  PetscCall(PetscFree(obs_global_idx_host));
  PetscCall(PetscFree(obs_coords_host_buf));

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

  Selected by the `PetscDALETKFCreateLocalizationMat()` dispatcher when the caller requests the Kokkos
  backend (i.e. when `da->R` has a Kokkos matrix type). The host counterpart
  `PetscDALETKFCreateLocalizationMat_AIJ()` produces a numerically identical Q for the polynomial kernels
  (`PETSCDA_LETKF_LOC_GASPARI_COHN`, `PETSCDA_LETKF_LOC_BOXCAR`) and a bit-comparable Q for
  `PETSCDA_LETKF_LOC_GAUSSIAN` modulo `exp()` rounding.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFCreateLocalizationMat_Kokkos(PetscDALETKFLocalizationType type, PetscReal radius, Vec xyz[], PetscReal bd[], Mat H, Mat *Q, PetscInt *max_nnz_local, PetscInt *n_nnz_local)
{
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using MemSpace     = ExecSpace::memory_space;
  using DevScalar2D  = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace>;
  using HostScalar2D = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using DevInt1D     = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>;
  using DevScalar1D  = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, MemSpace>;
  using HostInt1D    = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using HostScalar1D = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  PetscInt                                                  dim, n_vert_local, n_obs_global, n_obs_local, n_obs_cand;
  PetscInt                                                  total_nnz   = 0;
  PetscInt64                                                total_nnz64 = 0;
  PetscInt                                                  row_max     = 0;
  PetscInt                                                 *obs_global_idx_host;
  PetscReal                                                 cutoff, cutoff2;
  PetscReal                                                *obs_coords_host_buf;
  Vec                                                      *obs_vecs;
  DevScalar2D                                               vertex_coords_dev;
  HostScalar2D                                              vertex_coords_host;
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> obs_coords_dev;
  Kokkos::View<PetscInt *, MemSpace>                        obs_global_idx_dev;
  Kokkos::View<PetscReal *, MemSpace>                       bd_dev;
  Kokkos::View<PetscReal *, Kokkos::HostSpace>              bd_host;
  DevInt1D                                                  row_counts_dev, row_offsets_dev, col_indices_dev;
  DevScalar1D                                               values_dev;
  HostInt1D                                                 row_counts_host, row_offsets_host, col_indices_host;
  HostScalar1D                                              values_host;
#if PetscDefined(USE_DEBUG)
  DevInt1D  actual_counts_dev;
  HostInt1D actual_counts_host;
#endif

  PetscFunctionBegin;
  /* Single source of truth for the cutoff policy lives in letkf_kernels.h; see CPU
     dalocalizationletkf.c for why this is computed directly rather than via sqrt(cutoff^2). */
  cutoff  = LETKFCutoff(type, radius);
  cutoff2 = cutoff * cutoff;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatGetLocalSize(H, &n_obs_local, NULL));
  PetscCall(MatGetSize(H, &n_obs_global, NULL));
  PetscCall(PetscDALETKFComputeObsCoords(H, xyz, &dim, &obs_vecs));
  PetscCall(VecGetLocalSize(xyz[0], &n_vert_local));

  /* Copy vertex coordinates to device */
  PetscCallCXX(vertex_coords_dev = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace>("vertex_coords", n_vert_local, dim));
  PetscCallCXX(vertex_coords_host = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace>("vertex_coords_host", n_vert_local, dim));
  for (PetscInt d = 0; d < dim; ++d) {
    const PetscScalar *local_coords_array;
    PetscCall(VecGetArrayRead(xyz[d], &local_coords_array));
    for (PetscInt i = 0; i < n_vert_local; ++i) vertex_coords_host(i, d) = local_coords_array[i];
    PetscCall(VecRestoreArrayRead(xyz[d], &local_coords_array));
  }
  PetscCallCXX(Kokkos::deep_copy(vertex_coords_dev, vertex_coords_host));

  /* Bbox-pruned obs gather: each rank receives only obs whose location can fall within the kernel
     cutoff of any vertex it owns. Keeps obs_coords device memory bounded by the local working set
     instead of the global obs count. Output buffers are PetscMalloc1'd; we wrap them in Kokkos
     unmanaged host views and deep_copy onto the device. */
  PetscCall(PetscDALETKFGatherObsBbox(dim, xyz, bd, cutoff, H, obs_vecs, &n_obs_cand, &obs_global_idx_host, &obs_coords_host_buf));

  PetscCallCXX(obs_coords_dev = Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace>("obs_coords", n_obs_cand, dim));
  PetscCallCXX(obs_global_idx_dev = Kokkos::View<PetscInt *, MemSpace>("obs_global_idx", n_obs_cand));
  PetscCallCXX(Kokkos::deep_copy(obs_coords_dev, Kokkos::View<const PetscReal **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(obs_coords_host_buf, n_obs_cand, dim)));
  PetscCallCXX(Kokkos::deep_copy(obs_global_idx_dev, Kokkos::View<const PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(obs_global_idx_host, n_obs_cand)));

  /* Copy boundary data to device */
  PetscCallCXX(bd_dev = Kokkos::View<PetscReal *, MemSpace>("bd_dev", dim));
  PetscCallCXX(bd_host = Kokkos::View<PetscReal *, Kokkos::HostSpace>("bd_host", dim));
  for (PetscInt d = 0; d < dim; ++d) bd_host(d) = bd[d];
  PetscCallCXX(Kokkos::deep_copy(bd_dev, bd_host));

  /* Pass 1: Count nnz per row (only entries with positive kernel weight).
     Same trade-off as the host backend: re-evaluate the kernel in Pass 2 rather than
     allocate and compact at the cutoff-bbox upper bound. The arithmetic is cheap on the
     device; the global memory writes saved by exact sizing dominate. */
  PetscCallCXX(row_counts_dev = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("row_counts", n_vert_local));

  Kokkos::parallel_for(
    "CountNnzPerRow", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt dd = 0; dd < dim; ++dd) v_coords[dd] = PetscRealPart(vertex_coords_dev(i, dd));

      PetscInt count = 0;
      for (PetscInt j = 0; j < n_obs_cand; ++j) {
        if (LETKFRowWeight(type, radius, cutoff2, dim, v_coords, &obs_coords_dev(j, 0), bd_dev.data()) > 0.0) count++;
      }
      row_counts_dev(i) = count;
    });
  Kokkos::fence();

  /* Copy row counts to host for preallocation */
  PetscCallCXX(row_counts_host = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("row_counts_host", n_vert_local));
  PetscCallCXX(Kokkos::deep_copy(row_counts_host, row_counts_dev));

  /* Compute prefix sum on host for CSR row offsets */
  PetscCallCXX(row_offsets_dev = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("row_offsets", n_vert_local + 1));
  PetscCallCXX(row_offsets_host = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("row_offsets_host", n_vert_local + 1));
  row_offsets_host(0) = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) row_offsets_host(i + 1) = row_offsets_host(i) + row_counts_host(i);
  PetscCallCXX(Kokkos::deep_copy(row_offsets_dev, row_offsets_host));

  /* Total nnz for output arrays and per-rank max nnz for downstream sizing. Accumulate in
     64-bit and cast so we trip a clear error instead of silently wrapping when localization
     radius * obs density overflows PetscInt. The per-rank max feeds PetscDALETKFInstallQ()
     without a downstream MatGetRow walk. */
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    total_nnz64 += (PetscInt64)row_counts_host(i);
    if (row_counts_host(i) > row_max) row_max = row_counts_host(i);
  }
  PetscCall(PetscIntCast(total_nnz64, &total_nnz));

  /* Pass 2: Fill column indices and weights */
  PetscCallCXX(col_indices_dev = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("col_indices", total_nnz));
  PetscCallCXX(values_dev = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, MemSpace>("values", total_nnz));

#if PetscDefined(USE_DEBUG)
  /* Mirror the CPU backend's PetscAssert(pos == row_counts[i]): record the actual Pass-2 write
     count per row in a device View and verify on host that it matches Pass 1. Allocated only in
     debug builds; release builds skip both the View and the per-row write. */
  PetscCallCXX(actual_counts_dev = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, MemSpace>("actual_counts", n_vert_local));
#endif

  Kokkos::parallel_for(
    "FillLocalizationMatrix", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscInt offset = row_offsets_dev(i);

      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt dd = 0; dd < dim; ++dd) v_coords[dd] = PetscRealPart(vertex_coords_dev(i, dd));

      PetscInt pos = 0;
      for (PetscInt j = 0; j < n_obs_cand; ++j) {
        PetscReal w = LETKFRowWeight(type, radius, cutoff2, dim, v_coords, &obs_coords_dev(j, 0), bd_dev.data());
        if (w > 0.0) {
          col_indices_dev(offset + pos) = obs_global_idx_dev(j);
          values_dev(offset + pos)      = w;
          pos++;
        }
      }
#if PetscDefined(USE_DEBUG)
      actual_counts_dev(i) = pos;
#endif
    });
  Kokkos::fence();

#if PetscDefined(USE_DEBUG)
  PetscCallCXX(actual_counts_host = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("actual_counts_host", n_vert_local));
  PetscCallCXX(Kokkos::deep_copy(actual_counts_host, actual_counts_dev));
  for (PetscInt i = 0; i < n_vert_local; ++i)
    PetscCheck(actual_counts_host(i) == row_counts_host(i), PETSC_COMM_SELF, PETSC_ERR_PLIB, "LETKF localization Pass 1/2 mismatch on row %" PetscInt_FMT ": pass1=%" PetscInt_FMT " pass2=%" PetscInt_FMT, i, row_counts_host(i), actual_counts_host(i));
#endif

  /* Copy results to host */
  PetscCallCXX(col_indices_host = Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>("col_indices_host", total_nnz));
  PetscCallCXX(values_host = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace>("values_host", total_nnz));
  PetscCallCXX(Kokkos::deep_copy(col_indices_host, col_indices_dev));
  PetscCallCXX(Kokkos::deep_copy(values_host, values_dev));

  PetscCall(PetscDALETKFAssembleQFromCSR(H, n_vert_local, n_obs_local, n_obs_global, MATAIJKOKKOS, row_counts_host.data(), row_offsets_host.data(), col_indices_host.data(), values_host.data(), Q));
  PetscCall(PetscDALETKFLogQStats(*Q, type, radius, n_vert_local, n_obs_global, row_counts_host.data()));

  *max_nnz_local = row_max;
  *n_nnz_local   = total_nnz;
  PetscCall(PetscDALETKFDestroyObsCoords(dim, &obs_vecs));
  PetscCall(PetscFree(obs_global_idx_host));
  PetscCall(PetscFree(obs_coords_host_buf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

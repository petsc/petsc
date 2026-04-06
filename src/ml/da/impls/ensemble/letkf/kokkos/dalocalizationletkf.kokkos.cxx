#include <petsc.h>
#include <petscmat.h>
#include <petsc_kokkos.hpp>
#include <cmath>
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
static PetscReal GaspariCohn(PetscReal distance, PetscReal radius)
{
  if (radius <= 0.0) return 0.0;
  const PetscReal r = distance / radius;

  if (r >= 2.0) return 0.0;

  const PetscReal r2 = r * r;
  const PetscReal r3 = r2 * r;
  const PetscReal r4 = r3 * r;
  const PetscReal r5 = r4 * r;

  PetscReal val;
  if (r <= 1.0) {
    val = -0.25 * r5 + 0.5 * r4 + 0.625 * r3 - (5.0 / 3.0) * r2 + 1.0;
  } else {
    val = (1.0 / 12.0) * r5 - 0.5 * r4 + 0.625 * r3 + (5.0 / 3.0) * r2 - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
  }
  return val > PETSC_SMALL ? val : 0.0;
}

/*@
  PetscDALETKFGetLocalizationMatrix - Compute localization weight matrix for LETKF using radius-based Gaspari-Cohn weighting

  Collective

  Input Parameters:
+ radius  - Gaspari-Cohn cutoff half-width (must be positive; observations beyond 2*radius get zero weight)
. Vecxyz  - Array of vectors containing the vertex coordinates (one per spatial dimension)
. bd      - Array of domain extents per dimension (used for periodic wrapping; 0 = non-periodic)
- H       - Observation operator matrix

  Output Parameter:
. Q - Localization weight matrix (sparse, AIJ format)

  Level: intermediate

  Notes:
  The output matrix Q has dimensions (n_vert_global x n_obs_global). Each row contains
  a variable number of non-zero entries corresponding to observations within the cutoff
  distance 2*radius, weighted by the Gaspari-Cohn fifth-order piecewise rational function.

  For effectively no localization, use a radius larger than the domain diameter so that
  all observations fall within the cutoff and receive weight close to 1.0.

  Kokkos is required for this routine.

.seealso: [](ch_da), `PetscDALETKFSetLocalization()`, `PetscDALETKFSetLocalizationRadius()`
@*/
PetscErrorCode PetscDALETKFGetLocalizationMatrix(PetscReal radius, Vec Vecxyz[3], PetscReal bd[3], Mat H, Mat *Q)
{
  PetscInt dim = 0, n_vert_local, d, n_obs_global, n_obs_local;
  Vec     *obs_vecs;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscAssertPointer(Q, 5);

  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));
  PetscCheck(radius > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Localization radius must be positive, got %g. Use a large radius for effectively no localization.", (double)radius);
  PetscCall(MatGetLocalSize(H, &n_obs_local, NULL));
  PetscCall(MatGetSize(H, &n_obs_global, NULL));
  for (d = 0; d < 3; ++d) {
    if (Vecxyz[d]) dim++;
    else break;
  }
  PetscCall(VecGetLocalSize(Vecxyz[0], &n_vert_local));
  PetscCheck(dim > 0, comm, PETSC_ERR_ARG_WRONG, "Dim must be > 0");

  /* Compute observation coordinates: obs_locs[d] = H * Vecxyz[d] */
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  for (d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, Vecxyz[d], obs_vecs[d]));
  }

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;

  /* Copy vertex coordinates to device */
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace> vertex_coords_dev("vertex_coords", n_vert_local, dim);
  {
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> vertex_coords_host("vertex_coords_host", n_vert_local, dim);
    for (d = 0; d < dim; ++d) {
      const PetscScalar *local_coords_array;
      PetscCall(VecGetArrayRead(Vecxyz[d], &local_coords_array));
      for (PetscInt i = 0; i < n_vert_local; ++i) vertex_coords_host(i, d) = local_coords_array[i];
      PetscCall(VecRestoreArrayRead(Vecxyz[d], &local_coords_array));
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

  PetscReal cutoff2 = 4.0 * radius * radius; /* (2*radius)^2 */

  /* Pass 1: Count nnz per row (only entries with positive Gaspari-Cohn weight).
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
          if (bd_dev(dd) != 0) {
            PetscReal domain_size = bd_dev(dd);
            if (diff > 0.5 * domain_size) diff -= domain_size;
            else if (diff < -0.5 * domain_size) diff += domain_size;
          }
          dist2 += diff * diff;
        }
        if (dist2 < cutoff2 && GaspariCohn(Kokkos::sqrt(dist2), radius) > 0.0) count++;
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
  PetscInt total_nnz = 0;
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
          if (bd_dev(dd) != 0) {
            PetscReal domain_size = bd_dev(dd);
            if (diff > 0.5 * domain_size) diff -= domain_size;
            else if (diff < -0.5 * domain_size) diff += domain_size;
          }
          dist2 += diff * diff;
        }
        if (dist2 < cutoff2) {
          PetscReal w = GaspariCohn(Kokkos::sqrt(dist2), radius);
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
  PetscInt rstart;
  PetscCall(VecGetOwnershipRange(Vecxyz[0], &rstart, NULL));

  /* Determine column ownership range for diagonal/off-diagonal preallocation split */
  PetscInt    cstart, cend;
  PetscLayout cmap;
  PetscCall(PetscLayoutCreate(comm, &cmap));
  PetscCall(PetscLayoutSetLocalSize(cmap, n_obs_local));
  PetscCall(PetscLayoutSetSize(cmap, n_obs_global));
  PetscCall(PetscLayoutSetUp(cmap));
  PetscCall(PetscLayoutGetRange(cmap, &cstart, &cend));
  PetscCall(PetscLayoutDestroy(&cmap));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, MATAIJKOKKOS));
  PetscInt *d_nnz, *o_nnz;
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
  PetscCall(MatSetFromOptions(*Q));
  PetscCall(MatSetUp(*Q));

  /* Fill matrix row by row */
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt globalRow = rstart + i;
    PetscInt nnz       = row_counts_host(i);
    PetscInt off       = row_offsets_host(i);
    PetscCall(MatSetValues(*Q, 1, &globalRow, nnz, &col_indices_host(off), &values_host(off), INSERT_VALUES));
  }

  /* Log statistics */
  PetscInt local_min = n_obs_global, local_max = 0;
  for (PetscInt i = 0; i < n_vert_local; ++i) {
    if (row_counts_host(i) < local_min) local_min = row_counts_host(i);
    if (row_counts_host(i) > local_max) local_max = row_counts_host(i);
  }
  PetscCall(PetscInfo((PetscObject)*Q, "LETKF localization (radius=%g): %" PetscInt_FMT " vertices, %" PetscInt_FMT " obs, nnz/row min=%" PetscInt_FMT " max=%" PetscInt_FMT "\n", (double)radius, n_vert_local, n_obs_global, local_min, local_max));

  /* Cleanup */
  for (d = 0; d < dim; ++d) PetscCall(VecDestroy(&obs_vecs[d]));
  PetscCall(PetscFree(obs_vecs));

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

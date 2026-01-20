#include <petsc/private/dmpleximpl.h>
#include <petscdmplex.h>
#include <petscmat.h>
#include <petsc_kokkos.hpp>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <Kokkos_Core.hpp>

typedef struct {
  PetscReal distance;
  PetscInt  obs_index;
} DistObsPair;

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

  if (r <= 1.0) {
    // Region [0, 1]
    return -0.25 * r5 + 0.5 * r4 + 0.625 * r3 - (5.0 / 3.0) * r2 + 1.0;
  } else {
    // Region [1, 2]
    return (1.0 / 12.0) * r5 - 0.5 * r4 + 0.625 * r3 + (5.0 / 3.0) * r2 - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
  }
}

/*@
  DMPlexGetLETKFLocalizationMatrix - Compute localization weight matrix for LETKF [move to ml/da/interface]

  Collective

  Input Parameters:
+ n_obs_vertex - Number of nearest observations to use per vertex (eg, MAX_Q_NUM_LOCAL_OBSERVATIONS in LETKF)
. n_obs_local - Number of local observations
. n_dof - Number of degrees of freedom
. Vecxyz - Array of vectors containing the coordinates
- H - Observation operator matrix

  Output Parameter:
. Q - Localization weight matrix (sparse, AIJ format)

  Notes:
  The output matrix Q has dimensions (n_vert_global x n_obs_global) where
  n_vert_global is the number of vertices in the DMPlex. Each row contains
  exactly n_obs_vertex non-zero entries corresponding to the nearest
  observations, weighted by the Gaspari-Cohn fifth-order piecewise
  rational function.

  The observation locations are computed as H * V where V is the vector
  of vertex coordinates. The localization weights ensure smooth tapering
  of observation influence with distance.

  Kokkos is required for this routine.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetLETKFLocalizationMatrix(const PetscInt n_obs_vertex, const PetscInt n_obs_local, const PetscInt n_dof, Vec Vecxyz[3], Mat H, Mat *Q)
{
  PetscInt dim = 0, n_vert_local, d, N, n_obs_global, n_state_local;
  Vec     *obs_vecs;
  MPI_Comm comm;
  PetscInt n_state_global;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(H, MAT_CLASSID, 5);
  PetscAssertPointer(Q, 6);

  PetscCall(PetscKokkosInitializeCheck());

  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));

  /* Infer dim from the number of vectors in Vecxyz */
  for (d = 0; d < 3; ++d) {
    if (Vecxyz[d]) dim++;
    else break;
  }

  PetscCheck(dim > 0, comm, PETSC_ERR_ARG_WRONG, "Dim must be > 0");
  PetscCheck(n_obs_vertex > 0, comm, PETSC_ERR_ARG_WRONG, "n_obs_vertex must be > 0");

  PetscCall(VecGetSize(Vecxyz[0], &n_state_global));
  PetscCall(VecGetLocalSize(Vecxyz[0], &n_state_local));
  n_vert_local = n_state_local / n_dof;

  /* Check H dimensions */
  PetscCall(MatGetSize(H, &n_obs_global, &N));
  PetscCheck(N == n_state_global, comm, PETSC_ERR_ARG_SIZ, "H number of columns %" PetscInt_FMT " != global state size %" PetscInt_FMT, N, n_state_global);
  // If n_obs_global < n_obs_vertex, we will pad with -1 indices and 0.0 weights.
  // This is not an error condition, but rather a case where we have fewer observations than requested neighbors.

  /* Allocate storage for observation locations */
  PetscCall(PetscMalloc1(dim, &obs_vecs));

  /* Compute observation locations per dimension */
  for (d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, Vecxyz[d], obs_vecs[d]));
  }

  /* Create output matrix Q in N/n_dof x P */
  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*Q, n_obs_vertex, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*Q, n_obs_vertex, NULL, n_obs_vertex, NULL));
  PetscCall(MatSetFromOptions(*Q));
  PetscCall(MatSetUp(*Q));

  PetscCall(PetscInfo((PetscObject)*Q, "Computing LETKF localization matrix: %" PetscInt_FMT " vertices, %" PetscInt_FMT " observations, %" PetscInt_FMT " neighbors\n", n_vert_local, n_obs_global, n_obs_vertex));

  /* Prepare Kokkos Views */
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;

  /* Vertex Coordinates */
  // Use LayoutLeft for coalesced access on GPU (i is contiguous)
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace> vertex_coords_dev("vertex_coords", n_vert_local, dim);
  {
    // Host view must match the data layout from VecGetArray (d-major, i-minor implies LayoutLeft for (i,d) view)
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> vertex_coords_host("vertex_coords_host", n_vert_local, dim);
    for (d = 0; d < dim; ++d) {
      const PetscScalar *local_coords_array;
      PetscCall(VecGetArrayRead(Vecxyz[d], &local_coords_array));
      // Copy data. Since vertex_coords_host is LayoutLeft, &vertex_coords_host(0, d) is the start of column d.
      for (PetscInt i = 0; i < n_vert_local; ++i) vertex_coords_host(i, d) = local_coords_array[i];
      PetscCall(VecRestoreArrayRead(Vecxyz[d], &local_coords_array));
    }
    Kokkos::deep_copy(vertex_coords_dev, vertex_coords_host);
  }

  /* Observation Coordinates */
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

  PetscInt rstart;
  PetscCall(VecGetOwnershipRange(Vecxyz[0], &rstart, NULL));

  /* Output Views */
  // LayoutLeft for coalesced access on GPU
  Kokkos::View<PetscInt **, Kokkos::LayoutLeft, MemSpace>    indices_dev("indices", n_vert_local, n_obs_vertex);
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, MemSpace> values_dev("values", n_vert_local, n_obs_vertex);

  /* Temporary storage for top-k per vertex */
  // LayoutLeft for coalesced access on GPU.
  // Note: For the insertion sort within a thread, LayoutRight would offer better cache locality for the thread's private list.
  // However, LayoutLeft is preferred for coalesced access across threads during the final weight computation and initialization.
  // Given the random access nature of the sort (divergence), we stick to the default GPU layout (Left).
  Kokkos::View<PetscReal **, Kokkos::LayoutLeft, MemSpace> best_dists_dev("best_dists", n_vert_local, n_obs_vertex);
  Kokkos::View<PetscInt **, Kokkos::LayoutLeft, MemSpace>  best_idxs_dev("best_idxs", n_vert_local, n_obs_vertex);

  /* Main Kernel */
  Kokkos::parallel_for(
    "ComputeLocalization", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscReal current_max_dist = PETSC_MAX_REAL;

      // Cache vertex coordinates in registers to avoid repeated global memory access
      // dim is small (<= 3), so this fits easily in registers
      PetscReal v_coords[3] = {0.0, 0.0, 0.0};
      for (PetscInt d = 0; d < dim; ++d) v_coords[d] = PetscRealPart(vertex_coords_dev(i, d));

      // Initialize with infinity
      for (PetscInt k = 0; k < n_obs_vertex; ++k) {
        best_dists_dev(i, k) = PETSC_MAX_REAL;
        best_idxs_dev(i, k)  = -1;
      }

      // Iterate over all observations
      for (PetscInt j = 0; j < n_obs_global; ++j) {
        PetscReal dist2 = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
          PetscReal diff = v_coords[d] - obs_coords_dev(j, d);
          dist2 += diff * diff;
        }

        // Check if this observation is closer than the furthest stored observation
        if (dist2 < current_max_dist) {
          // Insert sorted
          PetscInt pos = n_obs_vertex - 1;
          while (pos > 0 && best_dists_dev(i, pos - 1) > dist2) {
            best_dists_dev(i, pos) = best_dists_dev(i, pos - 1);
            best_idxs_dev(i, pos)  = best_idxs_dev(i, pos - 1);
            pos--;
          }
          best_dists_dev(i, pos) = dist2;
          best_idxs_dev(i, pos)  = j;

          // Update current max distance
          current_max_dist = best_dists_dev(i, n_obs_vertex - 1);
        }
      }

      // Compute weights
      PetscReal radius2 = best_dists_dev(i, n_obs_vertex - 1);
      PetscReal radius  = std::sqrt(radius2);
      if (radius == 0.0) radius = 1.0;

      for (PetscInt k = 0; k < n_obs_vertex; ++k) {
        if (best_idxs_dev(i, k) != -1) {
          PetscReal dist    = std::sqrt(best_dists_dev(i, k));
          indices_dev(i, k) = best_idxs_dev(i, k);
          values_dev(i, k)  = GaspariCohn(dist, radius);
        } else {
          indices_dev(i, k) = -1; // Ignore this entry
          values_dev(i, k)  = 0.0;
        }
      }
    });

  /* Copy back to host and fill matrix */
  // Host views must be LayoutRight for MatSetValues (row-major)
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, Kokkos::HostSpace>    indices_host("indices_host", n_vert_local, n_obs_vertex);
  Kokkos::View<PetscScalar **, Kokkos::LayoutRight, Kokkos::HostSpace> values_host("values_host", n_vert_local, n_obs_vertex);

  // Deep copy will handle layout conversion (transpose) if device views are LayoutLeft
  // Note: Kokkos::deep_copy cannot copy between different layouts if the memory spaces are different (e.g. GPU to Host).
  // We need an intermediate mirror view on the host with the same layout as the device view.
  Kokkos::View<PetscInt **, Kokkos::LayoutLeft, Kokkos::HostSpace>    indices_host_left = Kokkos::create_mirror_view(indices_dev);
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> values_host_left  = Kokkos::create_mirror_view(values_dev);

  Kokkos::deep_copy(indices_host_left, indices_dev);
  Kokkos::deep_copy(values_host_left, values_dev);

  // Now copy from LayoutLeft host view to LayoutRight host view
  Kokkos::deep_copy(indices_host, indices_host_left);
  Kokkos::deep_copy(values_host, values_host_left);

  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt globalRow = rstart + i;
    PetscCall(MatSetValues(*Q, 1, &globalRow, n_obs_vertex, &indices_host(i, 0), &values_host(i, 0), INSERT_VALUES));
  }

  /* Cleanup Phase 2 storage */
  for (d = 0; d < dim; ++d) PetscCall(VecDestroy(&obs_vecs[d]));
  PetscCall(PetscFree(obs_vecs));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

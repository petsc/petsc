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
  PetscReal r      = distance / radius;
  PetscReal weight = 0.0;

  if (r >= 2.0) {
    weight = 0.0;
  } else if (r >= 1.0) {
    // Region [1, 2]
    PetscReal r2 = r * r;
    PetscReal r3 = r2 * r;
    PetscReal r4 = r3 * r;
    PetscReal r5 = r4 * r;
    weight       = (1.0 / 12.0) * r5 - (0.5) * r4 + (0.625) * r3 + (5.0 / 3.0) * r2 - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
  } else {
    // Region [0, 1]
    PetscReal r2 = r * r;
    PetscReal r3 = r2 * r;
    PetscReal r4 = r3 * r;
    PetscReal r5 = r4 * r;
    weight       = -0.25 * r5 + 0.5 * r4 + 0.625 * r3 - (5.0 / 3.0) * r2 + 1.0;
  }
  return weight;
}

/*@
  DMPlexGetLETKFLocalizationMatrix - Compute localization weight matrix for LETKF

  Collective

  Input Parameters:
+ plex - The DMPlex object
. numobservations - Number of nearest observations to use per vertex
. numglobalobs - Total number of observations
- H - Observation operator matrix

  Output Parameter:
. Q - Localization weight matrix (sparse, AIJ format)

  Notes:
  The output matrix Q has dimensions (numVertices x numglobalobs) where
  numVertices is the number of vertices in the DMPlex. Each row contains
  exactly numobservations non-zero entries corresponding to the nearest
  observations, weighted by the Gaspari-Cohn fifth-order piecewise
  rational function.

  The observation locations are computed as H * V where V is the vector
  of vertex coordinates. The localization weights ensure smooth tapering
  of observation influence with distance.

  Kokkos is required for this routine. LETKF has a lot of fine grain parallelism and is not useful without threads or GPUs.

  Level: intermediate

.seealso: `DMPLEX`, `DMPlexGetDepthStratum()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMPlexGetLETKFLocalizationMatrix(DM plex, PetscInt numobservations, PetscInt numglobalobs, Mat H, Mat *Q)
{
  PetscInt      dim, vStart, vEnd, numVertices, d;
  PetscInt      M, N;
  Vec           coordinates;
  Vec          *obs_vecs;
  PetscScalar **obs_coords;
  PetscInt      localRows, globalRows;
  MPI_Comm      comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(plex, DM_CLASSID, 1);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscAssertPointer(Q, 5);

  PetscCall(PetscKokkosInitializeCheck());

  PetscCall(PetscObjectGetComm((PetscObject)plex, &comm));
  PetscCall(DMGetCoordinateDim(plex, &dim));
  PetscCall(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
  numVertices = vEnd - vStart;

  /* Check H dimensions */
  PetscCall(MatGetSize(H, &M, &N));
  PetscCheck(M == numglobalobs, comm, PETSC_ERR_ARG_SIZ, "H matrix rows %" PetscInt_FMT " != numglobalobs %" PetscInt_FMT, M, numglobalobs);

  PetscCall(DMGetCoordinates(plex, &coordinates));
  PetscCheck(coordinates, comm, PETSC_ERR_ARG_WRONGSTATE, "DM must have coordinates");

  /* Allocate storage for observation locations */
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  PetscCall(PetscMalloc1(dim, &obs_coords));

  /* Compute observation locations per dimension */
  for (d = 0; d < dim; ++d) {
    Vec coord_comp;
    PetscCall(MatCreateVecs(H, &coord_comp, &obs_vecs[d]));
    PetscCall(VecStrideGather(coordinates, d, coord_comp, INSERT_VALUES));
    PetscCall(MatMult(H, coord_comp, obs_vecs[d]));
    PetscCall(VecGetArray(obs_vecs[d], &obs_coords[d]));
    PetscCall(VecDestroy(&coord_comp));
  }

  /* Create output matrix Q */
  localRows = numVertices;
  PetscCallMPI(MPIU_Allreduce(&localRows, &globalRows, 1, MPIU_INT, MPI_SUM, comm));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, localRows, PETSC_DECIDE, globalRows, numglobalobs));
  PetscCall(MatSetType(*Q, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocation(*Q, numobservations, NULL, numobservations, NULL));
  PetscCall(MatSetUp(*Q));

  /* Prepare Kokkos Views */
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;

  /* Vertex Coordinates */
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> vertex_coords_dev("vertex_coords", numVertices, dim);
  {
    Kokkos::View<PetscReal **, Kokkos::LayoutRight, Kokkos::HostSpace> vertex_coords_host("vertex_coords_host", numVertices, dim);
    Vec                                                                localCoords;
    PetscScalar                                                       *local_coords_array;
    PetscSection                                                       coordSection;
    PetscCall(DMGetCoordinatesLocal(plex, &localCoords));
    PetscCall(DMGetCoordinateSection(plex, &coordSection));
    PetscCall(VecGetArray(localCoords, &local_coords_array));

    for (PetscInt v = 0; v < numVertices; ++v) {
      PetscInt off;
      PetscCall(PetscSectionGetOffset(coordSection, vStart + v, &off));
      for (d = 0; d < dim; ++d) vertex_coords_host(v, d) = PetscRealPart(local_coords_array[off + d]);
    }
    PetscCall(VecRestoreArray(localCoords, &local_coords_array));
    Kokkos::deep_copy(vertex_coords_dev, vertex_coords_host);
  }

  /* Observation Coordinates */
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> obs_coords_dev("obs_coords", numglobalobs, dim);
  {
    Kokkos::View<PetscReal **, Kokkos::LayoutRight, Kokkos::HostSpace> obs_coords_host("obs_coords_host", numglobalobs, dim);
    for (PetscInt j = 0; j < numglobalobs; ++j) {
      for (d = 0; d < dim; ++d) obs_coords_host(j, d) = PetscRealPart(obs_coords[d][j]);
    }
    Kokkos::deep_copy(obs_coords_dev, obs_coords_host);
  }

  /* Global Rows */
  Kokkos::View<PetscInt *, MemSpace> global_rows_dev("global_rows", numVertices);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> global_rows_host("global_rows_host", numVertices);
    PetscSection                                globalSection;
    PetscCall(DMGetGlobalSection(plex, &globalSection));
    for (PetscInt v = 0; v < numVertices; ++v) {
      PetscInt globalRow;
      PetscCall(PetscSectionGetOffset(globalSection, vStart + v, &globalRow));
      global_rows_host(v) = globalRow;
    }
    Kokkos::deep_copy(global_rows_dev, global_rows_host);
  }

  /* Output Views */
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, MemSpace>    indices_dev("indices", numVertices, numobservations);
  Kokkos::View<PetscScalar **, Kokkos::LayoutRight, MemSpace> values_dev("values", numVertices, numobservations);

  /* Temporary storage for top-k per vertex */
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> best_dists_dev("best_dists", numVertices, numobservations);
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, MemSpace>  best_idxs_dev("best_idxs", numVertices, numobservations);

  Kokkos::deep_copy(best_dists_dev, 1.0e30);

  /* Main Kernel */
  Kokkos::parallel_for(
    "ComputeLocalization", Kokkos::RangePolicy<ExecSpace>(0, numVertices), KOKKOS_LAMBDA(const PetscInt i) {
      PetscReal current_max_dist = 1.0e30;
      PetscInt  count            = 0;

      // Iterate over all observations
      for (PetscInt j = 0; j < numglobalobs; ++j) {
        PetscReal dist2 = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
          PetscReal diff = vertex_coords_dev(i, d) - obs_coords_dev(j, d);
          dist2 += diff * diff;
        }

        if (count < numobservations) {
          // Insert sorted
          PetscInt pos = count;
          while (pos > 0 && best_dists_dev(i, pos - 1) > dist2) {
            best_dists_dev(i, pos) = best_dists_dev(i, pos - 1);
            best_idxs_dev(i, pos)  = best_idxs_dev(i, pos - 1);
            pos--;
          }
          best_dists_dev(i, pos) = dist2;
          best_idxs_dev(i, pos)  = j;
          count++;
          if (count == numobservations) current_max_dist = best_dists_dev(i, numobservations - 1);
        } else if (dist2 < current_max_dist) {
          // Insert sorted
          PetscInt pos = numobservations - 1;
          while (pos > 0 && best_dists_dev(i, pos - 1) > dist2) {
            best_dists_dev(i, pos) = best_dists_dev(i, pos - 1);
            best_idxs_dev(i, pos)  = best_idxs_dev(i, pos - 1);
            pos--;
          }
          best_dists_dev(i, pos) = dist2;
          best_idxs_dev(i, pos)  = j;
          current_max_dist       = best_dists_dev(i, numobservations - 1);
        }
      }

      // Compute weights
      PetscReal radius2 = best_dists_dev(i, numobservations - 1);
      PetscReal radius  = std::sqrt(radius2);
      if (radius == 0.0) radius = 1.0;

      for (PetscInt k = 0; k < numobservations; ++k) {
        PetscReal dist    = std::sqrt(best_dists_dev(i, k));
        indices_dev(i, k) = best_idxs_dev(i, k);
        values_dev(i, k)  = GaspariCohn(dist, radius);
      }
    });

  /* Copy back to host and fill matrix */
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, Kokkos::HostSpace>    indices_host     = Kokkos::create_mirror_view(indices_dev);
  Kokkos::View<PetscScalar **, Kokkos::LayoutRight, Kokkos::HostSpace> values_host      = Kokkos::create_mirror_view(values_dev);
  Kokkos::View<PetscInt *, Kokkos::HostSpace>                          global_rows_host = Kokkos::create_mirror_view(global_rows_dev);

  Kokkos::deep_copy(indices_host, indices_dev);
  Kokkos::deep_copy(values_host, values_dev);
  Kokkos::deep_copy(global_rows_host, global_rows_dev);

  for (PetscInt i = 0; i < numVertices; ++i) {
    PetscInt globalRow = global_rows_host(i);
    PetscCall(MatSetValues(*Q, 1, &globalRow, numobservations, &indices_host(i, 0), &values_host(i, 0), INSERT_VALUES));
  }

  /* Cleanup Phase 2 storage */
  for (d = 0; d < dim; ++d) {
    PetscCall(VecRestoreArray(obs_vecs[d], &obs_coords[d]));
    PetscCall(VecDestroy(&obs_vecs[d]));
  }
  PetscCall(PetscFree(obs_vecs));
  PetscCall(PetscFree(obs_coords));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

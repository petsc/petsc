static const char help[] = "Test KDTree\n\n";

#include <petsc.h>

static inline PetscReal Distance(PetscInt dim, const PetscReal *PETSC_RESTRICT x, const PetscReal *PETSC_RESTRICT y)
{
  PetscReal dist = 0;
  for (PetscInt j = 0; j < dim; j++) dist += PetscSqr(x[j] - y[j]);
  return PetscSqrtReal(dist);
}

int main(int argc, char **argv)
{
  MPI_Comm      comm;
  PetscInt      num_coords, dim, num_rand_points = 0, bucket_size = PETSC_DECIDE;
  PetscRandom   random;
  PetscReal    *coords;
  PetscInt      coords_size, num_points_queried = 0, num_trees_built = 0, loops = 1;
  PetscBool     view_tree = PETSC_FALSE, view_performance = PETSC_TRUE, test_tree_points = PETSC_FALSE, test_rand_points = PETSC_FALSE, query_rand_points = PETSC_FALSE;
  PetscCopyMode copy_mode = PETSC_OWN_POINTER;
  PetscKDTree   tree;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogDefaultBegin()); // So we can use PetscLogEventGetPerfInfo without -log_view
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Test Options", "none");
  PetscCall(PetscOptionsInt("-num_coords", "Number of coordinates", "", PETSC_FALSE, &num_coords, NULL));
  PetscCall(PetscOptionsInt("-dim", "Dimension of the coordinates", "", PETSC_FALSE, &dim, NULL));
  PetscCall(PetscOptionsInt("-bucket_size", "Size of the bucket at each leaf", "", bucket_size, &bucket_size, NULL));
  PetscCall(PetscOptionsBoundedInt("-loops", "Number of times to loop through KDTree creation and querying", "", loops, &loops, NULL, 1));
  PetscCall(PetscOptionsEnum("-copy_mode", "Copy mode to use with KDTree", "PetscKDTreeCreate", PetscCopyModes, (PetscEnum)copy_mode, (PetscEnum *)&copy_mode, NULL));
  PetscCall(PetscOptionsBool("-view_tree", "View the KDTree", "", view_tree, &view_tree, NULL));
  PetscCall(PetscOptionsBool("-view_performance", "View the performance speed of the KDTree", "", view_performance, &view_performance, NULL));
  PetscCall(PetscOptionsBool("-test_tree_points", "Test querying tree points against itself", "", test_tree_points, &test_tree_points, NULL));
  PetscCall(PetscOptionsBool("-test_rand_points", "Test querying random points via brute force", "", test_rand_points, &test_rand_points, NULL));
  PetscCall(PetscOptionsBool("-query_rand_points", "Query querying random points", "", query_rand_points, &query_rand_points, NULL));
  if (test_rand_points || query_rand_points) PetscCall(PetscOptionsInt("-num_rand_points", "Number of random points to test with", "", num_rand_points, &num_rand_points, NULL));
  PetscOptionsEnd();

  coords_size = num_coords * dim;
  PetscCall(PetscMalloc1(coords_size, &coords));
  PetscCall(PetscRandomCreate(comm, &random));

  for (PetscInt loop_count = 0; loop_count < loops; loop_count++) {
    PetscCall(PetscRandomGetValuesReal(random, coords_size, coords));

    PetscCall(PetscKDTreeCreate(num_coords, dim, coords, copy_mode, bucket_size, &tree));
    num_trees_built++;
    if (view_tree) PetscCall(PetscKDTreeView(tree, NULL));

    if (test_tree_points) { // Test querying the current coordinates
      PetscCount *indices;
      PetscReal  *distances;
      num_points_queried += num_coords;

      PetscCall(PetscMalloc2(num_coords, &indices, num_coords, &distances));
      PetscCall(PetscKDTreeQueryPointsNearestNeighbor(tree, num_coords, coords, PETSC_MACHINE_EPSILON * 1e3, indices, distances));
      for (PetscInt i = 0; i < num_coords; i++) {
        if (indices[i] != i) PetscCall(PetscPrintf(comm, "Query failed for coord %" PetscInt_FMT ", query returned index %" PetscCount_FMT " with distance %g\n", i, indices[i], (double)distances[i]));
      }
      PetscCall(PetscFree2(indices, distances));
    }

    if (num_rand_points > 0) {
      PetscCount *indices;
      PetscReal  *distances;
      PetscReal  *rand_points;
      PetscInt    rand_queries_size = num_rand_points * dim;

      num_points_queried += num_rand_points;
      PetscCall(PetscMalloc3(rand_queries_size, &rand_points, num_rand_points, &indices, num_rand_points, &distances));
      PetscCall(PetscRandomGetValuesReal(random, rand_queries_size, rand_points));
      PetscCall(PetscKDTreeQueryPointsNearestNeighbor(tree, num_rand_points, rand_points, 0., indices, distances));

      if (test_rand_points) {
        for (PetscInt i = 0; i < num_rand_points; i++) {
          PetscReal *rand_point = &rand_points[dim * i], nearest_distance = PETSC_MAX_REAL;
          PetscInt   index = -1;
          for (PetscInt j = 0; j < num_coords; j++) {
            PetscReal dist = Distance(dim, rand_point, &coords[dim * j]);
            if (dist < nearest_distance) {
              nearest_distance = dist;
              index            = j;
            }
          }
          if (indices[i] != index)
            PetscCall(PetscPrintf(comm, "Query failed for random point %" PetscInt_FMT ". Query returned index %" PetscCount_FMT " with distance %g, but coordinate %" PetscInt_FMT " with distance %g is closer\n", i, indices[i], (double)distances[i], index, (double)nearest_distance));
        }
      }
      PetscCall(PetscFree3(rand_points, indices, distances));
    }
  }

  if (view_performance) {
    PetscLogEvent      kdtree_build_log, kdtree_query_log;
    PetscEventPerfInfo build_perf_info, query_perf_info;

    PetscCall(PetscLogEventGetId("KDTreeBuild", &kdtree_build_log));
    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, kdtree_build_log, &build_perf_info));
    PetscCall(PetscLogEventGetId("KDTreeQuery", &kdtree_query_log));
    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, kdtree_query_log, &query_perf_info));
    PetscCall(PetscPrintf(comm, "KDTreeBuild %.6e for %" PetscInt_FMT " trees built\n", build_perf_info.time, num_trees_built));
    PetscCall(PetscPrintf(comm, "\tTime per tree: %.6e\n", build_perf_info.time / num_trees_built));
    PetscCall(PetscPrintf(comm, "KDTreeQuery %.6e for %" PetscCount_FMT " queries\n", query_perf_info.time, (PetscCount)num_points_queried));
    PetscCall(PetscPrintf(comm, "\tTime per query: %.6e\n", query_perf_info.time / num_points_queried));
  }

  if (copy_mode != PETSC_OWN_POINTER) PetscCall(PetscFree(coords));
  PetscCall(PetscKDTreeDestroy(&tree));
  PetscCall(PetscRandomDestroy(&random));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    suffix: kdtree
    args: -num_coords 35 -test_tree_points -test_rand_points -num_rand_points 300 -bucket_size 13 -view_performance false -view_tree true -kdtree_debug
    test:
      suffix: 1D
      args: -dim 1
    test:
      suffix: 2D
      args: -dim 2
    test:
      suffix: 3D
      args: -dim 3
    test:
      suffix: 3D_small
      args: -dim 3 -num_coords 13

  testset:
    suffix: kdtree_3D_large
    args: -dim 3 -num_coords 350 -test_tree_points -test_rand_points -num_rand_points 300 -view_performance false -kdtree_debug
    output_file: output/empty.out
    test:
    test:
      suffix: copy
      args: -copy_mode copy_values
    test:
      suffix: use
      args: -copy_mode use_pointer
TEST*/

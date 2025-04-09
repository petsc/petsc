#include <petscviewer.h>
#include <petscis.h>
#include <petsc/private/petscimpl.h>

// For accessing bitwise Boolean values in are_handles_leaves
#define GREATER_BIT    0
#define LESS_EQUAL_BIT 1

typedef struct {
  uint8_t    axis;
  char       are_handles_leaves;
  PetscReal  split;
  PetscCount greater_handle, less_equal_handle;
} KDStem;

typedef struct {
  PetscInt   count;
  PetscCount indices_handle, coords_handle;
} KDLeaf;

struct _n_PetscKDTree {
  PetscInt dim;
  PetscInt max_bucket_size;

  PetscBool  is_root_leaf;
  PetscCount root_handle;

  PetscCount       num_coords, num_leaves, num_stems, num_bucket_indices;
  const PetscReal *coords, *coords_owned; // Only free owned on Destroy
  KDLeaf          *leaves;
  KDStem          *stems;
  PetscCount      *bucket_indices;
};

/*@C
  PetscKDTreeDestroy - destroy a `PetscKDTree`

  Not Collective, No Fortran Support

  Input Parameters:
. tree - tree to destroy

  Level: advanced

.seealso: `PetscKDTree`, `PetscKDTreeCreate()`
@*/
PetscErrorCode PetscKDTreeDestroy(PetscKDTree *tree)
{
  PetscFunctionBeginUser;
  if (*tree == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree((*tree)->stems));
  PetscCall(PetscFree((*tree)->leaves));
  PetscCall(PetscFree((*tree)->bucket_indices));
  PetscCall(PetscFree((*tree)->coords_owned));
  PetscCall(PetscFree(*tree));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscLogEvent         PetscKDTree_Build, PetscKDTree_Query;
static PetscErrorCode PetscKDTreeRegisterLogEvents(void)
{
  static PetscBool is_initialized = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (is_initialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventRegister("KDTreeBuild", IS_CLASSID, &PetscKDTree_Build));
  PetscCall(PetscLogEventRegister("KDTreeQuery", IS_CLASSID, &PetscKDTree_Query));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// From http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
static inline uint32_t RoundToNextPowerOfTwo(uint32_t v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

typedef struct {
  uint8_t     initial_axis;
  PetscKDTree tree;
} *KDTreeSortContext;

// Sort nodes based on "superkey"
// See "Building a Balanced k-d Tree in O(kn log n) Time" https://jcgt.org/published/0004/01/03/
static inline int PetscKDTreeSortFunc(PetscCount left, PetscCount right, PetscKDTree tree, uint8_t axis)
{
  const PetscReal *coords = tree->coords;
  const PetscInt   dim    = tree->dim;

  for (PetscInt i = 0; i < dim; i++) {
    PetscReal diff = coords[left * dim + axis] - coords[right * dim + axis];
    if (PetscUnlikely(diff == 0)) {
      axis = (axis + 1) % dim;
      continue;
    } else return PetscSign(diff);
  }
  return 0; // All components are the same
}

static int PetscKDTreeTimSort(const void *l, const void *r, void *ctx)
{
  KDTreeSortContext kd_ctx = (KDTreeSortContext)ctx;
  return PetscKDTreeSortFunc(*(PetscCount *)l, *(PetscCount *)r, kd_ctx->tree, kd_ctx->initial_axis);
}

static PetscErrorCode PetscKDTreeVerifySortedIndices(PetscKDTree tree, PetscCount sorted_indices[], PetscCount temp[], PetscCount start, PetscCount end)
{
  PetscCount num_coords = tree->num_coords, range_size = end - start, location;
  PetscBool  has_duplicates;

  PetscFunctionBeginUser;
  PetscCall(PetscArraycpy(temp, &sorted_indices[0 * num_coords + start], range_size));
  PetscCall(PetscSortCount(range_size, temp));
  PetscCall(PetscSortedCheckDupsCount(range_size, temp, &has_duplicates));
  PetscCheck(has_duplicates == PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Sorted indices must have unique entries, but found duplicates");
  for (PetscInt d = 1; d < tree->dim; d++) {
    for (PetscCount i = start; i < end; i++) {
      PetscCall(PetscFindCount(sorted_indices[d * num_coords + i], range_size, temp, &location));
      PetscCheck(location > -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Sorted indices are not consistent. Could not find %" PetscCount_FMT " from %" PetscInt_FMT " dimensional index in 0th dimension", sorted_indices[d * num_coords + i], d);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscKDTree    tree;
  PetscSegBuffer stems, leaves, bucket_indices, bucket_coords;
  PetscBool      debug_build, copy_coords;
} *KDTreeBuild;

// The range is end exclusive, so [start,end).
static PetscErrorCode PetscKDTreeBuildStemAndLeaves(KDTreeBuild kd_build, PetscCount sorted_indices[], PetscCount temp[], PetscCount start, PetscCount end, PetscInt depth, PetscBool *is_node_leaf, PetscCount *node_handle)
{
  PetscKDTree tree = kd_build->tree;
  PetscInt    dim  = tree->dim;

  PetscFunctionBeginUser;
  PetscCheck(start <= end, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Start %" PetscCount_FMT " must be less than or equal to end %" PetscCount_FMT, start, end);
  if (kd_build->debug_build) PetscCall(PetscKDTreeVerifySortedIndices(tree, sorted_indices, temp, start, end));
  if (end - start <= tree->max_bucket_size) {
    KDLeaf     *leaf;
    PetscCount *bucket_indices;

    PetscCall(PetscSegBufferGetSize(kd_build->leaves, node_handle));
    PetscCall(PetscSegBufferGet(kd_build->leaves, 1, &leaf));
    PetscCall(PetscMemzero(leaf, sizeof(KDLeaf)));
    *is_node_leaf = PETSC_TRUE;

    PetscCall(PetscIntCast(end - start, &leaf->count));
    PetscCall(PetscSegBufferGetSize(kd_build->bucket_indices, &leaf->indices_handle));
    PetscCall(PetscSegBufferGet(kd_build->bucket_indices, leaf->count, &bucket_indices));
    PetscCall(PetscArraycpy(bucket_indices, &sorted_indices[start], leaf->count));
    if (kd_build->copy_coords) {
      PetscReal *bucket_coords;
      PetscCall(PetscSegBufferGetSize(kd_build->bucket_coords, &leaf->coords_handle));
      PetscCall(PetscSegBufferGet(kd_build->bucket_coords, leaf->count * dim, &bucket_coords));
      // Coords are saved in axis-major ordering for better vectorization
      for (PetscCount i = 0; i < leaf->count; i++) {
        for (PetscInt d = 0; d < dim; d++) bucket_coords[d * leaf->count + i] = tree->coords[bucket_indices[i] * dim + d];
      }
    } else leaf->coords_handle = -1;
  } else {
    KDStem    *stem;
    PetscCount num_coords = tree->num_coords;
    uint8_t    axis       = (uint8_t)(depth % dim);
    PetscBool  is_greater_leaf, is_less_equal_leaf;
    PetscCount median     = start + PetscCeilInt64(end - start, 2) - 1, lower;
    PetscCount median_idx = sorted_indices[median], medianp1_idx = sorted_indices[median + 1];

    PetscCall(PetscSegBufferGetSize(kd_build->stems, node_handle));
    PetscCall(PetscSegBufferGet(kd_build->stems, 1, &stem));
    PetscCall(PetscMemzero(stem, sizeof(KDStem)));
    *is_node_leaf = PETSC_FALSE;

    stem->axis = axis;
    // Place split halfway between the "boundary" nodes of the partitioning
    stem->split = (tree->coords[tree->dim * median_idx + axis] + tree->coords[tree->dim * medianp1_idx + axis]) / 2;
    PetscCall(PetscArraycpy(temp, &sorted_indices[0 * num_coords + start], end - start));
    lower = median; // Set lower in case dim == 1
    for (PetscInt d = 1; d < tree->dim; d++) {
      PetscCount upper = median;
      lower            = start - 1;
      for (PetscCount i = start; i < end; i++) {
        // In case of duplicate coord point equal to the median coord point, limit lower partition to median, ensuring balanced tree
        if (lower < median && PetscKDTreeSortFunc(sorted_indices[d * num_coords + i], median_idx, tree, axis) <= 0) {
          sorted_indices[(d - 1) * num_coords + (++lower)] = sorted_indices[d * num_coords + i];
        } else {
          sorted_indices[(d - 1) * num_coords + (++upper)] = sorted_indices[d * num_coords + i];
        }
      }
      PetscCheck(lower == median, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Partitioning into less_equal bin failed. Range upper bound should be %" PetscCount_FMT " but partitioning resulted in %" PetscCount_FMT, median, lower);
      PetscCheck(upper == end - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Partitioning into greater bin failed. Range upper bound should be %" PetscCount_FMT " but partitioning resulted in %" PetscCount_FMT, upper, end - 1);
    }
    PetscCall(PetscArraycpy(&sorted_indices[(tree->dim - 1) * num_coords + start], temp, end - start));

    PetscCall(PetscKDTreeBuildStemAndLeaves(kd_build, sorted_indices, temp, start, lower + 1, depth + 1, &is_less_equal_leaf, &stem->less_equal_handle));
    if (is_less_equal_leaf) PetscCall(PetscBTSet(&stem->are_handles_leaves, LESS_EQUAL_BIT));
    PetscCall(PetscKDTreeBuildStemAndLeaves(kd_build, sorted_indices, temp, lower + 1, end, depth + 1, &is_greater_leaf, &stem->greater_handle));
    if (is_greater_leaf) PetscCall(PetscBTSet(&stem->are_handles_leaves, GREATER_BIT));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscKDTreeCreate - create a `PetscKDTree`

  Not Collective, No Fortran Support

  Input Parameters:
+ num_coords      - number of coordinate points to build the `PetscKDTree`
. dim             - the dimension of the coordinates
. coords          - array of the coordinates, in point-major order
. copy_mode       - behavior handling `coords`, `PETSC_COPY_VALUES` generally more performant
- max_bucket_size - maximum number of points stored at each leaf

  Output Parameter:
. new_tree - the resulting `PetscKDTree`

  Level: advanced

  Note:
  When `copy_mode == PETSC_COPY_VALUES`, the coordinates are copied and organized to optimize vectorization and cache-coherency.
  It is recommended to run this way if the extra memory use is not a concern and it has very little impact on the `PetscKDTree` creation time.

  Developer Note:
  Building algorithm detailed in 'Building a Balanced k-d Tree in O(kn log n) Time' Brown, 2015

.seealso: `PetscKDTree`, `PetscKDTreeDestroy()`, `PetscKDTreeQueryPointsNearestNeighbor()`
@*/
PetscErrorCode PetscKDTreeCreate(PetscCount num_coords, PetscInt dim, const PetscReal coords[], PetscCopyMode copy_mode, PetscInt max_bucket_size, PetscKDTree *new_tree)
{
  PetscKDTree tree;
  PetscCount *sorted_indices, *temp;

  PetscFunctionBeginUser;
  PetscCall(PetscKDTreeRegisterLogEvents());
  PetscCall(PetscLogEventBegin(PetscKDTree_Build, 0, 0, 0, 0));
  PetscCheck(dim > 0, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Dimension of PetscKDTree must be greater than 0, received %" PetscInt_FMT, dim);
  PetscCheck(num_coords > -1, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Number of coordinates may not be negative, received %" PetscCount_FMT, num_coords);
  if (num_coords == 0) {
    *new_tree = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssertPointer(coords, 3);
  PetscAssertPointer(new_tree, 6);
  PetscCall(PetscNew(&tree));
  tree->dim             = dim;
  tree->max_bucket_size = max_bucket_size == PETSC_DECIDE ? 32 : max_bucket_size;
  tree->num_coords      = num_coords;

  switch (copy_mode) {
  case PETSC_OWN_POINTER:
    tree->coords_owned = coords; // fallthrough
  case PETSC_USE_POINTER:
    tree->coords = coords;
    break;
  case PETSC_COPY_VALUES:
    PetscCall(PetscMalloc1(num_coords * dim, &tree->coords_owned));
    PetscCall(PetscArraycpy((PetscReal *)tree->coords_owned, coords, num_coords * dim));
    tree->coords = tree->coords_owned;
    break;
  }

  KDTreeSortContext kd_ctx;
  PetscCall(PetscMalloc2(num_coords * dim, &sorted_indices, num_coords, &temp));
  PetscCall(PetscNew(&kd_ctx));
  kd_ctx->tree = tree;
  for (PetscInt j = 0; j < dim; j++) {
    for (PetscCount i = 0; i < num_coords; i++) sorted_indices[num_coords * j + i] = i;
    kd_ctx->initial_axis = (uint8_t)j;
    PetscCall(PetscTimSort((PetscInt)num_coords, &sorted_indices[num_coords * j], sizeof(*sorted_indices), PetscKDTreeTimSort, kd_ctx));
  }
  PetscCall(PetscFree(kd_ctx));

  PetscInt    num_leaves = (PetscInt)PetscCeilInt64(num_coords, tree->max_bucket_size);
  PetscInt    num_stems  = RoundToNextPowerOfTwo((uint32_t)num_leaves);
  KDTreeBuild kd_build;
  PetscCall(PetscNew(&kd_build));
  kd_build->tree        = tree;
  kd_build->copy_coords = copy_mode == PETSC_COPY_VALUES ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-kdtree_debug", &kd_build->debug_build, NULL));
  PetscCall(PetscSegBufferCreate(sizeof(KDStem), num_stems, &kd_build->stems));
  PetscCall(PetscSegBufferCreate(sizeof(KDLeaf), num_leaves, &kd_build->leaves));
  PetscCall(PetscSegBufferCreate(sizeof(PetscCount), num_coords, &kd_build->bucket_indices));
  if (kd_build->copy_coords) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), num_coords * dim, &kd_build->bucket_coords));

  PetscCall(PetscKDTreeBuildStemAndLeaves(kd_build, sorted_indices, temp, 0, num_coords, 0, &tree->is_root_leaf, &tree->root_handle));

  PetscCall(PetscSegBufferGetSize(kd_build->stems, &tree->num_stems));
  PetscCall(PetscSegBufferGetSize(kd_build->leaves, &tree->num_leaves));
  PetscCall(PetscSegBufferGetSize(kd_build->bucket_indices, &tree->num_bucket_indices));
  PetscCall(PetscSegBufferExtractAlloc(kd_build->stems, &tree->stems));
  PetscCall(PetscSegBufferExtractAlloc(kd_build->leaves, &tree->leaves));
  PetscCall(PetscSegBufferExtractAlloc(kd_build->bucket_indices, &tree->bucket_indices));
  if (kd_build->copy_coords) {
    PetscCall(PetscFree(tree->coords_owned));
    PetscCall(PetscSegBufferExtractAlloc(kd_build->bucket_coords, &tree->coords_owned));
    tree->coords = tree->coords_owned;
    PetscCall(PetscSegBufferDestroy(&kd_build->bucket_coords));
  }
  PetscCall(PetscSegBufferDestroy(&kd_build->stems));
  PetscCall(PetscSegBufferDestroy(&kd_build->leaves));
  PetscCall(PetscSegBufferDestroy(&kd_build->bucket_indices));
  PetscCall(PetscFree(kd_build));
  PetscCall(PetscFree2(sorted_indices, temp));
  *new_tree = tree;
  PetscCall(PetscLogEventEnd(PetscKDTree_Build, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscReal PetscSquareDistance(PetscInt dim, const PetscReal *PETSC_RESTRICT x, const PetscReal *PETSC_RESTRICT y)
{
  PetscReal dist = 0;
  for (PetscInt j = 0; j < dim; j++) dist += PetscSqr(x[j] - y[j]);
  return dist;
}

static inline PetscErrorCode PetscKDTreeQueryLeaf(PetscKDTree tree, KDLeaf leaf, const PetscReal point[], PetscCount *index, PetscReal *distance_sqr)
{
  PetscInt dim = tree->dim;

  PetscFunctionBeginUser;
  *distance_sqr = PETSC_MAX_REAL;
  *index        = -1;
  for (PetscInt i = 0; i < leaf.count; i++) {
    PetscCount point_index = tree->bucket_indices[leaf.indices_handle + i];
    PetscReal  dist        = PetscSquareDistance(dim, point, &tree->coords[point_index * dim]);
    if (dist < *distance_sqr) {
      *distance_sqr = dist;
      *index        = point_index;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscKDTreeQueryLeaf_CopyCoords(PetscKDTree tree, KDLeaf leaf, const PetscReal point[], PetscCount *index, PetscReal *distance_sqr)
{
  PetscInt dim = tree->dim;

  PetscFunctionBeginUser;
  *distance_sqr = PETSC_MAX_REAL;
  *index        = -1;
  for (PetscInt i = 0; i < leaf.count; i++) {
    // Coord data saved in axis-major ordering for vectorization
    PetscReal dist = 0.;
    for (PetscInt d = 0; d < dim; d++) dist += PetscSqr(point[d] - tree->coords[leaf.coords_handle + d * leaf.count + i]);
    if (dist < *distance_sqr) {
      *distance_sqr = dist;
      *index        = tree->bucket_indices[leaf.indices_handle + i];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Recursive point query from 'Algorithms for Fast Vector Quantization' by  Sunil Arya and David Mount
// Variant also implemented in pykdtree
static PetscErrorCode PetscKDTreeQuery_Recurse(PetscKDTree tree, const PetscReal point[], PetscCount node_handle, char is_node_leaf, PetscReal offset[], PetscReal rd, PetscReal tol_sqr, PetscCount *index, PetscReal *dist_sqr)
{
  PetscFunctionBeginUser;
  if (*dist_sqr < tol_sqr) PetscFunctionReturn(PETSC_SUCCESS);
  if (is_node_leaf) {
    KDLeaf     leaf = tree->leaves[node_handle];
    PetscReal  dist;
    PetscCount point_index;

    if (leaf.coords_handle > -1) PetscCall(PetscKDTreeQueryLeaf_CopyCoords(tree, leaf, point, &point_index, &dist));
    else PetscCall(PetscKDTreeQueryLeaf(tree, leaf, point, &point_index, &dist));
    if (dist < *dist_sqr) {
      *dist_sqr = dist;
      *index    = point_index;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  KDStem    stem       = tree->stems[node_handle];
  PetscReal old_offset = offset[stem.axis], new_offset = point[stem.axis] - stem.split;
  if (new_offset <= 0) {
    PetscCall(PetscKDTreeQuery_Recurse(tree, point, stem.less_equal_handle, PetscBTLookup(&stem.are_handles_leaves, LESS_EQUAL_BIT), offset, rd, tol_sqr, index, dist_sqr));
    rd += -PetscSqr(old_offset) + PetscSqr(new_offset);
    if (rd < *dist_sqr) {
      offset[stem.axis] = new_offset;
      PetscCall(PetscKDTreeQuery_Recurse(tree, point, stem.greater_handle, PetscBTLookup(&stem.are_handles_leaves, GREATER_BIT), offset, rd, tol_sqr, index, dist_sqr));
      offset[stem.axis] = old_offset;
    }
  } else {
    PetscCall(PetscKDTreeQuery_Recurse(tree, point, stem.greater_handle, PetscBTLookup(&stem.are_handles_leaves, GREATER_BIT), offset, rd, tol_sqr, index, dist_sqr));
    rd += -PetscSqr(old_offset) + PetscSqr(new_offset);
    if (rd < *dist_sqr) {
      offset[stem.axis] = new_offset;
      PetscCall(PetscKDTreeQuery_Recurse(tree, point, stem.less_equal_handle, PetscBTLookup(&stem.are_handles_leaves, LESS_EQUAL_BIT), offset, rd, tol_sqr, index, dist_sqr));
      offset[stem.axis] = old_offset;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscKDTreeQueryPointsNearestNeighbor - find the nearest neighbor in a `PetscKDTree`

  Not Collective, No Fortran Support

  Input Parameters:
+ tree       - tree to query
. num_points - number of points to query
. points     - array of the coordinates, in point-major order
- tolerance  - tolerance for nearest neighbor

  Output Parameters:
+ indices   - indices of the nearest neighbor to the query point
- distances - distance between the queried point and the nearest neighbor

  Level: advanced

  Notes:
  When traversing the tree, if a point has been found to be closer than the `tolerance`, the function short circuits and doesn't check for any closer points.

  The `indices` and `distances` arrays should be at least of size `num_points`.

.seealso: `PetscKDTree`, `PetscKDTreeCreate()`
@*/
PetscErrorCode PetscKDTreeQueryPointsNearestNeighbor(PetscKDTree tree, PetscCount num_points, const PetscReal points[], PetscReal tolerance, PetscCount indices[], PetscReal distances[])
{
  PetscReal *offsets, rd;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(PetscKDTree_Query, 0, 0, 0, 0));
  if (tree == NULL) {
    PetscCheck(num_points == 0, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "num_points may only be zero, if tree is NULL");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssertPointer(points, 3);
  PetscAssertPointer(indices, 5);
  PetscAssertPointer(distances, 6);
  PetscCall(PetscCalloc1(tree->dim, &offsets));

  for (PetscCount p = 0; p < num_points; p++) {
    rd           = 0.;
    distances[p] = PETSC_MAX_REAL;
    indices[p]   = -1;
    PetscCall(PetscKDTreeQuery_Recurse(tree, &points[p * tree->dim], tree->root_handle, (char)tree->is_root_leaf, offsets, rd, PetscSqr(tolerance), &indices[p], &distances[p]));
    distances[p] = PetscSqrtReal(distances[p]);
  }
  PetscCall(PetscFree(offsets));
  PetscCall(PetscLogEventEnd(PetscKDTree_Query, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscKDTreeView - view a `PetscKDTree`

  Not Collective, No Fortran Support

  Input Parameters:
+ tree   - tree to view
- viewer - visualization context

  Level: advanced

.seealso: `PetscKDTree`, `PetscKDTreeCreate()`, `PetscViewer`
@*/
PetscErrorCode PetscKDTreeView(PetscKDTree tree, PetscViewer viewer)
{
  PetscFunctionBeginUser;
  if (viewer) PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  else PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  if (tree == NULL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscViewerASCIIPrintf(viewer, "KDTree:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer)); // KDTree:
  PetscCall(PetscViewerASCIIPrintf(viewer, "Stems:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer)); // Stems:
  for (PetscCount i = 0; i < tree->num_stems; i++) {
    KDStem stem = tree->stems[i];
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stem %" PetscCount_FMT ": Axis=%" PetscInt_FMT " Split=%g Greater_%s=%" PetscCount_FMT " Lesser_Equal_%s=%" PetscCount_FMT "\n", i, (PetscInt)stem.axis, (double)stem.split, PetscBTLookup(&stem.are_handles_leaves, GREATER_BIT) ? "Leaf" : "Stem",
                                     stem.greater_handle, PetscBTLookup(&stem.are_handles_leaves, LESS_EQUAL_BIT) ? "Leaf" : "Stem", stem.less_equal_handle));
  }
  PetscCall(PetscViewerASCIIPopTab(viewer)); // Stems:

  PetscCall(PetscViewerASCIIPrintf(viewer, "Leaves:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer)); // Leaves:
  for (PetscCount i = 0; i < tree->num_leaves; i++) {
    KDLeaf leaf = tree->leaves[i];
    PetscCall(PetscViewerASCIIPrintf(viewer, "Leaf %" PetscCount_FMT ": Count=%" PetscInt_FMT, i, leaf.count));
    PetscCall(PetscViewerASCIIPushTab(viewer)); // Coords:
    for (PetscInt j = 0; j < leaf.count; j++) {
      PetscInt   tabs;
      PetscCount bucket_index = tree->bucket_indices[leaf.indices_handle + j];
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscCount_FMT ": ", bucket_index));

      PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
      PetscCall(PetscViewerASCIISetTab(viewer, 0));
      if (leaf.coords_handle > -1) {
        for (PetscInt k = 0; k < tree->dim; k++) PetscCall(PetscViewerASCIIPrintf(viewer, "%g ", (double)tree->coords[leaf.coords_handle + leaf.count * k + j]));
        PetscCall(PetscViewerASCIIPrintf(viewer, " (stored at leaf)"));
      } else {
        for (PetscInt k = 0; k < tree->dim; k++) PetscCall(PetscViewerASCIIPrintf(viewer, "%g ", (double)tree->coords[bucket_index * tree->dim + k]));
      }
      PetscCall(PetscViewerASCIISetTab(viewer, tabs));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer)); // Coords:
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscCall(PetscViewerASCIIPopTab(viewer)); // Leaves:
  PetscCall(PetscViewerASCIIPopTab(viewer)); // KDTree:
  PetscFunctionReturn(PETSC_SUCCESS);
}

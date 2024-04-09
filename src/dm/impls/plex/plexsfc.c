#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petscsf.h>
#include <petsc/private/hashset.h>

typedef uint64_t ZCode;

PETSC_HASH_SET(ZSet, ZCode, PetscHash_UInt64, PetscHashEqual)

typedef struct {
  PetscInt i, j, k;
} Ijk;

typedef struct {
  Ijk         eextent;
  Ijk         vextent;
  PetscMPIInt comm_size;
  ZCode      *zstarts;
} ZLayout;

static unsigned ZCodeSplit1(ZCode z)
{
  z = ((z & 01001001001001001) | ((z >> 2) & 02002002002002002) | ((z >> 4) & 04004004004004004));
  z = (z | (z >> 6) | (z >> 12)) & 0000000777000000777;
  z = (z | (z >> 18)) & 0777777;
  return (unsigned)z;
}

static ZCode ZEncode1(unsigned t)
{
  ZCode z = t;
  z       = (z | (z << 18)) & 0777000000777;
  z       = (z | (z << 6) | (z << 12)) & 07007007007007007;
  z       = (z | (z << 2) | (z << 4)) & 0111111111111111111;
  return z;
}

static Ijk ZCodeSplit(ZCode z)
{
  Ijk c;
  c.i = ZCodeSplit1(z >> 2);
  c.j = ZCodeSplit1(z >> 1);
  c.k = ZCodeSplit1(z >> 0);
  return c;
}

static ZCode ZEncode(Ijk c)
{
  ZCode z = (ZEncode1(c.i) << 2) | (ZEncode1(c.j) << 1) | ZEncode1(c.k);
  return z;
}

static PetscBool IjkActive(Ijk extent, Ijk l)
{
  if (l.i < extent.i && l.j < extent.j && l.k < extent.k) return PETSC_TRUE;
  return PETSC_FALSE;
}

// If z is not the base of an octet (last 3 bits 0), return 0.
//
// If z is the base of an octet, we recursively grow to the biggest structured octet. This is typically useful when a z
// is outside the domain and we wish to skip a (possibly recursively large) octet to find our next interesting point.
static ZCode ZStepOct(ZCode z)
{
  if (PetscUnlikely(z == 0)) return 0; // Infinite loop below if z == 0
  ZCode step = 07;
  for (; (z & step) == 0; step = (step << 3) | 07) { }
  return step >> 3;
}

// Since element/vertex box extents are typically not equal powers of 2, Z codes that lie within the domain are not contiguous.
static PetscErrorCode ZLayoutCreate(PetscMPIInt size, const PetscInt eextent[3], const PetscInt vextent[3], ZLayout *layout)
{
  PetscFunctionBegin;
  layout->eextent.i = eextent[0];
  layout->eextent.j = eextent[1];
  layout->eextent.k = eextent[2];
  layout->vextent.i = vextent[0];
  layout->vextent.j = vextent[1];
  layout->vextent.k = vextent[2];
  layout->comm_size = size;
  layout->zstarts   = NULL;
  PetscCall(PetscMalloc1(size + 1, &layout->zstarts));

  PetscInt total_elems = eextent[0] * eextent[1] * eextent[2];
  ZCode    z           = 0;
  layout->zstarts[0]   = 0;
  // This loop traverses all vertices in the global domain, so is worth making fast. We use ZStepBound
  for (PetscMPIInt r = 0; r < size; r++) {
    PetscInt elems_needed = (total_elems / size) + (total_elems % size > r), count;
    for (count = 0; count < elems_needed; z++) {
      ZCode skip = ZStepOct(z); // optimistically attempt a longer step
      for (ZCode s = skip;; s >>= 3) {
        Ijk trial = ZCodeSplit(z + s);
        if (IjkActive(layout->eextent, trial)) {
          while (count + s + 1 > (ZCode)elems_needed) s >>= 3; // Shrink the octet
          count += s + 1;
          z += s;
          break;
        }
        if (s == 0) { // the whole skip octet is inactive
          z += skip;
          break;
        }
      }
    }
    // Pick up any extra vertices in the Z ordering before the next rank's first owned element.
    //
    // This leads to poorly balanced vertices when eextent is a power of 2, since all the fringe vertices end up
    // on the last rank. A possible solution is to balance the Z-order vertices independently from the cells, which will
    // result in a lot of element closures being remote. We could finish marking boundary conditions, then do a round of
    // vertex ownership smoothing (which would reorder and redistribute vertices without touching element distribution).
    // Another would be to have an analytic ownership criteria for vertices in the fringe veextent - eextent. This would
    // complicate the job of identifying an owner and its offset.
    //
    // The current recommended approach is to let `-dm_distribute 1` (default) resolve vertex ownership. This is
    // *mandatory* with isoperiodicity (except in special cases) to remove standed vertices from local spaces. Here's
    // the issue:
    //
    // Consider this partition on rank 0 (left) and rank 1.
    //
    //    4 --------  5 -- 14 --10 -- 21 --11
    //                |          |          |
    // 7 -- 16 --  8  |          |          |
    // |           |  3 -------  7 -------  9
    // |           |             |          |
    // 4 --------  6 ------ 10   |          |
    // |           |         |   6 -- 16 -- 8
    // |           |         |
    // 3 ---11---  5 --18--  9
    //
    // The periodic face SF looks like
    // [0] Number of roots=21, leaves=1, remote ranks=1
    // [0] 16 <- (0,11)
    // [1] Number of roots=22, leaves=2, remote ranks=2
    // [1] 14 <- (0,18)
    // [1] 21 <- (1,16)
    //
    // In handling face (0,16), rank 0 learns that (0,7) and (0,8) map to (0,3) and (0,5) respectively, thus we won't use
    // the point SF links to (1,4) and (1,5). Rank 1 learns about the periodic mapping of (1,5) while handling face
    // (1,14), but never learns that vertex (1,4) has been mapped to (0,3) by face (0,16).
    //
    // We can relatively easily inform vertex (1,4) of this mapping, but it stays in rank 1's local space despite not
    // being in the closure and thus not being contributed to. This would be mostly harmless except that some viewer
    // routines expect all local points to be somehow significant. It is not easy to analytically remove the (1,4)
    // vertex because the point SF and isoperiodic face SF would need to be updated to account for removal of the
    // stranded vertices.
    for (; z <= ZEncode(layout->vextent); z++) {
      Ijk loc = ZCodeSplit(z);
      if (IjkActive(layout->eextent, loc)) break;
      z += ZStepOct(z);
    }
    layout->zstarts[r + 1] = z;
  }
  layout->zstarts[size] = ZEncode(layout->vextent);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt ZLayoutElementsOnRank(const ZLayout *layout, PetscMPIInt rank)
{
  PetscInt remote_elem = 0;
  for (ZCode rz = layout->zstarts[rank]; rz < layout->zstarts[rank + 1]; rz++) {
    Ijk loc = ZCodeSplit(rz);
    if (IjkActive(layout->eextent, loc)) remote_elem++;
    else rz += ZStepOct(rz);
  }
  return remote_elem;
}

static PetscInt ZCodeFind(ZCode key, PetscInt n, const ZCode X[])
{
  PetscInt lo = 0, hi = n;

  if (n == 0) return -1;
  while (hi - lo > 1) {
    PetscInt mid = lo + (hi - lo) / 2;
    if (key < X[mid]) hi = mid;
    else lo = mid;
  }
  return key == X[lo] ? lo : -(lo + (key > X[lo]) + 1);
}

static PetscErrorCode DMPlexCreateBoxMesh_Tensor_SFC_Periodicity_Private(DM dm, const ZLayout *layout, const ZCode *vert_z, PetscSegBuffer per_faces[3], const PetscReal *lower, const PetscReal *upper, const DMBoundaryType *periodicity, PetscSegBuffer donor_face_closure[3], PetscSegBuffer my_donor_faces[3])
{
  MPI_Comm    comm;
  PetscInt    dim, vStart, vEnd;
  PetscMPIInt size;
  PetscSF     face_sfs[3];
  PetscScalar transforms[3][4][4] = {{{0}}};

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetDimension(dm, &dim));
  const PetscInt csize = PetscPowInt(2, dim - 1);
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));

  PetscInt num_directions = 0;
  for (PetscInt direction = 0; direction < dim; direction++) {
    size_t       num_faces;
    PetscInt    *faces;
    ZCode       *donor_verts, *donor_minz;
    PetscSFNode *leaf;

    if (periodicity[direction] != DM_BOUNDARY_PERIODIC) continue;
    PetscCall(PetscSegBufferGetSize(per_faces[direction], &num_faces));
    PetscCall(PetscSegBufferExtractInPlace(per_faces[direction], &faces));
    PetscCall(PetscSegBufferExtractInPlace(donor_face_closure[direction], &donor_verts));
    PetscCall(PetscMalloc1(num_faces, &donor_minz));
    PetscCall(PetscMalloc1(num_faces, &leaf));
    for (PetscInt i = 0; i < (PetscInt)num_faces; i++) {
      ZCode minz = donor_verts[i * csize];
      for (PetscInt j = 1; j < csize; j++) minz = PetscMin(minz, donor_verts[i * csize + j]);
      donor_minz[i] = minz;
    }
    {
      PetscBool sorted;
      PetscCall(PetscSortedInt64(num_faces, (const PetscInt64 *)donor_minz, &sorted));
      // If a donor vertex were chosen to broker multiple faces, we would have a logic error.
      // Checking for sorting is a cheap check that there are no duplicates.
      PetscCheck(sorted, PETSC_COMM_SELF, PETSC_ERR_PLIB, "minz not sorted; possible duplicates not checked");
    }
    for (PetscInt i = 0; i < (PetscInt)num_faces;) {
      ZCode    z           = donor_minz[i];
      PetscInt remote_rank = ZCodeFind(z, size + 1, layout->zstarts), remote_count = 0;
      if (remote_rank < 0) remote_rank = -(remote_rank + 1) - 1;
      // Process all the vertices on this rank
      for (ZCode rz = layout->zstarts[remote_rank]; rz < layout->zstarts[remote_rank + 1]; rz++) {
        Ijk loc = ZCodeSplit(rz);
        if (rz == z) {
          leaf[i].rank  = remote_rank;
          leaf[i].index = remote_count;
          i++;
          if (i == (PetscInt)num_faces) break;
          z = donor_minz[i];
        }
        if (IjkActive(layout->vextent, loc)) remote_count++;
      }
    }
    PetscCall(PetscFree(donor_minz));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), &face_sfs[num_directions]));
    PetscCall(PetscSFSetGraph(face_sfs[num_directions], vEnd - vStart, num_faces, NULL, PETSC_USE_POINTER, leaf, PETSC_USE_POINTER));
    const PetscInt *my_donor_degree;
    PetscCall(PetscSFComputeDegreeBegin(face_sfs[num_directions], &my_donor_degree));
    PetscCall(PetscSFComputeDegreeEnd(face_sfs[num_directions], &my_donor_degree));
    PetscInt num_multiroots = 0;
    for (PetscInt i = 0; i < vEnd - vStart; i++) {
      num_multiroots += my_donor_degree[i];
      if (my_donor_degree[i] == 0) continue;
      PetscAssert(my_donor_degree[i] == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Local vertex has multiple faces");
    }
    PetscInt *my_donors, *donor_indices, *my_donor_indices;
    size_t    num_my_donors;
    PetscCall(PetscSegBufferGetSize(my_donor_faces[direction], &num_my_donors));
    PetscCheck((PetscInt)num_my_donors == num_multiroots, PETSC_COMM_SELF, PETSC_ERR_SUP, "Donor request does not match expected donors");
    PetscCall(PetscSegBufferExtractInPlace(my_donor_faces[direction], &my_donors));
    PetscCall(PetscMalloc1(vEnd - vStart, &my_donor_indices));
    for (PetscInt i = 0; i < (PetscInt)num_my_donors; i++) {
      PetscInt f = my_donors[i];
      PetscInt num_points, *points = NULL, minv = PETSC_MAX_INT;
      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &num_points, &points));
      for (PetscInt j = 0; j < num_points; j++) {
        PetscInt p = points[2 * j];
        if (p < vStart || vEnd <= p) continue;
        minv = PetscMin(minv, p);
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &num_points, &points));
      PetscAssert(my_donor_degree[minv - vStart] == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Local vertex not requested");
      my_donor_indices[minv - vStart] = f;
    }
    PetscCall(PetscMalloc1(num_faces, &donor_indices));
    PetscCall(PetscSFBcastBegin(face_sfs[num_directions], MPIU_INT, my_donor_indices, donor_indices, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(face_sfs[num_directions], MPIU_INT, my_donor_indices, donor_indices, MPI_REPLACE));
    PetscCall(PetscFree(my_donor_indices));
    // Modify our leafs so they point to donor faces instead of donor minz. Additionally, give them indices as faces.
    for (PetscInt i = 0; i < (PetscInt)num_faces; i++) leaf[i].index = donor_indices[i];
    PetscCall(PetscFree(donor_indices));
    PetscInt pStart, pEnd;
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscSFSetGraph(face_sfs[num_directions], pEnd - pStart, num_faces, faces, PETSC_COPY_VALUES, leaf, PETSC_OWN_POINTER));
    {
      char face_sf_name[PETSC_MAX_PATH_LEN];
      PetscCall(PetscSNPrintf(face_sf_name, sizeof face_sf_name, "Z-order Isoperiodic Faces #%" PetscInt_FMT, num_directions));
      PetscCall(PetscObjectSetName((PetscObject)face_sfs[num_directions], face_sf_name));
    }

    transforms[num_directions][0][0]         = 1;
    transforms[num_directions][1][1]         = 1;
    transforms[num_directions][2][2]         = 1;
    transforms[num_directions][3][3]         = 1;
    transforms[num_directions][direction][3] = upper[direction] - lower[direction];
    num_directions++;
  }

  PetscCall(DMPlexSetIsoperiodicFaceSF(dm, num_directions, face_sfs));
  PetscCall(DMPlexSetIsoperiodicFaceTransform(dm, num_directions, (PetscScalar *)transforms));

  for (PetscInt i = 0; i < num_directions; i++) PetscCall(PetscSFDestroy(&face_sfs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This is a DMGlobalToLocalHook that applies the affine offsets. When extended for rotated periodicity, it'll need to
// apply a rotatonal transform and similar operations will be needed for fields (e.g., to rotate a velocity vector).
// We use this crude approach here so we don't have to write new GPU kernels yet.
static PetscErrorCode DMCoordAddPeriodicOffsets_Private(DM dm, Vec g, InsertMode mode, Vec l, void *ctx)
{
  PetscFunctionBegin;
  // These `VecScatter`s should be merged to improve efficiency; the scatters cannot be overlapped.
  for (PetscInt i = 0; i < dm->periodic.num_affines; i++) {
    PetscCall(VecScatterBegin(dm->periodic.affine_to_local[i], dm->periodic.affine[i], l, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(dm->periodic.affine_to_local[i], dm->periodic.affine[i], l, ADD_VALUES, SCATTER_FORWARD));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Start with an SF for a positive depth (e.g., faces) and create a new SF for matched closure. The caller must ensure
// that both the donor (root) face and the periodic (leaf) face have consistent orientation, meaning that their closures
// are isomorphic. It may be useful/necessary for this restriction to be loosened.
//
// Output Arguments:
//
// + closure_sf - augmented point SF (see `DMGetPointSF()`) that includes the faces and all points in its closure. This
//   can be used to create a global section and section SF.
// - is_points - array of index sets for just the points in the closure of `face_sf`. These may be used to apply an affine
//   transformation to periodic dofs; see DMPeriodicCoordinateSetUp_Internal().
//
static PetscErrorCode DMPlexCreateIsoperiodicPointSF_Private(DM dm, PetscInt num_face_sfs, PetscSF *face_sfs, PetscSF *closure_sf, IS **is_points)
{
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscSF            point_sf;
  PetscInt           nroots, nleaves;
  const PetscInt    *filocal;
  const PetscSFNode *firemote;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetPointSF(dm, &point_sf)); // Point SF has remote points
  PetscCall(PetscMalloc1(num_face_sfs, is_points));

  for (PetscInt f = 0; f < num_face_sfs; f++) {
    PetscSF   face_sf = face_sfs[f];
    PetscInt *rootdata, *leafdata;

    PetscCall(PetscSFGetGraph(face_sf, &nroots, &nleaves, &filocal, &firemote));
    PetscCall(PetscCalloc2(2 * nroots, &rootdata, 2 * nroots, &leafdata));
    for (PetscInt i = 0; i < nleaves; i++) {
      PetscInt point = filocal[i], cl_size, *closure = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &cl_size, &closure));
      leafdata[point] = cl_size - 1;
      PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &cl_size, &closure));
    }
    PetscCall(PetscSFReduceBegin(face_sf, MPIU_INT, leafdata, rootdata + nroots, MPIU_SUM));
    PetscCall(PetscSFReduceEnd(face_sf, MPIU_INT, leafdata, rootdata + nroots, MPIU_SUM));

    PetscInt root_offset = 0;
    for (PetscInt p = 0; p < nroots; p++) {
      const PetscInt *donor_dof = rootdata + nroots;
      if (donor_dof[p] == 0) {
        rootdata[2 * p]     = -1;
        rootdata[2 * p + 1] = -1;
        continue;
      }
      PetscInt  cl_size;
      PetscInt *closure = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &cl_size, &closure));
      // cl_size - 1 = points not including self
      PetscAssert(donor_dof[p] == cl_size - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reduced leaf cone sizes do not match root cone sizes");
      rootdata[2 * p]     = root_offset;
      rootdata[2 * p + 1] = cl_size - 1;
      root_offset += cl_size - 1;
      PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &cl_size, &closure));
    }
    PetscCall(PetscSFBcastBegin(face_sf, MPIU_2INT, rootdata, leafdata, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(face_sf, MPIU_2INT, rootdata, leafdata, MPI_REPLACE));
    // Count how many leaves we need to communicate the closures
    PetscInt leaf_offset = 0;
    for (PetscInt i = 0; i < nleaves; i++) {
      PetscInt point = filocal[i];
      if (leafdata[2 * point + 1] < 0) continue;
      leaf_offset += leafdata[2 * point + 1];
    }

    PetscSFNode *closure_leaf;
    PetscCall(PetscMalloc1(leaf_offset, &closure_leaf));
    leaf_offset = 0;
    for (PetscInt i = 0; i < nleaves; i++) {
      PetscInt point   = filocal[i];
      PetscInt cl_size = leafdata[2 * point + 1];
      if (cl_size < 0) continue;
      for (PetscInt j = 0; j < cl_size; j++) {
        closure_leaf[leaf_offset].rank  = firemote[i].rank;
        closure_leaf[leaf_offset].index = leafdata[2 * point] + j;
        leaf_offset++;
      }
    }

    PetscSF sf_closure;
    PetscCall(PetscSFCreate(comm, &sf_closure));
    PetscCall(PetscSFSetGraph(sf_closure, root_offset, leaf_offset, NULL, PETSC_USE_POINTER, closure_leaf, PETSC_OWN_POINTER));

    PetscSFNode *leaf_donor_closure;
    { // Pack root buffer with owner for every point in the root cones
      PetscSFNode       *donor_closure;
      const PetscInt    *pilocal;
      const PetscSFNode *piremote;
      PetscInt           npoints;

      PetscCall(PetscSFGetGraph(point_sf, NULL, &npoints, &pilocal, &piremote));
      PetscCall(PetscCalloc1(root_offset, &donor_closure));
      root_offset = 0;
      for (PetscInt p = 0; p < nroots; p++) {
        if (rootdata[2 * p] < 0) continue;
        PetscInt  cl_size;
        PetscInt *closure = NULL;
        PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &cl_size, &closure));
        for (PetscInt j = 1; j < cl_size; j++) {
          PetscInt c = closure[2 * j];
          if (pilocal) {
            PetscInt found = -1;
            if (npoints > 0) PetscCall(PetscFindInt(c, npoints, pilocal, &found));
            if (found >= 0) {
              donor_closure[root_offset++] = piremote[found];
              continue;
            }
          }
          // we own c
          donor_closure[root_offset].rank  = rank;
          donor_closure[root_offset].index = c;
          root_offset++;
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &cl_size, &closure));
      }

      PetscCall(PetscMalloc1(leaf_offset, &leaf_donor_closure));
      PetscCall(PetscSFBcastBegin(sf_closure, MPIU_2INT, donor_closure, leaf_donor_closure, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf_closure, MPIU_2INT, donor_closure, leaf_donor_closure, MPI_REPLACE));
      PetscCall(PetscSFDestroy(&sf_closure));
      PetscCall(PetscFree(donor_closure));
    }

    PetscSFNode *new_iremote;
    PetscCall(PetscCalloc1(nroots, &new_iremote));
    for (PetscInt i = 0; i < nroots; i++) new_iremote[i].rank = -1;
    // Walk leaves and match vertices
    leaf_offset = 0;
    for (PetscInt i = 0; i < nleaves; i++) {
      PetscInt  point   = filocal[i], cl_size;
      PetscInt *closure = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &cl_size, &closure));
      for (PetscInt j = 1; j < cl_size; j++) { // TODO: should we send donor edge orientations so we can flip for consistency?
        PetscInt    c  = closure[2 * j];
        PetscSFNode lc = leaf_donor_closure[leaf_offset];
        // printf("[%d] face %d.%d: %d ?-- (%d,%d)\n", rank, point, j, c, lc.rank, lc.index);
        if (new_iremote[c].rank == -1) {
          new_iremote[c] = lc;
        } else PetscCheck(new_iremote[c].rank == lc.rank && new_iremote[c].index == lc.index, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched cone ordering between faces");
        leaf_offset++;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &cl_size, &closure));
    }
    PetscCall(PetscFree(leaf_donor_closure));

    // Include face points in closure SF
    for (PetscInt i = 0; i < nleaves; i++) new_iremote[filocal[i]] = firemote[i];
    // consolidate leaves
    PetscInt num_new_leaves = 0;
    for (PetscInt i = 0; i < nroots; i++) {
      if (new_iremote[i].rank == -1) continue;
      new_iremote[num_new_leaves] = new_iremote[i];
      leafdata[num_new_leaves]    = i;
      num_new_leaves++;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_new_leaves, leafdata, PETSC_COPY_VALUES, &(*is_points)[f]));

    PetscSF csf;
    PetscCall(PetscSFCreate(comm, &csf));
    PetscCall(PetscSFSetGraph(csf, nroots, num_new_leaves, leafdata, PETSC_COPY_VALUES, new_iremote, PETSC_COPY_VALUES));
    PetscCall(PetscFree(new_iremote)); // copy and delete because new_iremote is longer than it needs to be
    PetscCall(PetscFree2(rootdata, leafdata));

    PetscInt npoints;
    PetscCall(PetscSFGetGraph(point_sf, NULL, &npoints, NULL, NULL));
    if (npoints < 0) { // empty point_sf
      *closure_sf = csf;
    } else {
      PetscCall(PetscSFMerge(point_sf, csf, closure_sf));
      PetscCall(PetscSFDestroy(&csf));
    }
    if (f > 0) PetscCall(PetscSFDestroy(&point_sf)); // Only destroy if point_sf is from previous calls to PetscSFMerge
    point_sf = *closure_sf;                          // Use combined point + isoperiodic SF to define point ownership for further face_sf
  }
  PetscCall(PetscObjectSetName((PetscObject)*closure_sf, "Composed Periodic Points"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetIsoperiodicPointSF_Plex(DM dm, PetscSF *sf)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  if (!plex->periodic.composed_sf) PetscCall(DMPlexCreateIsoperiodicPointSF_Private(dm, plex->periodic.num_face_sfs, plex->periodic.face_sfs, &plex->periodic.composed_sf, &plex->periodic.periodic_points));
  if (sf) *sf = plex->periodic.composed_sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexMigrateIsoperiodicFaceSF_Internal(DM old_dm, DM dm, PetscSF sf_migration)
{
  DM_Plex    *plex = (DM_Plex *)old_dm->data;
  PetscSF     sf_point, *new_face_sfs;
  PetscMPIInt rank;

  PetscFunctionBegin;
  if (!plex->periodic.face_sfs) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetPointSF(dm, &sf_point));
  PetscCall(PetscMalloc1(plex->periodic.num_face_sfs, &new_face_sfs));

  for (PetscInt f = 0; f < plex->periodic.num_face_sfs; f++) {
    PetscInt           old_npoints, new_npoints, old_nleaf, new_nleaf, point_nleaf;
    PetscSFNode       *new_leafdata, *rootdata, *leafdata;
    const PetscInt    *old_local, *point_local;
    const PetscSFNode *old_remote, *point_remote;
    PetscCall(PetscSFGetGraph(plex->periodic.face_sfs[f], &old_npoints, &old_nleaf, &old_local, &old_remote));
    PetscCall(PetscSFGetGraph(sf_migration, NULL, &new_nleaf, NULL, NULL));
    PetscCall(PetscSFGetGraph(sf_point, &new_npoints, &point_nleaf, &point_local, &point_remote));
    PetscAssert(new_nleaf == new_npoints, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected migration leaf space to match new point root space");
    PetscCall(PetscMalloc3(old_npoints, &rootdata, old_npoints, &leafdata, new_npoints, &new_leafdata));

    // Fill new_leafdata with new owners of all points
    for (PetscInt i = 0; i < new_npoints; i++) {
      new_leafdata[i].rank  = rank;
      new_leafdata[i].index = i;
    }
    for (PetscInt i = 0; i < point_nleaf; i++) {
      PetscInt j      = point_local[i];
      new_leafdata[j] = point_remote[i];
    }
    // REPLACE is okay because every leaf agrees about the new owners
    PetscCall(PetscSFReduceBegin(sf_migration, MPIU_2INT, new_leafdata, rootdata, MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(sf_migration, MPIU_2INT, new_leafdata, rootdata, MPI_REPLACE));
    // rootdata now contains the new owners

    // Send to leaves of old space
    for (PetscInt i = 0; i < old_npoints; i++) {
      leafdata[i].rank  = -1;
      leafdata[i].index = -1;
    }
    PetscCall(PetscSFBcastBegin(plex->periodic.face_sfs[f], MPIU_2INT, rootdata, leafdata, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(plex->periodic.face_sfs[f], MPIU_2INT, rootdata, leafdata, MPI_REPLACE));

    // Send to new leaf space
    PetscCall(PetscSFBcastBegin(sf_migration, MPIU_2INT, leafdata, new_leafdata, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf_migration, MPIU_2INT, leafdata, new_leafdata, MPI_REPLACE));

    PetscInt     nface = 0, *new_local;
    PetscSFNode *new_remote;
    for (PetscInt i = 0; i < new_npoints; i++) nface += (new_leafdata[i].rank >= 0);
    PetscCall(PetscMalloc1(nface, &new_local));
    PetscCall(PetscMalloc1(nface, &new_remote));
    nface = 0;
    for (PetscInt i = 0; i < new_npoints; i++) {
      if (new_leafdata[i].rank == -1) continue;
      new_local[nface]  = i;
      new_remote[nface] = new_leafdata[i];
      nface++;
    }
    PetscCall(PetscFree3(rootdata, leafdata, new_leafdata));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), &new_face_sfs[f]));
    PetscCall(PetscSFSetGraph(new_face_sfs[f], new_npoints, nface, new_local, PETSC_OWN_POINTER, new_remote, PETSC_OWN_POINTER));
    {
      char new_face_sf_name[PETSC_MAX_PATH_LEN];
      PetscCall(PetscSNPrintf(new_face_sf_name, sizeof new_face_sf_name, "Migrated Isoperiodic Faces #%" PetscInt_FMT, f));
      PetscCall(PetscObjectSetName((PetscObject)new_face_sfs[f], new_face_sf_name));
    }
  }

  PetscCall(DMPlexSetIsoperiodicFaceSF(dm, plex->periodic.num_face_sfs, new_face_sfs));
  PetscCall(DMPlexSetIsoperiodicFaceTransform(dm, plex->periodic.num_face_sfs, (PetscScalar *)plex->periodic.transform));
  for (PetscInt f = 0; f < plex->periodic.num_face_sfs; f++) PetscCall(PetscSFDestroy(&new_face_sfs[f]));
  PetscCall(PetscFree(new_face_sfs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPeriodicCoordinateSetUp_Internal(DM dm)
{
  DM_Plex *plex = (DM_Plex *)dm->data;
  size_t   count;
  IS       isdof;
  PetscInt dim;

  PetscFunctionBegin;
  if (!plex->periodic.face_sfs) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetIsoperiodicPointSF_Plex(dm, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMGetIsoperiodicPointSF_C", DMGetIsoperiodicPointSF_Plex));

  PetscCall(DMGetDimension(dm, &dim));
  dm->periodic.num_affines = plex->periodic.num_face_sfs;
  PetscCall(PetscMalloc2(dm->periodic.num_affines, &dm->periodic.affine_to_local, dm->periodic.num_affines, &dm->periodic.affine));

  for (PetscInt f = 0; f < plex->periodic.num_face_sfs; f++) {
    {
      PetscInt        npoints;
      const PetscInt *points;
      IS              is = plex->periodic.periodic_points[f];
      PetscSegBuffer  seg;
      PetscSection    section;
      PetscCall(DMGetLocalSection(dm, &section));
      PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 32, &seg));
      PetscCall(ISGetSize(is, &npoints));
      PetscCall(ISGetIndices(is, &points));
      for (PetscInt i = 0; i < npoints; i++) {
        PetscInt point = points[i], off, dof;
        PetscCall(PetscSectionGetOffset(section, point, &off));
        PetscCall(PetscSectionGetDof(section, point, &dof));
        PetscAssert(dof % dim == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected dof %" PetscInt_FMT " not divisible by dimension %" PetscInt_FMT, dof, dim);
        for (PetscInt j = 0; j < dof / dim; j++) {
          PetscInt *slot;
          PetscCall(PetscSegBufferGetInts(seg, 1, &slot));
          *slot = off / dim + j;
        }
      }
      PetscInt *ind;
      PetscCall(PetscSegBufferGetSize(seg, &count));
      PetscCall(PetscSegBufferExtractAlloc(seg, &ind));
      PetscCall(PetscSegBufferDestroy(&seg));
      PetscCall(ISCreateBlock(PETSC_COMM_SELF, dim, count, ind, PETSC_OWN_POINTER, &isdof));
    }
    Vec        L, P;
    VecType    vec_type;
    VecScatter scatter;
    PetscCall(DMGetLocalVector(dm, &L));
    PetscCall(VecCreate(PETSC_COMM_SELF, &P));
    PetscCall(VecSetSizes(P, count * dim, count * dim));
    PetscCall(VecGetType(L, &vec_type));
    PetscCall(VecSetType(P, vec_type));
    PetscCall(VecScatterCreate(P, NULL, L, isdof, &scatter));
    PetscCall(DMRestoreLocalVector(dm, &L));
    PetscCall(ISDestroy(&isdof));

    {
      PetscScalar *x;
      PetscCall(VecGetArrayWrite(P, &x));
      for (PetscInt i = 0; i < (PetscInt)count; i++) {
        for (PetscInt j = 0; j < dim; j++) x[i * dim + j] = plex->periodic.transform[f][j][3];
      }
      PetscCall(VecRestoreArrayWrite(P, &x));
    }

    dm->periodic.affine_to_local[f] = scatter;
    dm->periodic.affine[f]          = P;
  }
  PetscCall(DMGlobalToLocalHookAdd(dm, NULL, DMCoordAddPeriodicOffsets_Private, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// We'll just orient all the edges, though only periodic boundary edges need orientation
static PetscErrorCode DMPlexOrientPositiveEdges_Private(DM dm)
{
  PetscInt dim, eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  if (dim < 3) PetscFunctionReturn(PETSC_SUCCESS); // not necessary
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  for (PetscInt e = eStart; e < eEnd; e++) {
    const PetscInt *cone;
    PetscCall(DMPlexGetCone(dm, e, &cone));
    if (cone[0] > cone[1]) PetscCall(DMPlexOrientPoint(dm, e, -1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCreateBoxMesh_Tensor_SFC_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate)
{
  PetscInt  eextent[3] = {1, 1, 1}, vextent[3] = {1, 1, 1};
  const Ijk closure_1[] = {
    {0, 0, 0},
    {1, 0, 0},
  };
  const Ijk closure_2[] = {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0},
  };
  const Ijk closure_3[] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
    {1, 0, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1},
    {0, 1, 1},
  };
  const Ijk *const closure_dim[] = {NULL, closure_1, closure_2, closure_3};
  // This must be kept consistent with DMPlexCreateCubeMesh_Internal
  const PetscInt        face_marker_1[]   = {1, 2};
  const PetscInt        face_marker_2[]   = {4, 2, 1, 3};
  const PetscInt        face_marker_3[]   = {6, 5, 3, 4, 1, 2};
  const PetscInt *const face_marker_dim[] = {NULL, face_marker_1, face_marker_2, face_marker_3};
  // Orient faces so the normal is in the positive axis and the first vertex is the one closest to zero.
  // These orientations can be determined by examining cones of a reference quad and hex element.
  const PetscInt        face_orient_1[]   = {0, 0};
  const PetscInt        face_orient_2[]   = {-1, 0, 0, -1};
  const PetscInt        face_orient_3[]   = {-2, 0, -2, 1, -2, 0};
  const PetscInt *const face_orient_dim[] = {NULL, face_orient_1, face_orient_2, face_orient_3};

  PetscFunctionBegin;
  PetscAssertPointer(dm, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  PetscCall(DMSetDimension(dm, dim));
  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  for (PetscInt i = 0; i < dim; i++) {
    eextent[i] = faces[i];
    vextent[i] = faces[i] + 1;
  }
  ZLayout layout;
  PetscCall(ZLayoutCreate(size, eextent, vextent, &layout));
  PetscZSet vset; // set of all vertices in the closure of the owned elements
  PetscCall(PetscZSetCreate(&vset));
  PetscInt local_elems = 0;
  for (ZCode z = layout.zstarts[rank]; z < layout.zstarts[rank + 1]; z++) {
    Ijk loc = ZCodeSplit(z);
    if (IjkActive(layout.vextent, loc)) PetscCall(PetscZSetAdd(vset, z));
    else {
      z += ZStepOct(z);
      continue;
    }
    if (IjkActive(layout.eextent, loc)) {
      local_elems++;
      // Add all neighboring vertices to set
      for (PetscInt n = 0; n < PetscPowInt(2, dim); n++) {
        Ijk   inc  = closure_dim[dim][n];
        Ijk   nloc = {loc.i + inc.i, loc.j + inc.j, loc.k + inc.k};
        ZCode v    = ZEncode(nloc);
        PetscCall(PetscZSetAdd(vset, v));
      }
    }
  }
  PetscInt local_verts, off = 0;
  ZCode   *vert_z;
  PetscCall(PetscZSetGetSize(vset, &local_verts));
  PetscCall(PetscMalloc1(local_verts, &vert_z));
  PetscCall(PetscZSetGetElems(vset, &off, vert_z));
  PetscCall(PetscZSetDestroy(&vset));
  // ZCode is unsigned for bitwise convenience, but highest bit should never be set, so can interpret as signed
  PetscCall(PetscSortInt64(local_verts, (PetscInt64 *)vert_z));

  PetscCall(DMPlexSetChart(dm, 0, local_elems + local_verts));
  for (PetscInt e = 0; e < local_elems; e++) PetscCall(DMPlexSetConeSize(dm, e, PetscPowInt(2, dim)));
  PetscCall(DMSetUp(dm));
  {
    PetscInt e = 0;
    for (ZCode z = layout.zstarts[rank]; z < layout.zstarts[rank + 1]; z++) {
      Ijk loc = ZCodeSplit(z);
      if (!IjkActive(layout.eextent, loc)) {
        z += ZStepOct(z);
        continue;
      }
      PetscInt cone[8], orient[8] = {0};
      for (PetscInt n = 0; n < PetscPowInt(2, dim); n++) {
        Ijk      inc  = closure_dim[dim][n];
        Ijk      nloc = {loc.i + inc.i, loc.j + inc.j, loc.k + inc.k};
        ZCode    v    = ZEncode(nloc);
        PetscInt ci   = ZCodeFind(v, local_verts, vert_z);
        PetscAssert(ci >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not find neighbor vertex in set");
        cone[n] = local_elems + ci;
      }
      PetscCall(DMPlexSetCone(dm, e, cone));
      PetscCall(DMPlexSetConeOrientation(dm, e, orient));
      e++;
    }
  }

  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));

  { // Create point SF
    PetscSF sf;
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), &sf));
    PetscInt owned_verts = ZCodeFind(layout.zstarts[rank + 1], local_verts, vert_z);
    if (owned_verts < 0) owned_verts = -(owned_verts + 1); // We don't care whether the key was found
    PetscInt     num_ghosts = local_verts - owned_verts;   // Due to sorting, owned vertices always come first
    PetscInt    *local_ghosts;
    PetscSFNode *ghosts;
    PetscCall(PetscMalloc1(num_ghosts, &local_ghosts));
    PetscCall(PetscMalloc1(num_ghosts, &ghosts));
    for (PetscInt i = 0; i < num_ghosts;) {
      ZCode    z           = vert_z[owned_verts + i];
      PetscInt remote_rank = ZCodeFind(z, size + 1, layout.zstarts), remote_count = 0;
      if (remote_rank < 0) remote_rank = -(remote_rank + 1) - 1;
      // We have a new remote rank; find all the ghost indices (which are contiguous in vert_z)

      // Count the elements on remote_rank
      PetscInt remote_elem = ZLayoutElementsOnRank(&layout, remote_rank);

      // Traverse vertices and make ghost links
      for (ZCode rz = layout.zstarts[remote_rank]; rz < layout.zstarts[remote_rank + 1]; rz++) {
        Ijk loc = ZCodeSplit(rz);
        if (rz == z) {
          local_ghosts[i] = local_elems + owned_verts + i;
          ghosts[i].rank  = remote_rank;
          ghosts[i].index = remote_elem + remote_count;
          i++;
          if (i == num_ghosts) break;
          z = vert_z[owned_verts + i];
        }
        if (IjkActive(layout.vextent, loc)) remote_count++;
        else rz += ZStepOct(rz);
      }
    }
    PetscCall(PetscSFSetGraph(sf, local_elems + local_verts, num_ghosts, local_ghosts, PETSC_OWN_POINTER, ghosts, PETSC_OWN_POINTER));
    PetscCall(PetscObjectSetName((PetscObject)sf, "SFC Point SF"));
    PetscCall(DMSetPointSF(dm, sf));
    PetscCall(PetscSFDestroy(&sf));
  }
  {
    Vec          coordinates;
    PetscScalar *coords;
    PetscSection coord_section;
    PetscInt     coord_size;
    PetscCall(DMGetCoordinateSection(dm, &coord_section));
    PetscCall(PetscSectionSetNumFields(coord_section, 1));
    PetscCall(PetscSectionSetFieldComponents(coord_section, 0, dim));
    PetscCall(PetscSectionSetChart(coord_section, local_elems, local_elems + local_verts));
    for (PetscInt v = 0; v < local_verts; v++) {
      PetscInt point = local_elems + v;
      PetscCall(PetscSectionSetDof(coord_section, point, dim));
      PetscCall(PetscSectionSetFieldDof(coord_section, point, 0, dim));
    }
    PetscCall(PetscSectionSetUp(coord_section));
    PetscCall(PetscSectionGetStorageSize(coord_section, &coord_size));
    PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
    PetscCall(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
    PetscCall(VecSetSizes(coordinates, coord_size, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(coordinates, dim));
    PetscCall(VecSetType(coordinates, VECSTANDARD));
    PetscCall(VecGetArray(coordinates, &coords));
    for (PetscInt v = 0; v < local_verts; v++) {
      Ijk loc             = ZCodeSplit(vert_z[v]);
      coords[v * dim + 0] = lower[0] + loc.i * (upper[0] - lower[0]) / layout.eextent.i;
      if (dim > 1) coords[v * dim + 1] = lower[1] + loc.j * (upper[1] - lower[1]) / layout.eextent.j;
      if (dim > 2) coords[v * dim + 2] = lower[2] + loc.k * (upper[2] - lower[2]) / layout.eextent.k;
    }
    PetscCall(VecRestoreArray(coordinates, &coords));
    PetscCall(DMSetCoordinatesLocal(dm, coordinates));
    PetscCall(VecDestroy(&coordinates));
  }
  if (interpolate) {
    PetscCall(DMPlexInterpolateInPlace_Internal(dm));
    // It's currently necessary to orient the donor and periodic edges consistently. An easy way to ensure that is ot
    // give all edges positive orientation. Since vertices are created in Z-order, all ranks will agree about the
    // ordering cone[0] < cone[1]. This is overkill and it would be nice to remove this preparation and make
    // DMPlexCreateIsoperiodicClosureSF_Private() more resilient, so it fixes any inconsistent orientations. That might
    // be needed in a general CGNS reader, for example.
    PetscCall(DMPlexOrientPositiveEdges_Private(dm));

    DMLabel label;
    PetscCall(DMCreateLabel(dm, "Face Sets"));
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscSegBuffer per_faces[3], donor_face_closure[3], my_donor_faces[3];
    for (PetscInt i = 0; i < 3; i++) {
      PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 64, &per_faces[i]));
      PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 64, &my_donor_faces[i]));
      PetscCall(PetscSegBufferCreate(sizeof(ZCode), 64 * PetscPowInt(2, dim), &donor_face_closure[i]));
    }
    PetscInt fStart, fEnd, vStart, vEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    for (PetscInt f = fStart; f < fEnd; f++) {
      PetscInt npoints, *points = NULL, num_fverts = 0, fverts[8];
      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &npoints, &points));
      PetscInt bc_count[6] = {0};
      for (PetscInt i = 0; i < npoints; i++) {
        PetscInt p = points[2 * i];
        if (p < vStart || vEnd <= p) continue;
        fverts[num_fverts++] = p;
        Ijk loc              = ZCodeSplit(vert_z[p - vStart]);
        // Convention here matches DMPlexCreateCubeMesh_Internal
        bc_count[0] += loc.i == 0;
        bc_count[1] += loc.i == layout.vextent.i - 1;
        bc_count[2] += loc.j == 0;
        bc_count[3] += loc.j == layout.vextent.j - 1;
        bc_count[4] += loc.k == 0;
        bc_count[5] += loc.k == layout.vextent.k - 1;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &npoints, &points));
      for (PetscInt bc = 0, bc_match = 0; bc < 2 * dim; bc++) {
        if (bc_count[bc] == PetscPowInt(2, dim - 1)) {
          PetscCall(DMPlexOrientPoint(dm, f, face_orient_dim[dim][bc]));
          if (periodicity[bc / 2] == DM_BOUNDARY_PERIODIC) {
            PetscInt *put;
            if (bc % 2 == 0) { // donor face; no label
              PetscCall(PetscSegBufferGet(my_donor_faces[bc / 2], 1, &put));
              *put = f;
            } else { // periodic face
              PetscCall(PetscSegBufferGet(per_faces[bc / 2], 1, &put));
              *put = f;
              ZCode *zput;
              PetscCall(PetscSegBufferGet(donor_face_closure[bc / 2], num_fverts, &zput));
              for (PetscInt i = 0; i < num_fverts; i++) {
                Ijk loc = ZCodeSplit(vert_z[fverts[i] - vStart]);
                switch (bc / 2) {
                case 0:
                  loc.i = 0;
                  break;
                case 1:
                  loc.j = 0;
                  break;
                case 2:
                  loc.k = 0;
                  break;
                }
                *zput++ = ZEncode(loc);
              }
            }
            continue;
          }
          PetscAssert(bc_match == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face matches multiple face sets");
          PetscCall(DMLabelSetValue(label, f, face_marker_dim[dim][bc]));
          bc_match++;
        }
      }
    }
    // Ensure that the Coordinate DM has our new boundary labels
    DM cdm;
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCopyLabels(dm, cdm, PETSC_COPY_VALUES, PETSC_FALSE, DM_COPY_LABELS_FAIL));
    if (periodicity[0] == DM_BOUNDARY_PERIODIC || (dim > 1 && periodicity[1] == DM_BOUNDARY_PERIODIC) || (dim > 2 && periodicity[2] == DM_BOUNDARY_PERIODIC)) {
      PetscCall(DMPlexCreateBoxMesh_Tensor_SFC_Periodicity_Private(dm, &layout, vert_z, per_faces, lower, upper, periodicity, donor_face_closure, my_donor_faces));
    }
    for (PetscInt i = 0; i < 3; i++) {
      PetscCall(PetscSegBufferDestroy(&per_faces[i]));
      PetscCall(PetscSegBufferDestroy(&donor_face_closure[i]));
      PetscCall(PetscSegBufferDestroy(&my_donor_faces[i]));
    }
  }
  PetscCall(PetscFree(layout.zstarts));
  PetscCall(PetscFree(vert_z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetIsoperiodicFaceSF - Express periodicity from an existing mesh

  Logically Collective

  Input Parameters:
+ dm           - The `DMPLEX` on which to set periodicity
. num_face_sfs - Number of `PetscSF`s in `face_sfs`
- face_sfs     - Array of `PetscSF` in which roots are (owned) donor faces and leaves are faces that must be matched to a (possibly remote) donor face.

  Level: advanced

  Note:
  One can use `-dm_plex_shape zbox` to use this mode of periodicity, wherein the periodic points are distinct both globally
  and locally, but are paired when creating a global dof space.

.seealso: [](ch_unstructured), `DMPLEX`, `DMGetGlobalSection()`, `DMPlexGetIsoperiodicFaceSF()`
@*/
PetscErrorCode DMPlexSetIsoperiodicFaceSF(DM dm, PetscInt num_face_sfs, PetscSF *face_sfs)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (face_sfs) PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMGetIsoperiodicPointSF_C", DMGetIsoperiodicPointSF_Plex));
  if (face_sfs == plex->periodic.face_sfs && num_face_sfs == plex->periodic.num_face_sfs) PetscFunctionReturn(PETSC_SUCCESS);

  for (PetscInt i = 0; i < num_face_sfs; i++) PetscCall(PetscObjectReference((PetscObject)face_sfs[i]));

  if (plex->periodic.num_face_sfs > 0) {
    for (PetscInt i = 0; i < plex->periodic.num_face_sfs; i++) PetscCall(PetscSFDestroy(&plex->periodic.face_sfs[i]));
    PetscCall(PetscFree(plex->periodic.face_sfs));
  }

  plex->periodic.num_face_sfs = num_face_sfs;
  PetscCall(PetscCalloc1(num_face_sfs, &plex->periodic.face_sfs));
  for (PetscInt i = 0; i < num_face_sfs; i++) plex->periodic.face_sfs[i] = face_sfs[i];

  DM cdm = dm->coordinates[0].dm; // Can't DMGetCoordinateDM because it automatically creates one
  if (cdm) {
    PetscCall(DMPlexSetIsoperiodicFaceSF(cdm, num_face_sfs, face_sfs));
    if (face_sfs) cdm->periodic.setup = DMPeriodicCoordinateSetUp_Internal;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetIsoperiodicFaceSF - Obtain periodicity for a mesh

  Logically Collective

  Input Parameter:
. dm - The `DMPLEX` for which to obtain periodic relation

  Output Parameters:
+ num_face_sfs - Number of `PetscSF`s in the array
- face_sfs     - Array of `PetscSF` in which roots are (owned) donor faces and leaves are faces that must be matched to a (possibly remote) donor face.

  Level: advanced

.seealso: [](ch_unstructured), `DMPLEX`, `DMGetGlobalSection()`, `DMPlexSetIsoperiodicFaceSF()`
@*/
PetscErrorCode DMPlexGetIsoperiodicFaceSF(DM dm, PetscInt *num_face_sfs, const PetscSF **face_sfs)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *face_sfs     = plex->periodic.face_sfs;
  *num_face_sfs = plex->periodic.num_face_sfs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexSetIsoperiodicFaceTransform - set geometric transform from donor to periodic points

  Logically Collective

  Input Parameters:
+ dm - `DMPLEX` that has been configured with `DMPlexSetIsoperiodicFaceSF()`
. n  - Number of transforms in array
- t  - Array of 4x4 affine transformation basis.

  Level: advanced

  Notes:
  Affine transforms are 4x4 matrices in which the leading 3x3 block expresses a rotation (or identity for no rotation),
  the last column contains a translation vector, and the bottom row is all zero except the last entry, which must always
  be 1. This representation is common in geometric modeling and allows affine transformations to be composed using
  simple matrix multiplication.

  Although the interface accepts a general affine transform, only affine translation is supported at present.

  Developer Notes:
  This interface should be replaced by making BasisTransform public, expanding it to support affine representations, and
  adding GPU implementations to apply the G2L/L2G transforms.

.seealso: [](ch_unstructured), `DMPLEX`, `DMGetGlobalSection()`, `DMPlexSetIsoperiodicFaceSF()`
@*/
PetscErrorCode DMPlexSetIsoperiodicFaceTransform(DM dm, PetscInt n, const PetscScalar *t)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(n == plex->periodic.num_face_sfs, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of transforms (%" PetscInt_FMT ") must equal number of isoperiodc face SFs (%" PetscInt_FMT ")", n, plex->periodic.num_face_sfs);

  PetscCall(PetscMalloc1(n, &plex->periodic.transform));
  for (PetscInt i = 0; i < n; i++) {
    for (PetscInt j = 0; j < 4; j++) {
      for (PetscInt k = 0; k < 4; k++) {
        PetscCheck(j != k || t[i * 16 + j * 4 + k] == 1., PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Rotated transforms not supported");
        plex->periodic.transform[i][j][k] = t[i * 16 + j * 4 + k];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

// Since element/vertex box extents are typically not equal powers of 2, Z codes that lie within the domain are not contiguous.
static PetscErrorCode ZLayoutCreate(PetscMPIInt size, const PetscInt eextent[3], const PetscInt vextent[3], ZLayout *zlayout)
{
  ZLayout layout;

  PetscFunctionBegin;
  layout.eextent.i = eextent[0];
  layout.eextent.j = eextent[1];
  layout.eextent.k = eextent[2];
  layout.vextent.i = vextent[0];
  layout.vextent.j = vextent[1];
  layout.vextent.k = vextent[2];
  layout.comm_size = size;
  PetscCall(PetscMalloc1(size + 1, &layout.zstarts));

  PetscInt total_elems = eextent[0] * eextent[1] * eextent[2];
  ZCode    z           = 0;
  layout.zstarts[0]    = 0;
  for (PetscMPIInt r = 0; r < size; r++) {
    PetscInt elems_needed = (total_elems / size) + (total_elems % size > r), count;
    for (count = 0; count < elems_needed; z++) {
      Ijk loc = ZCodeSplit(z);
      if (IjkActive(layout.eextent, loc)) count++;
    }
    // Pick up any extra vertices in the Z ordering before the next rank's first owned element.
    //
    // TODO: This leads to poorly balanced vertices when eextent is a power of 2, since all the fringe vertices end up
    // on the last rank. A possible solution is to balance the Z-order vertices independently from the cells, which will
    // result in a lot of element closures being remote. We could finish marking boundary conditions, then do a round of
    // vertex ownership smoothing (which would reorder and redistribute vertices without touching element distribution).
    // Another would be to have an analytic ownership criteria for vertices in the fringe veextent - eextent. This would
    // complicate the job of identifying an owner and its offset.
    for (; z <= ZEncode(layout.vextent); z++) {
      Ijk loc = ZCodeSplit(z);
      if (IjkActive(layout.eextent, loc)) break;
    }
    layout.zstarts[r + 1] = z;
  }
  *zlayout = layout;
  PetscFunctionReturn(0);
}

static PetscInt ZLayoutElementsOnRank(const ZLayout *layout, PetscMPIInt rank)
{
  PetscInt remote_elem = 0;
  for (ZCode rz = layout->zstarts[rank]; rz < layout->zstarts[rank + 1]; rz++) {
    Ijk loc = ZCodeSplit(rz);
    if (IjkActive(layout->eextent, loc)) remote_elem++;
  }
  return remote_elem;
}

PetscInt ZCodeFind(ZCode key, PetscInt n, const ZCode X[])
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

static PetscErrorCode DMPlexCreateBoxMesh_Tensor_SFC_Periodicity_Private(DM dm, const ZLayout *layout, const ZCode *vert_z, PetscSegBuffer per_faces, const DMBoundaryType *periodicity, PetscSegBuffer donor_face_closure, PetscSegBuffer my_donor_faces)
{
  MPI_Comm     comm;
  size_t       num_faces;
  PetscInt     dim, *faces, vStart, vEnd;
  PetscMPIInt  size;
  ZCode       *donor_verts, *donor_minz;
  PetscSFNode *leaf;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetDimension(dm, &dim));
  const PetscInt csize = PetscPowInt(2, dim - 1);
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscSegBufferGetSize(per_faces, &num_faces));
  PetscCall(PetscSegBufferExtractInPlace(per_faces, &faces));
  PetscCall(PetscSegBufferExtractInPlace(donor_face_closure, &donor_verts));
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
    PetscCheck(sorted, comm, PETSC_ERR_PLIB, "minz not sorted; periodicity in multiple dimensions not yet supported");
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
  PetscSF sfper;
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), &sfper));
  PetscCall(PetscSFSetGraph(sfper, vEnd - vStart, num_faces, PETSC_NULL, PETSC_USE_POINTER, leaf, PETSC_USE_POINTER));
  const PetscInt *my_donor_degree;
  PetscCall(PetscSFComputeDegreeBegin(sfper, &my_donor_degree));
  PetscCall(PetscSFComputeDegreeEnd(sfper, &my_donor_degree));
  PetscInt num_multiroots = 0;
  for (PetscInt i = 0; i < vEnd - vStart; i++) {
    num_multiroots += my_donor_degree[i];
    if (my_donor_degree[i] == 0) continue;
    PetscAssert(my_donor_degree[i] == 1, comm, PETSC_ERR_SUP, "Local vertex has multiple faces");
  }
  PetscInt *my_donors, *donor_indices, *my_donor_indices;
  size_t    num_my_donors;
  PetscCall(PetscSegBufferGetSize(my_donor_faces, &num_my_donors));
  PetscCheck((PetscInt)num_my_donors == num_multiroots, PETSC_COMM_SELF, PETSC_ERR_SUP, "Donor request does not match expected donors");
  PetscCall(PetscSegBufferExtractInPlace(my_donor_faces, &my_donors));
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
    PetscAssert(my_donor_degree[minv - vStart] == 1, comm, PETSC_ERR_SUP, "Local vertex not requested");
    my_donor_indices[minv - vStart] = f;
  }
  PetscCall(PetscMalloc1(num_faces, &donor_indices));
  PetscCall(PetscSFBcastBegin(sfper, MPIU_INT, my_donor_indices, donor_indices, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfper, MPIU_INT, my_donor_indices, donor_indices, MPI_REPLACE));
  PetscCall(PetscFree(my_donor_indices));
  // Modify our leafs so they point to donor faces instead of donor minz. Additionally, give them indices as faces.
  for (PetscInt i = 0; i < (PetscInt)num_faces; i++) leaf[i].index = donor_indices[i];
  PetscCall(PetscFree(donor_indices));
  PetscInt pStart, pEnd;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSFSetGraph(sfper, pEnd - pStart, num_faces, faces, PETSC_COPY_VALUES, leaf, PETSC_OWN_POINTER));
  PetscCall(PetscObjectSetName((PetscObject)sfper, "Periodic Faces"));
  PetscCall(PetscSFViewFromOptions(sfper, NULL, "-sfper_view"));

  PetscCall(DMPlexSetPeriodicFaceSF(dm, sfper));

  PetscScalar t[4][4] = {{0}};
  t[0][0]             = 1;
  t[1][1]             = 1;
  t[2][2]             = 1;
  t[3][3]             = 1;
  for (PetscInt i = 0; i < dim; i++)
    if (periodicity[i] == DM_BOUNDARY_PERIODIC) t[i][3] = 1;
  PetscCall(DMPlexSetPeriodicFaceTransform(dm, t));
  PetscCall(PetscSFDestroy(&sfper));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoordAddPeriodicOffsets_Private(DM dm, Vec g, InsertMode mode, Vec l, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecScatterBegin(dm->periodic.affine_to_local, dm->periodic.affine, l, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(dm->periodic.affine_to_local, dm->periodic.affine, l, ADD_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

// Start with an SF for a positive depth (e.g., faces) and create a new SF for matched closure.
//
// While the image face and corresponding donor face might not have the same orientation, it is assumed that the vertex
// numbering is consistent.
static PetscErrorCode DMPlexSFCreateClosureSF_Private(DM dm, PetscSF face_sf, PetscSF *closure_sf, IS *is_points)
{
  MPI_Comm           comm;
  PetscInt           nroots, nleaves, npoints;
  const PetscInt    *filocal, *pilocal;
  const PetscSFNode *firemote, *piremote;
  PetscMPIInt        rank;
  PetscSF            point_sf;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSFGetGraph(face_sf, &nroots, &nleaves, &filocal, &firemote));
  PetscCall(DMGetPointSF(dm, &point_sf)); // Point SF has remote points
  PetscCall(PetscSFGetGraph(point_sf, NULL, &npoints, &pilocal, &piremote));
  PetscInt *rootdata, *leafdata;
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
    PetscAssert(donor_dof[p] == cl_size - 1, comm, PETSC_ERR_PLIB, "Reduced leaf cone sizes do not match root cone sizes");
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

  // Pack root buffer with owner for every point in the root cones
  PetscSFNode *donor_closure;
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

  PetscSFNode *leaf_donor_closure;
  PetscCall(PetscMalloc1(leaf_offset, &leaf_donor_closure));
  PetscCall(PetscSFBcastBegin(sf_closure, MPIU_2INT, donor_closure, leaf_donor_closure, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf_closure, MPIU_2INT, donor_closure, leaf_donor_closure, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf_closure));
  PetscCall(PetscFree(donor_closure));

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
      } else PetscCheck(new_iremote[c].rank == lc.rank && new_iremote[c].index == lc.index, comm, PETSC_ERR_PLIB, "Mismatched cone ordering between faces");
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
  PetscCall(ISCreateGeneral(comm, num_new_leaves, leafdata, PETSC_COPY_VALUES, is_points));

  PetscSF csf;
  PetscCall(PetscSFCreate(comm, &csf));
  PetscCall(PetscSFSetGraph(csf, nroots, num_new_leaves, leafdata, PETSC_COPY_VALUES, new_iremote, PETSC_COPY_VALUES));
  PetscCall(PetscFree(new_iremote)); // copy and delete because new_iremote is longer than it needs to be
  PetscCall(PetscFree2(rootdata, leafdata));

  // TODO: this is a lie; it's only the periodic point SF; need to compose with standard point SF
  PetscCall(PetscObjectSetName((PetscObject)csf, "Composed Periodic Points"));
  PetscSFViewFromOptions(csf, NULL, "-csf_view");
  *closure_sf = csf;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetPointSFComposed_Plex(DM dm, PetscSF *sf)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  if (!plex->periodic.composed_sf) {
    PetscSF face_sf = plex->periodic.face_sf;

    PetscCall(DMPlexSFCreateClosureSF_Private(dm, face_sf, &plex->periodic.composed_sf, &plex->periodic.periodic_points));
  }
  if (sf) *sf = plex->periodic.composed_sf;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPeriodicCoordinateSetUp_Internal(DM dm)
{
  DM_Plex *plex = (DM_Plex *)dm->data;
  PetscFunctionBegin;
  if (!plex->periodic.face_sf) PetscFunctionReturn(0);
  PetscCall(DMGetPointSFComposed_Plex(dm, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMGetPointSFComposed_C", DMGetPointSFComposed_Plex));

  PetscInt dim;
  PetscCall(DMGetDimension(dm, &dim));
  size_t count;
  IS     isdof;
  {
    PetscInt        npoints;
    const PetscInt *points;
    IS              is = plex->periodic.periodic_points;
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
        *slot = off / dim;
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
      for (PetscInt j = 0; j < dim; j++) x[i * dim + j] = plex->periodic.transform[j][3];
    }
    PetscCall(VecRestoreArrayWrite(P, &x));
  }

  dm->periodic.affine_to_local = scatter;
  dm->periodic.affine          = P;
  PetscCall(DMGlobalToLocalHookAdd(dm, NULL, DMCoordAddPeriodicOffsets_Private, NULL));
  PetscFunctionReturn(0);
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
  PetscValidPointer(dm, 1);
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
    if (IjkActive(layout.vextent, loc)) PetscZSetAdd(vset, z);
    if (IjkActive(layout.eextent, loc)) {
      local_elems++;
      // Add all neighboring vertices to set
      for (PetscInt n = 0; n < PetscPowInt(2, dim); n++) {
        Ijk   inc  = closure_dim[dim][n];
        Ijk   nloc = {loc.i + inc.i, loc.j + inc.j, loc.k + inc.k};
        ZCode v    = ZEncode(nloc);
        PetscZSetAdd(vset, v);
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
      if (!IjkActive(layout.eextent, loc)) continue;
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
    PetscSFCreate(PetscObjectComm((PetscObject)dm), &sf);
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

    DMLabel label;
    PetscCall(DMCreateLabel(dm, "Face Sets"));
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscSegBuffer per_faces, donor_face_closure, my_donor_faces;
    PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 64, &per_faces));
    PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 64, &my_donor_faces));
    PetscCall(PetscSegBufferCreate(sizeof(ZCode), 64 * PetscPowInt(2, dim), &donor_face_closure));
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
              PetscCall(PetscSegBufferGet(my_donor_faces, 1, &put));
              *put = f;
            } else { // periodic face
              PetscCall(PetscSegBufferGet(per_faces, 1, &put));
              *put = f;
              ZCode *zput;
              PetscCall(PetscSegBufferGet(donor_face_closure, num_fverts, &zput));
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
      PetscCall(DMPlexCreateBoxMesh_Tensor_SFC_Periodicity_Private(dm, &layout, vert_z, per_faces, periodicity, donor_face_closure, my_donor_faces));
      PetscSF sfper;
      PetscCall(DMPlexGetPeriodicFaceSF(dm, &sfper));
      PetscCall(DMPlexSetPeriodicFaceSF(cdm, sfper));
      cdm->periodic.setup = DMPeriodicCoordinateSetUp_Internal;
    }
    PetscCall(PetscSegBufferDestroy(&per_faces));
    PetscCall(PetscSegBufferDestroy(&donor_face_closure));
    PetscCall(PetscSegBufferDestroy(&my_donor_faces));
  }
  PetscCall(PetscFree(layout.zstarts));
  PetscCall(PetscFree(vert_z));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetPeriodicFaceSF - Express periodicity from an existing mesh

  Logically collective

  Input Parameters:
+ dm - The `DMPLEX` on which to set periodicity
- face_sf - SF in which roots are (owned) donor faces and leaves are faces that must be matched to a (possibly remote) donor face.

  Level: advanced

  Notes:

  One can use `-dm_plex_box_sfc` to use this mode of periodicity, wherein the periodic points are distinct both globally
  and locally, but are paired when creating a global dof space.

.seealso: [](chapter_unstructured), `DMPLEX`, `DMGetGlobalSection()`, `DMPlexGetPeriodicFaceSF()`
@*/
PetscErrorCode DMPlexSetPeriodicFaceSF(DM dm, PetscSF face_sf)
{
  DM_Plex *plex = (DM_Plex *)dm->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectReference((PetscObject)face_sf));
  PetscCall(PetscSFDestroy(&plex->periodic.face_sf));
  plex->periodic.face_sf = face_sf;
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMGetPointSFComposed_C", DMGetPointSFComposed_Plex));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPeriodicFaceSF - Obtain periodicity for a mesh

  Logically collective

  Input Parameters:
. dm - The `DMPLEX` for which to obtain periodic relation

  Output Parameters:
. face_sf - SF in which roots are (owned) donor faces and leaves are faces that must be matched to a (possibly remote) donor face.

  Level: advanced

.seealso: [](chapter_unstructured), `DMPLEX`, `DMGetGlobalSection()`, `DMPlexSetPeriodicFaceSF()`
@*/
PetscErrorCode DMPlexGetPeriodicFaceSF(DM dm, PetscSF *face_sf)
{
  DM_Plex *plex = (DM_Plex *)dm->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *face_sf = plex->periodic.face_sf;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetPeriodicFaceTransform - set geometric transform from donor to periodic points

  Logically Collective

  Input Arguments:
+ dm - `DMPlex` that has been configured with `DMPlexSetPeriodicFaceSF()`
- t - 4x4 affine transformation basis.

@*/
PetscErrorCode DMPlexSetPeriodicFaceTransform(DM dm, const PetscScalar t[4][4])
{
  DM_Plex *plex = (DM_Plex *)dm->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (PetscInt i = 0; i < 4; i++) {
    for (PetscInt j = 0; j < 4; j++) {
      PetscCheck(i != j || t[i][j] == 1., PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Rotated transforms not supported");
      plex->periodic.transform[i][j] = t[i][j];
    }
  }
  PetscFunctionReturn(0);
}

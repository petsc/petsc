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

// ***** Overview of ZCode *******
// The SFC uses integer indexing for each dimension and encodes them into a single integer by interleaving the bits of each index.
// This is known as Morton encoding, and is referred to as ZCode in this code.
// So for index i having bits [i2,i1,i0], and similar for indexes j and k, the ZCode (Morton number) would be:
//    [k2,j2,i2,k1,j1,i1,k0,j0,i0]
// This encoding allows for easier traversal of the SFC structure (see https://en.wikipedia.org/wiki/Z-order_curve and `ZStepOct()`).
// `ZEncode()` is used to go from indices to ZCode, while `ZCodeSplit()` goes from ZCode back to indices.

// Decodes the leading interleaved index from a ZCode
// e.g. [k2,j2,i2,k1,j1,i1,k0,j0,i0] -> [i2,i1,i0]
// Magic numbers taken from https://stackoverflow.com/a/18528775/7564988 (translated to octal)
static unsigned ZCodeSplit1(ZCode z)
{
  z &= 0111111111111111111111;
  z = (z | z >> 2) & 0103030303030303030303;
  z = (z | z >> 4) & 0100170017001700170017;
  z = (z | z >> 8) & 0370000037700000377;
  z = (z | z >> 16) & 0370000000000177777;
  z = (z | z >> 32) & 07777777;
  return (unsigned)z;
}

// Encodes the leading interleaved index from a ZCode
// e.g. [i2,i1,i0] -> [0,0,i2,0,0,i1,0,0,i0]
static ZCode ZEncode1(unsigned t)
{
  ZCode z = t;
  z &= 07777777;
  z = (z | z << 32) & 0370000000000177777;
  z = (z | z << 16) & 0370000037700000377;
  z = (z | z << 8) & 0100170017001700170017;
  z = (z | z << 4) & 0103030303030303030303;
  z = (z | z << 2) & 0111111111111111111111;
  return z;
}

// Decodes i j k indices from a ZCode.
// Uses `ZCodeSplit1()` by shifting ZCode so that the leading index is the desired one to decode
static Ijk ZCodeSplit(ZCode z)
{
  Ijk c;
  c.i = ZCodeSplit1(z >> 2);
  c.j = ZCodeSplit1(z >> 1);
  c.k = ZCodeSplit1(z >> 0);
  return c;
}

// Encodes i j k indices to a ZCode.
// Uses `ZCodeEncode1()` by shifting resulting ZCode to the appropriate bit placement
static ZCode ZEncode(Ijk c)
{
  ZCode z = (ZEncode1((unsigned int)c.i) << 2) | (ZEncode1((unsigned int)c.j) << 1) | ZEncode1((unsigned int)c.k);
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

static inline PetscBool IsPointInsideStratum(PetscInt point, PetscInt pStart, PetscInt pEnd)
{
  return (point >= pStart && point < pEnd) ? PETSC_TRUE : PETSC_FALSE;
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
    PetscCount   num_faces;
    PetscInt    *faces;
    ZCode       *donor_verts, *donor_minz;
    PetscSFNode *leaf;
    PetscCount   num_multiroots = 0;
    PetscInt     pStart, pEnd;
    PetscBool    sorted;
    PetscInt     inum_faces;

    if (periodicity[direction] != DM_BOUNDARY_PERIODIC) continue;
    PetscCall(PetscSegBufferGetSize(per_faces[direction], &num_faces));
    PetscCall(PetscSegBufferExtractInPlace(per_faces[direction], &faces));
    PetscCall(PetscSegBufferExtractInPlace(donor_face_closure[direction], &donor_verts));
    PetscCall(PetscMalloc1(num_faces, &donor_minz));
    PetscCall(PetscMalloc1(num_faces, &leaf));
    for (PetscCount i = 0; i < num_faces; i++) {
      ZCode minz = donor_verts[i * csize];

      for (PetscInt j = 1; j < csize; j++) minz = PetscMin(minz, donor_verts[i * csize + j]);
      donor_minz[i] = minz;
    }
    PetscCall(PetscIntCast(num_faces, &inum_faces));
    PetscCall(PetscSortedInt64(inum_faces, (const PetscInt64 *)donor_minz, &sorted));
    // If a donor vertex were chosen to broker multiple faces, we would have a logic error.
    // Checking for sorting is a cheap check that there are no duplicates.
    PetscCheck(sorted, PETSC_COMM_SELF, PETSC_ERR_PLIB, "minz not sorted; possible duplicates not checked");
    for (PetscCount i = 0; i < num_faces;) {
      ZCode       z = donor_minz[i];
      PetscMPIInt remote_rank, remote_count = 0;

      PetscCall(PetscMPIIntCast(ZCodeFind(z, size + 1, layout->zstarts), &remote_rank));
      if (remote_rank < 0) remote_rank = -(remote_rank + 1) - 1;
      // Process all the vertices on this rank
      for (ZCode rz = layout->zstarts[remote_rank]; rz < layout->zstarts[remote_rank + 1]; rz++) {
        Ijk loc = ZCodeSplit(rz);

        if (rz == z) {
          leaf[i].rank  = remote_rank;
          leaf[i].index = remote_count;
          i++;
          if (i == num_faces) break;
          z = donor_minz[i];
        }
        if (IjkActive(layout->vextent, loc)) remote_count++;
      }
    }
    PetscCall(PetscFree(donor_minz));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), &face_sfs[num_directions]));
    PetscCall(PetscSFSetGraph(face_sfs[num_directions], vEnd - vStart, inum_faces, NULL, PETSC_USE_POINTER, leaf, PETSC_USE_POINTER));
    const PetscInt *my_donor_degree;
    PetscCall(PetscSFComputeDegreeBegin(face_sfs[num_directions], &my_donor_degree));
    PetscCall(PetscSFComputeDegreeEnd(face_sfs[num_directions], &my_donor_degree));

    for (PetscInt i = 0; i < vEnd - vStart; i++) {
      num_multiroots += my_donor_degree[i];
      if (my_donor_degree[i] == 0) continue;
      PetscAssert(my_donor_degree[i] == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Local vertex has multiple faces");
    }
    PetscInt  *my_donors, *donor_indices, *my_donor_indices;
    PetscCount num_my_donors;

    PetscCall(PetscSegBufferGetSize(my_donor_faces[direction], &num_my_donors));
    PetscCheck(num_my_donors == num_multiroots, PETSC_COMM_SELF, PETSC_ERR_SUP, "Donor request (%" PetscCount_FMT ") does not match expected donors (%" PetscCount_FMT ")", num_my_donors, num_multiroots);
    PetscCall(PetscSegBufferExtractInPlace(my_donor_faces[direction], &my_donors));
    PetscCall(PetscMalloc1(vEnd - vStart, &my_donor_indices));
    for (PetscCount i = 0; i < num_my_donors; i++) {
      PetscInt f = my_donors[i];
      PetscInt num_points, *points = NULL, minv = PETSC_INT_MAX;

      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &num_points, &points));
      for (PetscInt j = 0; j < num_points; j++) {
        PetscInt p = points[2 * j];
        if (!IsPointInsideStratum(p, vStart, vEnd)) continue;
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
    for (PetscCount i = 0; i < num_faces; i++) leaf[i].index = donor_indices[i];
    PetscCall(PetscFree(donor_indices));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscSFSetGraph(face_sfs[num_directions], pEnd - pStart, inum_faces, faces, PETSC_COPY_VALUES, leaf, PETSC_OWN_POINTER));
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

// Modify index array based on the transformation of `point` for the given section and field
// Used for correcting the sfNatural based on point reorientation
static PetscErrorCode DMPlexOrientFieldPointIndex(DM dm, PetscSection section, PetscInt field, PetscInt array_size, PetscInt array[], PetscInt point, PetscInt orientation)
{
  PetscInt        *copy;
  PetscInt         dof, off, point_ornt[2] = {point, orientation};
  const PetscInt **perms;

  PetscFunctionBeginUser;
  PetscCall(PetscSectionGetFieldPointSyms(section, field, 1, point_ornt, &perms, NULL));
  if (!perms) PetscFunctionReturn(PETSC_SUCCESS); // section may not have symmetries, such as Q2 finite elements
  PetscCall(PetscSectionGetDof(section, point, &dof));
  PetscCall(PetscSectionGetOffset(section, point, &off));
  PetscCheck(off + dof <= array_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Section indices exceed index array bounds");
  PetscCall(DMGetWorkArray(dm, dof, MPIU_INT, &copy));
  PetscArraycpy(copy, &array[off], dof);

  for (PetscInt i = 0; i < dof; i++) {
    if (perms[0]) array[off + perms[0][i]] = copy[i];
  }

  PetscCall(PetscSectionRestoreFieldPointSyms(section, field, 1, point_ornt, &perms, NULL));
  PetscCall(DMRestoreWorkArray(dm, dof, MPIU_INT, &copy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Modify Vec based on the transformation of `point` for the given section and field
static PetscErrorCode DMPlexOrientFieldPointVec(DM dm, PetscSection section, PetscInt field, Vec V, PetscInt point, PetscInt orientation)
{
  PetscScalar        *copy, *V_arr;
  PetscInt            dof, off, point_ornt[2] = {point, orientation};
  const PetscInt    **perms;
  const PetscScalar **rots;

  PetscFunctionBeginUser;
  PetscCall(PetscSectionGetFieldPointSyms(section, field, 1, point_ornt, &perms, &rots));
  if (!perms) PetscFunctionReturn(PETSC_SUCCESS); // section may not have symmetries, such as Q2 finite elements
  PetscCall(PetscSectionGetDof(section, point, &dof));
  PetscCall(PetscSectionGetOffset(section, point, &off));
  PetscCall(VecGetArray(V, &V_arr));
  PetscCall(DMGetWorkArray(dm, dof, MPIU_SCALAR, &copy));
  PetscArraycpy(copy, &V_arr[off], dof);

  for (PetscInt i = 0; i < dof; i++) {
    if (perms[0]) V_arr[off + perms[0][i]] = copy[i];
    if (rots[0]) V_arr[off + perms[0][i]] *= rots[0][i];
  }

  PetscCall(PetscSectionRestoreFieldPointSyms(section, field, 1, point_ornt, &perms, &rots));
  PetscCall(DMRestoreWorkArray(dm, dof, MPIU_SCALAR, &copy));
  PetscCall(VecRestoreArray(V, &V_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Reorient the point in the DMPlex while also applying necessary corrections to other structures (e.g. coordinates)
static PetscErrorCode DMPlexOrientPointWithCorrections(DM dm, PetscInt point, PetscInt ornt, PetscSection perm_section, PetscInt perm_length, PetscInt perm[])
{
  // TODO: Potential speed up if we early exit for ornt == 0 (i.e. if ornt is identity, we don't need to do anything)
  PetscFunctionBeginUser;
  PetscCall(DMPlexOrientPoint(dm, point, ornt));

  { // Correct coordinates based on new cone ordering
    DM           cdm;
    PetscSection csection;
    Vec          coordinates;
    PetscInt     pStart, pEnd;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetLocalSection(cdm, &csection));
    PetscCall(PetscSectionGetChart(csection, &pStart, &pEnd));
    if (IsPointInsideStratum(point, pStart, pEnd)) PetscCall(DMPlexOrientFieldPointVec(cdm, csection, 0, coordinates, point, ornt));
  }

  if (perm_section) {
    PetscInt num_fields;
    PetscCall(PetscSectionGetNumFields(perm_section, &num_fields));
    for (PetscInt f = 0; f < num_fields; f++) PetscCall(DMPlexOrientFieldPointIndex(dm, perm_section, f, perm_length, perm, point, ornt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates SF to communicate data from donor to periodic faces. The data can be different sizes per donor-periodic pair and is given in `point_sizes[]`
static PetscErrorCode CreateDonorToPeriodicSF(DM dm, PetscSF face_sf, PetscInt pStart, PetscInt pEnd, const PetscInt point_sizes[], PetscInt *rootbuffersize, PetscInt *leafbuffersize, PetscBT *rootbt, PetscSF *sf_closure)
{
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscInt           nroots, nleaves;
  PetscInt          *rootdata, *leafdata;
  const PetscInt    *filocal;
  const PetscSFNode *firemote;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscSFGetGraph(face_sf, &nroots, &nleaves, &filocal, &firemote));
  PetscCall(PetscCalloc2(2 * nroots, &rootdata, 2 * nroots, &leafdata));
  for (PetscInt i = 0; i < nleaves; i++) {
    PetscInt point = filocal[i];
    PetscCheck(IsPointInsideStratum(point, pStart, pEnd), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " in leaves exists outside of stratum [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, pStart, pEnd);
    leafdata[point] = point_sizes[point - pStart];
  }
  PetscCall(PetscSFReduceBegin(face_sf, MPIU_INT, leafdata, rootdata + nroots, MPIU_SUM));
  PetscCall(PetscSFReduceEnd(face_sf, MPIU_INT, leafdata, rootdata + nroots, MPIU_SUM));

  PetscInt root_offset = 0;
  PetscCall(PetscBTCreate(nroots, rootbt));
  for (PetscInt p = 0; p < nroots; p++) {
    const PetscInt *donor_dof = rootdata + nroots;
    if (donor_dof[p] == 0) {
      rootdata[2 * p]     = -1;
      rootdata[2 * p + 1] = -1;
      continue;
    }
    PetscCall(PetscBTSet(*rootbt, p));
    PetscCheck(IsPointInsideStratum(p, pStart, pEnd), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " in roots exists outside of stratum [%" PetscInt_FMT ", %" PetscInt_FMT ")", p, pStart, pEnd);
    PetscInt p_size = point_sizes[p - pStart];
    PetscCheck(donor_dof[p] == p_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reduced leaf data size (%" PetscInt_FMT ") does not match root data size (%" PetscInt_FMT ")", donor_dof[p], p_size);
    rootdata[2 * p]     = root_offset;
    rootdata[2 * p + 1] = p_size;
    root_offset += p_size;
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

  PetscCall(PetscSFCreate(comm, sf_closure));
  PetscCall(PetscSFSetGraph(*sf_closure, root_offset, leaf_offset, NULL, PETSC_USE_POINTER, closure_leaf, PETSC_OWN_POINTER));
  *rootbuffersize = root_offset;
  *leafbuffersize = leaf_offset;
  PetscCall(PetscFree2(rootdata, leafdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Determine if `key` is in `array`. `array` does NOT need to be sorted
static inline PetscBool SearchIntArray(PetscInt key, PetscInt array_size, const PetscInt array[])
{
  for (PetscInt i = 0; i < array_size; i++)
    if (array[i] == key) return PETSC_TRUE;
  return PETSC_FALSE;
}

// Translate a cone in periodic points to the cone in donor points based on the `periodic2donor` array
static inline PetscErrorCode TranslateConeP2D(const PetscInt periodic_cone[], PetscInt cone_size, const PetscInt periodic2donor[], PetscInt p2d_count, PetscInt p2d_cone[])
{
  PetscFunctionBeginUser;
  for (PetscInt p = 0; p < cone_size; p++) {
    PetscInt p2d_index = -1;
    for (PetscInt p2d = 0; p2d < p2d_count; p2d++) {
      if (periodic2donor[p2d * 2] == periodic_cone[p]) p2d_index = p2d;
    }
    PetscCheck(p2d_index >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find periodic point in periodic-to-donor array");
    p2d_cone[p] = periodic2donor[2 * p2d_index + 1];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Corrects the cone order of periodic faces (and their transitive closure's cones) to match their donor face pair.
//
// This is done by:
// 1. Communicating the donor's vertex coordinates and recursive cones (i.e. its own cone and those of it's constituent edges) to it's periodic pairs
//    - The donor vertices have the isoperiodic transform applied to them such that they should match exactly
// 2. Translating the periodic vertices into the donor vertices point IDs
// 3. Translating the cone of each periodic point into the donor point IDs
// 4. Comparing the periodic-to-donor cone to the donor cone for each point
// 5. Apply the necessary transformation to the periodic cone to make it match the donor cone
static PetscErrorCode DMPlexCorrectOrientationForIsoperiodic(DM dm)
{
  MPI_Comm        comm;
  DM_Plex        *plex = (DM_Plex *)dm->data;
  PetscInt        nroots, nleaves;
  PetscInt       *local_vec_perm = NULL, local_vec_length = 0, *global_vec_perm = NULL, global_vec_length = 0;
  const PetscInt *filocal, coords_field_id = 0;
  DM              cdm;
  PetscSection    csection, localSection = NULL;
  PetscSF         sfNatural_old = NULL;
  Vec             coordinates;
  PetscMPIInt     myrank;
  PetscBool       debug_printing = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &myrank));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCheck(coordinates, comm, PETSC_ERR_ARG_WRONGSTATE, "DM must have coordinates to setup isoperiodic");
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &csection));

  PetscCall(DMGetNaturalSF(dm, &sfNatural_old));
  // Prep data for naturalSF correction
  if (plex->periodic.num_face_sfs > 0 && sfNatural_old) {
    PetscSection       globalSection;
    PetscSF            pointSF, sectionSF;
    PetscInt           nleaves;
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;

    // Create global section with just pointSF and including constraints
    PetscCall(DMGetLocalSection(dm, &localSection));
    PetscCall(DMGetPointSF(dm, &pointSF));
    PetscCall(PetscSectionCreateGlobalSection(localSection, pointSF, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, &globalSection));

    // Set local_vec_perm to be negative values when that dof is not owned by the current rank
    // Dofs that are owned are set to their corresponding global Vec index
    PetscCall(PetscSectionGetStorageSize(globalSection, &global_vec_length));
    PetscCall(PetscSectionGetStorageSize(localSection, &local_vec_length));
    PetscCall(PetscMalloc2(global_vec_length, &global_vec_perm, local_vec_length, &local_vec_perm));
    for (PetscInt i = 0; i < global_vec_length; i++) global_vec_perm[i] = i;
    for (PetscInt i = 0; i < local_vec_length; i++) local_vec_perm[i] = -(i + 1);

    PetscCall(PetscSFCreate(comm, &sectionSF));
    PetscCall(PetscSFSetGraphSection(sectionSF, localSection, globalSection));
    PetscCall(PetscSFGetGraph(sectionSF, NULL, &nleaves, &ilocal, &iremote));
    for (PetscInt l = 0; l < nleaves; l++) {
      if (iremote[l].rank != myrank) continue;
      PetscInt local_index        = ilocal ? ilocal[l] : l;
      local_vec_perm[local_index] = global_vec_perm[iremote[l].index];
    }
    PetscCall(PetscSectionDestroy(&globalSection));
    PetscCall(PetscSFDestroy(&sectionSF));
  }

  // Create sf_vert_coords and sf_face_cones for communicating donor vertices and cones to periodic faces, respectively
  for (PetscInt f = 0; f < plex->periodic.num_face_sfs; f++) {
    PetscSF face_sf                   = plex->periodic.face_sfs[f];
    const PetscScalar (*transform)[4] = (const PetscScalar (*)[4])plex->periodic.transform[f];
    PetscInt *face_vertices_size, *face_cones_size;
    PetscInt  fStart, fEnd, vStart, vEnd, rootnumvert, leafnumvert, rootconesize, leafconesize, dim;
    PetscSF   sf_vert_coords, sf_face_cones;
    PetscBT   rootbt;

    PetscCall(DMGetCoordinateDim(dm, &dim));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(PetscCalloc2(fEnd - fStart, &face_vertices_size, fEnd - fStart, &face_cones_size));

    // Create SFs to communicate donor vertices and donor cones to periodic faces
    for (PetscInt f = fStart, index = 0; f < fEnd; f++, index++) {
      PetscInt cl_size, *closure = NULL, num_vertices = 0;
      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &cl_size, &closure));
      for (PetscInt p = 0; p < cl_size; p++) {
        PetscInt cl_point = closure[2 * p];
        if (IsPointInsideStratum(cl_point, vStart, vEnd)) num_vertices++;
        else {
          PetscInt cone_size;
          PetscCall(DMPlexGetConeSize(dm, cl_point, &cone_size));
          face_cones_size[index] += cone_size + 2;
        }
      }
      face_vertices_size[index] = num_vertices;
      face_cones_size[index] += num_vertices;
      PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &cl_size, &closure));
    }
    PetscCall(CreateDonorToPeriodicSF(dm, face_sf, fStart, fEnd, face_vertices_size, &rootnumvert, &leafnumvert, &rootbt, &sf_vert_coords));
    PetscCall(PetscBTDestroy(&rootbt));
    PetscCall(CreateDonorToPeriodicSF(dm, face_sf, fStart, fEnd, face_cones_size, &rootconesize, &leafconesize, &rootbt, &sf_face_cones));

    PetscCall(PetscSFGetGraph(face_sf, &nroots, &nleaves, &filocal, NULL));

    PetscReal *leaf_donor_coords;
    PetscInt  *leaf_donor_cones;

    { // Communicate donor coords and cones to the periodic faces
      PetscReal         *mydonor_vertices;
      PetscInt          *mydonor_cones;
      const PetscScalar *coords_arr;

      PetscCall(PetscCalloc2(rootnumvert * dim, &mydonor_vertices, rootconesize, &mydonor_cones));
      PetscCall(VecGetArrayRead(coordinates, &coords_arr));
      for (PetscInt donor_face = 0, donor_vert_offset = 0, donor_cone_offset = 0; donor_face < nroots; donor_face++) {
        if (!PetscBTLookup(rootbt, donor_face)) continue;
        PetscInt cl_size, *closure = NULL;

        PetscCall(DMPlexGetTransitiveClosure(dm, donor_face, PETSC_TRUE, &cl_size, &closure));
        // Pack vertex coordinates
        for (PetscInt p = 0; p < cl_size; p++) {
          PetscInt cl_point = closure[2 * p], dof, offset;
          if (!IsPointInsideStratum(cl_point, vStart, vEnd)) continue;
          mydonor_cones[donor_cone_offset++] = cl_point;
          PetscCall(PetscSectionGetFieldDof(csection, cl_point, coords_field_id, &dof));
          PetscCall(PetscSectionGetFieldOffset(csection, cl_point, coords_field_id, &offset));
          PetscAssert(dof == dim, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has dof size %" PetscInt_FMT ", but should match dimension size %" PetscInt_FMT, cl_point, dof, dim);
          // Apply isoperiodic transform to donor vertices such that corresponding periodic vertices should match exactly
          for (PetscInt d = 0; d < dof; d++) mydonor_vertices[donor_vert_offset * dim + d] = PetscRealPart(coords_arr[offset + d]) + PetscRealPart(transform[d][3]);
          donor_vert_offset++;
        }
        // Pack cones of face points (including face itself)
        for (PetscInt p = 0; p < cl_size; p++) {
          PetscInt        cl_point = closure[2 * p], cone_size, depth;
          const PetscInt *cone;

          PetscCall(DMPlexGetConeSize(dm, cl_point, &cone_size));
          PetscCall(DMPlexGetCone(dm, cl_point, &cone));
          PetscCall(DMPlexGetPointDepth(dm, cl_point, &depth));
          if (depth == 0) continue; // don't include vertex depth
          mydonor_cones[donor_cone_offset++] = cone_size;
          mydonor_cones[donor_cone_offset++] = cl_point;
          PetscArraycpy(&mydonor_cones[donor_cone_offset], cone, cone_size);
          donor_cone_offset += cone_size;
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm, donor_face, PETSC_TRUE, &cl_size, &closure));
      }
      PetscCall(VecRestoreArrayRead(coordinates, &coords_arr));
      PetscCall(PetscBTDestroy(&rootbt));

      MPI_Datatype vertex_unit;
      PetscMPIInt  n;
      PetscCall(PetscMPIIntCast(dim, &n));
      PetscCallMPI(MPI_Type_contiguous(n, MPIU_REAL, &vertex_unit));
      PetscCallMPI(MPI_Type_commit(&vertex_unit));
      PetscCall(PetscMalloc2(leafnumvert * 3, &leaf_donor_coords, leafconesize, &leaf_donor_cones));
      PetscCall(PetscSFBcastBegin(sf_vert_coords, vertex_unit, mydonor_vertices, leaf_donor_coords, MPI_REPLACE));
      PetscCall(PetscSFBcastBegin(sf_face_cones, MPIU_INT, mydonor_cones, leaf_donor_cones, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf_vert_coords, vertex_unit, mydonor_vertices, leaf_donor_coords, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf_face_cones, MPIU_INT, mydonor_cones, leaf_donor_cones, MPI_REPLACE));
      PetscCall(PetscSFDestroy(&sf_vert_coords));
      PetscCall(PetscSFDestroy(&sf_face_cones));
      PetscCallMPI(MPI_Type_free(&vertex_unit));
      PetscCall(PetscFree2(mydonor_vertices, mydonor_cones));
    }

    { // Determine periodic orientation w/rt donor vertices and reorient
      PetscReal tol = PetscSqr(PETSC_MACHINE_EPSILON * 1e3);
      PetscInt *periodic2donor, dm_depth, maxConeSize;
      PetscInt  coords_offset = 0, cones_offset = 0;

      PetscCall(DMPlexGetDepth(dm, &dm_depth));
      PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
      PetscCall(DMGetWorkArray(dm, 2 * PetscPowInt(maxConeSize, dm_depth - 1), MPIU_INT, &periodic2donor));

      // Translate the periodic face vertices into the donor vertices
      // Translation stored in periodic2donor
      for (PetscInt i = 0; i < nleaves; i++) {
        PetscInt  periodic_face = filocal[i], cl_size, num_verts = face_vertices_size[periodic_face - fStart];
        PetscInt  cones_size = face_cones_size[periodic_face - fStart], p2d_count = 0;
        PetscInt *closure = NULL;

        PetscCall(DMPlexGetTransitiveClosure(dm, periodic_face, PETSC_TRUE, &cl_size, &closure));
        for (PetscInt p = 0; p < cl_size; p++) {
          PetscInt     cl_point = closure[2 * p], coords_size, donor_vertex = -1;
          PetscScalar *coords = NULL;

          if (!IsPointInsideStratum(cl_point, vStart, vEnd)) continue;
          PetscCall(DMPlexVecGetClosure(dm, csection, coordinates, cl_point, &coords_size, &coords));
          PetscAssert(coords_size == dim, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has dof size %" PetscInt_FMT ", but should match dimension size %" PetscInt_FMT, cl_point, coords_size, dim);

          for (PetscInt v = 0; v < num_verts; v++) {
            PetscReal dist_sqr = 0;
            for (PetscInt d = 0; d < coords_size; d++) dist_sqr += PetscSqr(PetscRealPart(coords[d]) - leaf_donor_coords[(v + coords_offset) * dim + d]);
            if (dist_sqr < tol) {
              donor_vertex = leaf_donor_cones[cones_offset + v];
              break;
            }
          }
          PetscCheck(donor_vertex >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Periodic face %" PetscInt_FMT " could not find matching donor vertex for vertex %" PetscInt_FMT, periodic_face, cl_point);
          if (PetscDefined(USE_DEBUG)) {
            for (PetscInt c = 0; c < p2d_count; c++) PetscCheck(periodic2donor[2 * c + 1] != donor_vertex, comm, PETSC_ERR_PLIB, "Found repeated cone_point in periodic_ordering");
          }

          periodic2donor[2 * p2d_count + 0] = cl_point;
          periodic2donor[2 * p2d_count + 1] = donor_vertex;
          p2d_count++;
          PetscCall(DMPlexVecRestoreClosure(dm, csection, coordinates, cl_point, &coords_size, &coords));
        }
        coords_offset += num_verts;
        PetscCall(DMPlexRestoreTransitiveClosure(dm, periodic_face, PETSC_TRUE, &cl_size, &closure));

        { // Determine periodic orientation w/rt donor vertices and reorient
          PetscInt      depth, *p2d_cone, face_is_array[1] = {periodic_face};
          IS           *is_arrays, face_is;
          PetscSection *section_arrays;
          PetscInt     *donor_cone_array = &leaf_donor_cones[cones_offset + num_verts];

          PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1, face_is_array, PETSC_USE_POINTER, &face_is));
          PetscCall(DMPlexGetConeRecursive(dm, face_is, &depth, &is_arrays, &section_arrays));
          PetscCall(ISDestroy(&face_is));
          PetscCall(DMGetWorkArray(dm, maxConeSize, MPIU_INT, &p2d_cone));
          for (PetscInt d = 0; d < depth - 1; d++) {
            PetscInt        pStart, pEnd;
            PetscSection    section = section_arrays[d];
            const PetscInt *periodic_cone_arrays, *periodic_point_arrays;

            PetscCall(ISGetIndices(is_arrays[d], &periodic_cone_arrays));
            PetscCall(ISGetIndices(is_arrays[d + 1], &periodic_point_arrays)); // Points at d+1 correspond to the cones at d
            PetscCall(PetscSectionGetChart(section_arrays[d], &pStart, &pEnd));
            for (PetscInt p = pStart; p < pEnd; p++) {
              PetscInt periodic_cone_size, periodic_cone_offset, periodic_point = periodic_point_arrays[p];

              PetscCall(PetscSectionGetDof(section, p, &periodic_cone_size));
              PetscCall(PetscSectionGetOffset(section, p, &periodic_cone_offset));
              const PetscInt *periodic_cone = &periodic_cone_arrays[periodic_cone_offset];
              PetscCall(TranslateConeP2D(periodic_cone, periodic_cone_size, periodic2donor, p2d_count, p2d_cone));

              // Find the donor cone that matches the periodic point's cone
              PetscInt  donor_cone_offset = 0, donor_point = -1, *donor_cone = NULL;
              PetscBool cone_matches = PETSC_FALSE;
              while (donor_cone_offset < cones_size - num_verts) {
                PetscInt donor_cone_size = donor_cone_array[donor_cone_offset];
                donor_point              = donor_cone_array[donor_cone_offset + 1];
                donor_cone               = &donor_cone_array[donor_cone_offset + 2];

                if (donor_cone_size != periodic_cone_size) goto next_cone;
                for (PetscInt c = 0; c < periodic_cone_size; c++) {
                  cone_matches = SearchIntArray(donor_cone[c], periodic_cone_size, p2d_cone);
                  if (!cone_matches) goto next_cone;
                }
                // Save the found donor cone's point to the translation array. These will be used for higher depth points.
                // i.e. we save the edge translations for when we look for face cones
                periodic2donor[2 * p2d_count + 0] = periodic_point;
                periodic2donor[2 * p2d_count + 1] = donor_point;
                p2d_count++;
                break;

              next_cone:
                donor_cone_offset += donor_cone_size + 2;
              }
              PetscCheck(donor_cone_offset < cones_size - num_verts, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find donor cone equivalent to cone of periodic point %" PetscInt_FMT, periodic_point);

              { // Compare the donor cone with the translated periodic cone and reorient
                PetscInt       ornt;
                DMPolytopeType cell_type;
                PetscBool      found;
                PetscCall(DMPlexGetCellType(dm, periodic_point, &cell_type));
                PetscCall(DMPolytopeMatchOrientation(cell_type, donor_cone, p2d_cone, &ornt, &found));
                PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find transformation between donor (%" PetscInt_FMT ") and periodic (%" PetscInt_FMT ") cone's", periodic_point, donor_point);
                if (debug_printing) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Reorienting point %" PetscInt_FMT " by %" PetscInt_FMT "\n", periodic_point, ornt));
                PetscCall(DMPlexOrientPointWithCorrections(dm, periodic_point, ornt, localSection, local_vec_length, local_vec_perm));
              }
            }
            PetscCall(ISRestoreIndices(is_arrays[d], &periodic_cone_arrays));
            PetscCall(ISRestoreIndices(is_arrays[d + 1], &periodic_point_arrays));
          }
          PetscCall(DMRestoreWorkArray(dm, maxConeSize, MPIU_INT, &p2d_cone));
          PetscCall(DMPlexRestoreConeRecursive(dm, face_is, &depth, &is_arrays, &section_arrays));
        }

        PetscCall(DMPlexRestoreTransitiveClosure(dm, periodic_face, PETSC_TRUE, &cl_size, &closure));
        cones_offset += cones_size;
      }
      PetscCall(DMRestoreWorkArray(dm, 2 * PetscPowInt(maxConeSize, dm_depth - 1), MPIU_INT, &periodic2donor));
    }
    // Re-set local coordinates (i.e. destroy global coordinates if they were modified)
    PetscCall(DMSetCoordinatesLocal(dm, coordinates));

    PetscCall(PetscFree2(leaf_donor_coords, leaf_donor_cones));
    PetscCall(PetscFree2(face_vertices_size, face_cones_size));
  }

  if (sfNatural_old) { // Correct SFNatural based on the permutation of the local vector
    PetscSF      newglob_to_oldglob_sf, sfNatural_old, sfNatural_new;
    PetscSFNode *remote;

    { // Translate permutation of local Vec into permutation of global Vec
      PetscInt g = 0;
      PetscBT  global_vec_check; // Verify that global indices aren't doubled
      PetscCall(PetscBTCreate(global_vec_length, &global_vec_check));
      for (PetscInt l = 0; l < local_vec_length; l++) {
        PetscInt global_index = local_vec_perm[l];
        if (global_index < 0) continue;
        PetscCheck(!PetscBTLookupSet(global_vec_check, global_index), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Found duplicate global index %" PetscInt_FMT " in local_vec_perm", global_index);
        global_vec_perm[g++] = global_index;
      }
      PetscCheck(g == global_vec_length, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong number of non-negative local indices");
      PetscCall(PetscBTDestroy(&global_vec_check));
    }

    PetscCall(PetscMalloc1(global_vec_length, &remote));
    for (PetscInt i = 0; i < global_vec_length; i++) {
      remote[i].rank  = myrank;
      remote[i].index = global_vec_perm[i];
    }
    PetscCall(PetscFree2(global_vec_perm, local_vec_perm));
    PetscCall(PetscSFCreate(comm, &newglob_to_oldglob_sf));
    PetscCall(PetscSFSetGraph(newglob_to_oldglob_sf, global_vec_length, global_vec_length, NULL, PETSC_USE_POINTER, remote, PETSC_OWN_POINTER));
    PetscCall(DMGetNaturalSF(dm, &sfNatural_old));
    PetscCall(PetscSFCompose(newglob_to_oldglob_sf, sfNatural_old, &sfNatural_new));
    PetscCall(DMSetNaturalSF(dm, sfNatural_new));
    PetscCall(PetscSFDestroy(&sfNatural_new));
    PetscCall(PetscSFDestroy(&newglob_to_oldglob_sf));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Start with an SF for a positive depth (e.g., faces) and create a new SF for matched closure.
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

  PetscCall(DMPlexCorrectOrientationForIsoperiodic(dm));

  for (PetscInt f = 0; f < num_face_sfs; f++) {
    PetscSF   face_sf = face_sfs[f];
    PetscInt *cl_sizes;
    PetscInt  fStart, fEnd, rootbuffersize, leafbuffersize;
    PetscSF   sf_closure;
    PetscBT   rootbt;

    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(PetscMalloc1(fEnd - fStart, &cl_sizes));
    for (PetscInt f = fStart, index = 0; f < fEnd; f++, index++) {
      PetscInt cl_size, *closure = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &cl_size, &closure));
      cl_sizes[index] = cl_size - 1;
      PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &cl_size, &closure));
    }

    PetscCall(CreateDonorToPeriodicSF(dm, face_sf, fStart, fEnd, cl_sizes, &rootbuffersize, &leafbuffersize, &rootbt, &sf_closure));
    PetscCall(PetscFree(cl_sizes));
    PetscCall(PetscSFGetGraph(face_sf, &nroots, &nleaves, &filocal, &firemote));

    PetscSFNode *leaf_donor_closure;
    { // Pack root buffer with owner for every point in the root cones
      PetscSFNode       *donor_closure;
      const PetscInt    *pilocal;
      const PetscSFNode *piremote;
      PetscInt           npoints;

      PetscCall(PetscSFGetGraph(point_sf, NULL, &npoints, &pilocal, &piremote));
      PetscCall(PetscCalloc1(rootbuffersize, &donor_closure));
      for (PetscInt p = 0, root_offset = 0; p < nroots; p++) {
        if (!PetscBTLookup(rootbt, p)) continue;
        PetscInt cl_size, *closure = NULL;
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

      PetscCall(PetscMalloc1(leafbuffersize, &leaf_donor_closure));
      PetscCall(PetscSFBcastBegin(sf_closure, MPIU_SF_NODE, donor_closure, leaf_donor_closure, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf_closure, MPIU_SF_NODE, donor_closure, leaf_donor_closure, MPI_REPLACE));
      PetscCall(PetscSFDestroy(&sf_closure));
      PetscCall(PetscFree(donor_closure));
    }

    PetscSFNode *new_iremote;
    PetscCall(PetscCalloc1(nroots, &new_iremote));
    for (PetscInt i = 0; i < nroots; i++) new_iremote[i].rank = -1;
    // Walk leaves and match vertices
    for (PetscInt i = 0, leaf_offset = 0; i < nleaves; i++) {
      PetscInt  point   = filocal[i], cl_size;
      PetscInt *closure = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &cl_size, &closure));
      for (PetscInt j = 1; j < cl_size; j++) {
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
    PetscInt *leafdata;
    PetscCall(PetscMalloc1(nroots, &leafdata));
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
    PetscCall(PetscFree(leafdata));
    PetscCall(PetscBTDestroy(&rootbt));

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
    PetscCall(PetscSFReduceBegin(sf_migration, MPIU_SF_NODE, new_leafdata, rootdata, MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(sf_migration, MPIU_SF_NODE, new_leafdata, rootdata, MPI_REPLACE));
    // rootdata now contains the new owners

    // Send to leaves of old space
    for (PetscInt i = 0; i < old_npoints; i++) {
      leafdata[i].rank  = -1;
      leafdata[i].index = -1;
    }
    PetscCall(PetscSFBcastBegin(plex->periodic.face_sfs[f], MPIU_SF_NODE, rootdata, leafdata, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(plex->periodic.face_sfs[f], MPIU_SF_NODE, rootdata, leafdata, MPI_REPLACE));

    // Send to new leaf space
    PetscCall(PetscSFBcastBegin(sf_migration, MPIU_SF_NODE, leafdata, new_leafdata, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf_migration, MPIU_SF_NODE, leafdata, new_leafdata, MPI_REPLACE));

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
  DM_Plex   *plex = (DM_Plex *)dm->data;
  PetscCount count;
  IS         isdof;
  PetscInt   dim;

  PetscFunctionBegin;
  if (!plex->periodic.face_sfs) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(plex->periodic.periodic_points, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Isoperiodic PointSF must be created before this function is called");

  PetscCall(DMGetCoordinateDim(dm, &dim));
  dm->periodic.num_affines = plex->periodic.num_face_sfs;
  PetscCall(PetscFree2(dm->periodic.affine_to_local, dm->periodic.affine));
  PetscCall(PetscMalloc2(dm->periodic.num_affines, &dm->periodic.affine_to_local, dm->periodic.num_affines, &dm->periodic.affine));

  for (PetscInt f = 0; f < plex->periodic.num_face_sfs; f++) {
    PetscInt        npoints, vsize, isize;
    const PetscInt *points;
    IS              is = plex->periodic.periodic_points[f];
    PetscSegBuffer  seg;
    PetscSection    section;
    PetscInt       *ind;
    Vec             L, P;
    VecType         vec_type;
    VecScatter      scatter;
    PetscScalar    *x;

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
    PetscCall(PetscSegBufferGetSize(seg, &count));
    PetscCall(PetscSegBufferExtractAlloc(seg, &ind));
    PetscCall(PetscSegBufferDestroy(&seg));
    PetscCall(PetscIntCast(count, &isize));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, dim, isize, ind, PETSC_OWN_POINTER, &isdof));

    PetscCall(PetscIntCast(count * dim, &vsize));
    PetscCall(DMGetLocalVector(dm, &L));
    PetscCall(VecCreate(PETSC_COMM_SELF, &P));
    PetscCall(VecSetSizes(P, vsize, vsize));
    PetscCall(VecGetType(L, &vec_type));
    PetscCall(VecSetType(P, vec_type));
    PetscCall(VecScatterCreate(P, NULL, L, isdof, &scatter));
    PetscCall(DMRestoreLocalVector(dm, &L));
    PetscCall(ISDestroy(&isdof));

    PetscCall(VecGetArrayWrite(P, &x));
    for (PetscCount i = 0; i < count; i++) {
      for (PetscInt j = 0; j < dim; j++) x[i * dim + j] = plex->periodic.transform[f][j][3];
    }
    PetscCall(VecRestoreArrayWrite(P, &x));

    dm->periodic.affine_to_local[f] = scatter;
    dm->periodic.affine[f]          = P;
  }
  PetscCall(DMGlobalToLocalHookAdd(dm, NULL, DMCoordAddPeriodicOffsets_Private, NULL));
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
  PetscCall(PetscLogEventBegin(DMPLEX_CreateBoxSFC, dm, 0, 0, 0));
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
      ZCode       z = vert_z[owned_verts + i];
      PetscMPIInt remote_rank, remote_count = 0;

      PetscCall(PetscMPIIntCast(ZCodeFind(z, size + 1, layout.zstarts), &remote_rank));
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
        if (!IsPointInsideStratum(p, vStart, vEnd)) continue;
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
  PetscCall(PetscLogEventEnd(DMPLEX_CreateBoxSFC, dm, 0, 0, 0));
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
  if (num_face_sfs) {
    PetscAssertPointer(face_sfs, 3);
    PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMGetIsoperiodicPointSF_C", DMGetIsoperiodicPointSF_Plex));
  } else PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMGetIsoperiodicPointSF_C", NULL));
  if (num_face_sfs == plex->periodic.num_face_sfs && (num_face_sfs == 0 || face_sfs == plex->periodic.face_sfs)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMSetGlobalSection(dm, NULL));

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
  PetscBool isPlex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isPlex));
  if (isPlex) {
    DM_Plex *plex = (DM_Plex *)dm->data;
    if (face_sfs) *face_sfs = plex->periodic.face_sfs;
    if (num_face_sfs) *num_face_sfs = plex->periodic.num_face_sfs;
  } else {
    if (face_sfs) *face_sfs = NULL;
    if (num_face_sfs) *num_face_sfs = 0;
  }
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
PetscErrorCode DMPlexSetIsoperiodicFaceTransform(DM dm, PetscInt n, const PetscScalar t[])
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(n == plex->periodic.num_face_sfs, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of transforms (%" PetscInt_FMT ") must equal number of isoperiodc face SFs (%" PetscInt_FMT ")", n, plex->periodic.num_face_sfs);

  PetscCall(PetscFree(plex->periodic.transform));
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

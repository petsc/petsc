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
static ZLayout ZLayoutCreate(PetscMPIInt size, const PetscInt eextent[3], const PetscInt vextent[3])
{
  ZLayout layout;
  layout.eextent.i = eextent[0];
  layout.eextent.j = eextent[1];
  layout.eextent.k = eextent[2];
  layout.vextent.i = vextent[0];
  layout.vextent.j = vextent[1];
  layout.vextent.k = vextent[2];
  layout.comm_size = size;
  PetscMalloc1(size + 1, &layout.zstarts);

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
    for (; z <= ZEncode(layout.vextent); z++) {
      Ijk loc = ZCodeSplit(z);
      if (IjkActive(layout.eextent, loc)) break;
    }
    layout.zstarts[r + 1] = z;
  }
  return layout;
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
  ZLayout   layout = ZLayoutCreate(size, eextent, vextent);
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

  if (0) {
    DMLabel depth;
    PetscCall(DMCreateLabel(dm, "depth"));
    PetscCall(DMPlexGetDepthLabel(dm, &depth));
    PetscCall(DMLabelSetStratumBounds(depth, 0, local_elems, local_elems + local_verts));
    PetscCall(DMLabelSetStratumBounds(depth, 1, 0, local_elems));
  } else {
    PetscCall(DMPlexSymmetrize(dm));
    PetscCall(DMPlexStratify(dm));
  }
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
      PetscInt remote_elem = 0;
      for (ZCode rz = layout.zstarts[remote_rank]; rz < layout.zstarts[remote_rank + 1]; rz++) {
        Ijk loc = ZCodeSplit(rz);
        if (IjkActive(layout.eextent, loc)) remote_elem++;
      }

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
    PetscCall(PetscFree(layout.zstarts));
  }
  if (interpolate) {
    PetscCall(DMPlexInterpolateInPlace_Internal(dm));

    DMLabel label;
    PetscCall(DMCreateLabel(dm, "Face Sets"));
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscInt fStart, fEnd, vStart, vEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    for (PetscInt f = fStart; f < fEnd; f++) {
      PetscInt  npoints;
      PetscInt *points = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &npoints, &points));
      PetscInt bc_count[6] = {0};
      for (PetscInt i = 0; i < npoints; i++) {
        PetscInt p = points[2 * i];
        if (p < vStart || vEnd <= p) continue;
        Ijk loc = ZCodeSplit(vert_z[p - vStart]);
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
  }
  PetscCall(PetscFree(vert_z));
  PetscFunctionReturn(0);
}

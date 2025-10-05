#include <petsc/private/dmpleximpl.h>  /*I      "petscdmplex.h"   I*/
#include <petsc/private/petscfeimpl.h> /*I      "petscfe.h"       I*/
#include <petscblaslapack.h>
#include <petsctime.h>

const char *const DMPlexCoordMaps[] = {"none", "shear", "flare", "annulus", "shell", "sinusoid", "unknown", "DMPlexCoordMap", "DM_COORD_MAP_", NULL};

/*@
  DMPlexFindVertices - Try to find DAG points based on their coordinates.

  Not Collective (provided `DMGetCoordinatesLocalSetUp()` has been already called)

  Input Parameters:
+ dm          - The `DMPLEX` object
. coordinates - The `Vec` of coordinates of the sought points
- eps         - The tolerance or `PETSC_DEFAULT`

  Output Parameter:
. points - The `IS` of found DAG points or -1

  Level: intermediate

  Notes:
  The length of `Vec` coordinates must be npoints * dim where dim is the spatial dimension returned by `DMGetCoordinateDim()` and npoints is the number of sought points.

  The output `IS` is living on `PETSC_COMM_SELF` and its length is npoints.
  Each rank does the search independently.
  If this rank's local `DMPLEX` portion contains the DAG point corresponding to the i-th tuple of coordinates, the i-th entry of the output `IS` is set to that DAG point, otherwise to -1.

  The output `IS` must be destroyed by user.

  The tolerance is interpreted as the maximum Euclidean (L2) distance of the sought point from the specified coordinates.

  Complexity of this function is currently O(mn) with m number of vertices to find and n number of vertices in the local mesh. This could probably be improved if needed.

.seealso: `DMPLEX`, `DMPlexCreate()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMPlexFindVertices(DM dm, Vec coordinates, PetscReal eps, IS *points)
{
  PetscInt           c, cdim, i, j, o, p, vStart, vEnd;
  PetscInt           npoints;
  const PetscScalar *coord;
  Vec                allCoordsVec;
  const PetscScalar *allCoords;
  PetscInt          *dagPoints;

  PetscFunctionBegin;
  if (eps < 0) eps = PETSC_SQRT_MACHINE_EPSILON;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  {
    PetscInt n;

    PetscCall(VecGetLocalSize(coordinates, &n));
    PetscCheck(n % cdim == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Given coordinates Vec has local length %" PetscInt_FMT " not divisible by coordinate dimension %" PetscInt_FMT " of given DM", n, cdim);
    npoints = n / cdim;
  }
  PetscCall(DMGetCoordinatesLocal(dm, &allCoordsVec));
  PetscCall(VecGetArrayRead(allCoordsVec, &allCoords));
  PetscCall(VecGetArrayRead(coordinates, &coord));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  if (PetscDefined(USE_DEBUG)) {
    /* check coordinate section is consistent with DM dimension */
    PetscSection cs;
    PetscInt     ndof;

    PetscCall(DMGetCoordinateSection(dm, &cs));
    for (p = vStart; p < vEnd; p++) {
      PetscCall(PetscSectionGetDof(cs, p, &ndof));
      PetscCheck(ndof == cdim, PETSC_COMM_SELF, PETSC_ERR_PLIB, "point %" PetscInt_FMT ": ndof = %" PetscInt_FMT " != %" PetscInt_FMT " = cdim", p, ndof, cdim);
    }
  }
  PetscCall(PetscMalloc1(npoints, &dagPoints));
  if (eps == 0.0) {
    for (i = 0, j = 0; i < npoints; i++, j += cdim) {
      dagPoints[i] = -1;
      for (p = vStart, o = 0; p < vEnd; p++, o += cdim) {
        for (c = 0; c < cdim; c++) {
          if (coord[j + c] != allCoords[o + c]) break;
        }
        if (c == cdim) {
          dagPoints[i] = p;
          break;
        }
      }
    }
  } else {
    for (i = 0, j = 0; i < npoints; i++, j += cdim) {
      PetscReal norm;

      dagPoints[i] = -1;
      for (p = vStart, o = 0; p < vEnd; p++, o += cdim) {
        norm = 0.0;
        for (c = 0; c < cdim; c++) norm += PetscRealPart(PetscSqr(coord[j + c] - allCoords[o + c]));
        norm = PetscSqrtReal(norm);
        if (norm <= eps) {
          dagPoints[i] = p;
          break;
        }
      }
    }
  }
  PetscCall(VecRestoreArrayRead(allCoordsVec, &allCoords));
  PetscCall(VecRestoreArrayRead(coordinates, &coord));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, npoints, dagPoints, PETSC_OWN_POINTER, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode DMPlexGetLineIntersection_2D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], PetscReal intersection[], PetscBool *hasIntersection)
{
  const PetscReal p0_x  = segmentA[0 * 2 + 0];
  const PetscReal p0_y  = segmentA[0 * 2 + 1];
  const PetscReal p1_x  = segmentA[1 * 2 + 0];
  const PetscReal p1_y  = segmentA[1 * 2 + 1];
  const PetscReal p2_x  = segmentB[0 * 2 + 0];
  const PetscReal p2_y  = segmentB[0 * 2 + 1];
  const PetscReal p3_x  = segmentB[1 * 2 + 0];
  const PetscReal p3_y  = segmentB[1 * 2 + 1];
  const PetscReal s1_x  = p1_x - p0_x;
  const PetscReal s1_y  = p1_y - p0_y;
  const PetscReal s2_x  = p3_x - p2_x;
  const PetscReal s2_y  = p3_y - p2_y;
  const PetscReal denom = (-s2_x * s1_y + s1_x * s2_y);

  PetscFunctionBegin;
  *hasIntersection = PETSC_FALSE;
  /* Non-parallel lines */
  if (denom != 0.0) {
    const PetscReal s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denom;
    const PetscReal t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom;

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
      *hasIntersection = PETSC_TRUE;
      if (intersection) {
        intersection[0] = p0_x + (t * s1_x);
        intersection[1] = p0_y + (t * s1_y);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The plane is segmentB x segmentC: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection */
static PetscErrorCode DMPlexGetLinePlaneIntersection_3D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], const PetscReal segmentC[], PetscReal intersection[], PetscBool *hasIntersection)
{
  const PetscReal p0_x  = segmentA[0 * 3 + 0];
  const PetscReal p0_y  = segmentA[0 * 3 + 1];
  const PetscReal p0_z  = segmentA[0 * 3 + 2];
  const PetscReal p1_x  = segmentA[1 * 3 + 0];
  const PetscReal p1_y  = segmentA[1 * 3 + 1];
  const PetscReal p1_z  = segmentA[1 * 3 + 2];
  const PetscReal q0_x  = segmentB[0 * 3 + 0];
  const PetscReal q0_y  = segmentB[0 * 3 + 1];
  const PetscReal q0_z  = segmentB[0 * 3 + 2];
  const PetscReal q1_x  = segmentB[1 * 3 + 0];
  const PetscReal q1_y  = segmentB[1 * 3 + 1];
  const PetscReal q1_z  = segmentB[1 * 3 + 2];
  const PetscReal r0_x  = segmentC[0 * 3 + 0];
  const PetscReal r0_y  = segmentC[0 * 3 + 1];
  const PetscReal r0_z  = segmentC[0 * 3 + 2];
  const PetscReal r1_x  = segmentC[1 * 3 + 0];
  const PetscReal r1_y  = segmentC[1 * 3 + 1];
  const PetscReal r1_z  = segmentC[1 * 3 + 2];
  const PetscReal s0_x  = p1_x - p0_x;
  const PetscReal s0_y  = p1_y - p0_y;
  const PetscReal s0_z  = p1_z - p0_z;
  const PetscReal s1_x  = q1_x - q0_x;
  const PetscReal s1_y  = q1_y - q0_y;
  const PetscReal s1_z  = q1_z - q0_z;
  const PetscReal s2_x  = r1_x - r0_x;
  const PetscReal s2_y  = r1_y - r0_y;
  const PetscReal s2_z  = r1_z - r0_z;
  const PetscReal s3_x  = s1_y * s2_z - s1_z * s2_y; /* s1 x s2 */
  const PetscReal s3_y  = s1_z * s2_x - s1_x * s2_z;
  const PetscReal s3_z  = s1_x * s2_y - s1_y * s2_x;
  const PetscReal s4_x  = s0_y * s2_z - s0_z * s2_y; /* s0 x s2 */
  const PetscReal s4_y  = s0_z * s2_x - s0_x * s2_z;
  const PetscReal s4_z  = s0_x * s2_y - s0_y * s2_x;
  const PetscReal s5_x  = s1_y * s0_z - s1_z * s0_y; /* s1 x s0 */
  const PetscReal s5_y  = s1_z * s0_x - s1_x * s0_z;
  const PetscReal s5_z  = s1_x * s0_y - s1_y * s0_x;
  const PetscReal denom = -(s0_x * s3_x + s0_y * s3_y + s0_z * s3_z); /* -s0 . (s1 x s2) */

  PetscFunctionBegin;
  *hasIntersection = PETSC_FALSE;
  /* Line not parallel to plane */
  if (denom != 0.0) {
    const PetscReal t = (s3_x * (p0_x - q0_x) + s3_y * (p0_y - q0_y) + s3_z * (p0_z - q0_z)) / denom;
    const PetscReal u = (s4_x * (p0_x - q0_x) + s4_y * (p0_y - q0_y) + s4_z * (p0_z - q0_z)) / denom;
    const PetscReal v = (s5_x * (p0_x - q0_x) + s5_y * (p0_y - q0_y) + s5_z * (p0_z - q0_z)) / denom;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1) {
      *hasIntersection = PETSC_TRUE;
      if (intersection) {
        intersection[0] = p0_x + (t * s0_x);
        intersection[1] = p0_y + (t * s0_y);
        intersection[2] = p0_z + (t * s0_z);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode DMPlexGetPlaneSimplexIntersection_Coords_Internal(DM dm, PetscInt dim, PetscInt cdim, const PetscScalar coords[], const PetscReal p[], const PetscReal normal[], PetscBool *pos, PetscInt *Nint, PetscReal intPoints[])
{
  PetscReal d[4]; // distance of vertices to the plane
  PetscReal dp;   // distance from origin to the plane
  PetscInt  n = 0;

  PetscFunctionBegin;
  if (pos) *pos = PETSC_FALSE;
  if (Nint) *Nint = 0;
  if (PetscDefined(USE_DEBUG)) {
    PetscReal mag = DMPlex_NormD_Internal(cdim, normal);
    PetscCheck(PetscAbsReal(mag - (PetscReal)1.0) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Normal vector is not normalized: %g", (double)mag);
  }

  dp = DMPlex_DotRealD_Internal(cdim, normal, p);
  for (PetscInt v = 0; v < dim + 1; ++v) {
    // d[v] is positive, zero, or negative if vertex i is above, on, or below the plane
#if defined(PETSC_USE_COMPLEX)
    PetscReal c[4];
    for (PetscInt i = 0; i < cdim; ++i) c[i] = PetscRealPart(coords[v * cdim + i]);
    d[v] = DMPlex_DotRealD_Internal(cdim, normal, c);
#else
    d[v] = DMPlex_DotRealD_Internal(cdim, normal, &coords[v * cdim]);
#endif
    d[v] -= dp;
  }

  // If all d are positive or negative, no intersection
  {
    PetscInt v;
    for (v = 0; v < dim + 1; ++v)
      if (d[v] >= 0.) break;
    if (v == dim + 1) PetscFunctionReturn(PETSC_SUCCESS);
    for (v = 0; v < dim + 1; ++v)
      if (d[v] <= 0.) break;
    if (v == dim + 1) {
      if (pos) *pos = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  for (PetscInt v = 0; v < dim + 1; ++v) {
    // Points with zero distance are automatically added to the list.
    if (PetscAbsReal(d[v]) < PETSC_MACHINE_EPSILON) {
      for (PetscInt i = 0; i < cdim; ++i) intPoints[n * cdim + i] = PetscRealPart(coords[v * cdim + i]);
      ++n;
    } else {
      // For each point with nonzero distance, seek another point with opposite sign
      // and higher index, and compute the intersection of the line between those
      // points and the plane.
      for (PetscInt w = v + 1; w < dim + 1; ++w) {
        if (d[v] * d[w] < 0.) {
          PetscReal inv_dist = 1. / (d[v] - d[w]);
          for (PetscInt i = 0; i < cdim; ++i) intPoints[n * cdim + i] = (d[v] * PetscRealPart(coords[w * cdim + i]) - d[w] * PetscRealPart(coords[v * cdim + i])) * inv_dist;
          ++n;
        }
      }
    }
  }
  // TODO order output points if there are 4
  *Nint = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexGetPlaneSimplexIntersection_Internal(DM dm, PetscInt dim, PetscInt c, const PetscReal p[], const PetscReal normal[], PetscBool *pos, PetscInt *Nint, PetscReal intPoints[])
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords;
  PetscBool          isDG;
  PetscInt           cdim;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM has coordinates in %" PetscInt_FMT "D instead of %" PetscInt_FMT "D", cdim, dim);
  PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscCheck(numCoords == dim * (dim + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Tetrahedron should have %" PetscInt_FMT " coordinates, not %" PetscInt_FMT, dim * (dim + 1), numCoords);
  PetscCall(PetscArrayzero(intPoints, dim * (dim + 1)));

  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, coords, p, normal, pos, Nint, intPoints));

  PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexGetPlaneQuadIntersection_Internal(DM dm, PetscInt dim, PetscInt c, const PetscReal p[], const PetscReal normal[], PetscBool *pos, PetscInt *Nint, PetscReal intPoints[])
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords;
  PetscBool          isDG;
  PetscInt           cdim;
  PetscScalar        tcoords[6] = {0., 0., 0., 0., 0., 0.};
  const PetscInt     vertsA[3]  = {0, 1, 3};
  const PetscInt     vertsB[3]  = {1, 2, 3};
  PetscInt           NintA, NintB;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM has coordinates in %" PetscInt_FMT "D instead of %" PetscInt_FMT "D", cdim, dim);
  PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscCheck(numCoords == dim * 4, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Quadrilateral should have %" PetscInt_FMT " coordinates, not %" PetscInt_FMT, dim * 4, numCoords);
  PetscCall(PetscArrayzero(intPoints, dim * 4));

  for (PetscInt v = 0; v < 3; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsA[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintA, intPoints));
  for (PetscInt v = 0; v < 3; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsB[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintB, &intPoints[NintA * cdim]));
  *Nint = NintA + NintB;

  PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexGetPlaneHexIntersection_Internal(DM dm, PetscInt dim, PetscInt c, const PetscReal p[], const PetscReal normal[], PetscBool *pos, PetscInt *Nint, PetscReal intPoints[])
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords;
  PetscBool          isDG;
  PetscInt           cdim;
  PetscScalar        tcoords[12] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  // We split using the (2, 4) main diagonal, so all tets contain those vertices
  const PetscInt vertsA[4] = {0, 1, 2, 4};
  const PetscInt vertsB[4] = {0, 2, 3, 4};
  const PetscInt vertsC[4] = {1, 7, 2, 4};
  const PetscInt vertsD[4] = {2, 7, 6, 4};
  const PetscInt vertsE[4] = {3, 5, 4, 2};
  const PetscInt vertsF[4] = {4, 5, 6, 2};
  PetscInt       NintA, NintB, NintC, NintD, NintE, NintF, Nsum = 0;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM has coordinates in %" PetscInt_FMT "D instead of %" PetscInt_FMT "D", cdim, dim);
  PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscCheck(numCoords == dim * 8, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Hexahedron should have %" PetscInt_FMT " coordinates, not %" PetscInt_FMT, dim * 8, numCoords);
  PetscCall(PetscArrayzero(intPoints, dim * 18));

  for (PetscInt v = 0; v < 4; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsA[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintA, &intPoints[Nsum * cdim]));
  Nsum += NintA;
  for (PetscInt v = 0; v < 4; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsB[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintB, &intPoints[Nsum * cdim]));
  Nsum += NintB;
  for (PetscInt v = 0; v < 4; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsC[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintC, &intPoints[Nsum * cdim]));
  Nsum += NintC;
  for (PetscInt v = 0; v < 4; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsD[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintD, &intPoints[Nsum * cdim]));
  Nsum += NintD;
  for (PetscInt v = 0; v < 4; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsE[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintE, &intPoints[Nsum * cdim]));
  Nsum += NintE;
  for (PetscInt v = 0; v < 4; ++v)
    for (PetscInt d = 0; d < cdim; ++d) tcoords[v * cdim + d] = coords[vertsF[v] * cdim + d];
  PetscCall(DMPlexGetPlaneSimplexIntersection_Coords_Internal(dm, dim, cdim, tcoords, p, normal, pos, &NintF, &intPoints[Nsum * cdim]));
  Nsum += NintF;
  *Nint = Nsum;

  PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexGetPlaneCellIntersection_Internal - Finds the intersection of a plane with a cell

  Not collective

  Input Parameters:
+ dm     - the DM
. c      - the mesh point
. p      - a point on the plane.
- normal - a normal vector to the plane, must be normalized

  Output Parameters:
. pos       - `PETSC_TRUE` is the cell is on the positive side of the plane, `PETSC_FALSE` on the negative side
+ Nint      - the number of intersection points, in [0, 4]
- intPoints - the coordinates of the intersection points, should be length at least 12

  Note: The `pos` argument is only meaningful if the number of intersections is 0. The algorithmic idea comes from https://github.com/chrisk314/tet-plane-intersection.

  Level: developer

.seealso:
@*/
static PetscErrorCode DMPlexGetPlaneCellIntersection_Internal(DM dm, PetscInt c, const PetscReal p[], const PetscReal normal[], PetscBool *pos, PetscInt *Nint, PetscReal intPoints[])
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellType(dm, c, &ct));
  switch (ct) {
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(DMPlexGetPlaneSimplexIntersection_Internal(dm, DMPolytopeTypeGetDim(ct), c, p, normal, pos, Nint, intPoints));
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(DMPlexGetPlaneQuadIntersection_Internal(dm, DMPolytopeTypeGetDim(ct), c, p, normal, pos, Nint, intPoints));
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexGetPlaneHexIntersection_Internal(dm, DMPolytopeTypeGetDim(ct), c, p, normal, pos, Nint, intPoints));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No plane intersection for cell %" PetscInt_FMT " with type %s", c, DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexLocatePoint_Simplex_1D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscReal eps = PETSC_SQRT_MACHINE_EPSILON;
  const PetscReal x   = PetscRealPart(point[0]);
  PetscReal       v0, J, invJ, detJ;
  PetscReal       xi;

  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, &v0, &J, &invJ, &detJ));
  xi = invJ * (x - v0);

  if ((xi >= -eps) && (xi <= 2. + eps)) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexLocatePoint_Simplex_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscReal eps   = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal       xi[2] = {0., 0.};
  PetscReal       x[3], v0[3], J[9], invJ[9], detJ;
  PetscInt        embedDim;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &embedDim));
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  for (PetscInt j = 0; j < embedDim; ++j) x[j] = PetscRealPart(point[j]);
  for (PetscInt i = 0; i < 2; ++i) {
    for (PetscInt j = 0; j < embedDim; ++j) xi[i] += invJ[i * embedDim + j] * (x[j] - v0[j]);
  }
  if ((xi[0] >= -eps) && (xi[1] >= -eps) && (xi[0] + xi[1] <= 2.0 + eps)) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexClosestPoint_Simplex_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscReal cpoint[])
{
  const PetscInt embedDim = 2;
  PetscReal      x        = PetscRealPart(point[0]);
  PetscReal      y        = PetscRealPart(point[1]);
  PetscReal      v0[2], J[4], invJ[4], detJ;
  PetscReal      xi, eta, r;

  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  xi  = invJ[0 * embedDim + 0] * (x - v0[0]) + invJ[0 * embedDim + 1] * (y - v0[1]);
  eta = invJ[1 * embedDim + 0] * (x - v0[0]) + invJ[1 * embedDim + 1] * (y - v0[1]);

  xi  = PetscMax(xi, 0.0);
  eta = PetscMax(eta, 0.0);
  if (xi + eta > 2.0) {
    r = (xi + eta) / 2.0;
    xi /= r;
    eta /= r;
  }
  cpoint[0] = J[0 * embedDim + 0] * xi + J[0 * embedDim + 1] * eta + v0[0];
  cpoint[1] = J[1 * embedDim + 0] * xi + J[1 * embedDim + 1] * eta + v0[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This is the ray-casting, or even-odd algorithm: https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
static PetscErrorCode DMPlexLocatePoint_Quad_2D_Linear_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscScalar *array;
  PetscScalar       *coords    = NULL;
  const PetscInt     faces[8]  = {0, 1, 1, 2, 2, 3, 3, 0};
  PetscReal          x         = PetscRealPart(point[0]);
  PetscReal          y         = PetscRealPart(point[1]);
  PetscInt           crossings = 0, numCoords, embedDim;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  embedDim = numCoords / 4;
  PetscCheck(!(numCoords % 4), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Quadrilateral should have 8 coordinates, not %" PetscInt_FMT, numCoords);
  // Treat linear quads as Monge surfaces, so we just locate on the projection to x-y (could instead project to 2D)
  for (PetscInt f = 0; f < 4; ++f) {
    PetscReal x_i = PetscRealPart(coords[faces[2 * f + 0] * embedDim + 0]);
    PetscReal y_i = PetscRealPart(coords[faces[2 * f + 0] * embedDim + 1]);
    PetscReal x_j = PetscRealPart(coords[faces[2 * f + 1] * embedDim + 0]);
    PetscReal y_j = PetscRealPart(coords[faces[2 * f + 1] * embedDim + 1]);

    if ((x == x_j) && (y == y_j)) {
      // point is a corner
      crossings = 1;
      break;
    }
    if ((y_j > y) != (y_i > y)) {
      PetscReal slope = (x - x_j) * (y_i - y_j) - (x_i - x_j) * (y - y_j);
      if (slope == 0) {
        // point is a corner
        crossings = 1;
        break;
      }
      if ((slope < 0) != (y_i < y_j)) ++crossings;
    }
  }
  if (crossings % 2) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexLocatePoint_Quad_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  DM           cdm;
  PetscInt     degree, dimR, dimC;
  PetscFE      fe;
  PetscClassId id;
  PetscSpace   sp;
  PetscReal    pointR[3], ref[3], error;
  Vec          coords;
  PetscBool    found = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dimR));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDimension(cdm, &dimC));
  PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe));
  PetscCall(PetscObjectGetClassId((PetscObject)fe, &id));
  if (id != PETSCFE_CLASSID) degree = 1;
  else {
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));
  }
  if (degree == 1) {
    /* Use simple location method for linear elements*/
    PetscCall(DMPlexLocatePoint_Quad_2D_Linear_Internal(dm, point, c, cell));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Otherwise, we have to solve for the real to reference coordinates */
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  error = PETSC_SQRT_MACHINE_EPSILON;
  for (PetscInt d = 0; d < dimC; d++) pointR[d] = PetscRealPart(point[d]);
  PetscCall(DMPlexCoordinatesToReference_FE(cdm, fe, c, 1, pointR, ref, coords, dimC, dimR, 10, &error));
  if (error < PETSC_SQRT_MACHINE_EPSILON) found = PETSC_TRUE;
  if ((ref[0] > 1.0 + PETSC_SMALL) || (ref[0] < -1.0 - PETSC_SMALL) || (ref[1] > 1.0 + PETSC_SMALL) || (ref[1] < -1.0 - PETSC_SMALL)) found = PETSC_FALSE;
  if (PetscDefined(USE_DEBUG) && found) {
    PetscReal real[3], inverseError = 0, normPoint = DMPlex_NormD_Internal(dimC, pointR);

    normPoint = normPoint > PETSC_SMALL ? normPoint : 1.0;
    PetscCall(DMPlexReferenceToCoordinates_FE(cdm, fe, c, 1, ref, real, coords, dimC, dimR));
    inverseError = DMPlex_DistRealD_Internal(dimC, real, pointR);
    if (inverseError > PETSC_SQRT_MACHINE_EPSILON * normPoint) found = PETSC_FALSE;
    if (!found) PetscCall(PetscInfo(dm, "Point (%g, %g, %g) != Mapped Ref Coords (%g, %g, %g) with error %g\n", (double)pointR[0], (double)pointR[1], (double)pointR[2], (double)real[0], (double)real[1], (double)real[2], (double)inverseError));
  }
  if (found) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexLocatePoint_Simplex_3D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscInt  embedDim = 3;
  const PetscReal eps      = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal       v0[3], J[9], invJ[9], detJ;
  PetscReal       x = PetscRealPart(point[0]);
  PetscReal       y = PetscRealPart(point[1]);
  PetscReal       z = PetscRealPart(point[2]);
  PetscReal       xi, eta, zeta;

  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  xi   = invJ[0 * embedDim + 0] * (x - v0[0]) + invJ[0 * embedDim + 1] * (y - v0[1]) + invJ[0 * embedDim + 2] * (z - v0[2]);
  eta  = invJ[1 * embedDim + 0] * (x - v0[0]) + invJ[1 * embedDim + 1] * (y - v0[1]) + invJ[1 * embedDim + 2] * (z - v0[2]);
  zeta = invJ[2 * embedDim + 0] * (x - v0[0]) + invJ[2 * embedDim + 1] * (y - v0[1]) + invJ[2 * embedDim + 2] * (z - v0[2]);

  if ((xi >= -eps) && (eta >= -eps) && (zeta >= -eps) && (xi + eta + zeta <= 2.0 + eps)) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexLocatePoint_Hex_3D_Linear_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscScalar *array;
  PetscScalar       *coords    = NULL;
  const PetscInt     faces[24] = {0, 3, 2, 1, 5, 4, 7, 6, 3, 0, 4, 5, 1, 2, 6, 7, 3, 5, 6, 2, 0, 1, 7, 4};
  PetscBool          found     = PETSC_TRUE;
  PetscInt           numCoords, f;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscCheck(numCoords == 24, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Quadrilateral should have 8 coordinates, not %" PetscInt_FMT, numCoords);
  for (f = 0; f < 6; ++f) {
    /* Check the point is under plane */
    /*   Get face normal */
    PetscReal v_i[3];
    PetscReal v_j[3];
    PetscReal normal[3];
    PetscReal pp[3];
    PetscReal dot;

    v_i[0]    = PetscRealPart(coords[faces[f * 4 + 3] * 3 + 0] - coords[faces[f * 4 + 0] * 3 + 0]);
    v_i[1]    = PetscRealPart(coords[faces[f * 4 + 3] * 3 + 1] - coords[faces[f * 4 + 0] * 3 + 1]);
    v_i[2]    = PetscRealPart(coords[faces[f * 4 + 3] * 3 + 2] - coords[faces[f * 4 + 0] * 3 + 2]);
    v_j[0]    = PetscRealPart(coords[faces[f * 4 + 1] * 3 + 0] - coords[faces[f * 4 + 0] * 3 + 0]);
    v_j[1]    = PetscRealPart(coords[faces[f * 4 + 1] * 3 + 1] - coords[faces[f * 4 + 0] * 3 + 1]);
    v_j[2]    = PetscRealPart(coords[faces[f * 4 + 1] * 3 + 2] - coords[faces[f * 4 + 0] * 3 + 2]);
    normal[0] = v_i[1] * v_j[2] - v_i[2] * v_j[1];
    normal[1] = v_i[2] * v_j[0] - v_i[0] * v_j[2];
    normal[2] = v_i[0] * v_j[1] - v_i[1] * v_j[0];
    pp[0]     = PetscRealPart(coords[faces[f * 4 + 0] * 3 + 0] - point[0]);
    pp[1]     = PetscRealPart(coords[faces[f * 4 + 0] * 3 + 1] - point[1]);
    pp[2]     = PetscRealPart(coords[faces[f * 4 + 0] * 3 + 2] - point[2]);
    dot       = normal[0] * pp[0] + normal[1] * pp[1] + normal[2] * pp[2];

    /* Check that projected point is in face (2D location problem) */
    if (dot < 0.0) {
      found = PETSC_FALSE;
      break;
    }
  }
  if (found) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexLocatePoint_Hex_3D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  DM           cdm;
  PetscInt     degree, dimR, dimC;
  PetscFE      fe;
  PetscClassId id;
  PetscSpace   sp;
  PetscReal    pointR[3], ref[3], error;
  Vec          coords;
  PetscBool    found = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dimR));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDimension(cdm, &dimC));
  PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe));
  PetscCall(PetscObjectGetClassId((PetscObject)fe, &id));
  if (id != PETSCFE_CLASSID) degree = 1;
  else {
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));
  }
  if (degree == 1) {
    /* Use simple location method for linear elements*/
    PetscCall(DMPlexLocatePoint_Hex_3D_Linear_Internal(dm, point, c, cell));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Otherwise, we have to solve for the real to reference coordinates */
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  error = PETSC_SQRT_MACHINE_EPSILON;
  for (PetscInt d = 0; d < dimC; d++) pointR[d] = PetscRealPart(point[d]);
  PetscCall(DMPlexCoordinatesToReference_FE(cdm, fe, c, 1, pointR, ref, coords, dimC, dimR, 10, &error));
  if (error < PETSC_SQRT_MACHINE_EPSILON) found = PETSC_TRUE;
  if ((ref[0] > 1.0 + PETSC_SMALL) || (ref[0] < -1.0 - PETSC_SMALL) || (ref[1] > 1.0 + PETSC_SMALL) || (ref[1] < -1.0 - PETSC_SMALL) || (ref[2] > 1.0 + PETSC_SMALL) || (ref[2] < -1.0 - PETSC_SMALL)) found = PETSC_FALSE;
  if (PetscDefined(USE_DEBUG) && found) {
    PetscReal real[3], inverseError = 0, normPoint = DMPlex_NormD_Internal(dimC, pointR);

    normPoint = normPoint > PETSC_SMALL ? normPoint : 1.0;
    PetscCall(DMPlexReferenceToCoordinates_FE(cdm, fe, c, 1, ref, real, coords, dimC, dimR));
    inverseError = DMPlex_DistRealD_Internal(dimC, real, pointR);
    if (inverseError > PETSC_SQRT_MACHINE_EPSILON * normPoint) found = PETSC_FALSE;
    if (!found) PetscCall(PetscInfo(dm, "Point (%g, %g, %g) != Mapped Ref Coords (%g, %g, %g) with error %g\n", (double)pointR[0], (double)pointR[1], (double)pointR[2], (double)real[0], (double)real[1], (double)real[2], (double)inverseError));
  }
  if (found) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscGridHashInitialize_Internal(PetscGridHash box, PetscInt dim, const PetscScalar point[])
{
  PetscInt d;

  PetscFunctionBegin;
  box->dim = dim;
  for (d = 0; d < dim; ++d) box->lower[d] = box->upper[d] = point ? PetscRealPart(point[d]) : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscGridHashCreate(MPI_Comm comm, PetscInt dim, const PetscScalar point[], PetscGridHash *box)
{
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, box));
  PetscCall(PetscGridHashInitialize_Internal(*box, dim, point));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscGridHashEnlarge(PetscGridHash box, const PetscScalar point[])
{
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < box->dim; ++d) {
    box->lower[d] = PetscMin(box->lower[d], PetscRealPart(point[d]));
    box->upper[d] = PetscMax(box->upper[d], PetscRealPart(point[d]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCreateGridHash(DM dm, PetscGridHash *box)
{
  Vec                coordinates;
  const PetscScalar *a;
  PetscInt           cdim, cStart, cEnd;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));

  PetscCall(VecGetArrayRead(coordinates, &a));
  PetscCall(PetscGridHashCreate(PetscObjectComm((PetscObject)dm), cdim, a, box));
  PetscCall(VecRestoreArrayRead(coordinates, &a));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *array;
    PetscScalar       *coords = NULL;
    PetscInt           numCoords;
    PetscBool          isDG;

    PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
    for (PetscInt i = 0; i < numCoords / cdim; ++i) PetscCall(PetscGridHashEnlarge(*box, &coords[i * cdim]));
    PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscGridHashSetGrid - Divide the grid into boxes

  Not Collective

  Input Parameters:
+ box - The grid hash object
. n   - The number of boxes in each dimension, may use `PETSC_DETERMINE` for the entries
- h   - The box size in each dimension, only used if n[d] == `PETSC_DETERMINE`, if not needed you can pass in `NULL`

  Level: developer

.seealso: `DMPLEX`, `PetscGridHashCreate()`
@*/
PetscErrorCode PetscGridHashSetGrid(PetscGridHash box, const PetscInt n[], const PetscReal h[])
{
  PetscInt d;

  PetscFunctionBegin;
  PetscAssertPointer(n, 2);
  if (h) PetscAssertPointer(h, 3);
  for (d = 0; d < box->dim; ++d) {
    box->extent[d] = box->upper[d] - box->lower[d];
    if (n[d] == PETSC_DETERMINE) {
      PetscCheck(h, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing h");
      box->h[d] = h[d];
      box->n[d] = PetscCeilReal(box->extent[d] / h[d]);
    } else {
      box->n[d] = n[d];
      box->h[d] = box->extent[d] / n[d];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscGridHashGetEnclosingBox - Find the grid boxes containing each input point

  Not Collective

  Input Parameters:
+ box       - The grid hash object
. numPoints - The number of input points
- points    - The input point coordinates

  Output Parameters:
+ dboxes - An array of `numPoints` x `dim` integers expressing the enclosing box as (i_0, i_1, ..., i_dim)
- boxes  - An array of `numPoints` integers expressing the enclosing box as single number, or `NULL`

  Level: developer

  Note:
  This only guarantees that a box contains a point, not that a cell does.

.seealso: `DMPLEX`, `PetscGridHashCreate()`
@*/
PetscErrorCode PetscGridHashGetEnclosingBox(PetscGridHash box, PetscInt numPoints, const PetscScalar points[], PetscInt dboxes[], PetscInt boxes[])
{
  const PetscReal *lower = box->lower;
  const PetscReal *upper = box->upper;
  const PetscReal *h     = box->h;
  const PetscInt  *n     = box->n;
  const PetscInt   dim   = box->dim;
  PetscInt         d, p;

  PetscFunctionBegin;
  for (p = 0; p < numPoints; ++p) {
    for (d = 0; d < dim; ++d) {
      PetscInt dbox = PetscFloorReal((PetscRealPart(points[p * dim + d]) - lower[d]) / h[d]);

      if (dbox == n[d] && PetscAbsReal(PetscRealPart(points[p * dim + d]) - upper[d]) < 1.0e-9) dbox = n[d] - 1;
      if (dbox == -1 && PetscAbsReal(PetscRealPart(points[p * dim + d]) - lower[d]) < 1.0e-9) dbox = 0;
      PetscCheck(dbox >= 0 && dbox < n[d], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input point %" PetscInt_FMT " (%g, %g, %g) is outside of our bounding box (%g, %g, %g) - (%g, %g, %g)", p, (double)PetscRealPart(points[p * dim + 0]), dim > 1 ? (double)PetscRealPart(points[p * dim + 1]) : 0.0, dim > 2 ? (double)PetscRealPart(points[p * dim + 2]) : 0.0, (double)lower[0], (double)lower[1], (double)lower[2], (double)upper[0], (double)upper[1], (double)upper[2]);
      dboxes[p * dim + d] = dbox;
    }
    if (boxes)
      for (d = dim - 2, boxes[p] = dboxes[p * dim + dim - 1]; d >= 0; --d) boxes[p] = boxes[p] * n[d] + dboxes[p * dim + d];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscGridHashGetEnclosingBoxQuery - Find the grid boxes containing each input point

  Not Collective

  Input Parameters:
+ box         - The grid hash object
. cellSection - The PetscSection mapping cells to boxes
. numPoints   - The number of input points
- points      - The input point coordinates

  Output Parameters:
+ dboxes - An array of `numPoints`*`dim` integers expressing the enclosing box as (i_0, i_1, ..., i_dim)
. boxes  - An array of `numPoints` integers expressing the enclosing box as single number, or `NULL`
- found  - Flag indicating if point was located within a box

  Level: developer

  Note:
  This does an additional check that a cell actually contains the point, and found is `PETSC_FALSE` if no cell does. Thus, this function requires that `cellSection` is already constructed.

.seealso: `DMPLEX`, `PetscGridHashGetEnclosingBox()`
*/
static PetscErrorCode PetscGridHashGetEnclosingBoxQuery(PetscGridHash box, PetscSection cellSection, PetscInt numPoints, const PetscScalar points[], PetscInt dboxes[], PetscInt boxes[], PetscBool *found)
{
  const PetscReal *lower = box->lower;
  const PetscReal *upper = box->upper;
  const PetscReal *h     = box->h;
  const PetscInt  *n     = box->n;
  const PetscInt   dim   = box->dim;
  PetscInt         bStart, bEnd, d, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cellSection, PETSC_SECTION_CLASSID, 2);
  *found = PETSC_FALSE;
  PetscCall(PetscSectionGetChart(box->cellSection, &bStart, &bEnd));
  for (p = 0; p < numPoints; ++p) {
    for (d = 0; d < dim; ++d) {
      PetscInt dbox = PetscFloorReal((PetscRealPart(points[p * dim + d]) - lower[d]) / h[d]);

      if (dbox == n[d] && PetscAbsReal(PetscRealPart(points[p * dim + d]) - upper[d]) < 1.0e-9) dbox = n[d] - 1;
      if (dbox < 0 || dbox >= n[d]) PetscFunctionReturn(PETSC_SUCCESS);
      dboxes[p * dim + d] = dbox;
    }
    if (boxes)
      for (d = dim - 2, boxes[p] = dboxes[p * dim + dim - 1]; d >= 0; --d) boxes[p] = boxes[p] * n[d] + dboxes[p * dim + d];
    // It is possible for a box to overlap no grid cells
    if (boxes[p] < bStart || boxes[p] >= bEnd) PetscFunctionReturn(PETSC_SUCCESS);
  }
  *found = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscGridHashDestroy(PetscGridHash *box)
{
  PetscFunctionBegin;
  if (*box) {
    PetscCall(PetscSectionDestroy(&(*box)->cellSection));
    PetscCall(ISDestroy(&(*box)->cells));
    PetscCall(DMLabelDestroy(&(*box)->cellsSparse));
  }
  PetscCall(PetscFree(*box));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexLocatePoint_Internal(DM dm, PetscInt dim, const PetscScalar point[], PetscInt cellStart, PetscInt *cell)
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellType(dm, cellStart, &ct));
  switch (ct) {
  case DM_POLYTOPE_SEGMENT:
    PetscCall(DMPlexLocatePoint_Simplex_1D_Internal(dm, point, cellStart, cell));
    break;
  case DM_POLYTOPE_TRIANGLE:
    PetscCall(DMPlexLocatePoint_Simplex_2D_Internal(dm, point, cellStart, cell));
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(DMPlexLocatePoint_Quad_2D_Internal(dm, point, cellStart, cell));
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(DMPlexLocatePoint_Simplex_3D_Internal(dm, point, cellStart, cell));
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexLocatePoint_Hex_3D_Internal(dm, point, cellStart, cell));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for cell %" PetscInt_FMT " with type %s", cellStart, DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexClosestPoint_Internal - Returns the closest point in the cell to the given point
*/
static PetscErrorCode DMPlexClosestPoint_Internal(DM dm, PetscInt dim, const PetscScalar point[], PetscInt cell, PetscReal cpoint[])
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
  case DM_POLYTOPE_TRIANGLE:
    PetscCall(DMPlexClosestPoint_Simplex_2D_Internal(dm, point, cell, cpoint));
    break;
#if 0
    case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(DMPlexClosestPoint_General_2D_Internal(dm, point, cell, cpoint));break;
    case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(DMPlexClosestPoint_Simplex_3D_Internal(dm, point, cell, cpoint));break;
    case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexClosestPoint_General_3D_Internal(dm, point, cell, cpoint));break;
#endif
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No closest point location for cell %" PetscInt_FMT " with type %s", cell, DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexComputeGridHash_Internal - Create a grid hash structure covering the `DMPLEX`

  Collective

  Input Parameter:
. dm - The `DMPLEX`

  Output Parameter:
. localBox - The grid hash object

  Level: developer

  Notes:
  How do we determine all boxes intersecting a given cell?

  1) Get convex body enclosing cell. We will use a box called the box-hull.

  2) Get smallest brick of boxes enclosing the box-hull

  3) Each box is composed of 6 planes, 3 lower and 3 upper. We loop over dimensions, and
     for each new plane determine whether the cell is on the negative side, positive side, or intersects it.

     a) If the cell is on the negative side of the lower planes, it is not in the box

     b) If the cell is on the positive side of the upper planes, it is not in the box

     c) If there is no intersection, it is in the box

     d) If any intersection point is within the box limits, it is in the box

.seealso: `DMPLEX`, `PetscGridHashCreate()`, `PetscGridHashGetEnclosingBox()`
*/
static PetscErrorCode DMPlexComputeGridHash_Internal(DM dm, PetscGridHash *localBox)
{
  PetscInt        debug = ((DM_Plex *)dm->data)->printLocate;
  PetscGridHash   lbox;
  PetscSF         sf;
  const PetscInt *leaves;
  PetscInt       *dboxes, *boxes;
  PetscInt        cdim, cStart, cEnd, Nl = -1;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexCreateGridHash(dm, &lbox));
  {
    PetscInt n[3], d;

    PetscCall(PetscOptionsGetIntArray(NULL, ((PetscObject)dm)->prefix, "-dm_plex_hash_box_faces", n, &d, &flg));
    if (flg) {
      for (PetscInt i = d; i < cdim; ++i) n[i] = n[d - 1];
    } else {
      for (PetscInt i = 0; i < cdim; ++i) n[i] = PetscMax(2, PetscFloorReal(PetscPowReal((PetscReal)(cEnd - cStart), 1.0 / cdim) * 0.8));
    }
    PetscCall(PetscGridHashSetGrid(lbox, n, NULL));
    if (debug)
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "GridHash:\n  (%g, %g, %g) -- (%g, %g, %g)\n  n %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n  h %g %g %g\n", (double)lbox->lower[0], (double)lbox->lower[1], cdim > 2 ? (double)lbox->lower[2] : 0.,
                            (double)lbox->upper[0], (double)lbox->upper[1], cdim > 2 ? (double)lbox->upper[2] : 0, n[0], n[1], cdim > 2 ? n[2] : 0, (double)lbox->h[0], (double)lbox->h[1], cdim > 2 ? (double)lbox->h[2] : 0.));
  }

  PetscCall(DMGetPointSF(dm, &sf));
  if (sf) PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
  Nl = PetscMax(Nl, 0);
  PetscCall(PetscCalloc2(16 * cdim, &dboxes, 16, &boxes));

  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "cells", &lbox->cellsSparse));
  PetscCall(DMLabelCreateIndex(lbox->cellsSparse, cStart, cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscReal          intPoints[6 * 6 * 6 * 3];
    const PetscScalar *array;
    PetscScalar       *coords            = NULL;
    const PetscReal   *h                 = lbox->h;
    PetscReal          normal[9]         = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    PetscReal         *lowerIntPoints[3] = {&intPoints[0 * 6 * 6 * 3], &intPoints[1 * 6 * 6 * 3], &intPoints[2 * 6 * 6 * 3]};
    PetscReal         *upperIntPoints[3] = {&intPoints[3 * 6 * 6 * 3], &intPoints[4 * 6 * 6 * 3], &intPoints[5 * 6 * 6 * 3]};
    PetscReal          lp[3], up[3], *tmp;
    PetscInt           numCoords, idx, dlim[6], lowerInt[3], upperInt[3];
    PetscBool          isDG, lower[3], upper[3];

    PetscCall(PetscFindInt(c, Nl, leaves, &idx));
    if (idx >= 0) continue;
    // Get grid of boxes containing the cell
    PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
    PetscCall(PetscGridHashGetEnclosingBox(lbox, numCoords / cdim, coords, dboxes, boxes));
    PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
    for (PetscInt d = 0; d < cdim; ++d) dlim[d * 2 + 0] = dlim[d * 2 + 1] = dboxes[d];
    for (PetscInt d = cdim; d < 3; ++d) dlim[d * 2 + 0] = dlim[d * 2 + 1] = 0;
    for (PetscInt e = 1; e < numCoords / cdim; ++e) {
      for (PetscInt d = 0; d < cdim; ++d) {
        dlim[d * 2 + 0] = PetscMin(dlim[d * 2 + 0], dboxes[e * cdim + d]);
        dlim[d * 2 + 1] = PetscMax(dlim[d * 2 + 1], dboxes[e * cdim + d]);
      }
    }
    if (debug > 4) {
      for (PetscInt d = 0; d < cdim; ++d) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " direction %" PetscInt_FMT " box limits %" PetscInt_FMT "--%" PetscInt_FMT "\n", c, d, dlim[d * 2 + 0], dlim[d * 2 + 1]));
    }
    // Initialize with lower planes for first box
    for (PetscInt d = 0; d < cdim; ++d) {
      lp[d] = lbox->lower[d] + dlim[d * 2 + 0] * h[d];
      up[d] = lp[d] + h[d];
    }
    for (PetscInt d = 0; d < cdim; ++d) {
      PetscCall(DMPlexGetPlaneCellIntersection_Internal(dm, c, lp, &normal[d * 3], &lower[d], &lowerInt[d], lowerIntPoints[d]));
      if (debug > 4) {
        if (!lowerInt[d])
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " lower direction %" PetscInt_FMT " (%g, %g, %g) does not intersect %s\n", c, d, (double)lp[0], (double)lp[1], cdim > 2 ? (double)lp[2] : 0., lower[d] ? "positive" : "negative"));
        else PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " lower direction %" PetscInt_FMT " (%g, %g, %g) intersects %" PetscInt_FMT " times\n", c, d, (double)lp[0], (double)lp[1], cdim > 2 ? (double)lp[2] : 0., lowerInt[d]));
      }
    }
    // Loop over grid
    for (PetscInt k = dlim[2 * 2 + 0]; k <= dlim[2 * 2 + 1]; ++k, lp[2] = up[2], up[2] += h[2]) {
      if (cdim > 2) PetscCall(DMPlexGetPlaneCellIntersection_Internal(dm, c, up, &normal[3 * 2], &upper[2], &upperInt[2], upperIntPoints[2]));
      if (cdim > 2 && debug > 4) {
        if (!upperInt[2]) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " upper direction 2 (%g, %g, %g) does not intersect %s\n", c, (double)up[0], (double)up[1], cdim > 2 ? (double)up[2] : 0., upper[2] ? "positive" : "negative"));
        else PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " upper direction 2 (%g, %g, %g) intersects %" PetscInt_FMT " times\n", c, (double)up[0], (double)up[1], cdim > 2 ? (double)up[2] : 0., upperInt[2]));
      }
      for (PetscInt j = dlim[1 * 2 + 0]; j <= dlim[1 * 2 + 1]; ++j, lp[1] = up[1], up[1] += h[1]) {
        if (cdim > 1) PetscCall(DMPlexGetPlaneCellIntersection_Internal(dm, c, up, &normal[3 * 1], &upper[1], &upperInt[1], upperIntPoints[1]));
        if (cdim > 1 && debug > 4) {
          if (!upperInt[1])
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " upper direction 1 (%g, %g, %g) does not intersect %s\n", c, (double)up[0], (double)up[1], cdim > 2 ? (double)up[2] : 0., upper[1] ? "positive" : "negative"));
          else PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " upper direction 1 (%g, %g, %g) intersects %" PetscInt_FMT " times\n", c, (double)up[0], (double)up[1], cdim > 2 ? (double)up[2] : 0., upperInt[1]));
        }
        for (PetscInt i = dlim[0 * 2 + 0]; i <= dlim[0 * 2 + 1]; ++i, lp[0] = up[0], up[0] += h[0]) {
          const PetscInt box    = (k * lbox->n[1] + j) * lbox->n[0] + i;
          PetscBool      excNeg = PETSC_TRUE;
          PetscBool      excPos = PETSC_TRUE;
          PetscInt       NlInt  = 0;
          PetscInt       NuInt  = 0;

          PetscCall(DMPlexGetPlaneCellIntersection_Internal(dm, c, up, &normal[3 * 0], &upper[0], &upperInt[0], upperIntPoints[0]));
          if (debug > 4) {
            if (!upperInt[0])
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " upper direction 0 (%g, %g, %g) does not intersect %s\n", c, (double)up[0], (double)up[1], cdim > 2 ? (double)up[2] : 0., upper[0] ? "positive" : "negative"));
            else PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " upper direction 0 (%g, %g, %g) intersects %" PetscInt_FMT " times\n", c, (double)up[0], (double)up[1], cdim > 2 ? (double)up[2] : 0., upperInt[0]));
          }
          for (PetscInt d = 0; d < cdim; ++d) {
            NlInt += lowerInt[d];
            NuInt += upperInt[d];
          }
          // If there is no intersection...
          if (!NlInt && !NuInt) {
            // If the cell is on the negative side of the lower planes, it is not in the box
            for (PetscInt d = 0; d < cdim; ++d)
              if (lower[d]) {
                excNeg = PETSC_FALSE;
                break;
              }
            // If the cell is on the positive side of the upper planes, it is not in the box
            for (PetscInt d = 0; d < cdim; ++d)
              if (!upper[d]) {
                excPos = PETSC_FALSE;
                break;
              }
            if (excNeg || excPos) {
              if (debug && excNeg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " is on the negative side of the lower plane\n", c));
              if (debug && excPos) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " is on the positive side of the upper plane\n", c));
              continue;
            }
            // Otherwise it is in the box
            if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " is contained in box %" PetscInt_FMT "\n", c, box));
            PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
            continue;
          }
          /*
            If any intersection point is within the box limits, it is in the box
            We need to have tolerances here since intersection point calculations can introduce errors
            Initialize a count to track which planes have intersection outside the box.
            if two adjacent planes have intersection points upper and lower all outside the box, look
            first at if another plane has intersection points outside the box, if so, it is inside the cell
            look next if no intersection points exist on the other planes, and check if the planes are on the
            outside of the intersection points but on opposite ends. If so, the box cuts through the cell.
          */
          PetscInt outsideCount[6] = {0, 0, 0, 0, 0, 0};
          for (PetscInt plane = 0; plane < cdim; ++plane) {
            for (PetscInt ip = 0; ip < lowerInt[plane]; ++ip) {
              PetscInt d;

              for (d = 0; d < cdim; ++d) {
                if ((lowerIntPoints[plane][ip * cdim + d] < (lp[d] - PETSC_SMALL)) || (lowerIntPoints[plane][ip * cdim + d] > (up[d] + PETSC_SMALL))) {
                  if (lowerIntPoints[plane][ip * cdim + d] < (lp[d] - PETSC_SMALL)) outsideCount[d]++; // The lower point is to the left of this box, and we count it
                  break;
                }
              }
              if (d == cdim) {
                if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " intersected lower plane %" PetscInt_FMT " of box %" PetscInt_FMT "\n", c, plane, box));
                PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
                goto end;
              }
            }
            for (PetscInt ip = 0; ip < upperInt[plane]; ++ip) {
              PetscInt d;

              for (d = 0; d < cdim; ++d) {
                if ((upperIntPoints[plane][ip * cdim + d] < (lp[d] - PETSC_SMALL)) || (upperIntPoints[plane][ip * cdim + d] > (up[d] + PETSC_SMALL))) {
                  if (upperIntPoints[plane][ip * cdim + d] > (up[d] + PETSC_SMALL)) outsideCount[cdim + d]++; // The upper point is to the right of this box, and we count it
                  break;
                }
              }
              if (d == cdim) {
                if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " intersected upper plane %" PetscInt_FMT " of box %" PetscInt_FMT "\n", c, plane, box));
                PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
                goto end;
              }
            }
          }
          /*
             Check the planes with intersections
             in 2D, check if the square falls in the middle of a cell
             ie all four planes have intersection points outside of the box
             You do not want to be doing this, because it means your grid hashing is finer than your grid,
             but we should still support it I guess
          */
          if (cdim == 2) {
            PetscInt nIntersects = 0;
            for (PetscInt d = 0; d < cdim; ++d) nIntersects += (outsideCount[d] + outsideCount[d + cdim]);
            // if the count adds up to 8, that means each plane has 2 external intersections and thus it is in the cell
            if (nIntersects == 8) {
              PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
              goto end;
            }
          }
          /*
             In 3 dimensions, if two adjacent planes have at least 3 intersections outside the cell in the appropriate direction,
             we then check the 3rd planar dimension. If a plane falls between intersection points, the cell belongs to that box.
             If the planes are on opposite sides of the intersection points, the cell belongs to that box and it passes through the cell.
          */
          if (cdim == 3) {
            PetscInt faces[3] = {0, 0, 0}, checkInternalFace = 0;
            // Find two adjacent planes with at least 3 intersection points in the upper and lower
            // if the third plane has 3 intersection points or more, a pyramid base is formed on that plane and it is in the cell
            for (PetscInt d = 0; d < cdim; ++d)
              if (outsideCount[d] >= 3 && outsideCount[cdim + d] >= 3) {
                faces[d]++;
                checkInternalFace++;
              }
            if (checkInternalFace == 3) {
              // All planes have 3 intersection points, add it.
              PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
              goto end;
            }
            // Gross, figure out which adjacent faces have at least 3 points
            PetscInt nonIntersectingFace = -1;
            if (faces[0] == faces[1]) nonIntersectingFace = 2;
            if (faces[0] == faces[2]) nonIntersectingFace = 1;
            if (faces[1] == faces[2]) nonIntersectingFace = 0;
            if (nonIntersectingFace >= 0) {
              for (PetscInt plane = 0; plane < cdim; ++plane) {
                if (!lowerInt[nonIntersectingFace] && !upperInt[nonIntersectingFace]) continue;
                // If we have 2 adjacent sides with pyramids of intersection outside of them, and there is a point between the end caps at all, it must be between the two non intersecting ends, and the box is inside the cell.
                for (PetscInt ip = 0; ip < lowerInt[nonIntersectingFace]; ++ip) {
                  if (lowerIntPoints[plane][ip * cdim + nonIntersectingFace] > lp[nonIntersectingFace] - PETSC_SMALL || lowerIntPoints[plane][ip * cdim + nonIntersectingFace] < up[nonIntersectingFace] + PETSC_SMALL) goto setpoint;
                }
                for (PetscInt ip = 0; ip < upperInt[nonIntersectingFace]; ++ip) {
                  if (upperIntPoints[plane][ip * cdim + nonIntersectingFace] > lp[nonIntersectingFace] - PETSC_SMALL || upperIntPoints[plane][ip * cdim + nonIntersectingFace] < up[nonIntersectingFace] + PETSC_SMALL) goto setpoint;
                }
                goto end;
              }
              // The points are within the bonds of the non intersecting planes, add it.
            setpoint:
              PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
              goto end;
            }
          }
        end:
          lower[0]          = upper[0];
          lowerInt[0]       = upperInt[0];
          tmp               = lowerIntPoints[0];
          lowerIntPoints[0] = upperIntPoints[0];
          upperIntPoints[0] = tmp;
        }
        lp[0]             = lbox->lower[0] + dlim[0 * 2 + 0] * h[0];
        up[0]             = lp[0] + h[0];
        lower[1]          = upper[1];
        lowerInt[1]       = upperInt[1];
        tmp               = lowerIntPoints[1];
        lowerIntPoints[1] = upperIntPoints[1];
        upperIntPoints[1] = tmp;
      }
      lp[1]             = lbox->lower[1] + dlim[1 * 2 + 0] * h[1];
      up[1]             = lp[1] + h[1];
      lower[2]          = upper[2];
      lowerInt[2]       = upperInt[2];
      tmp               = lowerIntPoints[2];
      lowerIntPoints[2] = upperIntPoints[2];
      upperIntPoints[2] = tmp;
    }
  }
  PetscCall(PetscFree2(dboxes, boxes));

  if (debug) PetscCall(DMLabelView(lbox->cellsSparse, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(DMLabelConvertToSection(lbox->cellsSparse, &lbox->cellSection, &lbox->cells));
  PetscCall(DMLabelDestroy(&lbox->cellsSparse));
  *localBox = lbox;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMLocatePoints_Plex(DM dm, Vec v, DMPointLocationType ltype, PetscSF cellSF)
{
  PetscInt        debug = ((DM_Plex *)dm->data)->printLocate;
  DM_Plex        *mesh  = (DM_Plex *)dm->data;
  PetscBool       hash = mesh->useHashLocation, reuse = PETSC_FALSE;
  PetscInt        bs, numPoints, numFound, *found = NULL;
  PetscInt        cdim, Nl = 0, cStart, cEnd, numCells;
  PetscSF         sf;
  const PetscInt *leaves;
  const PetscInt *boxCells;
  PetscSFNode    *cells;
  PetscScalar    *a;
  PetscMPIInt     result;
  PetscLogDouble  t0, t1;
  PetscReal       gmin[3], gmax[3];
  PetscInt        terminating_query_type[] = {0, 0, 0};
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(PetscLogEventBegin(DMPLEX_LocatePoints, 0, 0, 0, 0));
  PetscCall(PetscTime(&t0));
  PetscCheck(ltype != DM_POINTLOCATION_NEAREST || hash, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Nearest point location only supported with grid hashing. Use -dm_plex_hash_location to enable it.");
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(VecGetBlockSize(v, &bs));
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)cellSF), PETSC_COMM_SELF, &result));
  PetscCheck(result == MPI_IDENT || result == MPI_CONGRUENT, PetscObjectComm((PetscObject)cellSF), PETSC_ERR_SUP, "Trying parallel point location: only local point location supported");
  // We ignore extra coordinates
  PetscCheck(bs >= cdim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Block size for point vector %" PetscInt_FMT " must be the mesh coordinate dimension %" PetscInt_FMT, bs, cdim);
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetPointSF(dm, &sf));
  if (sf) PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
  Nl = PetscMax(Nl, 0);
  PetscCall(VecGetLocalSize(v, &numPoints));
  PetscCall(VecGetArray(v, &a));
  numPoints /= bs;
  {
    const PetscSFNode *sf_cells;

    PetscCall(PetscSFGetGraph(cellSF, NULL, NULL, NULL, &sf_cells));
    if (sf_cells) {
      PetscCall(PetscInfo(dm, "[DMLocatePoints_Plex] Re-using existing StarForest node list\n"));
      cells = (PetscSFNode *)sf_cells;
      reuse = PETSC_TRUE;
    } else {
      PetscCall(PetscInfo(dm, "[DMLocatePoints_Plex] Creating and initializing new StarForest node list\n"));
      PetscCall(PetscMalloc1(numPoints, &cells));
      /* initialize cells if created */
      for (PetscInt p = 0; p < numPoints; p++) {
        cells[p].rank  = 0;
        cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
      }
    }
  }
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  if (hash) {
    if (!mesh->lbox) {
      PetscCall(PetscInfo(dm, "Initializing grid hashing\n"));
      PetscCall(DMPlexComputeGridHash_Internal(dm, &mesh->lbox));
    }
    /* Designate the local box for each point */
    /* Send points to correct process */
    /* Search cells that lie in each subbox */
    /*   Should we bin points before doing search? */
    PetscCall(ISGetIndices(mesh->lbox->cells, &boxCells));
  }
  numFound = 0;
  for (PetscInt p = 0; p < numPoints; ++p) {
    const PetscScalar *point   = &a[p * bs];
    PetscInt           dbin[3] = {-1, -1, -1}, bin, cell = -1, cellOffset;
    PetscBool          point_outside_domain = PETSC_FALSE;

    /* check bounding box of domain */
    for (PetscInt d = 0; d < cdim; d++) {
      if (PetscRealPart(point[d]) < gmin[d]) {
        point_outside_domain = PETSC_TRUE;
        break;
      }
      if (PetscRealPart(point[d]) > gmax[d]) {
        point_outside_domain = PETSC_TRUE;
        break;
      }
    }
    if (point_outside_domain) {
      cells[p].rank  = 0;
      cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
      terminating_query_type[0]++;
      continue;
    }

    /* check initial values in cells[].index - abort early if found */
    if (cells[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      PetscInt c = cells[p].index;

      cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
      PetscCall(DMPlexLocatePoint_Internal(dm, cdim, point, c, &cell));
      if (cell >= 0) {
        cells[p].rank  = 0;
        cells[p].index = cell;
        numFound++;
      }
    }
    if (cells[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      terminating_query_type[1]++;
      continue;
    }

    if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]Checking point %" PetscInt_FMT " (%.2g, %.2g, %.2g)\n", rank, p, (double)PetscRealPart(point[0]), (double)PetscRealPart(point[1]), cdim > 2 ? (double)PetscRealPart(point[2]) : 0.));
    if (hash) {
      PetscBool found_box;

      /* allow for case that point is outside box - abort early */
      PetscCall(PetscGridHashGetEnclosingBoxQuery(mesh->lbox, mesh->lbox->cellSection, 1, point, dbin, &bin, &found_box));
      if (found_box) {
        if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]  Found point in box %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n", rank, bin, dbin[0], dbin[1], cdim > 2 ? dbin[2] : 0));
        /* TODO Lay an interface over this so we can switch between Section (dense) and Label (sparse) */
        PetscCall(PetscSectionGetDof(mesh->lbox->cellSection, bin, &numCells));
        PetscCall(PetscSectionGetOffset(mesh->lbox->cellSection, bin, &cellOffset));
        for (PetscInt c = cellOffset; c < cellOffset + numCells; ++c) {
          if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]    Checking for point in cell %" PetscInt_FMT "\n", rank, boxCells[c]));
          PetscCall(DMPlexLocatePoint_Internal(dm, cdim, point, boxCells[c], &cell));
          if (cell >= 0) {
            if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]      FOUND in cell %" PetscInt_FMT "\n", rank, cell));
            cells[p].rank  = 0;
            cells[p].index = cell;
            numFound++;
            terminating_query_type[2]++;
            break;
          }
        }
      }
    } else {
      PetscBool found = PETSC_FALSE;
      for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt idx;

        PetscCall(PetscFindInt(c, Nl, leaves, &idx));
        if (idx >= 0) continue;
        PetscCall(DMPlexLocatePoint_Internal(dm, cdim, point, c, &cell));
        if (cell >= 0) {
          cells[p].rank  = 0;
          cells[p].index = cell;
          numFound++;
          terminating_query_type[2]++;
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) terminating_query_type[0]++;
    }
  }
  if (hash) PetscCall(ISRestoreIndices(mesh->lbox->cells, &boxCells));
  if (ltype == DM_POINTLOCATION_NEAREST && hash && numFound < numPoints) {
    for (PetscInt p = 0; p < numPoints; p++) {
      const PetscScalar *point     = &a[p * bs];
      PetscReal          cpoint[3] = {0, 0, 0}, diff[3], best[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL}, dist, distMax = PETSC_MAX_REAL;
      PetscInt           dbin[3] = {-1, -1, -1}, bin, cellOffset, bestc = -1;

      if (cells[p].index < 0) {
        PetscCall(PetscGridHashGetEnclosingBox(mesh->lbox, 1, point, dbin, &bin));
        PetscCall(PetscSectionGetDof(mesh->lbox->cellSection, bin, &numCells));
        PetscCall(PetscSectionGetOffset(mesh->lbox->cellSection, bin, &cellOffset));
        for (PetscInt c = cellOffset; c < cellOffset + numCells; ++c) {
          PetscCall(DMPlexClosestPoint_Internal(dm, cdim, point, boxCells[c], cpoint));
          for (PetscInt d = 0; d < cdim; ++d) diff[d] = cpoint[d] - PetscRealPart(point[d]);
          dist = DMPlex_NormD_Internal(cdim, diff);
          if (dist < distMax) {
            for (PetscInt d = 0; d < cdim; ++d) best[d] = cpoint[d];
            bestc   = boxCells[c];
            distMax = dist;
          }
        }
        if (distMax < PETSC_MAX_REAL) {
          ++numFound;
          cells[p].rank  = 0;
          cells[p].index = bestc;
          for (PetscInt d = 0; d < cdim; ++d) a[p * bs + d] = best[d];
        }
      }
    }
  }
  /* This code is only be relevant when interfaced to parallel point location */
  /* Check for highest numbered proc that claims a point (do we care?) */
  if (ltype == DM_POINTLOCATION_REMOVE && numFound < numPoints) {
    PetscCall(PetscMalloc1(numFound, &found));
    numFound = 0;
    for (PetscInt p = 0; p < numPoints; p++) {
      if (cells[p].rank >= 0 && cells[p].index >= 0) {
        if (numFound < p) cells[numFound] = cells[p];
        found[numFound++] = p;
      }
    }
  }
  PetscCall(VecRestoreArray(v, &a));
  if (!reuse) PetscCall(PetscSFSetGraph(cellSF, cEnd - cStart, numFound, found, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER));
  PetscCall(PetscTime(&t1));
  if (hash) {
    PetscCall(PetscInfo(dm, "[DMLocatePoints_Plex] terminating_query_type : %" PetscInt_FMT " [outside domain] : %" PetscInt_FMT " [inside initial cell] : %" PetscInt_FMT " [hash]\n", terminating_query_type[0], terminating_query_type[1], terminating_query_type[2]));
  } else {
    PetscCall(PetscInfo(dm, "[DMLocatePoints_Plex] terminating_query_type : %" PetscInt_FMT " [outside domain] : %" PetscInt_FMT " [inside initial cell] : %" PetscInt_FMT " [brute-force]\n", terminating_query_type[0], terminating_query_type[1], terminating_query_type[2]));
  }
  PetscCall(PetscInfo(dm, "[DMLocatePoints_Plex] npoints %" PetscInt_FMT " : time(rank0) %1.2e (sec): points/sec %1.4e\n", numPoints, t1 - t0, numPoints / (t1 - t0)));
  PetscCall(PetscLogEventEnd(DMPLEX_LocatePoints, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexComputeProjection2Dto1D - Rewrite coordinates to be the 1D projection of the 2D coordinates

  Not Collective

  Input/Output Parameter:
. coords - The coordinates of a segment, on output the new y-coordinate, and 0 for x, an array of size 4, last two entries are unchanged

  Output Parameter:
. R - The rotation which accomplishes the projection, array of size 4

  Level: developer

.seealso: `DMPLEX`, `DMPlexComputeProjection3Dto1D()`, `DMPlexComputeProjection3Dto2D()`
@*/
PetscErrorCode DMPlexComputeProjection2Dto1D(PetscScalar coords[], PetscReal R[])
{
  const PetscReal x = PetscRealPart(coords[2] - coords[0]);
  const PetscReal y = PetscRealPart(coords[3] - coords[1]);
  const PetscReal r = PetscSqrtReal(x * x + y * y), c = x / r, s = y / r;

  PetscFunctionBegin;
  R[0]      = c;
  R[1]      = -s;
  R[2]      = s;
  R[3]      = c;
  coords[0] = 0.0;
  coords[1] = r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexComputeProjection3Dto1D - Rewrite coordinates to be the 1D projection of the 3D coordinates

  Not Collective

  Input/Output Parameter:
. coords - The coordinates of a segment; on output, the new y-coordinate, and 0 for x and z, an array of size 6, the other entries are unchanged

  Output Parameter:
. R - The rotation which accomplishes the projection, an array of size 9

  Level: developer

  Note:
  This uses the basis completion described by Frisvad {cite}`frisvad2012building`

.seealso: `DMPLEX`, `DMPlexComputeProjection2Dto1D()`, `DMPlexComputeProjection3Dto2D()`
@*/
PetscErrorCode DMPlexComputeProjection3Dto1D(PetscScalar coords[], PetscReal R[])
{
  PetscReal x    = PetscRealPart(coords[3] - coords[0]);
  PetscReal y    = PetscRealPart(coords[4] - coords[1]);
  PetscReal z    = PetscRealPart(coords[5] - coords[2]);
  PetscReal r    = PetscSqrtReal(x * x + y * y + z * z);
  PetscReal rinv = 1. / r;

  PetscFunctionBegin;
  x *= rinv;
  y *= rinv;
  z *= rinv;
  if (x > 0.) {
    PetscReal inv1pX = 1. / (1. + x);

    R[0] = x;
    R[1] = -y;
    R[2] = -z;
    R[3] = y;
    R[4] = 1. - y * y * inv1pX;
    R[5] = -y * z * inv1pX;
    R[6] = z;
    R[7] = -y * z * inv1pX;
    R[8] = 1. - z * z * inv1pX;
  } else {
    PetscReal inv1mX = 1. / (1. - x);

    R[0] = x;
    R[1] = z;
    R[2] = y;
    R[3] = y;
    R[4] = -y * z * inv1mX;
    R[5] = 1. - y * y * inv1mX;
    R[6] = z;
    R[7] = 1. - z * z * inv1mX;
    R[8] = -y * z * inv1mX;
  }
  coords[0] = 0.0;
  coords[1] = r;
  coords[2] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexComputeProjection3Dto2D - Rewrite coordinates of 3 or more coplanar 3D points to a common 2D basis for the
  plane.  The normal is defined by positive orientation of the first 3 points.

  Not Collective

  Input Parameter:
. coordSize - Length of coordinate array (3x number of points); must be at least 9 (3 points)

  Input/Output Parameter:
. coords - The interlaced coordinates of each coplanar 3D point; on output the first
           2*coordSize/3 entries contain interlaced 2D points, with the rest undefined

  Output Parameter:
. R - 3x3 row-major rotation matrix whose columns are the tangent basis [t1, t2, n].  Multiplying by R^T transforms from original frame to tangent frame.

  Level: developer

.seealso: `DMPLEX`, `DMPlexComputeProjection2Dto1D()`, `DMPlexComputeProjection3Dto1D()`
@*/
PetscErrorCode DMPlexComputeProjection3Dto2D(PetscInt coordSize, PetscScalar coords[], PetscReal R[])
{
  PetscReal      x1[3], x2[3], n[3], c[3], norm;
  const PetscInt dim = 3;
  PetscInt       d, p;

  PetscFunctionBegin;
  /* 0) Calculate normal vector */
  for (d = 0; d < dim; ++d) {
    x1[d] = PetscRealPart(coords[1 * dim + d] - coords[0 * dim + d]);
    x2[d] = PetscRealPart(coords[2 * dim + d] - coords[0 * dim + d]);
  }
  // n = x1 \otimes x2
  n[0] = x1[1] * x2[2] - x1[2] * x2[1];
  n[1] = x1[2] * x2[0] - x1[0] * x2[2];
  n[2] = x1[0] * x2[1] - x1[1] * x2[0];
  norm = PetscSqrtReal(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (d = 0; d < dim; d++) n[d] /= norm;
  norm = PetscSqrtReal(x1[0] * x1[0] + x1[1] * x1[1] + x1[2] * x1[2]);
  for (d = 0; d < dim; d++) x1[d] /= norm;
  // x2 = n \otimes x1
  x2[0] = n[1] * x1[2] - n[2] * x1[1];
  x2[1] = n[2] * x1[0] - n[0] * x1[2];
  x2[2] = n[0] * x1[1] - n[1] * x1[0];
  for (d = 0; d < dim; d++) {
    R[d * dim + 0] = x1[d];
    R[d * dim + 1] = x2[d];
    R[d * dim + 2] = n[d];
    c[d]           = PetscRealPart(coords[0 * dim + d]);
  }
  for (p = 0; p < coordSize / dim; p++) {
    PetscReal y[3];
    for (d = 0; d < dim; d++) y[d] = PetscRealPart(coords[p * dim + d]) - c[d];
    for (d = 0; d < 2; d++) coords[p * 2 + d] = R[0 * dim + d] * y[0] + R[1 * dim + d] * y[1] + R[2 * dim + d] * y[2];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static inline void Volume_Triangle_Internal(PetscReal *vol, PetscReal coords[])
{
  /* Signed volume is 1/2 the determinant

   |  1  1  1 |
   | x0 x1 x2 |
   | y0 y1 y2 |

     but if x0,y0 is the origin, we have

   | x1 x2 |
   | y1 y2 |
  */
  const PetscReal x1 = coords[2] - coords[0], y1 = coords[3] - coords[1];
  const PetscReal x2 = coords[4] - coords[0], y2 = coords[5] - coords[1];
  PetscReal       M[4], detM;
  M[0] = x1;
  M[1] = x2;
  M[2] = y1;
  M[3] = y2;
  DMPlex_Det2D_Internal(&detM, M);
  *vol = 0.5 * detM;
  (void)PetscLogFlops(5.0);
}

PETSC_UNUSED static inline void Volume_Tetrahedron_Internal(PetscReal *vol, PetscReal coords[])
{
  /* Signed volume is 1/6th of the determinant

   |  1  1  1  1 |
   | x0 x1 x2 x3 |
   | y0 y1 y2 y3 |
   | z0 z1 z2 z3 |

     but if x0,y0,z0 is the origin, we have

   | x1 x2 x3 |
   | y1 y2 y3 |
   | z1 z2 z3 |
  */
  const PetscReal x1 = coords[3] - coords[0], y1 = coords[4] - coords[1], z1 = coords[5] - coords[2];
  const PetscReal x2 = coords[6] - coords[0], y2 = coords[7] - coords[1], z2 = coords[8] - coords[2];
  const PetscReal x3 = coords[9] - coords[0], y3 = coords[10] - coords[1], z3 = coords[11] - coords[2];
  const PetscReal onesixth = ((PetscReal)1. / (PetscReal)6.);
  PetscReal       M[9], detM;
  M[0] = x1;
  M[1] = x2;
  M[2] = x3;
  M[3] = y1;
  M[4] = y2;
  M[5] = y3;
  M[6] = z1;
  M[7] = z2;
  M[8] = z3;
  DMPlex_Det3D_Internal(&detM, M);
  *vol = -onesixth * detM;
  (void)PetscLogFlops(10.0);
}

static inline void Volume_Tetrahedron_Origin_Internal(PetscReal *vol, PetscReal coords[])
{
  const PetscReal onesixth = ((PetscReal)1. / (PetscReal)6.);
  DMPlex_Det3D_Internal(vol, coords);
  *vol *= -onesixth;
}

static PetscErrorCode DMPlexComputePointGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  PetscInt           dim, d, off;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionGetDof(coordSection, e, &dim));
  if (!dim) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSectionGetOffset(coordSection, e, &off));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  if (v0) {
    for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[off + d]);
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  *detJ = 1.;
  if (J) {
    for (d = 0; d < dim * dim; d++) J[d] = 0.;
    for (d = 0; d < dim; d++) J[d * dim + d] = 1.;
    if (invJ) {
      for (d = 0; d < dim * dim; d++) invJ[d] = 0.;
      for (d = 0; d < dim; d++) invJ[d * dim + d] = 1.;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetCellCoordinates - Get coordinates for a cell, taking into account periodicity

  Not Collective

  Input Parameters:
+ dm   - The `DMPLEX`
- cell - The cell number

  Output Parameters:
+ isDG   - Using cellwise coordinates
. Nc     - The number of coordinates
. array  - The coordinate array
- coords - The cell coordinates

  Level: developer

.seealso: `DMPLEX`, `DMPlexRestoreCellCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCellCoordinatesLocal()`
@*/
PetscErrorCode DMPlexGetCellCoordinates(DM dm, PetscInt cell, PetscBool *isDG, PetscInt *Nc, const PetscScalar *array[], PetscScalar *coords[])
{
  DM                 cdm;
  Vec                coordinates;
  PetscSection       cs;
  const PetscScalar *ccoords;
  PetscInt           pStart, pEnd;

  PetscFunctionBeginHot;
  *isDG   = PETSC_FALSE;
  *Nc     = 0;
  *array  = NULL;
  *coords = NULL;
  /* Check for cellwise coordinates */
  PetscCall(DMGetCellCoordinateSection(dm, &cs));
  if (!cs) goto cg;
  /* Check that the cell exists in the cellwise section */
  PetscCall(PetscSectionGetChart(cs, &pStart, &pEnd));
  if (cell < pStart || cell >= pEnd) goto cg;
  /* Check for cellwise coordinates for this cell */
  PetscCall(PetscSectionGetDof(cs, cell, Nc));
  if (!*Nc) goto cg;
  /* Check for cellwise coordinates */
  PetscCall(DMGetCellCoordinatesLocalNoncollective(dm, &coordinates));
  if (!coordinates) goto cg;
  /* Get cellwise coordinates */
  PetscCall(DMGetCellCoordinateDM(dm, &cdm));
  PetscCall(VecGetArrayRead(coordinates, array));
  PetscCall(DMPlexPointLocalRead(cdm, cell, *array, &ccoords));
  PetscCall(DMGetWorkArray(cdm, *Nc, MPIU_SCALAR, coords));
  PetscCall(PetscArraycpy(*coords, ccoords, *Nc));
  PetscCall(VecRestoreArrayRead(coordinates, array));
  *isDG = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
cg:
  /* Use continuous coordinates */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateSection(dm, &cs));
  PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coordinates));
  PetscCall(DMPlexVecGetOrientedClosure(cdm, cs, PETSC_FALSE, coordinates, cell, 0, Nc, coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexRestoreCellCoordinates - Get coordinates for a cell, taking into account periodicity

  Not Collective

  Input Parameters:
+ dm   - The `DMPLEX`
- cell - The cell number

  Output Parameters:
+ isDG   - Using cellwise coordinates
. Nc     - The number of coordinates
. array  - The coordinate array
- coords - The cell coordinates

  Level: developer

.seealso: `DMPLEX`, `DMPlexGetCellCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCellCoordinatesLocal()`
@*/
PetscErrorCode DMPlexRestoreCellCoordinates(DM dm, PetscInt cell, PetscBool *isDG, PetscInt *Nc, const PetscScalar *array[], PetscScalar *coords[])
{
  DM           cdm;
  PetscSection cs;
  Vec          coordinates;

  PetscFunctionBeginHot;
  if (*isDG) {
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    PetscCall(DMRestoreWorkArray(cdm, *Nc, MPIU_SCALAR, coords));
  } else {
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateSection(dm, &cs));
    PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coordinates));
    PetscCall(DMPlexVecRestoreClosure(cdm, cs, coordinates, cell, Nc, coords));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeLineGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscCheck(!invJ || J, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  *detJ = 0.0;
  if (numCoords == 6) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0;

    if (v0) {
      for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
    }
    PetscCall(DMPlexComputeProjection3Dto1D(coords, R));
    if (J) {
      J0   = 0.5 * PetscRealPart(coords[1]);
      J[0] = R[0] * J0;
      J[1] = R[1];
      J[2] = R[2];
      J[3] = R[3] * J0;
      J[4] = R[4];
      J[5] = R[5];
      J[6] = R[6] * J0;
      J[7] = R[7];
      J[8] = R[8];
      DMPlex_Det3D_Internal(detJ, J);
      if (invJ) DMPlex_Invert3D_Internal(invJ, J, *detJ);
    }
  } else if (numCoords == 4) {
    const PetscInt dim = 2;
    PetscReal      R[4], J0;

    if (v0) {
      for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
    }
    PetscCall(DMPlexComputeProjection2Dto1D(coords, R));
    if (J) {
      J0   = 0.5 * PetscRealPart(coords[1]);
      J[0] = R[0] * J0;
      J[1] = R[1];
      J[2] = R[2] * J0;
      J[3] = R[3];
      DMPlex_Det2D_Internal(detJ, J);
      if (invJ) DMPlex_Invert2D_Internal(invJ, J, *detJ);
    }
  } else if (numCoords == 2) {
    const PetscInt dim = 1;

    if (v0) {
      for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
    }
    if (J) {
      J[0]  = 0.5 * (PetscRealPart(coords[1]) - PetscRealPart(coords[0]));
      *detJ = J[0];
      PetscCall(PetscLogFlops(2.0));
      if (invJ) {
        invJ[0] = 1.0 / J[0];
        PetscCall(PetscLogFlops(1.0));
      }
    }
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for segment %" PetscInt_FMT " is %" PetscInt_FMT " != 2 or 4 or 6", e, numCoords);
  PetscCall(DMPlexRestoreCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeTriangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscCheck(!invJ || J, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  *detJ = 0.0;
  if (numCoords == 9) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    if (v0) {
      for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
    }
    PetscCall(DMPlexComputeProjection3Dto2D(numCoords, coords, R));
    if (J) {
      const PetscInt pdim = 2;

      for (d = 0; d < pdim; d++) {
        for (PetscInt f = 0; f < pdim; f++) J0[d * dim + f] = 0.5 * (PetscRealPart(coords[(f + 1) * pdim + d]) - PetscRealPart(coords[0 * pdim + d]));
      }
      PetscCall(PetscLogFlops(8.0));
      DMPlex_Det3D_Internal(detJ, J0);
      for (d = 0; d < dim; d++) {
        for (PetscInt f = 0; f < dim; f++) {
          J[d * dim + f] = 0.0;
          for (PetscInt g = 0; g < dim; g++) J[d * dim + f] += R[d * dim + g] * J0[g * dim + f];
        }
      }
      PetscCall(PetscLogFlops(18.0));
    }
    if (invJ) DMPlex_Invert3D_Internal(invJ, J, *detJ);
  } else if (numCoords == 6) {
    const PetscInt dim = 2;

    if (v0) {
      for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
    }
    if (J) {
      for (d = 0; d < dim; d++) {
        for (PetscInt f = 0; f < dim; f++) J[d * dim + f] = 0.5 * (PetscRealPart(coords[(f + 1) * dim + d]) - PetscRealPart(coords[0 * dim + d]));
      }
      PetscCall(PetscLogFlops(8.0));
      DMPlex_Det2D_Internal(detJ, J);
    }
    if (invJ) DMPlex_Invert2D_Internal(invJ, J, *detJ);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this triangle is %" PetscInt_FMT " != 6 or 9", numCoords);
  PetscCall(DMPlexRestoreCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeRectangleGeometry_Internal(DM dm, PetscInt e, PetscBool isTensor, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscCheck(!invJ || J, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  if (!Nq) {
    PetscInt vorder[4] = {0, 1, 2, 3};

    if (isTensor) {
      vorder[2] = 3;
      vorder[3] = 2;
    }
    *detJ = 0.0;
    if (numCoords == 12) {
      const PetscInt dim = 3;
      PetscReal      R[9], J0[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

      if (v) {
        for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);
      }
      PetscCall(DMPlexComputeProjection3Dto2D(numCoords, coords, R));
      if (J) {
        const PetscInt pdim = 2;

        for (d = 0; d < pdim; d++) {
          J0[d * dim + 0] = 0.5 * (PetscRealPart(coords[vorder[1] * pdim + d]) - PetscRealPart(coords[vorder[0] * pdim + d]));
          J0[d * dim + 1] = 0.5 * (PetscRealPart(coords[vorder[2] * pdim + d]) - PetscRealPart(coords[vorder[1] * pdim + d]));
        }
        PetscCall(PetscLogFlops(8.0));
        DMPlex_Det3D_Internal(detJ, J0);
        for (d = 0; d < dim; d++) {
          for (PetscInt f = 0; f < dim; f++) {
            J[d * dim + f] = 0.0;
            for (PetscInt g = 0; g < dim; g++) J[d * dim + f] += R[d * dim + g] * J0[g * dim + f];
          }
        }
        PetscCall(PetscLogFlops(18.0));
      }
      if (invJ) DMPlex_Invert3D_Internal(invJ, J, *detJ);
    } else if (numCoords == 8) {
      const PetscInt dim = 2;

      if (v) {
        for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);
      }
      if (J) {
        for (d = 0; d < dim; d++) {
          J[d * dim + 0] = 0.5 * (PetscRealPart(coords[vorder[1] * dim + d]) - PetscRealPart(coords[vorder[0] * dim + d]));
          J[d * dim + 1] = 0.5 * (PetscRealPart(coords[vorder[3] * dim + d]) - PetscRealPart(coords[vorder[0] * dim + d]));
        }
        PetscCall(PetscLogFlops(8.0));
        DMPlex_Det2D_Internal(detJ, J);
      }
      if (invJ) DMPlex_Invert2D_Internal(invJ, J, *detJ);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %" PetscInt_FMT " != 8 or 12", numCoords);
  } else {
    const PetscInt Nv         = 4;
    const PetscInt dimR       = 2;
    PetscInt       zToPlex[4] = {0, 1, 3, 2};
    PetscReal      zOrder[12];
    PetscReal      zCoeff[12];
    PetscInt       i, j, k, l, dim;

    if (isTensor) {
      zToPlex[2] = 2;
      zToPlex[3] = 3;
    }
    if (numCoords == 12) {
      dim = 3;
    } else if (numCoords == 8) {
      dim = 2;
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %" PetscInt_FMT " != 8 or 12", numCoords);
    for (i = 0; i < Nv; i++) {
      PetscInt zi = zToPlex[i];

      for (j = 0; j < dim; j++) zOrder[dim * i + j] = PetscRealPart(coords[dim * zi + j]);
    }
    for (j = 0; j < dim; j++) {
      /* Nodal basis for evaluation at the vertices: (1 \mp xi) (1 \mp eta):
           \phi^0 = (1 - xi - eta + xi eta) --> 1      = 1/4 ( \phi^0 + \phi^1 + \phi^2 + \phi^3)
           \phi^1 = (1 + xi - eta - xi eta) --> xi     = 1/4 (-\phi^0 + \phi^1 - \phi^2 + \phi^3)
           \phi^2 = (1 - xi + eta - xi eta) --> eta    = 1/4 (-\phi^0 - \phi^1 + \phi^2 + \phi^3)
           \phi^3 = (1 + xi + eta + xi eta) --> xi eta = 1/4 ( \phi^0 - \phi^1 - \phi^2 + \phi^3)
      */
      zCoeff[dim * 0 + j] = 0.25 * (zOrder[dim * 0 + j] + zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
      zCoeff[dim * 1 + j] = 0.25 * (-zOrder[dim * 0 + j] + zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
      zCoeff[dim * 2 + j] = 0.25 * (-zOrder[dim * 0 + j] - zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
      zCoeff[dim * 3 + j] = 0.25 * (zOrder[dim * 0 + j] - zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
    }
    for (i = 0; i < Nq; i++) {
      PetscReal xi = points[dimR * i], eta = points[dimR * i + 1];

      if (v) {
        PetscReal extPoint[4];

        extPoint[0] = 1.;
        extPoint[1] = xi;
        extPoint[2] = eta;
        extPoint[3] = xi * eta;
        for (j = 0; j < dim; j++) {
          PetscReal val = 0.;

          for (k = 0; k < Nv; k++) val += extPoint[k] * zCoeff[dim * k + j];
          v[i * dim + j] = val;
        }
      }
      if (J) {
        PetscReal extJ[8];

        extJ[0] = 0.;
        extJ[1] = 0.;
        extJ[2] = 1.;
        extJ[3] = 0.;
        extJ[4] = 0.;
        extJ[5] = 1.;
        extJ[6] = eta;
        extJ[7] = xi;
        for (j = 0; j < dim; j++) {
          for (k = 0; k < dimR; k++) {
            PetscReal val = 0.;

            for (l = 0; l < Nv; l++) val += zCoeff[dim * l + j] * extJ[dimR * l + k];
            J[i * dim * dim + dim * j + k] = val;
          }
        }
        if (dim == 3) { /* put the cross product in the third component of the Jacobian */
          PetscReal  x, y, z;
          PetscReal *iJ = &J[i * dim * dim];
          PetscReal  norm;

          x     = iJ[1 * dim + 0] * iJ[2 * dim + 1] - iJ[1 * dim + 1] * iJ[2 * dim + 0];
          y     = iJ[0 * dim + 1] * iJ[2 * dim + 0] - iJ[0 * dim + 0] * iJ[2 * dim + 1];
          z     = iJ[0 * dim + 0] * iJ[1 * dim + 1] - iJ[0 * dim + 1] * iJ[1 * dim + 0];
          norm  = PetscSqrtReal(x * x + y * y + z * z);
          iJ[2] = x / norm;
          iJ[5] = y / norm;
          iJ[8] = z / norm;
          DMPlex_Det3D_Internal(&detJ[i], &J[i * dim * dim]);
          if (invJ) DMPlex_Invert3D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);
        } else {
          DMPlex_Det2D_Internal(&detJ[i], &J[i * dim * dim]);
          if (invJ) DMPlex_Invert2D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);
        }
      }
    }
  }
  PetscCall(DMPlexRestoreCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeTetrahedronGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  const PetscInt     dim    = 3;
  PetscInt           numCoords, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscCheck(!invJ || J, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  *detJ = 0.0;
  if (v0) {
    for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      /* I orient with outward face normals */
      J[d * dim + 0] = 0.5 * (PetscRealPart(coords[2 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
      J[d * dim + 1] = 0.5 * (PetscRealPart(coords[1 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
      J[d * dim + 2] = 0.5 * (PetscRealPart(coords[3 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
    }
    PetscCall(PetscLogFlops(18.0));
    DMPlex_Det3D_Internal(detJ, J);
  }
  if (invJ) DMPlex_Invert3D_Internal(invJ, J, *detJ);
  PetscCall(DMPlexRestoreCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeHexahedronGeometry_Internal(DM dm, PetscInt e, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  const PetscInt     dim    = 3;
  PetscInt           numCoords, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscCheck(!invJ || J, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  if (!Nq) {
    *detJ = 0.0;
    if (v) {
      for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);
    }
    if (J) {
      for (d = 0; d < dim; d++) {
        J[d * dim + 0] = 0.5 * (PetscRealPart(coords[3 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
        J[d * dim + 1] = 0.5 * (PetscRealPart(coords[1 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
        J[d * dim + 2] = 0.5 * (PetscRealPart(coords[4 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
      }
      PetscCall(PetscLogFlops(18.0));
      DMPlex_Det3D_Internal(detJ, J);
    }
    if (invJ) DMPlex_Invert3D_Internal(invJ, J, *detJ);
  } else {
    const PetscInt Nv         = 8;
    const PetscInt zToPlex[8] = {0, 3, 1, 2, 4, 5, 7, 6};
    const PetscInt dim        = 3;
    const PetscInt dimR       = 3;
    PetscReal      zOrder[24];
    PetscReal      zCoeff[24];
    PetscInt       i, j, k, l;

    for (i = 0; i < Nv; i++) {
      PetscInt zi = zToPlex[i];

      for (j = 0; j < dim; j++) zOrder[dim * i + j] = PetscRealPart(coords[dim * zi + j]);
    }
    for (j = 0; j < dim; j++) {
      zCoeff[dim * 0 + j] = 0.125 * (zOrder[dim * 0 + j] + zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j] + zOrder[dim * 4 + j] + zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 1 + j] = 0.125 * (-zOrder[dim * 0 + j] + zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j] - zOrder[dim * 4 + j] + zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 2 + j] = 0.125 * (-zOrder[dim * 0 + j] - zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j] - zOrder[dim * 4 + j] - zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 3 + j] = 0.125 * (zOrder[dim * 0 + j] - zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j] + zOrder[dim * 4 + j] - zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 4 + j] = 0.125 * (-zOrder[dim * 0 + j] - zOrder[dim * 1 + j] - zOrder[dim * 2 + j] - zOrder[dim * 3 + j] + zOrder[dim * 4 + j] + zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 5 + j] = 0.125 * (+zOrder[dim * 0 + j] - zOrder[dim * 1 + j] + zOrder[dim * 2 + j] - zOrder[dim * 3 + j] - zOrder[dim * 4 + j] + zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 6 + j] = 0.125 * (+zOrder[dim * 0 + j] + zOrder[dim * 1 + j] - zOrder[dim * 2 + j] - zOrder[dim * 3 + j] - zOrder[dim * 4 + j] - zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 7 + j] = 0.125 * (-zOrder[dim * 0 + j] + zOrder[dim * 1 + j] + zOrder[dim * 2 + j] - zOrder[dim * 3 + j] + zOrder[dim * 4 + j] - zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
    }
    for (i = 0; i < Nq; i++) {
      PetscReal xi = points[dimR * i], eta = points[dimR * i + 1], theta = points[dimR * i + 2];

      if (v) {
        PetscReal extPoint[8];

        extPoint[0] = 1.;
        extPoint[1] = xi;
        extPoint[2] = eta;
        extPoint[3] = xi * eta;
        extPoint[4] = theta;
        extPoint[5] = theta * xi;
        extPoint[6] = theta * eta;
        extPoint[7] = theta * eta * xi;
        for (j = 0; j < dim; j++) {
          PetscReal val = 0.;

          for (k = 0; k < Nv; k++) val += extPoint[k] * zCoeff[dim * k + j];
          v[i * dim + j] = val;
        }
      }
      if (J) {
        PetscReal extJ[24];

        extJ[0]  = 0.;
        extJ[1]  = 0.;
        extJ[2]  = 0.;
        extJ[3]  = 1.;
        extJ[4]  = 0.;
        extJ[5]  = 0.;
        extJ[6]  = 0.;
        extJ[7]  = 1.;
        extJ[8]  = 0.;
        extJ[9]  = eta;
        extJ[10] = xi;
        extJ[11] = 0.;
        extJ[12] = 0.;
        extJ[13] = 0.;
        extJ[14] = 1.;
        extJ[15] = theta;
        extJ[16] = 0.;
        extJ[17] = xi;
        extJ[18] = 0.;
        extJ[19] = theta;
        extJ[20] = eta;
        extJ[21] = theta * eta;
        extJ[22] = theta * xi;
        extJ[23] = eta * xi;

        for (j = 0; j < dim; j++) {
          for (k = 0; k < dimR; k++) {
            PetscReal val = 0.;

            for (l = 0; l < Nv; l++) val += zCoeff[dim * l + j] * extJ[dimR * l + k];
            J[i * dim * dim + dim * j + k] = val;
          }
        }
        DMPlex_Det3D_Internal(&detJ[i], &J[i * dim * dim]);
        if (invJ) DMPlex_Invert3D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);
      }
    }
  }
  PetscCall(DMPlexRestoreCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeTriangularPrismGeometry_Internal(DM dm, PetscInt e, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  const PetscInt     dim    = 3;
  PetscInt           numCoords, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscCheck(!invJ || J, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  if (!Nq) {
    /* Assume that the map to the reference is affine */
    *detJ = 0.0;
    if (v) {
      for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);
    }
    if (J) {
      for (d = 0; d < dim; d++) {
        J[d * dim + 0] = 0.5 * (PetscRealPart(coords[2 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
        J[d * dim + 1] = 0.5 * (PetscRealPart(coords[1 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
        J[d * dim + 2] = 0.5 * (PetscRealPart(coords[4 * dim + d]) - PetscRealPart(coords[0 * dim + d]));
      }
      PetscCall(PetscLogFlops(18.0));
      DMPlex_Det3D_Internal(detJ, J);
    }
    if (invJ) DMPlex_Invert3D_Internal(invJ, J, *detJ);
  } else {
    const PetscInt dim  = 3;
    const PetscInt dimR = 3;
    const PetscInt Nv   = 6;
    PetscReal      verts[18];
    PetscReal      coeff[18];
    PetscInt       i, j, k, l;

    for (i = 0; i < Nv; ++i)
      for (j = 0; j < dim; ++j) verts[dim * i + j] = PetscRealPart(coords[dim * i + j]);
    for (j = 0; j < dim; ++j) {
      /* Check for triangle,
           phi^0 = -1/2 (xi + eta)  chi^0 = delta(-1, -1)   x(xi) = \sum_k x_k phi^k(xi) = \sum_k chi^k(x) phi^k(xi)
           phi^1 =  1/2 (1 + xi)    chi^1 = delta( 1, -1)   y(xi) = \sum_k y_k phi^k(xi) = \sum_k chi^k(y) phi^k(xi)
           phi^2 =  1/2 (1 + eta)   chi^2 = delta(-1,  1)

           phi^0 + phi^1 + phi^2 = 1    coef_1   = 1/2 (         chi^1 + chi^2)
          -phi^0 + phi^1 - phi^2 = xi   coef_xi  = 1/2 (-chi^0 + chi^1)
          -phi^0 - phi^1 + phi^2 = eta  coef_eta = 1/2 (-chi^0         + chi^2)

          < chi_0 chi_1 chi_2> A /  1  1  1 \ / phi_0 \   <chi> I <phi>^T  so we need the inverse transpose
                                 | -1  1 -1 | | phi_1 | =
                                 \ -1 -1  1 / \ phi_2 /

          Check phi^0: 1/2 (phi^0 chi^1 + phi^0 chi^2 + phi^0 chi^0 - phi^0 chi^1 + phi^0 chi^0 - phi^0 chi^2) = phi^0 chi^0
      */
      /* Nodal basis for evaluation at the vertices: {-xi - eta, 1 + xi, 1 + eta} (1 \mp zeta):
           \phi^0 = 1/4 (   -xi - eta        + xi zeta + eta zeta) --> /  1  1  1  1  1  1 \ 1
           \phi^1 = 1/4 (1      + eta - zeta           - eta zeta) --> | -1  1 -1 -1 -1  1 | eta
           \phi^2 = 1/4 (1 + xi       - zeta - xi zeta)            --> | -1 -1  1 -1  1 -1 | xi
           \phi^3 = 1/4 (   -xi - eta        - xi zeta - eta zeta) --> | -1 -1 -1  1  1  1 | zeta
           \phi^4 = 1/4 (1 + xi       + zeta + xi zeta)            --> |  1  1 -1 -1  1 -1 | xi zeta
           \phi^5 = 1/4 (1      + eta + zeta           + eta zeta) --> \  1 -1  1 -1 -1  1 / eta zeta
           1/4 /  0  1  1  0  1  1 \
               | -1  1  0 -1  0  1 |
               | -1  0  1 -1  1  0 |
               |  0 -1 -1  0  1  1 |
               |  1  0 -1 -1  1  0 |
               \  1 -1  0 -1  0  1 /
      */
      coeff[dim * 0 + j] = (1. / 4.) * (verts[dim * 1 + j] + verts[dim * 2 + j] + verts[dim * 4 + j] + verts[dim * 5 + j]);
      coeff[dim * 1 + j] = (1. / 4.) * (-verts[dim * 0 + j] + verts[dim * 1 + j] - verts[dim * 3 + j] + verts[dim * 5 + j]);
      coeff[dim * 2 + j] = (1. / 4.) * (-verts[dim * 0 + j] + verts[dim * 2 + j] - verts[dim * 3 + j] + verts[dim * 4 + j]);
      coeff[dim * 3 + j] = (1. / 4.) * (-verts[dim * 1 + j] - verts[dim * 2 + j] + verts[dim * 4 + j] + verts[dim * 5 + j]);
      coeff[dim * 4 + j] = (1. / 4.) * (verts[dim * 0 + j] - verts[dim * 2 + j] - verts[dim * 3 + j] + verts[dim * 4 + j]);
      coeff[dim * 5 + j] = (1. / 4.) * (verts[dim * 0 + j] - verts[dim * 1 + j] - verts[dim * 3 + j] + verts[dim * 5 + j]);
      /* For reference prism:
      {0, 0, 0}
      {0, 1, 0}
      {1, 0, 0}
      {0, 0, 1}
      {0, 0, 0}
      {0, 0, 0}
      */
    }
    for (i = 0; i < Nq; ++i) {
      const PetscReal xi = points[dimR * i], eta = points[dimR * i + 1], zeta = points[dimR * i + 2];

      if (v) {
        PetscReal extPoint[6];
        PetscInt  c;

        extPoint[0] = 1.;
        extPoint[1] = eta;
        extPoint[2] = xi;
        extPoint[3] = zeta;
        extPoint[4] = xi * zeta;
        extPoint[5] = eta * zeta;
        for (c = 0; c < dim; ++c) {
          PetscReal val = 0.;

          for (k = 0; k < Nv; ++k) val += extPoint[k] * coeff[k * dim + c];
          v[i * dim + c] = val;
        }
      }
      if (J) {
        PetscReal extJ[18];

        extJ[0]  = 0.;
        extJ[1]  = 0.;
        extJ[2]  = 0.;
        extJ[3]  = 0.;
        extJ[4]  = 1.;
        extJ[5]  = 0.;
        extJ[6]  = 1.;
        extJ[7]  = 0.;
        extJ[8]  = 0.;
        extJ[9]  = 0.;
        extJ[10] = 0.;
        extJ[11] = 1.;
        extJ[12] = zeta;
        extJ[13] = 0.;
        extJ[14] = xi;
        extJ[15] = 0.;
        extJ[16] = zeta;
        extJ[17] = eta;

        for (j = 0; j < dim; j++) {
          for (k = 0; k < dimR; k++) {
            PetscReal val = 0.;

            for (l = 0; l < Nv; l++) val += coeff[dim * l + j] * extJ[dimR * l + k];
            J[i * dim * dim + dim * j + k] = val;
          }
        }
        DMPlex_Det3D_Internal(&detJ[i], &J[i * dim * dim]);
        if (invJ) DMPlex_Invert3D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);
      }
    }
  }
  PetscCall(DMPlexRestoreCellCoordinates(dm, e, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeCellGeometryFEM_Implicit(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal *v, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  DMPolytopeType   ct;
  PetscInt         depth, dim, coordDim, coneSize, i;
  PetscInt         Nq     = 0;
  const PetscReal *points = NULL;
  DMLabel          depthLabel;
  PetscReal        xi0[3]   = {-1., -1., -1.}, v0[3], J0[9], detJ0;
  PetscBool        isAffine = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetValue(depthLabel, cell, &dim));
  if (depth == 1 && dim == 1) PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  PetscCheck(coordDim <= 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported coordinate dimension %" PetscInt_FMT " > 3", coordDim);
  if (quad) PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &points, NULL));
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
  case DM_POLYTOPE_POINT:
    PetscCall(DMPlexComputePointGeometry_Internal(dm, cell, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
    if (Nq) PetscCall(DMPlexComputeLineGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0));
    else PetscCall(DMPlexComputeLineGeometry_Internal(dm, cell, v, J, invJ, detJ));
    break;
  case DM_POLYTOPE_TRIANGLE:
    if (Nq) PetscCall(DMPlexComputeTriangleGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0));
    else PetscCall(DMPlexComputeTriangleGeometry_Internal(dm, cell, v, J, invJ, detJ));
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(DMPlexComputeRectangleGeometry_Internal(dm, cell, PETSC_FALSE, Nq, points, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
    PetscCall(DMPlexComputeRectangleGeometry_Internal(dm, cell, PETSC_TRUE, Nq, points, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    if (Nq) PetscCall(DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0));
    else PetscCall(DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v, J, invJ, detJ));
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexComputeHexahedronGeometry_Internal(dm, cell, Nq, points, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
  case DM_POLYTOPE_TRI_PRISM:
    PetscCall(DMPlexComputeTriangularPrismGeometry_Internal(dm, cell, Nq, points, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No element geometry for cell %" PetscInt_FMT " with type %s", cell, DMPolytopeTypes[PetscMax(0, PetscMin(ct, DM_NUM_POLYTOPES))]);
  }
  if (isAffine && Nq) {
    if (v) {
      for (i = 0; i < Nq; i++) CoordinatesRefToReal(coordDim, dim, xi0, v0, J0, &points[dim * i], &v[coordDim * i]);
    }
    if (detJ) {
      for (i = 0; i < Nq; i++) detJ[i] = detJ0;
    }
    if (J) {
      PetscInt k;

      for (i = 0, k = 0; i < Nq; i++) {
        PetscInt j;

        for (j = 0; j < coordDim * coordDim; j++, k++) J[k] = J0[j];
      }
    }
    if (invJ) {
      PetscInt k;
      switch (coordDim) {
      case 0:
        break;
      case 1:
        invJ[0] = 1. / J0[0];
        break;
      case 2:
        DMPlex_Invert2D_Internal(invJ, J0, detJ0);
        break;
      case 3:
        DMPlex_Invert3D_Internal(invJ, J0, detJ0);
        break;
      }
      for (i = 1, k = coordDim * coordDim; i < Nq; i++) {
        PetscInt j;

        for (j = 0; j < coordDim * coordDim; j++, k++) invJ[k] = invJ[j];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexComputeCellGeometryAffineFEM - Assuming an affine map, compute the Jacobian, inverse Jacobian, and Jacobian determinant for a given cell

  Collective

  Input Parameters:
+ dm   - the `DMPLEX`
- cell - the cell

  Output Parameters:
+ v0   - the translation part of this affine transform, meaning the translation to the origin (not the first vertex of the reference cell)
. J    - the Jacobian of the transform from the reference element
. invJ - the inverse of the Jacobian
- detJ - the Jacobian determinant

  Level: advanced

.seealso: `DMPLEX`, `DMPlexComputeCellGeometryFEM()`, `DMGetCoordinateSection()`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexComputeCellGeometryAffineFEM(DM dm, PetscInt cell, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM_Implicit(dm, cell, NULL, v0, J, invJ, detJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeCellGeometryFEM_FE(DM dm, PetscFE fe, PetscInt point, PetscQuadrature quad, PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           numCoords;
  PetscBool          isDG;
  PetscQuadrature    feQuad;
  const PetscReal   *quadPoints;
  PetscTabulation    T;
  PetscInt           dim, cdim, pdim, qdim, Nq, q;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetCellCoordinates(dm, point, &isDG, &numCoords, &array, &coords));
  if (!quad) { /* use the first point of the first functional of the dual space */
    PetscDualSpace dsp;

    PetscCall(PetscFEGetDualSpace(fe, &dsp));
    PetscCall(PetscDualSpaceGetFunctional(dsp, 0, &quad));
    PetscCall(PetscQuadratureGetData(quad, &qdim, NULL, &Nq, &quadPoints, NULL));
    Nq = 1;
  } else {
    PetscCall(PetscQuadratureGetData(quad, &qdim, NULL, &Nq, &quadPoints, NULL));
  }
  PetscCall(PetscFEGetDimension(fe, &pdim));
  PetscCall(PetscFEGetQuadrature(fe, &feQuad));
  if (feQuad == quad) {
    PetscCall(PetscFEGetCellTabulation(fe, J ? 1 : 0, &T));
    PetscCheck(numCoords == pdim * cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "There are %" PetscInt_FMT " coordinates for point %" PetscInt_FMT " != %" PetscInt_FMT "*%" PetscInt_FMT, numCoords, point, pdim, cdim);
  } else {
    PetscCall(PetscFECreateTabulation(fe, 1, Nq, quadPoints, J ? 1 : 0, &T));
  }
  PetscCheck(qdim == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Point dimension %" PetscInt_FMT " != quadrature dimension %" PetscInt_FMT, dim, qdim);
  {
    const PetscReal *basis    = T->T[0];
    const PetscReal *basisDer = T->T[1];
    PetscReal        detJt;

    PetscAssert(Nq == T->Np, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Np %" PetscInt_FMT " != %" PetscInt_FMT, Nq, T->Np);
    PetscAssert(pdim == T->Nb, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nb %" PetscInt_FMT " != %" PetscInt_FMT, pdim, T->Nb);
    PetscAssert(cdim == T->Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nc %" PetscInt_FMT " != %" PetscInt_FMT, cdim, T->Nc);
    PetscAssert(dim == T->cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "cdim %" PetscInt_FMT " != %" PetscInt_FMT, dim, T->cdim);
    if (v) {
      PetscCall(PetscArrayzero(v, Nq * cdim));
      for (q = 0; q < Nq; ++q) {
        PetscInt i, k;

        for (k = 0; k < pdim; ++k) {
          const PetscInt vertex = k / cdim;
          for (i = 0; i < cdim; ++i) v[q * cdim + i] += basis[(q * pdim + k) * cdim + i] * PetscRealPart(coords[vertex * cdim + i]);
        }
        PetscCall(PetscLogFlops(2.0 * pdim * cdim));
      }
    }
    if (J) {
      PetscCall(PetscArrayzero(J, Nq * cdim * cdim));
      for (q = 0; q < Nq; ++q) {
        PetscInt i, j, k, c, r;

        /* J = dx_i/d\xi_j = sum[k=0,n-1] dN_k/d\xi_j * x_i(k) */
        for (k = 0; k < pdim; ++k) {
          const PetscInt vertex = k / cdim;
          for (j = 0; j < dim; ++j) {
            for (i = 0; i < cdim; ++i) J[(q * cdim + i) * cdim + j] += basisDer[((q * pdim + k) * cdim + i) * dim + j] * PetscRealPart(coords[vertex * cdim + i]);
          }
        }
        PetscCall(PetscLogFlops(2.0 * pdim * dim * cdim));
        if (cdim > dim) {
          for (c = dim; c < cdim; ++c)
            for (r = 0; r < cdim; ++r) J[r * cdim + c] = r == c ? 1.0 : 0.0;
        }
        if (!detJ && !invJ) continue;
        detJt = 0.;
        switch (cdim) {
        case 3:
          DMPlex_Det3D_Internal(&detJt, &J[q * cdim * dim]);
          if (invJ) DMPlex_Invert3D_Internal(&invJ[q * cdim * dim], &J[q * cdim * dim], detJt);
          break;
        case 2:
          DMPlex_Det2D_Internal(&detJt, &J[q * cdim * dim]);
          if (invJ) DMPlex_Invert2D_Internal(&invJ[q * cdim * dim], &J[q * cdim * dim], detJt);
          break;
        case 1:
          detJt = J[q * cdim * dim];
          if (invJ) invJ[q * cdim * dim] = 1.0 / detJt;
        }
        if (detJ) detJ[q] = detJt;
      }
    } else PetscCheck(!detJ && !invJ, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Need J to compute invJ or detJ");
  }
  if (feQuad != quad) PetscCall(PetscTabulationDestroy(&T));
  PetscCall(DMPlexRestoreCellCoordinates(dm, point, &isDG, &numCoords, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexComputeCellGeometryFEM - Compute the Jacobian, inverse Jacobian, and Jacobian determinant at each quadrature point in the given cell

  Collective

  Input Parameters:
+ dm   - the `DMPLEX`
. cell - the cell
- quad - the quadrature containing the points in the reference element where the geometry will be evaluated.  If `quad` is `NULL`, geometry will be
         evaluated at the first vertex of the reference element

  Output Parameters:
+ v    - the image of the transformed quadrature points, otherwise the image of the first vertex in the closure of the reference element. This is a
         one-dimensional array of size $cdim * Nq$ where $cdim$ is the dimension of the `DM` coordinate space and $Nq$ is the number of quadrature points
. J    - the Jacobian of the transform from the reference element at each quadrature point. This is a one-dimensional array of size $Nq * cdim * cdim$ containing
         each Jacobian in column-major order.
. invJ - the inverse of the Jacobian at each quadrature point. This is a one-dimensional array of size $Nq * cdim * cdim$ containing
         each inverse Jacobian in column-major order.
- detJ - the Jacobian determinant at each quadrature point. This is a one-dimensional array of size $Nq$.

  Level: advanced

  Note:
  Implicit cell geometry must be used when the topological mesh dimension is not equal to the coordinate dimension, for instance for embedded manifolds.

.seealso: `DMPLEX`, `DMGetCoordinateSection()`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexComputeCellGeometryFEM(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal detJ[])
{
  DM       cdm;
  PetscFE  fe = NULL;
  PetscInt dim, cdim;

  PetscFunctionBegin;
  PetscAssertPointer(detJ, 7);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  if (cdm) {
    PetscClassId id;
    PetscInt     numFields;
    PetscDS      prob;
    PetscObject  disc;

    PetscCall(DMGetNumFields(cdm, &numFields));
    if (numFields) {
      PetscCall(DMGetDS(cdm, &prob));
      PetscCall(PetscDSGetDiscretization(prob, 0, &disc));
      PetscCall(PetscObjectGetClassId(disc, &id));
      if (id == PETSCFE_CLASSID) fe = (PetscFE)disc;
    }
  }
  if (!fe || (dim != cdim)) PetscCall(DMPlexComputeCellGeometryFEM_Implicit(dm, cell, quad, v, J, invJ, detJ));
  else PetscCall(DMPlexComputeCellGeometryFEM_FE(dm, fe, cell, quad, v, J, invJ, detJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeGeometryFVM_0D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords = NULL;
  PetscInt           d, dof, off;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(VecGetArrayRead(coordinates, &coords));

  /* for a point the centroid is just the coord */
  if (centroid) {
    PetscCall(PetscSectionGetDof(coordSection, cell, &dof));
    PetscCall(PetscSectionGetOffset(coordSection, cell, &off));
    for (d = 0; d < dof; d++) centroid[d] = PetscRealPart(coords[off + d]);
  }
  if (normal) {
    const PetscInt *support, *cones;
    PetscInt        supportSize;
    PetscReal       norm, sign;

    /* compute the norm based upon the support centroids */
    PetscCall(DMPlexGetSupportSize(dm, cell, &supportSize));
    PetscCall(DMPlexGetSupport(dm, cell, &support));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, support[0], NULL, normal, NULL));

    /* Take the normal from the centroid of the support to the vertex*/
    PetscCall(PetscSectionGetDof(coordSection, cell, &dof));
    PetscCall(PetscSectionGetOffset(coordSection, cell, &off));
    for (d = 0; d < dof; d++) normal[d] -= PetscRealPart(coords[off + d]);

    /* Determine the sign of the normal based upon its location in the support */
    PetscCall(DMPlexGetCone(dm, support[0], &cones));
    sign = cones[0] == cell ? 1.0 : -1.0;

    norm = DMPlex_NormD_Internal(dim, normal);
    for (d = 0; d < dim; ++d) normal[d] /= (norm * sign);
  }
  if (vol) *vol = 1.0;
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexComputeGeometryFVM_1D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           cdim, coordSize, d;
  PetscBool          isDG;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &coordSize, &array, &coords));
  PetscCheck(coordSize == cdim * 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge has %" PetscInt_FMT " coordinates != %" PetscInt_FMT, coordSize, cdim * 2);
  if (centroid) {
    for (d = 0; d < cdim; ++d) centroid[d] = 0.5 * PetscRealPart(coords[d] + coords[cdim + d]);
  }
  if (normal) {
    PetscReal norm;

    switch (cdim) {
    case 3:
      normal[2] = 0.; /* fall through */
    case 2:
      normal[0] = -PetscRealPart(coords[1] - coords[cdim + 1]);
      normal[1] = PetscRealPart(coords[0] - coords[cdim + 0]);
      break;
    case 1:
      normal[0] = 1.0;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT " not supported", cdim);
    }
    norm = DMPlex_NormD_Internal(cdim, normal);
    for (d = 0; d < cdim; ++d) normal[d] /= norm;
  }
  if (vol) {
    *vol = 0.0;
    for (d = 0; d < cdim; ++d) *vol += PetscSqr(PetscRealPart(coords[d] - coords[cdim + d]));
    *vol = PetscSqrtReal(*vol);
  }
  PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &coordSize, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Centroid_i = (\sum_n A_n Cn_i) / A */
static PetscErrorCode DMPlexComputeGeometryFVM_2D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  DMPolytopeType     ct;
  const PetscScalar *array;
  PetscScalar       *coords = NULL;
  PetscInt           coordSize;
  PetscBool          isDG;
  PetscInt           fv[4] = {0, 1, 2, 3};
  PetscInt           cdim, numCorners, p, d;

  PetscFunctionBegin;
  /* Must check for hybrid cells because prisms have a different orientation scheme */
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
    fv[2] = 3;
    fv[3] = 2;
    break;
  default:
    break;
  }
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetConeSize(dm, cell, &numCorners));
  PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &coordSize, &array, &coords));
  {
    PetscReal c[3] = {0., 0., 0.}, n[3] = {0., 0., 0.}, origin[3] = {0., 0., 0.}, norm;

    for (d = 0; d < cdim; d++) origin[d] = PetscRealPart(coords[d]);
    for (p = 0; p < numCorners - 2; ++p) {
      PetscReal e0[3] = {0., 0., 0.}, e1[3] = {0., 0., 0.};
      for (d = 0; d < cdim; d++) {
        e0[d] = PetscRealPart(coords[cdim * fv[p + 1] + d]) - origin[d];
        e1[d] = PetscRealPart(coords[cdim * fv[p + 2] + d]) - origin[d];
      }
      const PetscReal dx = e0[1] * e1[2] - e0[2] * e1[1];
      const PetscReal dy = e0[2] * e1[0] - e0[0] * e1[2];
      const PetscReal dz = e0[0] * e1[1] - e0[1] * e1[0];
      const PetscReal a  = PetscSqrtReal(dx * dx + dy * dy + dz * dz);

      n[0] += dx;
      n[1] += dy;
      n[2] += dz;
      for (d = 0; d < cdim; d++) c[d] += a * PetscRealPart(origin[d] + coords[cdim * fv[p + 1] + d] + coords[cdim * fv[p + 2] + d]) / 3.;
    }
    norm = PetscSqrtReal(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    // Allow zero volume cells
    if (norm != 0) {
      n[0] /= norm;
      n[1] /= norm;
      n[2] /= norm;
      c[0] /= norm;
      c[1] /= norm;
      c[2] /= norm;
    }
    if (vol) *vol = 0.5 * norm;
    if (centroid)
      for (d = 0; d < cdim; ++d) centroid[d] = c[d];
    if (normal)
      for (d = 0; d < cdim; ++d) normal[d] = n[d];
  }
  PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &coordSize, &array, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Centroid_i = (\sum_n V_n Cn_i) / V */
static PetscErrorCode DMPlexComputeGeometryFVM_3D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  DMPolytopeType        ct;
  const PetscScalar    *array;
  PetscScalar          *coords = NULL;
  PetscInt              coordSize;
  PetscBool             isDG;
  PetscReal             vsum      = 0.0, vtmp, coordsTmp[3 * 3], origin[3];
  const PetscInt        order[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const PetscInt       *cone, *faceSizes, *faces;
  const DMPolytopeType *faceTypes;
  PetscBool             isHybrid = PETSC_FALSE;
  PetscInt              numFaces, f, fOff = 0, p, d;

  PetscFunctionBegin;
  PetscCheck(dim <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No support for dim %" PetscInt_FMT " > 3", dim);
  /* Must check for hybrid cells because prisms have a different orientation scheme */
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
  case DM_POLYTOPE_TRI_PRISM_TENSOR:
  case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    isHybrid = PETSC_TRUE;
  default:
    break;
  }

  if (centroid)
    for (d = 0; d < dim; ++d) centroid[d] = 0.0;
  PetscCall(DMPlexGetCone(dm, cell, &cone));

  // Using the closure of faces for coordinates does not work in periodic geometries, so we index into the cell coordinates
  PetscCall(DMPlexGetRawFaces_Internal(dm, ct, order, &numFaces, &faceTypes, &faceSizes, &faces));
  PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &coordSize, &array, &coords));
  for (f = 0; f < numFaces; ++f) {
    PetscBool flip = isHybrid && f == 0 ? PETSC_TRUE : PETSC_FALSE; /* The first hybrid face is reversed */

    // If using zero as the origin vertex for each tetrahedron, an element far from the origin will have positive and
    // negative volumes that nearly cancel, thus incurring rounding error. Here we define origin[] as the first vertex
    // so that all tetrahedra have positive volume.
    if (f == 0)
      for (d = 0; d < dim; d++) origin[d] = PetscRealPart(coords[d]);
    switch (faceTypes[f]) {
    case DM_POLYTOPE_TRIANGLE:
      for (d = 0; d < dim; ++d) {
        coordsTmp[0 * dim + d] = PetscRealPart(coords[faces[fOff + 0] * dim + d]) - origin[d];
        coordsTmp[1 * dim + d] = PetscRealPart(coords[faces[fOff + 1] * dim + d]) - origin[d];
        coordsTmp[2 * dim + d] = PetscRealPart(coords[faces[fOff + 2] * dim + d]) - origin[d];
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (flip) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) { /* Centroid of OABC = (a+b+c)/4 */
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p * dim + d] * vtmp;
        }
      }
      break;
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_SEG_PRISM_TENSOR: {
      PetscInt fv[4] = {0, 1, 2, 3};

      /* Side faces for hybrid cells are stored as tensor products */
      if (isHybrid && f > 1) {
        fv[2] = 3;
        fv[3] = 2;
      }
      /* DO FOR PYRAMID */
      /* First tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0 * dim + d] = PetscRealPart(coords[faces[fOff + fv[0]] * dim + d]) - origin[d];
        coordsTmp[1 * dim + d] = PetscRealPart(coords[faces[fOff + fv[1]] * dim + d]) - origin[d];
        coordsTmp[2 * dim + d] = PetscRealPart(coords[faces[fOff + fv[3]] * dim + d]) - origin[d];
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (flip) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p * dim + d] * vtmp;
        }
      }
      /* Second tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0 * dim + d] = PetscRealPart(coords[faces[fOff + fv[1]] * dim + d]) - origin[d];
        coordsTmp[1 * dim + d] = PetscRealPart(coords[faces[fOff + fv[2]] * dim + d]) - origin[d];
        coordsTmp[2 * dim + d] = PetscRealPart(coords[faces[fOff + fv[3]] * dim + d]) - origin[d];
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (flip) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p * dim + d] * vtmp;
        }
      }
      break;
    }
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle face %" PetscInt_FMT " of type %s", cone[f], DMPolytopeTypes[ct]);
    }
    fOff += faceSizes[f];
  }
  PetscCall(DMPlexRestoreRawFaces_Internal(dm, ct, order, &numFaces, &faceTypes, &faceSizes, &faces));
  PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &coordSize, &array, &coords));
  if (vol) *vol = PetscAbsReal(vsum);
  if (normal)
    for (d = 0; d < dim; ++d) normal[d] = 0.0;
  if (centroid)
    for (d = 0; d < dim; ++d) centroid[d] = centroid[d] / (vsum * 4) + origin[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexComputeCellGeometryFVM - Compute the volume for a given cell

  Collective

  Input Parameters:
+ dm   - the `DMPLEX`
- cell - the cell

  Output Parameters:
+ vol      - the cell volume
. centroid - the cell centroid
- normal   - the cell normal, if appropriate

  Level: advanced

.seealso: `DMPLEX`, `DMGetCoordinateSection()`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexComputeCellGeometryFVM(DM dm, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscInt depth, dim;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(depth == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated");
  PetscCall(DMPlexGetPointDepth(dm, cell, &depth));
  switch (depth) {
  case 0:
    PetscCall(DMPlexComputeGeometryFVM_0D_Internal(dm, dim, cell, vol, centroid, normal));
    break;
  case 1:
    PetscCall(DMPlexComputeGeometryFVM_1D_Internal(dm, dim, cell, vol, centroid, normal));
    break;
  case 2:
    PetscCall(DMPlexComputeGeometryFVM_2D_Internal(dm, dim, cell, vol, centroid, normal));
    break;
  case 3:
    PetscCall(DMPlexComputeGeometryFVM_3D_Internal(dm, dim, cell, vol, centroid, normal));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT " (depth %" PetscInt_FMT ") for element geometry computation", dim, depth);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexComputeGeometryFVM - Computes the cell and face geometry for a finite volume method

  Input Parameter:
. dm - The `DMPLEX`

  Output Parameters:
+ cellgeom - A `Vec` of `PetscFVCellGeom` data
- facegeom - A `Vec` of `PetscFVFaceGeom` data

  Level: developer

.seealso: `DMPLEX`, `PetscFVFaceGeom`, `PetscFVCellGeom`
@*/
PetscErrorCode DMPlexComputeGeometryFVM(DM dm, Vec *cellgeom, Vec *facegeom)
{
  DM           dmFace, dmCell;
  DMLabel      ghostLabel;
  PetscSection sectionFace, sectionCell;
  PetscSection coordSection;
  Vec          coordinates;
  PetscScalar *fgeom, *cgeom;
  PetscReal    minradius, gminradius;
  PetscInt     dim, cStart, cEnd, cEndInterior, c, fStart, fEnd, f;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  /* Make cell centroids and volumes */
  PetscCall(DMClone(dm, &dmCell));
  PetscCall(DMSetCoordinateSection(dmCell, PETSC_DETERMINE, coordSection));
  PetscCall(DMSetCoordinatesLocal(dmCell, coordinates));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionCell));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, NULL));
  PetscCall(PetscSectionSetChart(sectionCell, cStart, cEnd));
  for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionCell, c, (PetscInt)PetscCeilReal(((PetscReal)sizeof(PetscFVCellGeom)) / sizeof(PetscScalar))));
  PetscCall(PetscSectionSetUp(sectionCell));
  PetscCall(DMSetLocalSection(dmCell, sectionCell));
  PetscCall(PetscSectionDestroy(&sectionCell));
  PetscCall(DMCreateLocalVector(dmCell, cellgeom));
  if (cEndInterior < 0) cEndInterior = cEnd;
  PetscCall(VecGetArray(*cellgeom, &cgeom));
  for (c = cStart; c < cEndInterior; ++c) {
    PetscFVCellGeom *cg;

    PetscCall(DMPlexPointLocalRef(dmCell, c, cgeom, &cg));
    PetscCall(PetscArrayzero(cg, 1));
    PetscCall(DMPlexComputeCellGeometryFVM(dmCell, c, &cg->volume, cg->centroid, NULL));
  }
  /* Compute face normals and minimum cell radius */
  PetscCall(DMClone(dm, &dmFace));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionFace));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(PetscSectionSetChart(sectionFace, fStart, fEnd));
  for (f = fStart; f < fEnd; ++f) PetscCall(PetscSectionSetDof(sectionFace, f, (PetscInt)PetscCeilReal(((PetscReal)sizeof(PetscFVFaceGeom)) / sizeof(PetscScalar))));
  PetscCall(PetscSectionSetUp(sectionFace));
  PetscCall(DMSetLocalSection(dmFace, sectionFace));
  PetscCall(PetscSectionDestroy(&sectionFace));
  PetscCall(DMCreateLocalVector(dmFace, facegeom));
  PetscCall(VecGetArray(*facegeom, &fgeom));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  minradius = PETSC_MAX_REAL;
  for (f = fStart; f < fEnd; ++f) {
    PetscFVFaceGeom *fg;
    PetscReal        area;
    const PetscInt  *cells;
    PetscInt         ncells, ghost = -1, d, numChildren;

    if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
    PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
    PetscCall(DMPlexGetSupport(dm, f, &cells));
    PetscCall(DMPlexGetSupportSize(dm, f, &ncells));
    /* It is possible to get a face with no support when using partition overlap */
    if (!ncells || ghost >= 0 || numChildren) continue;
    PetscCall(DMPlexPointLocalRef(dmFace, f, fgeom, &fg));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &area, fg->centroid, fg->normal));
    for (d = 0; d < dim; ++d) fg->normal[d] *= area;
    /* Flip face orientation if necessary to match ordering in support, and Update minimum radius */
    {
      PetscFVCellGeom *cL, *cR;
      PetscReal       *lcentroid, *rcentroid;
      PetscReal        l[3], r[3], v[3];

      PetscCall(DMPlexPointLocalRead(dmCell, cells[0], cgeom, &cL));
      lcentroid = cells[0] >= cEndInterior ? fg->centroid : cL->centroid;
      if (ncells > 1) {
        PetscCall(DMPlexPointLocalRead(dmCell, cells[1], cgeom, &cR));
        rcentroid = cells[1] >= cEndInterior ? fg->centroid : cR->centroid;
      } else {
        rcentroid = fg->centroid;
      }
      PetscCall(DMLocalizeCoordinateReal_Internal(dm, dim, fg->centroid, lcentroid, l));
      PetscCall(DMLocalizeCoordinateReal_Internal(dm, dim, fg->centroid, rcentroid, r));
      DMPlex_WaxpyD_Internal(dim, -1, l, r, v);
      if (DMPlex_DotRealD_Internal(dim, fg->normal, v) < 0) {
        for (d = 0; d < dim; ++d) fg->normal[d] = -fg->normal[d];
      }
      if (DMPlex_DotRealD_Internal(dim, fg->normal, v) <= 0) {
        PetscCheck(dim != 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Direction for face %" PetscInt_FMT " could not be fixed, normal (%g,%g) v (%g,%g)", f, (double)fg->normal[0], (double)fg->normal[1], (double)v[0], (double)v[1]);
        PetscCheck(dim != 3, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Direction for face %" PetscInt_FMT " could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", f, (double)fg->normal[0], (double)fg->normal[1], (double)fg->normal[2], (double)v[0], (double)v[1], (double)v[2]);
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Direction for face %" PetscInt_FMT " could not be fixed", f);
      }
      if (cells[0] < cEndInterior) {
        DMPlex_WaxpyD_Internal(dim, -1, fg->centroid, cL->centroid, v);
        minradius = PetscMin(minradius, DMPlex_NormD_Internal(dim, v));
      }
      if (ncells > 1 && cells[1] < cEndInterior) {
        DMPlex_WaxpyD_Internal(dim, -1, fg->centroid, cR->centroid, v);
        minradius = PetscMin(minradius, DMPlex_NormD_Internal(dim, v));
      }
    }
  }
  PetscCallMPI(MPIU_Allreduce(&minradius, &gminradius, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm)));
  PetscCall(DMPlexSetMinRadius(dm, gminradius));
  /* Compute centroids of ghost cells */
  for (c = cEndInterior; c < cEnd; ++c) {
    PetscFVFaceGeom *fg;
    const PetscInt  *cone, *support;
    PetscInt         coneSize, supportSize, s;

    PetscCall(DMPlexGetConeSize(dmCell, c, &coneSize));
    PetscCheck(coneSize == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %" PetscInt_FMT " has cone size %" PetscInt_FMT " != 1", c, coneSize);
    PetscCall(DMPlexGetCone(dmCell, c, &cone));
    PetscCall(DMPlexGetSupportSize(dmCell, cone[0], &supportSize));
    PetscCheck(supportSize == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " has support size %" PetscInt_FMT " != 2", cone[0], supportSize);
    PetscCall(DMPlexGetSupport(dmCell, cone[0], &support));
    PetscCall(DMPlexPointLocalRef(dmFace, cone[0], fgeom, &fg));
    for (s = 0; s < 2; ++s) {
      /* Reflect ghost centroid across plane of face */
      if (support[s] == c) {
        PetscFVCellGeom *ci;
        PetscFVCellGeom *cg;
        PetscReal        c2f[3], a;

        PetscCall(DMPlexPointLocalRead(dmCell, support[(s + 1) % 2], cgeom, &ci));
        DMPlex_WaxpyD_Internal(dim, -1, ci->centroid, fg->centroid, c2f); /* cell to face centroid */
        a = DMPlex_DotRealD_Internal(dim, c2f, fg->normal) / DMPlex_DotRealD_Internal(dim, fg->normal, fg->normal);
        PetscCall(DMPlexPointLocalRef(dmCell, support[s], cgeom, &cg));
        DMPlex_WaxpyD_Internal(dim, 2 * a, fg->normal, ci->centroid, cg->centroid);
        cg->volume = ci->volume;
      }
    }
  }
  PetscCall(VecRestoreArray(*facegeom, &fgeom));
  PetscCall(VecRestoreArray(*cellgeom, &cgeom));
  PetscCall(DMDestroy(&dmCell));
  PetscCall(DMDestroy(&dmFace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetMinRadius - Returns the minimum distance from any cell centroid to a face

  Not Collective

  Input Parameter:
. dm - the `DMPLEX`

  Output Parameter:
. minradius - the minimum cell radius

  Level: developer

.seealso: `DMPLEX`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexGetMinRadius(DM dm, PetscReal *minradius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(minradius, 2);
  *minradius = ((DM_Plex *)dm->data)->minradius;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetMinRadius - Sets the minimum distance from the cell centroid to a face

  Logically Collective

  Input Parameters:
+ dm        - the `DMPLEX`
- minradius - the minimum cell radius

  Level: developer

.seealso: `DMPLEX`, `DMSetCoordinates()`
@*/
PetscErrorCode DMPlexSetMinRadius(DM dm, PetscReal minradius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ((DM_Plex *)dm->data)->minradius = minradius;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetCoordinateMap - Returns the function used to map coordinates of newly generated mesh points

  Not Collective

  Input Parameter:
. dm - the `DMPLEX`

  Output Parameter:
. coordFunc - the mapping function

  Level: developer

  Note:
  This function maps from the generated coordinate for the new point to the actual coordinate. Thus it is only practical for manifolds with a nice analytical definition that you can get to from any starting point, like a sphere,

.seealso: `DMPLEX`, `DMGetCoordinates()`, `DMPlexSetCoordinateMap()`, `PetscPointFn`
@*/
PetscErrorCode DMPlexGetCoordinateMap(DM dm, PetscPointFn **coordFunc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(coordFunc, 2);
  *coordFunc = ((DM_Plex *)dm->data)->coordFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexSetCoordinateMap - Sets the function used to map coordinates of newly generated mesh points

  Logically Collective

  Input Parameters:
+ dm        - the `DMPLEX`
- coordFunc - the mapping function

  Level: developer

  Note:
  This function maps from the generated coordinate for the new point to the actual coordinate. Thus it is only practical for manifolds with a nice analytical definition that you can get to from any starting point, like a sphere,

.seealso: `DMPLEX`, `DMSetCoordinates()`, `DMPlexGetCoordinateMap()`, `PetscPointFn`
@*/
PetscErrorCode DMPlexSetCoordinateMap(DM dm, PetscPointFn *coordFunc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ((DM_Plex *)dm->data)->coordFunc = coordFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildGradientReconstruction_Internal(DM dm, PetscFV fvm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  DMLabel      ghostLabel;
  PetscScalar *dx, *grad, **gref;
  PetscInt     dim, cStart, cEnd, c, cEndInterior, maxNumFaces;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, NULL));
  cEndInterior = cEndInterior < 0 ? cEnd : cEndInterior;
  PetscCall(DMPlexGetMaxSizes(dm, &maxNumFaces, NULL));
  PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
  for (c = cStart; c < cEndInterior; c++) {
    const PetscInt  *faces;
    PetscInt         numFaces, usedFaces, f, d;
    PetscFVCellGeom *cg;
    PetscBool        boundary;
    PetscInt         ghost;

    // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
    PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
    if (ghost >= 0) continue;

    PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
    PetscCall(DMPlexGetConeSize(dm, c, &numFaces));
    PetscCall(DMPlexGetCone(dm, c, &faces));
    PetscCheck(numFaces >= dim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      PetscFVCellGeom *cg1;
      PetscFVFaceGeom *fg;
      const PetscInt  *fcells;
      PetscInt         ncell, side;

      PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
      PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
      if ((ghost >= 0) || boundary) continue;
      PetscCall(DMPlexGetSupport(dm, faces[f], &fcells));
      side  = (c != fcells[0]); /* c is on left=0 or right=1 of face */
      ncell = fcells[!side];    /* the neighbor */
      PetscCall(DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg));
      PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
      for (d = 0; d < dim; ++d) dx[usedFaces * dim + d] = cg1->centroid[d] - cg->centroid[d];
      gref[usedFaces++] = fg->grad[side]; /* Gradient reconstruction term will go here */
    }
    PetscCheck(usedFaces, PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
    PetscCall(PetscFVComputeGradient(fvm, usedFaces, dx, grad));
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
      PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
      if ((ghost >= 0) || boundary) continue;
      for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces * dim + d];
      ++usedFaces;
    }
  }
  PetscCall(PetscFree3(dx, grad, gref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildGradientReconstruction_Internal_Tree(DM dm, PetscFV fvm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  DMLabel      ghostLabel;
  PetscScalar *dx, *grad, **gref;
  PetscInt     dim, cStart, cEnd, c, cEndInterior, fStart, fEnd, f, nStart, nEnd, maxNumFaces = 0;
  PetscSection neighSec;
  PetscInt (*neighbors)[2];
  PetscInt *counter;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, NULL));
  if (cEndInterior < 0) cEndInterior = cEnd;
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &neighSec));
  PetscCall(PetscSectionSetChart(neighSec, cStart, cEndInterior));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  for (f = fStart; f < fEnd; f++) {
    const PetscInt *fcells;
    PetscBool       boundary;
    PetscInt        ghost = -1;
    PetscInt        numChildren, numCells, c;

    if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
    PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
    PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
    if ((ghost >= 0) || boundary || numChildren) continue;
    PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
    if (numCells == 2) {
      PetscCall(DMPlexGetSupport(dm, f, &fcells));
      for (c = 0; c < 2; c++) {
        PetscInt cell = fcells[c];

        if (cell >= cStart && cell < cEndInterior) PetscCall(PetscSectionAddDof(neighSec, cell, 1));
      }
    }
  }
  PetscCall(PetscSectionSetUp(neighSec));
  PetscCall(PetscSectionGetMaxDof(neighSec, &maxNumFaces));
  PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
  nStart = 0;
  PetscCall(PetscSectionGetStorageSize(neighSec, &nEnd));
  PetscCall(PetscMalloc1(nEnd - nStart, &neighbors));
  PetscCall(PetscCalloc1(cEndInterior - cStart, &counter));
  for (f = fStart; f < fEnd; f++) {
    const PetscInt *fcells;
    PetscBool       boundary;
    PetscInt        ghost = -1;
    PetscInt        numChildren, numCells, c;

    if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
    PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
    PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
    if ((ghost >= 0) || boundary || numChildren) continue;
    PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
    if (numCells == 2) {
      PetscCall(DMPlexGetSupport(dm, f, &fcells));
      for (c = 0; c < 2; c++) {
        PetscInt cell = fcells[c], off;

        if (cell >= cStart && cell < cEndInterior) {
          PetscCall(PetscSectionGetOffset(neighSec, cell, &off));
          off += counter[cell - cStart]++;
          neighbors[off][0] = f;
          neighbors[off][1] = fcells[1 - c];
        }
      }
    }
  }
  PetscCall(PetscFree(counter));
  PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
  for (c = cStart; c < cEndInterior; c++) {
    PetscInt         numFaces, f, d, off, ghost = -1;
    PetscFVCellGeom *cg;

    PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
    PetscCall(PetscSectionGetDof(neighSec, c, &numFaces));
    PetscCall(PetscSectionGetOffset(neighSec, c, &off));

    // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
    if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
    if (ghost >= 0) continue;

    PetscCheck(numFaces >= dim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0; f < numFaces; ++f) {
      PetscFVCellGeom *cg1;
      PetscFVFaceGeom *fg;
      const PetscInt  *fcells;
      PetscInt         ncell, side, nface;

      nface = neighbors[off + f][0];
      ncell = neighbors[off + f][1];
      PetscCall(DMPlexGetSupport(dm, nface, &fcells));
      side = (c != fcells[0]);
      PetscCall(DMPlexPointLocalRef(dmFace, nface, fgeom, &fg));
      PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
      for (d = 0; d < dim; ++d) dx[f * dim + d] = cg1->centroid[d] - cg->centroid[d];
      gref[f] = fg->grad[side]; /* Gradient reconstruction term will go here */
    }
    PetscCall(PetscFVComputeGradient(fvm, numFaces, dx, grad));
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) gref[f][d] = grad[f * dim + d];
    }
  }
  PetscCall(PetscFree3(dx, grad, gref));
  PetscCall(PetscSectionDestroy(&neighSec));
  PetscCall(PetscFree(neighbors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexComputeGradientFVM - Compute geometric factors for gradient reconstruction, which are stored in the geometry data, and compute layout for gradient data

  Collective

  Input Parameters:
+ dm           - The `DMPLEX`
. fvm          - The `PetscFV`
- cellGeometry - The face geometry from `DMPlexComputeCellGeometryFVM()`

  Input/Output Parameter:
. faceGeometry - The face geometry from `DMPlexComputeFaceGeometryFVM()`; on output
                 the geometric factors for gradient calculation are inserted

  Output Parameter:
. dmGrad - The `DM` describing the layout of gradient data

  Level: developer

.seealso: `DMPLEX`, `DMPlexGetFaceGeometryFVM()`, `DMPlexGetCellGeometryFVM()`
@*/
PetscErrorCode DMPlexComputeGradientFVM(DM dm, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM *dmGrad)
{
  DM           dmFace, dmCell;
  PetscScalar *fgeom, *cgeom;
  PetscSection sectionGrad, parentSection;
  PetscInt     dim, pdim, cStart, cEnd, cEndInterior, c;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFVGetNumComponents(fvm, &pdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, NULL));
  /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
  PetscCall(VecGetDM(faceGeometry, &dmFace));
  PetscCall(VecGetDM(cellGeometry, &dmCell));
  PetscCall(VecGetArray(faceGeometry, &fgeom));
  PetscCall(VecGetArray(cellGeometry, &cgeom));
  PetscCall(DMPlexGetTree(dm, &parentSection, NULL, NULL, NULL, NULL));
  if (!parentSection) {
    PetscCall(BuildGradientReconstruction_Internal(dm, fvm, dmFace, fgeom, dmCell, cgeom));
  } else {
    PetscCall(BuildGradientReconstruction_Internal_Tree(dm, fvm, dmFace, fgeom, dmCell, cgeom));
  }
  PetscCall(VecRestoreArray(faceGeometry, &fgeom));
  PetscCall(VecRestoreArray(cellGeometry, &cgeom));
  /* Create storage for gradients */
  PetscCall(DMClone(dm, dmGrad));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionGrad));
  PetscCall(PetscSectionSetChart(sectionGrad, cStart, cEnd));
  for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionGrad, c, pdim * dim));
  PetscCall(PetscSectionSetUp(sectionGrad));
  PetscCall(DMSetLocalSection(*dmGrad, sectionGrad));
  PetscCall(PetscSectionDestroy(&sectionGrad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetDataFVM - Retrieve precomputed cell geometry

  Collective

  Input Parameters:
+ dm - The `DM`
- fv - The `PetscFV`

  Output Parameters:
+ cellgeom - The cell geometry
. facegeom - The face geometry
- gradDM   - The gradient matrices

  Level: developer

.seealso: `DMPLEX`, `DMPlexComputeGeometryFVM()`
@*/
PetscErrorCode DMPlexGetDataFVM(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM)
{
  PetscObject cellgeomobj, facegeomobj;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)dm, "DMPlex_cellgeom_fvm", &cellgeomobj));
  if (!cellgeomobj) {
    Vec cellgeomInt, facegeomInt;

    PetscCall(DMPlexComputeGeometryFVM(dm, &cellgeomInt, &facegeomInt));
    PetscCall(PetscObjectCompose((PetscObject)dm, "DMPlex_cellgeom_fvm", (PetscObject)cellgeomInt));
    PetscCall(PetscObjectCompose((PetscObject)dm, "DMPlex_facegeom_fvm", (PetscObject)facegeomInt));
    PetscCall(VecDestroy(&cellgeomInt));
    PetscCall(VecDestroy(&facegeomInt));
    PetscCall(PetscObjectQuery((PetscObject)dm, "DMPlex_cellgeom_fvm", &cellgeomobj));
  }
  PetscCall(PetscObjectQuery((PetscObject)dm, "DMPlex_facegeom_fvm", &facegeomobj));
  if (cellgeom) *cellgeom = (Vec)cellgeomobj;
  if (facegeom) *facegeom = (Vec)facegeomobj;
  if (gradDM) {
    PetscObject gradobj;
    PetscBool   computeGradients;

    PetscCall(PetscFVGetComputeGradients(fv, &computeGradients));
    if (!computeGradients) {
      *gradDM = NULL;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscObjectQuery((PetscObject)dm, "DMPlex_dmgrad_fvm", &gradobj));
    if (!gradobj) {
      DM dmGradInt;

      PetscCall(DMPlexComputeGradientFVM(dm, fv, (Vec)facegeomobj, (Vec)cellgeomobj, &dmGradInt));
      PetscCall(PetscObjectCompose((PetscObject)dm, "DMPlex_dmgrad_fvm", (PetscObject)dmGradInt));
      PetscCall(DMDestroy(&dmGradInt));
      PetscCall(PetscObjectQuery((PetscObject)dm, "DMPlex_dmgrad_fvm", &gradobj));
    }
    *gradDM = (DM)gradobj;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCoordinatesToReference_NewtonUpdate(PetscInt dimC, PetscInt dimR, PetscScalar *J, PetscScalar *invJ, PetscScalar *work, PetscReal *resNeg, PetscReal *guess)
{
  PetscInt l, m;

  PetscFunctionBeginHot;
  if (dimC == dimR && dimR <= 3) {
    /* invert Jacobian, multiply */
    PetscScalar det, idet;

    switch (dimR) {
    case 1:
      invJ[0] = 1. / J[0];
      break;
    case 2:
      det     = J[0] * J[3] - J[1] * J[2];
      idet    = 1. / det;
      invJ[0] = J[3] * idet;
      invJ[1] = -J[1] * idet;
      invJ[2] = -J[2] * idet;
      invJ[3] = J[0] * idet;
      break;
    case 3: {
      invJ[0] = J[4] * J[8] - J[5] * J[7];
      invJ[1] = J[2] * J[7] - J[1] * J[8];
      invJ[2] = J[1] * J[5] - J[2] * J[4];
      det     = invJ[0] * J[0] + invJ[1] * J[3] + invJ[2] * J[6];
      idet    = 1. / det;
      invJ[0] *= idet;
      invJ[1] *= idet;
      invJ[2] *= idet;
      invJ[3] = idet * (J[5] * J[6] - J[3] * J[8]);
      invJ[4] = idet * (J[0] * J[8] - J[2] * J[6]);
      invJ[5] = idet * (J[2] * J[3] - J[0] * J[5]);
      invJ[6] = idet * (J[3] * J[7] - J[4] * J[6]);
      invJ[7] = idet * (J[1] * J[6] - J[0] * J[7]);
      invJ[8] = idet * (J[0] * J[4] - J[1] * J[3]);
    } break;
    }
    for (l = 0; l < dimR; l++) {
      for (m = 0; m < dimC; m++) guess[l] += PetscRealPart(invJ[l * dimC + m]) * resNeg[m];
    }
  } else {
#if defined(PETSC_USE_COMPLEX)
    char transpose = 'C';
#else
    char transpose = 'T';
#endif
    PetscBLASInt m, n, one = 1, worksize, info;

    PetscCall(PetscBLASIntCast(dimR, &m));
    PetscCall(PetscBLASIntCast(dimC, &n));
    PetscCall(PetscBLASIntCast(dimC * dimC, &worksize));
    for (l = 0; l < dimC; l++) invJ[l] = resNeg[l];

    PetscCallBLAS("LAPACKgels", LAPACKgels_(&transpose, &m, &n, &one, J, &m, invJ, &n, work, &worksize, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to GELS %" PetscBLASInt_FMT, info);

    for (l = 0; l < dimR; l++) guess[l] += PetscRealPart(invJ[l]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCoordinatesToReference_Tensor(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[], Vec coords, PetscInt dimC, PetscInt dimR)
{
  PetscInt     coordSize, i, j, k, l, m, maxIts = 7, numV = (1 << dimR);
  PetscScalar *coordsScalar = NULL;
  PetscReal   *cellData, *cellCoords, *cellCoeffs, *extJ, *resNeg;
  PetscScalar *J, *invJ, *work;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscCheck(coordSize >= dimC * numV, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expecting at least %" PetscInt_FMT " coordinates, got %" PetscInt_FMT, dimC * (1 << dimR), coordSize);
  PetscCall(DMGetWorkArray(dm, 2 * coordSize + dimR + dimC, MPIU_REAL, &cellData));
  PetscCall(DMGetWorkArray(dm, 3 * dimR * dimC, MPIU_SCALAR, &J));
  cellCoords = &cellData[0];
  cellCoeffs = &cellData[coordSize];
  extJ       = &cellData[2 * coordSize];
  resNeg     = &cellData[2 * coordSize + dimR];
  invJ       = &J[dimR * dimC];
  work       = &J[2 * dimR * dimC];
  if (dimR == 2) {
    const PetscInt zToPlex[4] = {0, 1, 3, 2};

    for (i = 0; i < 4; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
    }
  } else if (dimR == 3) {
    const PetscInt zToPlex[8] = {0, 3, 1, 2, 4, 5, 7, 6};

    for (i = 0; i < 8; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
    }
  } else {
    for (i = 0; i < coordSize; i++) cellCoords[i] = PetscRealPart(coordsScalar[i]);
  }
  /* Perform the shuffling transform that converts values at the corners of [-1,1]^d to coefficients */
  for (i = 0; i < dimR; i++) {
    PetscReal *swap;

    for (j = 0; j < (numV / 2); j++) {
      for (k = 0; k < dimC; k++) {
        cellCoeffs[dimC * j + k]                = 0.5 * (cellCoords[dimC * (2 * j + 1) + k] + cellCoords[dimC * 2 * j + k]);
        cellCoeffs[dimC * (j + (numV / 2)) + k] = 0.5 * (cellCoords[dimC * (2 * j + 1) + k] - cellCoords[dimC * 2 * j + k]);
      }
    }

    if (i < dimR - 1) {
      swap       = cellCoeffs;
      cellCoeffs = cellCoords;
      cellCoords = swap;
    }
  }
  PetscCall(PetscArrayzero(refCoords, numPoints * dimR));
  for (j = 0; j < numPoints; j++) {
    for (i = 0; i < maxIts; i++) {
      PetscReal *guess = &refCoords[dimR * j];

      /* compute -residual and Jacobian */
      for (k = 0; k < dimC; k++) resNeg[k] = realCoords[dimC * j + k];
      for (k = 0; k < dimC * dimR; k++) J[k] = 0.;
      for (k = 0; k < numV; k++) {
        PetscReal extCoord = 1.;
        for (l = 0; l < dimR; l++) {
          PetscReal coord = guess[l];
          PetscInt  dep   = (k & (1 << l)) >> l;

          extCoord *= dep * coord + !dep;
          extJ[l] = dep;

          for (m = 0; m < dimR; m++) {
            PetscReal coord = guess[m];
            PetscInt  dep   = ((k & (1 << m)) >> m) && (m != l);
            PetscReal mult  = dep * coord + !dep;

            extJ[l] *= mult;
          }
        }
        for (l = 0; l < dimC; l++) {
          PetscReal coeff = cellCoeffs[dimC * k + l];

          resNeg[l] -= coeff * extCoord;
          for (m = 0; m < dimR; m++) J[dimR * l + m] += coeff * extJ[m];
        }
      }
      if (0 && PetscDefined(USE_DEBUG)) {
        PetscReal maxAbs = 0.;

        for (l = 0; l < dimC; l++) maxAbs = PetscMax(maxAbs, PetscAbsReal(resNeg[l]));
        PetscCall(PetscInfo(dm, "cell %" PetscInt_FMT ", point %" PetscInt_FMT ", iter %" PetscInt_FMT ": res %g\n", cell, j, i, (double)maxAbs));
      }

      PetscCall(DMPlexCoordinatesToReference_NewtonUpdate(dimC, dimR, J, invJ, work, resNeg, guess));
    }
  }
  PetscCall(DMRestoreWorkArray(dm, 3 * dimR * dimC, MPIU_SCALAR, &J));
  PetscCall(DMRestoreWorkArray(dm, 2 * coordSize + dimR + dimC, MPIU_REAL, &cellData));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexReferenceToCoordinates_Tensor(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[], Vec coords, PetscInt dimC, PetscInt dimR)
{
  PetscInt     coordSize, i, j, k, l, numV = (1 << dimR);
  PetscScalar *coordsScalar = NULL;
  PetscReal   *cellData, *cellCoords, *cellCoeffs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscCheck(coordSize >= dimC * numV, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expecting at least %" PetscInt_FMT " coordinates, got %" PetscInt_FMT, dimC * (1 << dimR), coordSize);
  PetscCall(DMGetWorkArray(dm, 2 * coordSize, MPIU_REAL, &cellData));
  cellCoords = &cellData[0];
  cellCoeffs = &cellData[coordSize];
  if (dimR == 2) {
    const PetscInt zToPlex[4] = {0, 1, 3, 2};

    for (i = 0; i < 4; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
    }
  } else if (dimR == 3) {
    const PetscInt zToPlex[8] = {0, 3, 1, 2, 4, 5, 7, 6};

    for (i = 0; i < 8; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
    }
  } else {
    for (i = 0; i < coordSize; i++) cellCoords[i] = PetscRealPart(coordsScalar[i]);
  }
  /* Perform the shuffling transform that converts values at the corners of [-1,1]^d to coefficients */
  for (i = 0; i < dimR; i++) {
    PetscReal *swap;

    for (j = 0; j < (numV / 2); j++) {
      for (k = 0; k < dimC; k++) {
        cellCoeffs[dimC * j + k]                = 0.5 * (cellCoords[dimC * (2 * j + 1) + k] + cellCoords[dimC * 2 * j + k]);
        cellCoeffs[dimC * (j + (numV / 2)) + k] = 0.5 * (cellCoords[dimC * (2 * j + 1) + k] - cellCoords[dimC * 2 * j + k]);
      }
    }

    if (i < dimR - 1) {
      swap       = cellCoeffs;
      cellCoeffs = cellCoords;
      cellCoords = swap;
    }
  }
  PetscCall(PetscArrayzero(realCoords, numPoints * dimC));
  for (j = 0; j < numPoints; j++) {
    const PetscReal *guess  = &refCoords[dimR * j];
    PetscReal       *mapped = &realCoords[dimC * j];

    for (k = 0; k < numV; k++) {
      PetscReal extCoord = 1.;
      for (l = 0; l < dimR; l++) {
        PetscReal coord = guess[l];
        PetscInt  dep   = (k & (1 << l)) >> l;

        extCoord *= dep * coord + !dep;
      }
      for (l = 0; l < dimC; l++) {
        PetscReal coeff = cellCoeffs[dimC * k + l];

        mapped[l] += coeff * extCoord;
      }
    }
  }
  PetscCall(DMRestoreWorkArray(dm, 2 * coordSize, MPIU_REAL, &cellData));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCoordinatesToReference_FE(DM dm, PetscFE fe, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[], Vec coords, PetscInt Nc, PetscInt dimR, PetscInt maxIter, PetscReal *tol)
{
  PetscInt     numComp, pdim, i, j, k, l, m, coordSize;
  PetscScalar *nodes = NULL;
  PetscReal   *invV, *modes;
  PetscReal   *B, *D, *resNeg;
  PetscScalar *J, *invJ, *work;
  PetscReal    tolerance = tol == NULL ? 0.0 : *tol;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDimension(fe, &pdim));
  PetscCall(PetscFEGetNumComponents(fe, &numComp));
  PetscCheck(numComp == Nc, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "coordinate discretization must have as many components (%" PetscInt_FMT ") as embedding dimension (!= %" PetscInt_FMT ")", numComp, Nc);
  /* we shouldn't apply inverse closure permutation, if one exists */
  PetscCall(DMPlexVecGetOrientedClosure(dm, NULL, PETSC_FALSE, coords, cell, 0, &coordSize, &nodes));
  /* convert nodes to values in the stable evaluation basis */
  PetscCall(DMGetWorkArray(dm, pdim, MPIU_REAL, &modes));
  invV = fe->invV;
  for (i = 0; i < pdim; ++i) {
    modes[i] = 0.;
    for (j = 0; j < pdim; ++j) modes[i] += invV[i * pdim + j] * PetscRealPart(nodes[j]);
  }
  PetscCall(DMGetWorkArray(dm, pdim * Nc + pdim * Nc * dimR + Nc, MPIU_REAL, &B));
  D      = &B[pdim * Nc];
  resNeg = &D[pdim * Nc * dimR];
  PetscCall(DMGetWorkArray(dm, 3 * Nc * dimR, MPIU_SCALAR, &J));
  invJ = &J[Nc * dimR];
  work = &invJ[Nc * dimR];
  for (i = 0; i < numPoints * dimR; i++) refCoords[i] = 0.;
  for (j = 0; j < numPoints; j++) {
    PetscReal normPoint = DMPlex_NormD_Internal(Nc, &realCoords[j * Nc]);
    normPoint           = normPoint > PETSC_SMALL ? normPoint : 1.0;
    for (i = 0; i < maxIter; i++) { /* we could batch this so that we're not making big B and D arrays all the time */
      PetscReal *guess = &refCoords[j * dimR], error = 0;
      PetscCall(PetscSpaceEvaluate(fe->basisSpace, 1, guess, B, D, NULL));
      for (k = 0; k < Nc; k++) resNeg[k] = realCoords[j * Nc + k];
      for (k = 0; k < Nc * dimR; k++) J[k] = 0.;
      for (k = 0; k < pdim; k++) {
        for (l = 0; l < Nc; l++) {
          resNeg[l] -= modes[k] * B[k * Nc + l];
          for (m = 0; m < dimR; m++) J[l * dimR + m] += modes[k] * D[(k * Nc + l) * dimR + m];
        }
      }
      if (0 && PetscDefined(USE_DEBUG)) {
        PetscReal maxAbs = 0.;

        for (l = 0; l < Nc; l++) maxAbs = PetscMax(maxAbs, PetscAbsReal(resNeg[l]));
        PetscCall(PetscInfo(dm, "cell %" PetscInt_FMT ", point %" PetscInt_FMT ", iter %" PetscInt_FMT ": res %g\n", cell, j, i, (double)maxAbs));
      }
      error = DMPlex_NormD_Internal(Nc, resNeg);
      if (error < tolerance * normPoint) {
        if (tol) *tol = error / normPoint;
        break;
      }
      PetscCall(DMPlexCoordinatesToReference_NewtonUpdate(Nc, dimR, J, invJ, work, resNeg, guess));
    }
  }
  PetscCall(DMRestoreWorkArray(dm, 3 * Nc * dimR, MPIU_SCALAR, &J));
  PetscCall(DMRestoreWorkArray(dm, pdim * Nc + pdim * Nc * dimR + Nc, MPIU_REAL, &B));
  PetscCall(DMRestoreWorkArray(dm, pdim, MPIU_REAL, &modes));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &nodes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: TOBY please fix this for Nc > 1 */
PetscErrorCode DMPlexReferenceToCoordinates_FE(DM dm, PetscFE fe, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[], Vec coords, PetscInt Nc, PetscInt dimR)
{
  PetscInt     numComp, pdim, i, j, k, l, coordSize;
  PetscScalar *nodes = NULL;
  PetscReal   *invV, *modes;
  PetscReal   *B;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDimension(fe, &pdim));
  PetscCall(PetscFEGetNumComponents(fe, &numComp));
  PetscCheck(numComp == Nc, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "coordinate discretization must have as many components (%" PetscInt_FMT ") as embedding dimension (!= %" PetscInt_FMT ")", numComp, Nc);
  /* we shouldn't apply inverse closure permutation, if one exists */
  PetscCall(DMPlexVecGetOrientedClosure(dm, NULL, PETSC_FALSE, coords, cell, 0, &coordSize, &nodes));
  /* convert nodes to values in the stable evaluation basis */
  PetscCall(DMGetWorkArray(dm, pdim, MPIU_REAL, &modes));
  invV = fe->invV;
  for (i = 0; i < pdim; ++i) {
    modes[i] = 0.;
    for (j = 0; j < pdim; ++j) modes[i] += invV[i * pdim + j] * PetscRealPart(nodes[j]);
  }
  PetscCall(DMGetWorkArray(dm, numPoints * pdim * Nc, MPIU_REAL, &B));
  PetscCall(PetscSpaceEvaluate(fe->basisSpace, numPoints, refCoords, B, NULL, NULL));
  for (i = 0; i < numPoints * Nc; i++) realCoords[i] = 0.;
  for (j = 0; j < numPoints; j++) {
    PetscReal *mapped = &realCoords[j * Nc];

    for (k = 0; k < pdim; k++) {
      for (l = 0; l < Nc; l++) mapped[l] += modes[k] * B[(j * pdim + k) * Nc + l];
    }
  }
  PetscCall(DMRestoreWorkArray(dm, numPoints * pdim * Nc, MPIU_REAL, &B));
  PetscCall(DMRestoreWorkArray(dm, pdim, MPIU_REAL, &modes));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &nodes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCoordinatesToReference - Pull coordinates back from the mesh to the reference element
  using a single element map.

  Not Collective

  Input Parameters:
+ dm         - The mesh, with coordinate maps defined either by a `PetscDS` for the coordinate `DM` (see `DMGetCoordinateDM()`) or
               implicitly by the coordinates of the corner vertices of the cell: as an affine map for simplicial elements, or
               as a multilinear map for tensor-product elements
. cell       - the cell whose map is used.
. numPoints  - the number of points to locate
- realCoords - (numPoints x coordinate dimension) array of coordinates (see `DMGetCoordinateDim()`)

  Output Parameter:
. refCoords - (`numPoints` x `dimension`) array of reference coordinates (see `DMGetDimension()`)

  Level: intermediate

  Notes:
  This inversion will be accurate inside the reference element, but may be inaccurate for
  mappings that do not extend uniquely outside the reference cell (e.g, most non-affine maps)

.seealso: `DMPLEX`, `DMPlexReferenceToCoordinates()`
@*/
PetscErrorCode DMPlexCoordinatesToReference(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[])
{
  PetscInt       dimC, dimR, depth, i, cellHeight, height;
  DMPolytopeType ct;
  DM             coordDM = NULL;
  Vec            coords;
  PetscFE        fe = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDimension(dm, &dimR));
  PetscCall(DMGetCoordinateDim(dm, &dimC));
  if (dimR <= 0 || dimC <= 0 || numPoints <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  if (coordDM) {
    PetscInt coordFields;

    PetscCall(DMGetNumFields(coordDM, &coordFields));
    if (coordFields) {
      PetscClassId id;
      PetscObject  disc;

      PetscCall(DMGetField(coordDM, 0, NULL, &disc));
      PetscCall(PetscObjectGetClassId(disc, &id));
      if (id == PETSCFE_CLASSID) fe = (PetscFE)disc;
    }
  }
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  PetscCall(DMPlexGetPointHeight(dm, cell, &height));
  PetscCheck(height == cellHeight, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %" PetscInt_FMT " not in a cell, height = %" PetscInt_FMT, cell, height);
  PetscCheck(!DMPolytopeTypeIsHybrid(ct) && ct != DM_POLYTOPE_FV_GHOST, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %" PetscInt_FMT " is unsupported cell type %s", cell, DMPolytopeTypes[ct]);
  if (!fe) { /* implicit discretization: affine or multilinear */
    PetscInt  coneSize;
    PetscBool isSimplex, isTensor;

    PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
    isSimplex = (coneSize == (dimR + 1)) ? PETSC_TRUE : PETSC_FALSE;
    isTensor  = (coneSize == ((depth == 1) ? (1 << dimR) : (2 * dimR))) ? PETSC_TRUE : PETSC_FALSE;
    if (isSimplex) {
      PetscReal detJ, *v0, *J, *invJ;

      PetscCall(DMGetWorkArray(dm, dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
      J    = &v0[dimC];
      invJ = &J[dimC * dimC];
      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, cell, v0, J, invJ, &detJ));
      for (i = 0; i < numPoints; i++) { /* Apply the inverse affine transformation for each point */
        const PetscReal x0[3] = {-1., -1., -1.};

        CoordinatesRealToRef(dimC, dimR, x0, v0, invJ, &realCoords[dimC * i], &refCoords[dimR * i]);
      }
      PetscCall(DMRestoreWorkArray(dm, dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
    } else if (isTensor) {
      PetscCall(DMPlexCoordinatesToReference_Tensor(coordDM, cell, numPoints, realCoords, refCoords, coords, dimC, dimR));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unrecognized cone size %" PetscInt_FMT, coneSize);
  } else {
    PetscCall(DMPlexCoordinatesToReference_FE(coordDM, fe, cell, numPoints, realCoords, refCoords, coords, dimC, dimR, 7, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexReferenceToCoordinates - Map references coordinates to coordinates in the mesh for a single element map.

  Not Collective

  Input Parameters:
+ dm        - The mesh, with coordinate maps defined either by a PetscDS for the coordinate `DM` (see `DMGetCoordinateDM()`) or
               implicitly by the coordinates of the corner vertices of the cell: as an affine map for simplicial elements, or
               as a multilinear map for tensor-product elements
. cell      - the cell whose map is used.
. numPoints - the number of points to locate
- refCoords - (numPoints x dimension) array of reference coordinates (see `DMGetDimension()`)

  Output Parameter:
. realCoords - (numPoints x coordinate dimension) array of coordinates (see `DMGetCoordinateDim()`)

  Level: intermediate

.seealso: `DMPLEX`, `DMPlexCoordinatesToReference()`
@*/
PetscErrorCode DMPlexReferenceToCoordinates(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[])
{
  PetscInt       dimC, dimR, depth, i, cellHeight, height;
  DMPolytopeType ct;
  DM             coordDM = NULL;
  Vec            coords;
  PetscFE        fe = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDimension(dm, &dimR));
  PetscCall(DMGetCoordinateDim(dm, &dimC));
  if (dimR <= 0 || dimC <= 0 || numPoints <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  if (coordDM) {
    PetscInt coordFields;

    PetscCall(DMGetNumFields(coordDM, &coordFields));
    if (coordFields) {
      PetscClassId id;
      PetscObject  disc;

      PetscCall(DMGetField(coordDM, 0, NULL, &disc));
      PetscCall(PetscObjectGetClassId(disc, &id));
      if (id == PETSCFE_CLASSID) fe = (PetscFE)disc;
    }
  }
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  PetscCall(DMPlexGetPointHeight(dm, cell, &height));
  PetscCheck(height == cellHeight, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %" PetscInt_FMT " not in a cell, height = %" PetscInt_FMT, cell, height);
  PetscCheck(!DMPolytopeTypeIsHybrid(ct) && ct != DM_POLYTOPE_FV_GHOST, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %" PetscInt_FMT " is unsupported cell type %s", cell, DMPolytopeTypes[ct]);
  if (!fe) { /* implicit discretization: affine or multilinear */
    PetscInt  coneSize;
    PetscBool isSimplex, isTensor;

    PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
    isSimplex = (coneSize == (dimR + 1)) ? PETSC_TRUE : PETSC_FALSE;
    isTensor  = (coneSize == ((depth == 1) ? (1 << dimR) : (2 * dimR))) ? PETSC_TRUE : PETSC_FALSE;
    if (isSimplex) {
      PetscReal detJ, *v0, *J;

      PetscCall(DMGetWorkArray(dm, dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
      J = &v0[dimC];
      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, cell, v0, J, NULL, &detJ));
      for (i = 0; i < numPoints; i++) { /* Apply the affine transformation for each point */
        const PetscReal xi0[3] = {-1., -1., -1.};

        CoordinatesRefToReal(dimC, dimR, xi0, v0, J, &refCoords[dimR * i], &realCoords[dimC * i]);
      }
      PetscCall(DMRestoreWorkArray(dm, dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
    } else if (isTensor) {
      PetscCall(DMPlexReferenceToCoordinates_Tensor(coordDM, cell, numPoints, refCoords, realCoords, coords, dimC, dimR));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unrecognized cone size %" PetscInt_FMT, coneSize);
  } else {
    PetscCall(DMPlexReferenceToCoordinates_FE(coordDM, fe, cell, numPoints, refCoords, realCoords, coords, dimC, dimR));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

void coordMap_identity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  PetscInt       c;

  for (c = 0; c < Nc; ++c) f0[c] = u[c];
}

/* Shear applies the transformation, assuming we fix z,
  / 1  0  m_0 \
  | 0  1  m_1 |
  \ 0  0   1  /
*/
void coordMap_shear(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar coords[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  const PetscInt ax = (PetscInt)PetscRealPart(constants[0]);
  PetscInt       c;

  for (c = 0; c < Nc; ++c) coords[c] = u[c] + constants[c + 1] * u[ax];
}

/* Flare applies the transformation, assuming we fix x_f,

   x_i = x_i * alpha_i x_f
*/
void coordMap_flare(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar coords[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  const PetscInt cf = (PetscInt)PetscRealPart(constants[0]);
  PetscInt       c;

  for (c = 0; c < Nc; ++c) coords[c] = u[c] * (c == cf ? 1.0 : constants[c + 1] * u[cf]);
}

/*
  We would like to map the unit square to a quarter of the annulus between circles of radius 1 and 2. We start by mapping the straight sections, which
  will correspond to the top and bottom of our square. So

    (0,0)--(1,0)  ==>  (1,0)--(2,0)      Just a shift of (1,0)
    (0,1)--(1,1)  ==>  (0,1)--(0,2)      Switch x and y

  So it looks like we want to map each layer in y to a ray, so x is the radius and y is the angle:

    (x, y)  ==>  (x+1, \pi/2 y)                           in (r', \theta') space
            ==>  ((x+1) cos(\pi/2 y), (x+1) sin(\pi/2 y)) in (x', y') space
*/
void coordMap_annulus(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal ri = PetscRealPart(constants[0]);
  const PetscReal ro = PetscRealPart(constants[1]);

  xp[0] = (x[0] * (ro - ri) + ri) * PetscCosReal(0.5 * PETSC_PI * x[1]);
  xp[1] = (x[0] * (ro - ri) + ri) * PetscSinReal(0.5 * PETSC_PI * x[1]);
}

/*
  We would like to map the unit cube to a hemisphere of the spherical shell between balls of radius 1 and 2. We want to map the bottom surface onto the
  lower hemisphere and the upper surface onto the top, letting z be the radius.

    (x, y)  ==>  ((z+3)/2, \pi/2 (|x| or |y|), arctan y/x)                                                  in (r', \theta', \phi') space
            ==>  ((z+3)/2 \cos(\theta') cos(\phi'), (z+3)/2 \cos(\theta') sin(\phi'), (z+3)/2 sin(\theta')) in (x', y', z') space
*/
void coordMap_shell(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal pi4    = PETSC_PI / 4.0;
  const PetscReal ri     = PetscRealPart(constants[0]);
  const PetscReal ro     = PetscRealPart(constants[1]);
  const PetscReal rp     = (x[2] + 1) * 0.5 * (ro - ri) + ri;
  const PetscReal phip   = PetscAtan2Real(x[1], x[0]);
  const PetscReal thetap = 0.5 * PETSC_PI * (1.0 - ((((phip <= pi4) && (phip >= -pi4)) || ((phip >= 3.0 * pi4) || (phip <= -3.0 * pi4))) ? PetscAbsReal(x[0]) : PetscAbsReal(x[1])));

  xp[0] = rp * PetscCosReal(thetap) * PetscCosReal(phip);
  xp[1] = rp * PetscCosReal(thetap) * PetscSinReal(phip);
  xp[2] = rp * PetscSinReal(thetap);
}

void coordMap_sinusoid(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal c = PetscRealPart(constants[0]);
  const PetscReal m = PetscRealPart(constants[1]);
  const PetscReal n = PetscRealPart(constants[2]);

  xp[0] = x[0];
  xp[1] = x[1];
  if (dim > 2) xp[2] = c * PetscCosReal(2. * m * PETSC_PI * x[0]) * PetscCosReal(2. * n * PETSC_PI * x[1]);
}

/*@C
  DMPlexRemapGeometry - This function maps the original `DM` coordinates to new coordinates.

  Not Collective

  Input Parameters:
+ dm   - The `DM`
. time - The time
- func - The function transforming current coordinates to new coordinates

  Calling sequence of `func`:
+ dim          - The spatial dimension
. Nf           - The number of input fields (here 1)
. NfAux        - The number of input auxiliary fields
. uOff         - The offset of the coordinates in u[] (here 0)
. uOff_x       - The offset of the coordinates in u_x[] (here 0)
. u            - The coordinate values at this point in space
. u_t          - The coordinate time derivative at this point in space (here `NULL`)
. u_x          - The coordinate derivatives at this point in space
. aOff         - The offset of each auxiliary field in u[]
. aOff_x       - The offset of each auxiliary field in u_x[]
. a            - The auxiliary field values at this point in space
. a_t          - The auxiliary field time derivative at this point in space (or `NULL`)
. a_x          - The auxiliary field derivatives at this point in space
. t            - The current time
. x            - The coordinates of this point (here not used)
. numConstants - The number of constants
. constants    - The value of each constant
- f            - The new coordinates at this point in space

  Level: intermediate

.seealso: `DMPLEX`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCoordinateDM()`, `DMProjectFieldLocal()`, `DMProjectFieldLabelLocal()`
@*/
PetscErrorCode DMPlexRemapGeometry(DM dm, PetscReal time, void (*func)(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]))
{
  DM           cdm;
  PetscDS      cds;
  DMField      cf;
  PetscObject  obj;
  PetscClassId id;
  Vec          lCoords, tmpCoords;

  PetscFunctionBegin;
  if (!func) PetscCall(DMPlexGetCoordinateMap(dm, &func));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &lCoords));
  PetscCall(DMGetDS(cdm, &cds));
  PetscCall(PetscDSGetDiscretization(cds, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  if (id != PETSCFE_CLASSID) {
    PetscSection       cSection;
    const PetscScalar *constants;
    PetscScalar       *coords, f[16];
    PetscInt           dim, cdim, Nc, vStart, vEnd;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCheck(cdim <= 16, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Affine version of DMPlexRemapGeometry is currently limited to dimensions <= 16, not %" PetscInt_FMT, cdim);
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(DMGetCoordinateSection(dm, &cSection));
    PetscCall(PetscDSGetConstants(cds, &Nc, &constants));
    PetscCall(VecGetArrayWrite(lCoords, &coords));
    for (PetscInt v = vStart; v < vEnd; ++v) {
      PetscInt uOff[2] = {0, cdim};
      PetscInt off, c;

      PetscCall(PetscSectionGetOffset(cSection, v, &off));
      (*func)(dim, 1, 0, uOff, NULL, &coords[off], NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.0, NULL, Nc, constants, f);
      for (c = 0; c < cdim; ++c) coords[off + c] = f[c];
    }
    PetscCall(VecRestoreArrayWrite(lCoords, &coords));
  } else {
    PetscCall(DMGetLocalVector(cdm, &tmpCoords));
    PetscCall(VecCopy(lCoords, tmpCoords));
    /* We have to do the coordinate field manually right now since the coordinate DM will not have its own */
    PetscCall(DMGetCoordinateField(dm, &cf));
    cdm->coordinates[0].field = cf;
    PetscCall(DMProjectFieldLocal(cdm, time, tmpCoords, &func, INSERT_VALUES, lCoords));
    cdm->coordinates[0].field = NULL;
    PetscCall(DMRestoreLocalVector(cdm, &tmpCoords));
    PetscCall(DMSetCoordinatesLocal(dm, lCoords));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexShearGeometry - This shears the domain, meaning adds a multiple of the shear coordinate to all other coordinates.

  Not Collective

  Input Parameters:
+ dm          - The `DMPLEX`
. direction   - The shear coordinate direction, e.g. `DM_X` is the x-axis
- multipliers - The multiplier m for each direction which is not the shear direction

  Level: intermediate

.seealso: `DMPLEX`, `DMPlexRemapGeometry()`, `DMDirection`, `DM_X`, `DM_Y`, `DM_Z`
@*/
PetscErrorCode DMPlexShearGeometry(DM dm, DMDirection direction, PetscReal multipliers[])
{
  DM             cdm;
  PetscDS        cds;
  PetscScalar   *moduli;
  const PetscInt dir = (PetscInt)direction;
  PetscInt       dE, d, e;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(PetscMalloc1(dE + 1, &moduli));
  moduli[0] = dir;
  for (d = 0, e = 0; d < dE; ++d) moduli[d + 1] = d == dir ? 0.0 : (multipliers ? multipliers[e++] : 1.0);
  PetscCall(DMGetDS(cdm, &cds));
  PetscCall(PetscDSSetConstants(cds, dE + 1, moduli));
  PetscCall(DMPlexRemapGeometry(dm, 0.0, coordMap_shear));
  PetscCall(PetscFree(moduli));
  PetscFunctionReturn(PETSC_SUCCESS);
}

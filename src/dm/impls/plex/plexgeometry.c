#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/petscfeimpl.h>  /*I      "petscfe.h"       I*/
#include <petscblaslapack.h>
#include <petsctime.h>

/*@
  DMPlexFindVertices - Try to find DAG points based on their coordinates.

  Not Collective (provided DMGetCoordinatesLocalSetUp() has been called already)

  Input Parameters:
+ dm - The DMPlex object
. coordinates - The Vec of coordinates of the sought points
- eps - The tolerance or PETSC_DEFAULT

  Output Parameters:
. points - The IS of found DAG points or -1

  Level: intermediate

  Notes:
  The length of Vec coordinates must be npoints * dim where dim is the spatial dimension returned by DMGetCoordinateDim() and npoints is the number of sought points.

  The output IS is living on PETSC_COMM_SELF and its length is npoints.
  Each rank does the search independently.
  If this rank's local DMPlex portion contains the DAG point corresponding to the i-th tuple of coordinates, the i-th entry of the output IS is set to that DAG point, otherwise to -1.

  The output IS must be destroyed by user.

  The tolerance is interpreted as the maximum Euclidean (L2) distance of the sought point from the specified coordinates.

  Complexity of this function is currently O(mn) with m number of vertices to find and n number of vertices in the local mesh. This could probably be improved if needed.

.seealso: `DMPlexCreate()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMPlexFindVertices(DM dm, Vec coordinates, PetscReal eps, IS *points)
{
  PetscInt          c, cdim, i, j, o, p, vStart, vEnd;
  PetscInt          npoints;
  const PetscScalar *coord;
  Vec               allCoordsVec;
  const PetscScalar *allCoords;
  PetscInt          *dagPoints;

  PetscFunctionBegin;
  if (eps < 0) eps = PETSC_SQRT_MACHINE_EPSILON;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  {
    PetscInt n;

    PetscCall(VecGetLocalSize(coordinates, &n));
    PetscCheck(n % cdim == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Given coordinates Vec has local length %" PetscInt_FMT " not divisible by coordinate dimension %" PetscInt_FMT " of given DM", n, cdim);
    npoints = n / cdim;
  }
  PetscCall(DMGetCoordinatesLocal(dm, &allCoordsVec));
  PetscCall(VecGetArrayRead(allCoordsVec, &allCoords));
  PetscCall(VecGetArrayRead(coordinates, &coord));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  if (PetscDefined(USE_DEBUG)) {
    /* check coordinate section is consistent with DM dimension */
    PetscSection      cs;
    PetscInt          ndof;

    PetscCall(DMGetCoordinateSection(dm, &cs));
    for (p = vStart; p < vEnd; p++) {
      PetscCall(PetscSectionGetDof(cs, p, &ndof));
      PetscCheck(ndof == cdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "point %" PetscInt_FMT ": ndof = %" PetscInt_FMT " != %" PetscInt_FMT " = cdim", p, ndof, cdim);
    }
  }
  PetscCall(PetscMalloc1(npoints, &dagPoints));
  if (eps == 0.0) {
    for (i=0,j=0; i < npoints; i++,j+=cdim) {
      dagPoints[i] = -1;
      for (p = vStart,o=0; p < vEnd; p++,o+=cdim) {
        for (c = 0; c < cdim; c++) {
          if (coord[j+c] != allCoords[o+c]) break;
        }
        if (c == cdim) {
          dagPoints[i] = p;
          break;
        }
      }
    }
  } else {
    for (i=0,j=0; i < npoints; i++,j+=cdim) {
      PetscReal         norm;

      dagPoints[i] = -1;
      for (p = vStart,o=0; p < vEnd; p++,o+=cdim) {
        norm = 0.0;
        for (c = 0; c < cdim; c++) {
          norm += PetscRealPart(PetscSqr(coord[j+c] - allCoords[o+c]));
        }
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
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetLineIntersection_2D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], PetscReal intersection[], PetscBool *hasIntersection)
{
  const PetscReal p0_x  = segmentA[0*2+0];
  const PetscReal p0_y  = segmentA[0*2+1];
  const PetscReal p1_x  = segmentA[1*2+0];
  const PetscReal p1_y  = segmentA[1*2+1];
  const PetscReal p2_x  = segmentB[0*2+0];
  const PetscReal p2_y  = segmentB[0*2+1];
  const PetscReal p3_x  = segmentB[1*2+0];
  const PetscReal p3_y  = segmentB[1*2+1];
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
    const PetscReal t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom;

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
      *hasIntersection = PETSC_TRUE;
      if (intersection) {
        intersection[0] = p0_x + (t * s1_x);
        intersection[1] = p0_y + (t * s1_y);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* The plane is segmentB x segmentC: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection */
static PetscErrorCode DMPlexGetLinePlaneIntersection_3D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], const PetscReal segmentC[], PetscReal intersection[], PetscBool *hasIntersection)
{
  const PetscReal p0_x  = segmentA[0*3+0];
  const PetscReal p0_y  = segmentA[0*3+1];
  const PetscReal p0_z  = segmentA[0*3+2];
  const PetscReal p1_x  = segmentA[1*3+0];
  const PetscReal p1_y  = segmentA[1*3+1];
  const PetscReal p1_z  = segmentA[1*3+2];
  const PetscReal q0_x  = segmentB[0*3+0];
  const PetscReal q0_y  = segmentB[0*3+1];
  const PetscReal q0_z  = segmentB[0*3+2];
  const PetscReal q1_x  = segmentB[1*3+0];
  const PetscReal q1_y  = segmentB[1*3+1];
  const PetscReal q1_z  = segmentB[1*3+2];
  const PetscReal r0_x  = segmentC[0*3+0];
  const PetscReal r0_y  = segmentC[0*3+1];
  const PetscReal r0_z  = segmentC[0*3+2];
  const PetscReal r1_x  = segmentC[1*3+0];
  const PetscReal r1_y  = segmentC[1*3+1];
  const PetscReal r1_z  = segmentC[1*3+2];
  const PetscReal s0_x  = p1_x - p0_x;
  const PetscReal s0_y  = p1_y - p0_y;
  const PetscReal s0_z  = p1_z - p0_z;
  const PetscReal s1_x  = q1_x - q0_x;
  const PetscReal s1_y  = q1_y - q0_y;
  const PetscReal s1_z  = q1_z - q0_z;
  const PetscReal s2_x  = r1_x - r0_x;
  const PetscReal s2_y  = r1_y - r0_y;
  const PetscReal s2_z  = r1_z - r0_z;
  const PetscReal s3_x  = s1_y*s2_z - s1_z*s2_y; /* s1 x s2 */
  const PetscReal s3_y  = s1_z*s2_x - s1_x*s2_z;
  const PetscReal s3_z  = s1_x*s2_y - s1_y*s2_x;
  const PetscReal s4_x  = s0_y*s2_z - s0_z*s2_y; /* s0 x s2 */
  const PetscReal s4_y  = s0_z*s2_x - s0_x*s2_z;
  const PetscReal s4_z  = s0_x*s2_y - s0_y*s2_x;
  const PetscReal s5_x  = s1_y*s0_z - s1_z*s0_y; /* s1 x s0 */
  const PetscReal s5_y  = s1_z*s0_x - s1_x*s0_z;
  const PetscReal s5_z  = s1_x*s0_y - s1_y*s0_x;
  const PetscReal denom = -(s0_x*s3_x + s0_y*s3_y + s0_z*s3_z); /* -s0 . (s1 x s2) */

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
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLocatePoint_Simplex_1D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscReal eps = PETSC_SQRT_MACHINE_EPSILON;
  const PetscReal x   = PetscRealPart(point[0]);
  PetscReal       v0, J, invJ, detJ;
  PetscReal       xi;

  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, &v0, &J, &invJ, &detJ));
  xi   = invJ*(x - v0);

  if ((xi >= -eps) && (xi <= 2.+eps)) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLocatePoint_Simplex_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscInt  embedDim = 2;
  const PetscReal eps      = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal       x        = PetscRealPart(point[0]);
  PetscReal       y        = PetscRealPart(point[1]);
  PetscReal       v0[2], J[4], invJ[4], detJ;
  PetscReal       xi, eta;

  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  xi  = invJ[0*embedDim+0]*(x - v0[0]) + invJ[0*embedDim+1]*(y - v0[1]);
  eta = invJ[1*embedDim+0]*(x - v0[0]) + invJ[1*embedDim+1]*(y - v0[1]);

  if ((xi >= -eps) && (eta >= -eps) && (xi + eta <= 2.0+eps)) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexClosestPoint_Simplex_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscReal cpoint[])
{
  const PetscInt  embedDim = 2;
  PetscReal       x        = PetscRealPart(point[0]);
  PetscReal       y        = PetscRealPart(point[1]);
  PetscReal       v0[2], J[4], invJ[4], detJ;
  PetscReal       xi, eta, r;

  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  xi  = invJ[0*embedDim+0]*(x - v0[0]) + invJ[0*embedDim+1]*(y - v0[1]);
  eta = invJ[1*embedDim+0]*(x - v0[0]) + invJ[1*embedDim+1]*(y - v0[1]);

  xi  = PetscMax(xi,  0.0);
  eta = PetscMax(eta, 0.0);
  if (xi + eta > 2.0) {
    r    = (xi + eta)/2.0;
    xi  /= r;
    eta /= r;
  }
  cpoint[0] = J[0*embedDim+0]*xi + J[0*embedDim+1]*eta + v0[0];
  cpoint[1] = J[1*embedDim+0]*xi + J[1*embedDim+1]*eta + v0[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLocatePoint_Quad_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  PetscSection       coordSection;
  Vec             coordsLocal;
  PetscScalar    *coords = NULL;
  const PetscInt  faces[8]  = {0, 1, 1, 2, 2, 3, 3, 0};
  PetscReal       x         = PetscRealPart(point[0]);
  PetscReal       y         = PetscRealPart(point[1]);
  PetscInt        crossings = 0, f;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, NULL, &coords));
  for (f = 0; f < 4; ++f) {
    PetscReal x_i   = PetscRealPart(coords[faces[2*f+0]*2+0]);
    PetscReal y_i   = PetscRealPart(coords[faces[2*f+0]*2+1]);
    PetscReal x_j   = PetscRealPart(coords[faces[2*f+1]*2+0]);
    PetscReal y_j   = PetscRealPart(coords[faces[2*f+1]*2+1]);
    PetscReal slope = (y_j - y_i) / (x_j - x_i);
    PetscBool cond1 = (x_i <= x) && (x < x_j) ? PETSC_TRUE : PETSC_FALSE;
    PetscBool cond2 = (x_j <= x) && (x < x_i) ? PETSC_TRUE : PETSC_FALSE;
    PetscBool above = (y < slope * (x - x_i) + y_i) ? PETSC_TRUE : PETSC_FALSE;
    if ((cond1 || cond2)  && above) ++crossings;
  }
  if (crossings % 2) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &coords));
  PetscFunctionReturn(0);
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
  xi   = invJ[0*embedDim+0]*(x - v0[0]) + invJ[0*embedDim+1]*(y - v0[1]) + invJ[0*embedDim+2]*(z - v0[2]);
  eta  = invJ[1*embedDim+0]*(x - v0[0]) + invJ[1*embedDim+1]*(y - v0[1]) + invJ[1*embedDim+2]*(z - v0[2]);
  zeta = invJ[2*embedDim+0]*(x - v0[0]) + invJ[2*embedDim+1]*(y - v0[1]) + invJ[2*embedDim+2]*(z - v0[2]);

  if ((xi >= -eps) && (eta >= -eps) && (zeta >= -eps) && (xi + eta + zeta <= 2.0+eps)) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLocatePoint_General_3D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  PetscSection   coordSection;
  Vec            coordsLocal;
  PetscScalar   *coords = NULL;
  const PetscInt faces[24] = {0, 3, 2, 1,  5, 4, 7, 6,  3, 0, 4, 5,
                              1, 2, 6, 7,  3, 5, 6, 2,  0, 1, 7, 4};
  PetscBool      found = PETSC_TRUE;
  PetscInt       f;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, NULL, &coords));
  for (f = 0; f < 6; ++f) {
    /* Check the point is under plane */
    /*   Get face normal */
    PetscReal v_i[3];
    PetscReal v_j[3];
    PetscReal normal[3];
    PetscReal pp[3];
    PetscReal dot;

    v_i[0]    = PetscRealPart(coords[faces[f*4+3]*3+0]-coords[faces[f*4+0]*3+0]);
    v_i[1]    = PetscRealPart(coords[faces[f*4+3]*3+1]-coords[faces[f*4+0]*3+1]);
    v_i[2]    = PetscRealPart(coords[faces[f*4+3]*3+2]-coords[faces[f*4+0]*3+2]);
    v_j[0]    = PetscRealPart(coords[faces[f*4+1]*3+0]-coords[faces[f*4+0]*3+0]);
    v_j[1]    = PetscRealPart(coords[faces[f*4+1]*3+1]-coords[faces[f*4+0]*3+1]);
    v_j[2]    = PetscRealPart(coords[faces[f*4+1]*3+2]-coords[faces[f*4+0]*3+2]);
    normal[0] = v_i[1]*v_j[2] - v_i[2]*v_j[1];
    normal[1] = v_i[2]*v_j[0] - v_i[0]*v_j[2];
    normal[2] = v_i[0]*v_j[1] - v_i[1]*v_j[0];
    pp[0]     = PetscRealPart(coords[faces[f*4+0]*3+0] - point[0]);
    pp[1]     = PetscRealPart(coords[faces[f*4+0]*3+1] - point[1]);
    pp[2]     = PetscRealPart(coords[faces[f*4+0]*3+2] - point[2]);
    dot       = normal[0]*pp[0] + normal[1]*pp[1] + normal[2]*pp[2];

    /* Check that projected point is in face (2D location problem) */
    if (dot < 0.0) {
      found = PETSC_FALSE;
      break;
    }
  }
  if (found) *cell = c;
  else *cell = DMLOCATEPOINT_POINT_NOT_FOUND;
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscGridHashInitialize_Internal(PetscGridHash box, PetscInt dim, const PetscScalar point[])
{
  PetscInt d;

  PetscFunctionBegin;
  box->dim = dim;
  for (d = 0; d < dim; ++d) box->lower[d] = box->upper[d] = PetscRealPart(point[d]);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGridHashCreate(MPI_Comm comm, PetscInt dim, const PetscScalar point[], PetscGridHash *box)
{
  PetscFunctionBegin;
  PetscCall(PetscMalloc1(1, box));
  PetscCall(PetscGridHashInitialize_Internal(*box, dim, point));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGridHashEnlarge(PetscGridHash box, const PetscScalar point[])
{
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < box->dim; ++d) {
    box->lower[d] = PetscMin(box->lower[d], PetscRealPart(point[d]));
    box->upper[d] = PetscMax(box->upper[d], PetscRealPart(point[d]));
  }
  PetscFunctionReturn(0);
}

/*
  PetscGridHashSetGrid - Divide the grid into boxes

  Not collective

  Input Parameters:
+ box - The grid hash object
. n   - The number of boxes in each dimension, or PETSC_DETERMINE
- h   - The box size in each dimension, only used if n[d] == PETSC_DETERMINE

  Level: developer

.seealso: `PetscGridHashCreate()`
*/
PetscErrorCode PetscGridHashSetGrid(PetscGridHash box, const PetscInt n[], const PetscReal h[])
{
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < box->dim; ++d) {
    box->extent[d] = box->upper[d] - box->lower[d];
    if (n[d] == PETSC_DETERMINE) {
      box->h[d] = h[d];
      box->n[d] = PetscCeilReal(box->extent[d]/h[d]);
    } else {
      box->n[d] = n[d];
      box->h[d] = box->extent[d]/n[d];
    }
  }
  PetscFunctionReturn(0);
}

/*
  PetscGridHashGetEnclosingBox - Find the grid boxes containing each input point

  Not collective

  Input Parameters:
+ box       - The grid hash object
. numPoints - The number of input points
- points    - The input point coordinates

  Output Parameters:
+ dboxes    - An array of numPoints*dim integers expressing the enclosing box as (i_0, i_1, ..., i_dim)
- boxes     - An array of numPoints integers expressing the enclosing box as single number, or NULL

  Level: developer

.seealso: `PetscGridHashCreate()`
*/
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
      PetscInt dbox = PetscFloorReal((PetscRealPart(points[p*dim+d]) - lower[d])/h[d]);

      if (dbox == n[d] && PetscAbsReal(PetscRealPart(points[p*dim+d]) - upper[d]) < 1.0e-9) dbox = n[d]-1;
      if (dbox == -1   && PetscAbsReal(PetscRealPart(points[p*dim+d]) - lower[d]) < 1.0e-9) dbox = 0;
      PetscCheck(dbox >= 0 && dbox < n[d],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input point %" PetscInt_FMT " (%g, %g, %g) is outside of our bounding box",
                                             p, (double) PetscRealPart(points[p*dim+0]), dim > 1 ? (double) PetscRealPart(points[p*dim+1]) : 0.0, dim > 2 ? (double) PetscRealPart(points[p*dim+2]) : 0.0);
      dboxes[p*dim+d] = dbox;
    }
    if (boxes) for (d = dim-2, boxes[p] = dboxes[p*dim+dim-1]; d >= 0; --d) boxes[p] = boxes[p]*n[d] + dboxes[p*dim+d];
  }
  PetscFunctionReturn(0);
}

/*
 PetscGridHashGetEnclosingBoxQuery - Find the grid boxes containing each input point

 Not collective

  Input Parameters:
+ box       - The grid hash object
. numPoints - The number of input points
- points    - The input point coordinates

  Output Parameters:
+ dboxes    - An array of numPoints*dim integers expressing the enclosing box as (i_0, i_1, ..., i_dim)
. boxes     - An array of numPoints integers expressing the enclosing box as single number, or NULL
- found     - Flag indicating if point was located within a box

  Level: developer

.seealso: `PetscGridHashGetEnclosingBox()`
*/
PetscErrorCode PetscGridHashGetEnclosingBoxQuery(PetscGridHash box, PetscInt numPoints, const PetscScalar points[], PetscInt dboxes[], PetscInt boxes[],PetscBool *found)
{
  const PetscReal *lower = box->lower;
  const PetscReal *upper = box->upper;
  const PetscReal *h     = box->h;
  const PetscInt  *n     = box->n;
  const PetscInt   dim   = box->dim;
  PetscInt         d, p;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  for (p = 0; p < numPoints; ++p) {
    for (d = 0; d < dim; ++d) {
      PetscInt dbox = PetscFloorReal((PetscRealPart(points[p*dim+d]) - lower[d])/h[d]);

      if (dbox == n[d] && PetscAbsReal(PetscRealPart(points[p*dim+d]) - upper[d]) < 1.0e-9) dbox = n[d]-1;
      if (dbox < 0 || dbox >= n[d]) {
        PetscFunctionReturn(0);
      }
      dboxes[p*dim+d] = dbox;
    }
    if (boxes) for (d = dim-2, boxes[p] = dboxes[p*dim+dim-1]; d >= 0; --d) boxes[p] = boxes[p]*n[d] + dboxes[p*dim+d];
  }
  *found = PETSC_TRUE;
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLocatePoint_Internal(DM dm, PetscInt dim, const PetscScalar point[], PetscInt cellStart, PetscInt *cell)
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellType(dm, cellStart, &ct));
  switch (ct) {
    case DM_POLYTOPE_SEGMENT:
    PetscCall(DMPlexLocatePoint_Simplex_1D_Internal(dm, point, cellStart, cell));break;
    case DM_POLYTOPE_TRIANGLE:
    PetscCall(DMPlexLocatePoint_Simplex_2D_Internal(dm, point, cellStart, cell));break;
    case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(DMPlexLocatePoint_Quad_2D_Internal(dm, point, cellStart, cell));break;
    case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(DMPlexLocatePoint_Simplex_3D_Internal(dm, point, cellStart, cell));break;
    case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexLocatePoint_General_3D_Internal(dm, point, cellStart, cell));break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for cell %" PetscInt_FMT " with type %s", cellStart, DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexClosestPoint_Internal - Returns the closest point in the cell to the given point
*/
PetscErrorCode DMPlexClosestPoint_Internal(DM dm, PetscInt dim, const PetscScalar point[], PetscInt cell, PetscReal cpoint[])
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
    PetscCall(DMPlexClosestPoint_Simplex_2D_Internal(dm, point, cell, cpoint));break;
#if 0
    case DM_POLYTOPE_QUADRILATERAL:
    PetscCall(DMPlexClosestPoint_General_2D_Internal(dm, point, cell, cpoint));break;
    case DM_POLYTOPE_TETRAHEDRON:
    PetscCall(DMPlexClosestPoint_Simplex_3D_Internal(dm, point, cell, cpoint));break;
    case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexClosestPoint_General_3D_Internal(dm, point, cell, cpoint));break;
#endif
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No closest point location for cell %" PetscInt_FMT " with type %s", cell, DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexComputeGridHash_Internal - Create a grid hash structure covering the Plex

  Collective on dm

  Input Parameter:
. dm - The Plex

  Output Parameter:
. localBox - The grid hash object

  Level: developer

.seealso: `PetscGridHashCreate()`, `PetscGridHashGetEnclosingBox()`
*/
PetscErrorCode DMPlexComputeGridHash_Internal(DM dm, PetscGridHash *localBox)
{
  const PetscInt     debug = 0;
  MPI_Comm           comm;
  PetscGridHash      lbox;
  Vec                coordinates;
  PetscSection       coordSection;
  Vec                coordsLocal;
  const PetscScalar *coords;
  PetscScalar       *edgeCoords;
  PetscInt          *dboxes, *boxes;
  PetscInt           n[3] = {2, 2, 2};
  PetscInt           dim, N, maxConeSize, cStart, cEnd, c, eStart, eEnd, i;
  PetscBool          flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(VecGetLocalSize(coordinates, &N));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  PetscCall(PetscGridHashCreate(comm, dim, coords, &lbox));
  for (i = 0; i < N; i += dim) PetscCall(PetscGridHashEnlarge(lbox, &coords[i]));
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  c    = dim;
  PetscCall(PetscOptionsGetIntArray(NULL, ((PetscObject) dm)->prefix, "-dm_plex_hash_box_faces", n, &c, &flg));
  if (flg) {for (i = c; i < dim; ++i) n[i] = n[c-1];}
  else     {for (i = 0; i < dim; ++i) n[i] = PetscMax(2, PetscFloorReal(PetscPowReal((PetscReal) (cEnd - cStart), 1.0/dim) * 0.8));}
  PetscCall(PetscGridHashSetGrid(lbox, n, NULL));
#if 0
  /* Could define a custom reduction to merge these */
  PetscCall(MPIU_Allreduce(lbox->lower, gbox->lower, 3, MPIU_REAL, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(lbox->upper, gbox->upper, 3, MPIU_REAL, MPI_MAX, comm));
#endif
  /* Is there a reason to snap the local bounding box to a division of the global box? */
  /* Should we compute all overlaps of local boxes? We could do this with a rendevouz scheme partitioning the global box */
  /* Create label */
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  if (dim < 2) eStart = eEnd = -1;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "cells", &lbox->cellsSparse));
  PetscCall(DMLabelCreateIndex(lbox->cellsSparse, cStart, cEnd));
  /* Compute boxes which overlap each cell: https://stackoverflow.com/questions/13790208/triangle-square-intersection-test-in-2d */
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscCalloc3(16 * dim, &dboxes, 16, &boxes, PetscPowInt(maxConeSize, dim) * dim, &edgeCoords));
  for (c = cStart; c < cEnd; ++c) {
    const PetscReal *h       = lbox->h;
    PetscScalar     *ccoords = NULL;
    PetscInt         csize   = 0;
    PetscInt        *closure = NULL;
    PetscInt         Ncl, cl, Ne = 0;
    PetscScalar      point[3];
    PetscInt         dlim[6], d, e, i, j, k;

    /* Get all edges in cell */
    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &Ncl, &closure));
    for (cl = 0; cl < Ncl*2; ++cl) {
      if ((closure[cl] >= eStart) && (closure[cl] < eEnd)) {
        PetscScalar *ecoords = &edgeCoords[Ne*dim*2];
        PetscInt     ecsize  = dim*2;

        PetscCall(DMPlexVecGetClosure(dm, coordSection, coordsLocal, closure[cl], &ecsize, &ecoords));
        PetscCheck(ecsize == dim*2,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Got %" PetscInt_FMT " coords for edge, instead of %" PetscInt_FMT, ecsize, dim*2);
        ++Ne;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &Ncl, &closure));
    /* Find boxes enclosing each vertex */
    PetscCall(DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, &csize, &ccoords));
    PetscCall(PetscGridHashGetEnclosingBox(lbox, csize/dim, ccoords, dboxes, boxes));
    /* Mark cells containing the vertices */
    for (e = 0; e < csize/dim; ++e) {
      if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT " has vertex in box %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n", c, boxes[e], dboxes[e*dim+0], dim > 1 ? dboxes[e*dim+1] : -1, dim > 2 ? dboxes[e*dim+2] : -1));
      PetscCall(DMLabelSetValue(lbox->cellsSparse, c, boxes[e]));
    }
    /* Get grid of boxes containing these */
    for (d = 0;   d < dim; ++d) {dlim[d*2+0] = dlim[d*2+1] = dboxes[d];}
    for (d = dim; d < 3;   ++d) {dlim[d*2+0] = dlim[d*2+1] = 0;}
    for (e = 1; e < dim+1; ++e) {
      for (d = 0; d < dim; ++d) {
        dlim[d*2+0] = PetscMin(dlim[d*2+0], dboxes[e*dim+d]);
        dlim[d*2+1] = PetscMax(dlim[d*2+1], dboxes[e*dim+d]);
      }
    }
    /* Check for intersection of box with cell */
    for (k = dlim[2*2+0], point[2] = lbox->lower[2] + k*h[2]; k <= dlim[2*2+1]; ++k, point[2] += h[2]) {
      for (j = dlim[1*2+0], point[1] = lbox->lower[1] + j*h[1]; j <= dlim[1*2+1]; ++j, point[1] += h[1]) {
        for (i = dlim[0*2+0], point[0] = lbox->lower[0] + i*h[0]; i <= dlim[0*2+1]; ++i, point[0] += h[0]) {
          const PetscInt box = (k*lbox->n[1] + j)*lbox->n[0] + i;
          PetscScalar    cpoint[3];
          PetscInt       cell, edge, ii, jj, kk;

          if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Box %" PetscInt_FMT ": (%.2g, %.2g, %.2g) -- (%.2g, %.2g, %.2g)\n", box, (double)PetscRealPart(point[0]), (double)PetscRealPart(point[1]), (double)PetscRealPart(point[2]), (double)PetscRealPart(point[0] + h[0]), (double)PetscRealPart(point[1] + h[1]), (double)PetscRealPart(point[2] + h[2])));
          /* Check whether cell contains any vertex of this subbox TODO vectorize this */
          for (kk = 0, cpoint[2] = point[2]; kk < (dim > 2 ? 2 : 1); ++kk, cpoint[2] += h[2]) {
            for (jj = 0, cpoint[1] = point[1]; jj < (dim > 1 ? 2 : 1); ++jj, cpoint[1] += h[1]) {
              for (ii = 0, cpoint[0] = point[0]; ii < 2; ++ii, cpoint[0] += h[0]) {

                PetscCall(DMPlexLocatePoint_Internal(dm, dim, cpoint, c, &cell));
                if (cell >= 0) {
                  if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " contains vertex (%.2g, %.2g, %.2g) of box %" PetscInt_FMT "\n", c, (double)PetscRealPart(cpoint[0]), (double)PetscRealPart(cpoint[1]), (double)PetscRealPart(cpoint[2]), box));
                  PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box));
                  jj = kk = 2;
                  break;
                }
              }
            }
          }
          /* Check whether cell edge intersects any face of these subboxes TODO vectorize this */
          for (edge = 0; edge < Ne; ++edge) {
            PetscReal segA[6] = {0.,0.,0.,0.,0.,0.};
            PetscReal segB[6] = {0.,0.,0.,0.,0.,0.};
            PetscReal segC[6] = {0.,0.,0.,0.,0.,0.};

            PetscCheck(dim <= 3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected dim %" PetscInt_FMT " > 3",dim);
            for (d = 0; d < dim*2; ++d) segA[d] = PetscRealPart(edgeCoords[edge*dim*2+d]);
            /* 1D: (x) -- (x+h)               0 -- 1
               2D: (x,   y)   -- (x,   y+h)   (0, 0) -- (0, 1)
                   (x+h, y)   -- (x+h, y+h)   (1, 0) -- (1, 1)
                   (x,   y)   -- (x+h, y)     (0, 0) -- (1, 0)
                   (x,   y+h) -- (x+h, y+h)   (0, 1) -- (1, 1)
               3D: (x,   y,   z)   -- (x,   y+h, z),   (x,   y,   z)   -- (x,   y,   z+h) (0, 0, 0) -- (0, 1, 0), (0, 0, 0) -- (0, 0, 1)
                   (x+h, y,   z)   -- (x+h, y+h, z),   (x+h, y,   z)   -- (x+h, y,   z+h) (1, 0, 0) -- (1, 1, 0), (1, 0, 0) -- (1, 0, 1)
                   (x,   y,   z)   -- (x+h, y,   z),   (x,   y,   z)   -- (x,   y,   z+h) (0, 0, 0) -- (1, 0, 0), (0, 0, 0) -- (0, 0, 1)
                   (x,   y+h, z)   -- (x+h, y+h, z),   (x,   y+h, z)   -- (x,   y+h, z+h) (0, 1, 0) -- (1, 1, 0), (0, 1, 0) -- (0, 1, 1)
                   (x,   y,   z)   -- (x+h, y,   z),   (x,   y,   z)   -- (x,   y+h, z)   (0, 0, 0) -- (1, 0, 0), (0, 0, 0) -- (0, 1, 0)
                   (x,   y,   z+h) -- (x+h, y,   z+h), (x,   y,   z+h) -- (x,   y+h, z+h) (0, 0, 1) -- (1, 0, 1), (0, 0, 1) -- (0, 1, 1)
             */
            /* Loop over faces with normal in direction d */
            for (d = 0; d < dim; ++d) {
              PetscBool intersects = PETSC_FALSE;
              PetscInt  e = (d+1)%dim;
              PetscInt  f = (d+2)%dim;

              /* There are two faces in each dimension */
              for (ii = 0; ii < 2; ++ii) {
                segB[d]     = PetscRealPart(point[d] + ii*h[d]);
                segB[dim+d] = PetscRealPart(point[d] + ii*h[d]);
                segC[d]     = PetscRealPart(point[d] + ii*h[d]);
                segC[dim+d] = PetscRealPart(point[d] + ii*h[d]);
                if (dim > 1) {
                  segB[e]     = PetscRealPart(point[e] + 0*h[e]);
                  segB[dim+e] = PetscRealPart(point[e] + 1*h[e]);
                  segC[e]     = PetscRealPart(point[e] + 0*h[e]);
                  segC[dim+e] = PetscRealPart(point[e] + 0*h[e]);
                }
                if (dim > 2) {
                  segB[f]     = PetscRealPart(point[f] + 0*h[f]);
                  segB[dim+f] = PetscRealPart(point[f] + 0*h[f]);
                  segC[f]     = PetscRealPart(point[f] + 0*h[f]);
                  segC[dim+f] = PetscRealPart(point[f] + 1*h[f]);
                }
                if (dim == 2) {
                  PetscCall(DMPlexGetLineIntersection_2D_Internal(segA, segB, NULL, &intersects));
                } else if (dim == 3) {
                  PetscCall(DMPlexGetLinePlaneIntersection_3D_Internal(segA, segB, segC, NULL, &intersects));
                }
                if (intersects) {
                  if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cell %" PetscInt_FMT " edge %" PetscInt_FMT " (%.2g, %.2g, %.2g)--(%.2g, %.2g, %.2g) intersects box %" PetscInt_FMT ", face (%.2g, %.2g, %.2g)--(%.2g, %.2g, %.2g) (%.2g, %.2g, %.2g)--(%.2g, %.2g, %.2g)\n", c, edge, (double)segA[0], (double)segA[1], (double)segA[2], (double)segA[3], (double)segA[4], (double)segA[5], box, (double)segB[0], (double)segB[1], (double)segB[2], (double)segB[3], (double)segB[4], (double)segB[5], (double)segC[0], (double)segC[1], (double)segC[2], (double)segC[3], (double)segC[4], (double)segC[5]));
                  PetscCall(DMLabelSetValue(lbox->cellsSparse, c, box)); edge = Ne; break;
                }
              }
            }
          }
        }
      }
    }
    PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &ccoords));
  }
  PetscCall(PetscFree3(dboxes, boxes, edgeCoords));
  if (debug) PetscCall(DMLabelView(lbox->cellsSparse, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(DMLabelConvertToSection(lbox->cellsSparse, &lbox->cellSection, &lbox->cells));
  PetscCall(DMLabelDestroy(&lbox->cellsSparse));
  *localBox = lbox;
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocatePoints_Plex(DM dm, Vec v, DMPointLocationType ltype, PetscSF cellSF)
{
  const PetscInt  debug = 0;
  DM_Plex        *mesh = (DM_Plex *) dm->data;
  PetscBool       hash = mesh->useHashLocation, reuse = PETSC_FALSE;
  PetscInt        bs, numPoints, p, numFound, *found = NULL;
  PetscInt        dim, cStart, cEnd, numCells, c, d;
  const PetscInt *boxCells;
  PetscSFNode    *cells;
  PetscScalar    *a;
  PetscMPIInt     result;
  PetscLogDouble  t0,t1;
  PetscReal       gmin[3],gmax[3];
  PetscInt        terminating_query_type[] = { 0, 0, 0 };

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_LocatePoints,0,0,0,0));
  PetscCall(PetscTime(&t0));
  PetscCheck(ltype != DM_POINTLOCATION_NEAREST || hash,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Nearest point location only supported with grid hashing. Use -dm_plex_hash_location to enable it.");
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(VecGetBlockSize(v, &bs));
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)cellSF),PETSC_COMM_SELF,&result));
  PetscCheck(result == MPI_IDENT || result == MPI_CONGRUENT,PetscObjectComm((PetscObject)cellSF),PETSC_ERR_SUP, "Trying parallel point location: only local point location supported");
  PetscCheck(bs == dim,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Block size for point vector %" PetscInt_FMT " must be the mesh coordinate dimension %" PetscInt_FMT, bs, dim);
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(VecGetLocalSize(v, &numPoints));
  PetscCall(VecGetArray(v, &a));
  numPoints /= bs;
  {
    const PetscSFNode *sf_cells;

    PetscCall(PetscSFGetGraph(cellSF,NULL,NULL,NULL,&sf_cells));
    if (sf_cells) {
      PetscCall(PetscInfo(dm,"[DMLocatePoints_Plex] Re-using existing StarForest node list\n"));
      cells = (PetscSFNode*)sf_cells;
      reuse = PETSC_TRUE;
    } else {
      PetscCall(PetscInfo(dm,"[DMLocatePoints_Plex] Creating and initializing new StarForest node list\n"));
      PetscCall(PetscMalloc1(numPoints, &cells));
      /* initialize cells if created */
      for (p=0; p<numPoints; p++) {
        cells[p].rank  = 0;
        cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
      }
    }
  }
  /* define domain bounding box */
  {
    Vec coorglobal;

    PetscCall(DMGetCoordinates(dm,&coorglobal));
    PetscCall(VecStrideMaxAll(coorglobal,NULL,gmax));
    PetscCall(VecStrideMinAll(coorglobal,NULL,gmin));
  }
  if (hash) {
    if (!mesh->lbox) {PetscCall(PetscInfo(dm, "Initializing grid hashing"));PetscCall(DMPlexComputeGridHash_Internal(dm, &mesh->lbox));}
    /* Designate the local box for each point */
    /* Send points to correct process */
    /* Search cells that lie in each subbox */
    /*   Should we bin points before doing search? */
    PetscCall(ISGetIndices(mesh->lbox->cells, &boxCells));
  }
  for (p = 0, numFound = 0; p < numPoints; ++p) {
    const PetscScalar *point = &a[p*bs];
    PetscInt           dbin[3] = {-1,-1,-1}, bin, cell = -1, cellOffset;
    PetscBool          point_outside_domain = PETSC_FALSE;

    /* check bounding box of domain */
    for (d=0; d<dim; d++) {
      if (PetscRealPart(point[d]) < gmin[d]) { point_outside_domain = PETSC_TRUE; break; }
      if (PetscRealPart(point[d]) > gmax[d]) { point_outside_domain = PETSC_TRUE; break; }
    }
    if (point_outside_domain) {
      cells[p].rank = 0;
      cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
      terminating_query_type[0]++;
      continue;
    }

    /* check initial values in cells[].index - abort early if found */
    if (cells[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      c = cells[p].index;
      cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
      PetscCall(DMPlexLocatePoint_Internal(dm, dim, point, c, &cell));
      if (cell >= 0) {
        cells[p].rank = 0;
        cells[p].index = cell;
        numFound++;
      }
    }
    if (cells[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      terminating_query_type[1]++;
      continue;
    }

    if (hash) {
      PetscBool found_box;

      if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Checking point %" PetscInt_FMT " (%.2g, %.2g, %.2g)\n", p, (double)PetscRealPart(point[0]), (double)PetscRealPart(point[1]), (double)PetscRealPart(point[2])));
      /* allow for case that point is outside box - abort early */
      PetscCall(PetscGridHashGetEnclosingBoxQuery(mesh->lbox, 1, point, dbin, &bin,&found_box));
      if (found_box) {
        if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Found point in box %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n", bin, dbin[0], dbin[1], dbin[2]));
        /* TODO Lay an interface over this so we can switch between Section (dense) and Label (sparse) */
        PetscCall(PetscSectionGetDof(mesh->lbox->cellSection, bin, &numCells));
        PetscCall(PetscSectionGetOffset(mesh->lbox->cellSection, bin, &cellOffset));
        for (c = cellOffset; c < cellOffset + numCells; ++c) {
          if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "    Checking for point in cell %" PetscInt_FMT "\n", boxCells[c]));
          PetscCall(DMPlexLocatePoint_Internal(dm, dim, point, boxCells[c], &cell));
          if (cell >= 0) {
            if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "      FOUND in cell %" PetscInt_FMT "\n", cell));
            cells[p].rank = 0;
            cells[p].index = cell;
            numFound++;
            terminating_query_type[2]++;
            break;
          }
        }
      }
    } else {
      for (c = cStart; c < cEnd; ++c) {
        PetscCall(DMPlexLocatePoint_Internal(dm, dim, point, c, &cell));
        if (cell >= 0) {
          cells[p].rank = 0;
          cells[p].index = cell;
          numFound++;
          terminating_query_type[2]++;
          break;
        }
      }
    }
  }
  if (hash) PetscCall(ISRestoreIndices(mesh->lbox->cells, &boxCells));
  if (ltype == DM_POINTLOCATION_NEAREST && hash && numFound < numPoints) {
    for (p = 0; p < numPoints; p++) {
      const PetscScalar *point = &a[p*bs];
      PetscReal          cpoint[3], diff[3], best[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL}, dist, distMax = PETSC_MAX_REAL;
      PetscInt           dbin[3] = {-1,-1,-1}, bin, cellOffset, d, bestc = -1;

      if (cells[p].index < 0) {
        PetscCall(PetscGridHashGetEnclosingBox(mesh->lbox, 1, point, dbin, &bin));
        PetscCall(PetscSectionGetDof(mesh->lbox->cellSection, bin, &numCells));
        PetscCall(PetscSectionGetOffset(mesh->lbox->cellSection, bin, &cellOffset));
        for (c = cellOffset; c < cellOffset + numCells; ++c) {
          PetscCall(DMPlexClosestPoint_Internal(dm, dim, point, boxCells[c], cpoint));
          for (d = 0; d < dim; ++d) diff[d] = cpoint[d] - PetscRealPart(point[d]);
          dist = DMPlex_NormD_Internal(dim, diff);
          if (dist < distMax) {
            for (d = 0; d < dim; ++d) best[d] = cpoint[d];
            bestc = boxCells[c];
            distMax = dist;
          }
        }
        if (distMax < PETSC_MAX_REAL) {
          ++numFound;
          cells[p].rank  = 0;
          cells[p].index = bestc;
          for (d = 0; d < dim; ++d) a[p*bs+d] = best[d];
        }
      }
    }
  }
  /* This code is only be relevant when interfaced to parallel point location */
  /* Check for highest numbered proc that claims a point (do we care?) */
  if (ltype == DM_POINTLOCATION_REMOVE && numFound < numPoints) {
    PetscCall(PetscMalloc1(numFound,&found));
    for (p = 0, numFound = 0; p < numPoints; p++) {
      if (cells[p].rank >= 0 && cells[p].index >= 0) {
        if (numFound < p) {
          cells[numFound] = cells[p];
        }
        found[numFound++] = p;
      }
    }
  }
  PetscCall(VecRestoreArray(v, &a));
  if (!reuse) {
    PetscCall(PetscSFSetGraph(cellSF, cEnd - cStart, numFound, found, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER));
  }
  PetscCall(PetscTime(&t1));
  if (hash) {
    PetscCall(PetscInfo(dm,"[DMLocatePoints_Plex] terminating_query_type : %" PetscInt_FMT " [outside domain] : %" PetscInt_FMT " [inside initial cell] : %" PetscInt_FMT " [hash]\n",terminating_query_type[0],terminating_query_type[1],terminating_query_type[2]));
  } else {
    PetscCall(PetscInfo(dm,"[DMLocatePoints_Plex] terminating_query_type : %" PetscInt_FMT " [outside domain] : %" PetscInt_FMT " [inside initial cell] : %" PetscInt_FMT " [brute-force]\n",terminating_query_type[0],terminating_query_type[1],terminating_query_type[2]));
  }
  PetscCall(PetscInfo(dm,"[DMLocatePoints_Plex] npoints %" PetscInt_FMT " : time(rank0) %1.2e (sec): points/sec %1.4e\n",numPoints,t1-t0,(double)((double)numPoints/(t1-t0))));
  PetscCall(PetscLogEventEnd(DMPLEX_LocatePoints,0,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeProjection2Dto1D - Rewrite coordinates to be the 1D projection of the 2D coordinates

  Not collective

  Input/Output Parameter:
. coords - The coordinates of a segment, on output the new y-coordinate, and 0 for x

  Output Parameter:
. R - The rotation which accomplishes the projection

  Level: developer

.seealso: `DMPlexComputeProjection3Dto1D()`, `DMPlexComputeProjection3Dto2D()`
@*/
PetscErrorCode DMPlexComputeProjection2Dto1D(PetscScalar coords[], PetscReal R[])
{
  const PetscReal x = PetscRealPart(coords[2] - coords[0]);
  const PetscReal y = PetscRealPart(coords[3] - coords[1]);
  const PetscReal r = PetscSqrtReal(x*x + y*y), c = x/r, s = y/r;

  PetscFunctionBegin;
  R[0] = c; R[1] = -s;
  R[2] = s; R[3] =  c;
  coords[0] = 0.0;
  coords[1] = r;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeProjection3Dto1D - Rewrite coordinates to be the 1D projection of the 3D coordinates

  Not collective

  Input/Output Parameter:
. coords - The coordinates of a segment; on output, the new y-coordinate, and 0 for x and z

  Output Parameter:
. R - The rotation which accomplishes the projection

  Note: This uses the basis completion described by Frisvad in http://www.imm.dtu.dk/~jerf/papers/abstracts/onb.html, DOI:10.1080/2165347X.2012.689606

  Level: developer

.seealso: `DMPlexComputeProjection2Dto1D()`, `DMPlexComputeProjection3Dto2D()`
@*/
PetscErrorCode DMPlexComputeProjection3Dto1D(PetscScalar coords[], PetscReal R[])
{
  PetscReal      x    = PetscRealPart(coords[3] - coords[0]);
  PetscReal      y    = PetscRealPart(coords[4] - coords[1]);
  PetscReal      z    = PetscRealPart(coords[5] - coords[2]);
  PetscReal      r    = PetscSqrtReal(x*x + y*y + z*z);
  PetscReal      rinv = 1. / r;
  PetscFunctionBegin;

  x *= rinv; y *= rinv; z *= rinv;
  if (x > 0.) {
    PetscReal inv1pX   = 1./ (1. + x);

    R[0] = x; R[1] = -y;              R[2] = -z;
    R[3] = y; R[4] = 1. - y*y*inv1pX; R[5] =     -y*z*inv1pX;
    R[6] = z; R[7] =     -y*z*inv1pX; R[8] = 1. - z*z*inv1pX;
  }
  else {
    PetscReal inv1mX   = 1./ (1. - x);

    R[0] = x; R[1] = z;               R[2] = y;
    R[3] = y; R[4] =     -y*z*inv1mX; R[5] = 1. - y*y*inv1mX;
    R[6] = z; R[7] = 1. - z*z*inv1mX; R[8] =     -y*z*inv1mX;
  }
  coords[0] = 0.0;
  coords[1] = r;
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeProjection3Dto2D - Rewrite coordinates of 3 or more coplanar 3D points to a common 2D basis for the
    plane.  The normal is defined by positive orientation of the first 3 points.

  Not collective

  Input Parameter:
. coordSize - Length of coordinate array (3x number of points); must be at least 9 (3 points)

  Input/Output Parameter:
. coords - The interlaced coordinates of each coplanar 3D point; on output the first
           2*coordSize/3 entries contain interlaced 2D points, with the rest undefined

  Output Parameter:
. R - 3x3 row-major rotation matrix whose columns are the tangent basis [t1, t2, n].  Multiplying by R^T transforms from original frame to tangent frame.

  Level: developer

.seealso: `DMPlexComputeProjection2Dto1D()`, `DMPlexComputeProjection3Dto1D()`
@*/
PetscErrorCode DMPlexComputeProjection3Dto2D(PetscInt coordSize, PetscScalar coords[], PetscReal R[])
{
  PetscReal x1[3], x2[3], n[3], c[3], norm;
  const PetscInt dim = 3;
  PetscInt       d, p;

  PetscFunctionBegin;
  /* 0) Calculate normal vector */
  for (d = 0; d < dim; ++d) {
    x1[d] = PetscRealPart(coords[1*dim+d] - coords[0*dim+d]);
    x2[d] = PetscRealPart(coords[2*dim+d] - coords[0*dim+d]);
  }
  // n = x1 \otimes x2
  n[0] = x1[1]*x2[2] - x1[2]*x2[1];
  n[1] = x1[2]*x2[0] - x1[0]*x2[2];
  n[2] = x1[0]*x2[1] - x1[1]*x2[0];
  norm = PetscSqrtReal(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  for (d = 0; d < dim; d++) n[d] /= norm;
  norm = PetscSqrtReal(x1[0] * x1[0] + x1[1] * x1[1] + x1[2] * x1[2]);
  for (d = 0; d < dim; d++) x1[d] /= norm;
  // x2 = n \otimes x1
  x2[0] = n[1] * x1[2] - n[2] * x1[1];
  x2[1] = n[2] * x1[0] - n[0] * x1[2];
  x2[2] = n[0] * x1[1] - n[1] * x1[0];
  for (d=0; d<dim; d++) {
    R[d * dim + 0] = x1[d];
    R[d * dim + 1] = x2[d];
    R[d * dim + 2] = n[d];
    c[d] = PetscRealPart(coords[0*dim + d]);
  }
  for (p=0; p<coordSize/dim; p++) {
    PetscReal y[3];
    for (d=0; d<dim; d++) y[d] = PetscRealPart(coords[p*dim + d]) - c[d];
    for (d=0; d<2; d++) coords[p*2+d] = R[0*dim + d] * y[0] + R[1*dim + d] * y[1] + R[2*dim + d] * y[2];
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED
static inline void Volume_Triangle_Internal(PetscReal *vol, PetscReal coords[])
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
  M[0] = x1; M[1] = x2;
  M[2] = y1; M[3] = y2;
  DMPlex_Det2D_Internal(&detM, M);
  *vol = 0.5*detM;
  (void)PetscLogFlops(5.0);
}

PETSC_UNUSED
static inline void Volume_Tetrahedron_Internal(PetscReal *vol, PetscReal coords[])
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
  const PetscReal x1 = coords[3] - coords[0], y1 = coords[4]  - coords[1], z1 = coords[5]  - coords[2];
  const PetscReal x2 = coords[6] - coords[0], y2 = coords[7]  - coords[1], z2 = coords[8]  - coords[2];
  const PetscReal x3 = coords[9] - coords[0], y3 = coords[10] - coords[1], z3 = coords[11] - coords[2];
  const PetscReal onesixth = ((PetscReal)1./(PetscReal)6.);
  PetscReal       M[9], detM;
  M[0] = x1; M[1] = x2; M[2] = x3;
  M[3] = y1; M[4] = y2; M[5] = y3;
  M[6] = z1; M[7] = z2; M[8] = z3;
  DMPlex_Det3D_Internal(&detM, M);
  *vol = -onesixth*detM;
  (void)PetscLogFlops(10.0);
}

static inline void Volume_Tetrahedron_Origin_Internal(PetscReal *vol, PetscReal coords[])
{
  const PetscReal onesixth = ((PetscReal)1./(PetscReal)6.);
  DMPlex_Det3D_Internal(vol, coords);
  *vol *= -onesixth;
}

static PetscErrorCode DMPlexComputePointGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  const PetscScalar *coords;
  PetscInt       dim, d, off;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionGetDof(coordSection,e,&dim));
  if (!dim) PetscFunctionReturn(0);
  PetscCall(PetscSectionGetOffset(coordSection,e,&off));
  PetscCall(VecGetArrayRead(coordinates,&coords));
  if (v0) {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[off + d]);}
  PetscCall(VecRestoreArrayRead(coordinates,&coords));
  *detJ = 1.;
  if (J) {
    for (d = 0; d < dim * dim; d++) J[d] = 0.;
    for (d = 0; d < dim; d++) J[d * dim + d] = 1.;
    if (invJ) {
      for (d = 0; d < dim * dim; d++) invJ[d] = 0.;
      for (d = 0; d < dim; d++) invJ[d * dim + d] = 1.;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeLineGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, d, pStart, pEnd, numSelfCoords = 0;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionGetChart(coordSection,&pStart,&pEnd));
  if (e >= pStart && e < pEnd) PetscCall(PetscSectionGetDof(coordSection,e,&numSelfCoords));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords));
  numCoords = numSelfCoords ? numSelfCoords : numCoords;
  PetscCheck(!invJ || J,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  *detJ = 0.0;
  if (numCoords == 6) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    PetscCall(DMPlexComputeProjection3Dto1D(coords, R));
    if (J)    {
      J0   = 0.5*PetscRealPart(coords[1]);
      J[0] = R[0]*J0; J[1] = R[1]; J[2] = R[2];
      J[3] = R[3]*J0; J[4] = R[4]; J[5] = R[5];
      J[6] = R[6]*J0; J[7] = R[7]; J[8] = R[8];
      DMPlex_Det3D_Internal(detJ, J);
      if (invJ) {DMPlex_Invert2D_Internal(invJ, J, *detJ);}
    }
  } else if (numCoords == 4) {
    const PetscInt dim = 2;
    PetscReal      R[4], J0;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    PetscCall(DMPlexComputeProjection2Dto1D(coords, R));
    if (J)    {
      J0   = 0.5*PetscRealPart(coords[1]);
      J[0] = R[0]*J0; J[1] = R[1];
      J[2] = R[2]*J0; J[3] = R[3];
      DMPlex_Det2D_Internal(detJ, J);
      if (invJ) {DMPlex_Invert2D_Internal(invJ, J, *detJ);}
    }
  } else if (numCoords == 2) {
    const PetscInt dim = 1;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    if (J)    {
      J[0]  = 0.5*(PetscRealPart(coords[1]) - PetscRealPart(coords[0]));
      *detJ = J[0];
      PetscCall(PetscLogFlops(2.0));
      if (invJ) {invJ[0] = 1.0/J[0]; PetscCall(PetscLogFlops(1.0));}
    }
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this segment is %" PetscInt_FMT " != 2", numCoords);
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeTriangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, numSelfCoords = 0, d, f, g, pStart, pEnd;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionGetChart(coordSection,&pStart,&pEnd));
  if (e >= pStart && e < pEnd) PetscCall(PetscSectionGetDof(coordSection,e,&numSelfCoords));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords));
  numCoords = numSelfCoords ? numSelfCoords : numCoords;
  *detJ = 0.0;
  if (numCoords == 9) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    PetscCall(DMPlexComputeProjection3Dto2D(numCoords, coords, R));
    if (J)    {
      const PetscInt pdim = 2;

      for (d = 0; d < pdim; d++) {
        for (f = 0; f < pdim; f++) {
          J0[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*pdim+d]) - PetscRealPart(coords[0*pdim+d]));
        }
      }
      PetscCall(PetscLogFlops(8.0));
      DMPlex_Det3D_Internal(detJ, J0);
      for (d = 0; d < dim; d++) {
        for (f = 0; f < dim; f++) {
          J[d*dim+f] = 0.0;
          for (g = 0; g < dim; g++) {
            J[d*dim+f] += R[d*dim+g]*J0[g*dim+f];
          }
        }
      }
      PetscCall(PetscLogFlops(18.0));
    }
    if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
  } else if (numCoords == 6) {
    const PetscInt dim = 2;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    if (J)    {
      for (d = 0; d < dim; d++) {
        for (f = 0; f < dim; f++) {
          J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
        }
      }
      PetscCall(PetscLogFlops(8.0));
      DMPlex_Det2D_Internal(detJ, J);
    }
    if (invJ) {DMPlex_Invert2D_Internal(invJ, J, *detJ);}
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this triangle is %" PetscInt_FMT " != 6 or 9", numCoords);
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeRectangleGeometry_Internal(DM dm, PetscInt e, PetscBool isTensor, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, numSelfCoords = 0, d, f, g, pStart, pEnd;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionGetChart(coordSection,&pStart,&pEnd));
  if (e >= pStart && e < pEnd) PetscCall(PetscSectionGetDof(coordSection,e,&numSelfCoords));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords));
  numCoords = numSelfCoords ? numSelfCoords : numCoords;
  if (!Nq) {
    PetscInt vorder[4] = {0, 1, 2, 3};

    if (isTensor) {vorder[2] = 3; vorder[3] = 2;}
    *detJ = 0.0;
    if (numCoords == 12) {
      const PetscInt dim = 3;
      PetscReal      R[9], J0[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};

      if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
      PetscCall(DMPlexComputeProjection3Dto2D(numCoords, coords, R));
      if (J)    {
        const PetscInt pdim = 2;

        for (d = 0; d < pdim; d++) {
          J0[d*dim+0] = 0.5*(PetscRealPart(coords[vorder[1]*pdim+d]) - PetscRealPart(coords[vorder[0]*pdim+d]));
          J0[d*dim+1] = 0.5*(PetscRealPart(coords[vorder[2]*pdim+d]) - PetscRealPart(coords[vorder[1]*pdim+d]));
        }
        PetscCall(PetscLogFlops(8.0));
        DMPlex_Det3D_Internal(detJ, J0);
        for (d = 0; d < dim; d++) {
          for (f = 0; f < dim; f++) {
            J[d*dim+f] = 0.0;
            for (g = 0; g < dim; g++) {
              J[d*dim+f] += R[d*dim+g]*J0[g*dim+f];
            }
          }
        }
        PetscCall(PetscLogFlops(18.0));
      }
      if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
    } else if (numCoords == 8) {
      const PetscInt dim = 2;

      if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
      if (J)    {
        for (d = 0; d < dim; d++) {
          J[d*dim+0] = 0.5*(PetscRealPart(coords[vorder[1]*dim+d]) - PetscRealPart(coords[vorder[0]*dim+d]));
          J[d*dim+1] = 0.5*(PetscRealPart(coords[vorder[3]*dim+d]) - PetscRealPart(coords[vorder[0]*dim+d]));
        }
        PetscCall(PetscLogFlops(8.0));
        DMPlex_Det2D_Internal(detJ, J);
      }
      if (invJ) {DMPlex_Invert2D_Internal(invJ, J, *detJ);}
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %" PetscInt_FMT " != 8 or 12", numCoords);
  } else {
    const PetscInt Nv = 4;
    const PetscInt dimR = 2;
    PetscInt  zToPlex[4] = {0, 1, 3, 2};
    PetscReal zOrder[12];
    PetscReal zCoeff[12];
    PetscInt  i, j, k, l, dim;

    if (isTensor) {zToPlex[2] = 2; zToPlex[3] = 3;}
    if (numCoords == 12) {
      dim = 3;
    } else if (numCoords == 8) {
      dim = 2;
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %" PetscInt_FMT " != 8 or 12", numCoords);
    for (i = 0; i < Nv; i++) {
      PetscInt zi = zToPlex[i];

      for (j = 0; j < dim; j++) {
        zOrder[dim * i + j] = PetscRealPart(coords[dim * zi + j]);
      }
    }
    for (j = 0; j < dim; j++) {
      /* Nodal basis for evaluation at the vertices: (1 \mp xi) (1 \mp eta):
           \phi^0 = (1 - xi - eta + xi eta) --> 1      = 1/4 ( \phi^0 + \phi^1 + \phi^2 + \phi^3)
           \phi^1 = (1 + xi - eta - xi eta) --> xi     = 1/4 (-\phi^0 + \phi^1 - \phi^2 + \phi^3)
           \phi^2 = (1 - xi + eta - xi eta) --> eta    = 1/4 (-\phi^0 - \phi^1 + \phi^2 + \phi^3)
           \phi^3 = (1 + xi + eta + xi eta) --> xi eta = 1/4 ( \phi^0 - \phi^1 - \phi^2 + \phi^3)
      */
      zCoeff[dim * 0 + j] = 0.25 * (  zOrder[dim * 0 + j] + zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
      zCoeff[dim * 1 + j] = 0.25 * (- zOrder[dim * 0 + j] + zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
      zCoeff[dim * 2 + j] = 0.25 * (- zOrder[dim * 0 + j] - zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
      zCoeff[dim * 3 + j] = 0.25 * (  zOrder[dim * 0 + j] - zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j]);
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

          for (k = 0; k < Nv; k++) {
            val += extPoint[k] * zCoeff[dim * k + j];
          }
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

            for (l = 0; l < Nv; l++) {
              val += zCoeff[dim * l + j] * extJ[dimR * l + k];
            }
            J[i * dim * dim + dim * j + k] = val;
          }
        }
        if (dim == 3) { /* put the cross product in the third component of the Jacobian */
          PetscReal x, y, z;
          PetscReal *iJ = &J[i * dim * dim];
          PetscReal norm;

          x = iJ[1 * dim + 0] * iJ[2 * dim + 1] - iJ[1 * dim + 1] * iJ[2 * dim + 0];
          y = iJ[0 * dim + 1] * iJ[2 * dim + 0] - iJ[0 * dim + 0] * iJ[2 * dim + 1];
          z = iJ[0 * dim + 0] * iJ[1 * dim + 1] - iJ[0 * dim + 1] * iJ[1 * dim + 0];
          norm = PetscSqrtReal(x * x + y * y + z * z);
          iJ[2] = x / norm;
          iJ[5] = y / norm;
          iJ[8] = z / norm;
          DMPlex_Det3D_Internal(&detJ[i], &J[i * dim * dim]);
          if (invJ) {DMPlex_Invert3D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);}
        } else {
          DMPlex_Det2D_Internal(&detJ[i], &J[i * dim * dim]);
          if (invJ) {DMPlex_Invert2D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);}
        }
      }
    }
  }
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeTetrahedronGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  const PetscInt dim = 3;
  PetscInt       d;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords));
  *detJ = 0.0;
  if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
  if (J)    {
    for (d = 0; d < dim; d++) {
      /* I orient with outward face normals */
      J[d*dim+0] = 0.5*(PetscRealPart(coords[2*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+1] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+2] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
    }
    PetscCall(PetscLogFlops(18.0));
    DMPlex_Det3D_Internal(detJ, J);
  }
  if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeHexahedronGeometry_Internal(DM dm, PetscInt e, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  const PetscInt dim = 3;
  PetscInt       d;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords));
  if (!Nq) {
    *detJ = 0.0;
    if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
    if (J)    {
      for (d = 0; d < dim; d++) {
        J[d*dim+0] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+1] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+2] = 0.5*(PetscRealPart(coords[4*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
      PetscCall(PetscLogFlops(18.0));
      DMPlex_Det3D_Internal(detJ, J);
    }
    if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
  } else {
    const PetscInt Nv = 8;
    const PetscInt zToPlex[8] = {0, 3, 1, 2, 4, 5, 7, 6};
    const PetscInt dim = 3;
    const PetscInt dimR = 3;
    PetscReal zOrder[24];
    PetscReal zCoeff[24];
    PetscInt  i, j, k, l;

    for (i = 0; i < Nv; i++) {
      PetscInt zi = zToPlex[i];

      for (j = 0; j < dim; j++) {
        zOrder[dim * i + j] = PetscRealPart(coords[dim * zi + j]);
      }
    }
    for (j = 0; j < dim; j++) {
      zCoeff[dim * 0 + j] = 0.125 * (  zOrder[dim * 0 + j] + zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j] + zOrder[dim * 4 + j] + zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 1 + j] = 0.125 * (- zOrder[dim * 0 + j] + zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j] - zOrder[dim * 4 + j] + zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 2 + j] = 0.125 * (- zOrder[dim * 0 + j] - zOrder[dim * 1 + j] + zOrder[dim * 2 + j] + zOrder[dim * 3 + j] - zOrder[dim * 4 + j] - zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 3 + j] = 0.125 * (  zOrder[dim * 0 + j] - zOrder[dim * 1 + j] - zOrder[dim * 2 + j] + zOrder[dim * 3 + j] + zOrder[dim * 4 + j] - zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 4 + j] = 0.125 * (- zOrder[dim * 0 + j] - zOrder[dim * 1 + j] - zOrder[dim * 2 + j] - zOrder[dim * 3 + j] + zOrder[dim * 4 + j] + zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 5 + j] = 0.125 * (+ zOrder[dim * 0 + j] - zOrder[dim * 1 + j] + zOrder[dim * 2 + j] - zOrder[dim * 3 + j] - zOrder[dim * 4 + j] + zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 6 + j] = 0.125 * (+ zOrder[dim * 0 + j] + zOrder[dim * 1 + j] - zOrder[dim * 2 + j] - zOrder[dim * 3 + j] - zOrder[dim * 4 + j] - zOrder[dim * 5 + j] + zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
      zCoeff[dim * 7 + j] = 0.125 * (- zOrder[dim * 0 + j] + zOrder[dim * 1 + j] + zOrder[dim * 2 + j] - zOrder[dim * 3 + j] + zOrder[dim * 4 + j] - zOrder[dim * 5 + j] - zOrder[dim * 6 + j] + zOrder[dim * 7 + j]);
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

          for (k = 0; k < Nv; k++) {
            val += extPoint[k] * zCoeff[dim * k + j];
          }
          v[i * dim + j] = val;
        }
      }
      if (J) {
        PetscReal extJ[24];

        extJ[0]  = 0.         ; extJ[1]  = 0.        ; extJ[2]  = 0.      ;
        extJ[3]  = 1.         ; extJ[4]  = 0.        ; extJ[5]  = 0.      ;
        extJ[6]  = 0.         ; extJ[7]  = 1.        ; extJ[8]  = 0.      ;
        extJ[9]  = eta        ; extJ[10] = xi        ; extJ[11] = 0.      ;
        extJ[12] = 0.         ; extJ[13] = 0.        ; extJ[14] = 1.      ;
        extJ[15] = theta      ; extJ[16] = 0.        ; extJ[17] = xi      ;
        extJ[18] = 0.         ; extJ[19] = theta     ; extJ[20] = eta     ;
        extJ[21] = theta * eta; extJ[22] = theta * xi; extJ[23] = eta * xi;

        for (j = 0; j < dim; j++) {
          for (k = 0; k < dimR; k++) {
            PetscReal val = 0.;

            for (l = 0; l < Nv; l++) {
              val += zCoeff[dim * l + j] * extJ[dimR * l + k];
            }
            J[i * dim * dim + dim * j + k] = val;
          }
        }
        DMPlex_Det3D_Internal(&detJ[i], &J[i * dim * dim]);
        if (invJ) {DMPlex_Invert3D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);}
      }
    }
  }
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeTriangularPrismGeometry_Internal(DM dm, PetscInt e, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  const PetscInt dim = 3;
  PetscInt       d;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords));
  if (!Nq) {
    /* Assume that the map to the reference is affine */
    *detJ = 0.0;
    if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
    if (J)    {
      for (d = 0; d < dim; d++) {
        J[d*dim+0] = 0.5*(PetscRealPart(coords[2*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+1] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+2] = 0.5*(PetscRealPart(coords[4*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
      PetscCall(PetscLogFlops(18.0));
      DMPlex_Det3D_Internal(detJ, J);
    }
    if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
  } else {
    const PetscInt dim  = 3;
    const PetscInt dimR = 3;
    const PetscInt Nv   = 6;
    PetscReal verts[18];
    PetscReal coeff[18];
    PetscInt  i, j, k, l;

    for (i = 0; i < Nv; ++i) for (j = 0; j < dim; ++j) verts[dim * i + j] = PetscRealPart(coords[dim * i + j]);
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
      coeff[dim * 0 + j] = (1./4.) * (                      verts[dim * 1 + j] + verts[dim * 2 + j]                      + verts[dim * 4 + j] + verts[dim * 5 + j]);
      coeff[dim * 1 + j] = (1./4.) * (-verts[dim * 0 + j] + verts[dim * 1 + j]                      - verts[dim * 3 + j]                      + verts[dim * 5 + j]);
      coeff[dim * 2 + j] = (1./4.) * (-verts[dim * 0 + j]                      + verts[dim * 2 + j] - verts[dim * 3 + j] + verts[dim * 4 + j]);
      coeff[dim * 3 + j] = (1./4.) * (                    - verts[dim * 1 + j] - verts[dim * 2 + j]                      + verts[dim * 4 + j] + verts[dim * 5 + j]);
      coeff[dim * 4 + j] = (1./4.) * ( verts[dim * 0 + j]                      - verts[dim * 2 + j] - verts[dim * 3 + j] + verts[dim * 4 + j]);
      coeff[dim * 5 + j] = (1./4.) * ( verts[dim * 0 + j] - verts[dim * 1 + j]                      - verts[dim * 3 + j]                      + verts[dim * 5 + j]);
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

          for (k = 0; k < Nv; ++k) {
            val += extPoint[k] * coeff[k*dim + c];
          }
          v[i*dim + c] = val;
        }
      }
      if (J) {
        PetscReal extJ[18];

        extJ[0]  = 0.  ; extJ[1]  = 0.  ; extJ[2]  = 0. ;
        extJ[3]  = 0.  ; extJ[4]  = 1.  ; extJ[5]  = 0. ;
        extJ[6]  = 1.  ; extJ[7]  = 0.  ; extJ[8]  = 0. ;
        extJ[9]  = 0.  ; extJ[10] = 0.  ; extJ[11] = 1. ;
        extJ[12] = zeta; extJ[13] = 0.  ; extJ[14] = xi ;
        extJ[15] = 0.  ; extJ[16] = zeta; extJ[17] = eta;

        for (j = 0; j < dim; j++) {
          for (k = 0; k < dimR; k++) {
            PetscReal val = 0.;

            for (l = 0; l < Nv; l++) {
              val += coeff[dim * l + j] * extJ[dimR * l + k];
            }
            J[i * dim * dim + dim * j + k] = val;
          }
        }
        DMPlex_Det3D_Internal(&detJ[i], &J[i * dim * dim]);
        if (invJ) {DMPlex_Invert3D_Internal(&invJ[i * dim * dim], &J[i * dim * dim], detJ[i]);}
      }
    }
  }
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeCellGeometryFEM_Implicit(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal *v, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  DMPolytopeType  ct;
  PetscInt        depth, dim, coordDim, coneSize, i;
  PetscInt        Nq = 0;
  const PetscReal *points = NULL;
  DMLabel         depthLabel;
  PetscReal       xi0[3] = {-1.,-1.,-1.}, v0[3], J0[9], detJ0;
  PetscBool       isAffine = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetValue(depthLabel, cell, &dim));
  if (depth == 1 && dim == 1) {
    PetscCall(DMGetDimension(dm, &dim));
  }
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  PetscCheck(coordDim <= 3,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported coordinate dimension %" PetscInt_FMT " > 3", coordDim);
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
    else    PetscCall(DMPlexComputeLineGeometry_Internal(dm, cell, v,  J,  invJ,  detJ));
    break;
    case DM_POLYTOPE_TRIANGLE:
    if (Nq) PetscCall(DMPlexComputeTriangleGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0));
    else    PetscCall(DMPlexComputeTriangleGeometry_Internal(dm, cell, v,  J,  invJ,  detJ));
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
    else    PetscCall(DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v,  J,  invJ,  detJ));
    break;
    case DM_POLYTOPE_HEXAHEDRON:
    PetscCall(DMPlexComputeHexahedronGeometry_Internal(dm, cell, Nq, points, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
    case DM_POLYTOPE_TRI_PRISM:
    PetscCall(DMPlexComputeTriangularPrismGeometry_Internal(dm, cell, Nq, points, v, J, invJ, detJ));
    isAffine = PETSC_FALSE;
    break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No element geometry for cell %" PetscInt_FMT " with type %s", cell, DMPolytopeTypes[PetscMax(0, PetscMin(ct, DM_NUM_POLYTOPES))]);
  }
  if (isAffine && Nq) {
    if (v) {
      for (i = 0; i < Nq; i++) {
        CoordinatesRefToReal(coordDim, dim, xi0, v0, J0, &points[dim * i], &v[coordDim * i]);
      }
    }
    if (detJ) {
      for (i = 0; i < Nq; i++) {
        detJ[i] = detJ0;
      }
    }
    if (J) {
      PetscInt k;

      for (i = 0, k = 0; i < Nq; i++) {
        PetscInt j;

        for (j = 0; j < coordDim * coordDim; j++, k++) {
          J[k] = J0[j];
        }
      }
    }
    if (invJ) {
      PetscInt k;
      switch (coordDim) {
      case 0:
        break;
      case 1:
        invJ[0] = 1./J0[0];
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

        for (j = 0; j < coordDim * coordDim; j++, k++) {
          invJ[k] = invJ[j];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeCellGeometryAffineFEM - Assuming an affine map, compute the Jacobian, inverse Jacobian, and Jacobian determinant for a given cell

  Collective on dm

  Input Parameters:
+ dm   - the DM
- cell - the cell

  Output Parameters:
+ v0   - the translation part of this affine transform
. J    - the Jacobian of the transform from the reference element
. invJ - the inverse of the Jacobian
- detJ - the Jacobian determinant

  Level: advanced

  Fortran Notes:
  Since it returns arrays, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: `DMPlexComputeCellGeometryFEM()`, `DMGetCoordinateSection()`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexComputeCellGeometryAffineFEM(DM dm, PetscInt cell, PetscReal *v0, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  PetscFunctionBegin;
  PetscCall(DMPlexComputeCellGeometryFEM_Implicit(dm,cell,NULL,v0,J,invJ,detJ));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeCellGeometryFEM_FE(DM dm, PetscFE fe, PetscInt point, PetscQuadrature quad, PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscQuadrature   feQuad;
  PetscSection      coordSection;
  Vec               coordinates;
  PetscScalar      *coords = NULL;
  const PetscReal  *quadPoints;
  PetscTabulation T;
  PetscInt          dim, cdim, pdim, qdim, Nq, numCoords, q;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, point, &numCoords, &coords));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
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
    PetscCheck(numCoords == pdim*cdim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "There are %" PetscInt_FMT " coordinates for point %" PetscInt_FMT " != %" PetscInt_FMT "*%" PetscInt_FMT, numCoords, point, pdim, cdim);
  } else {
    PetscCall(PetscFECreateTabulation(fe, 1, Nq, quadPoints, J ? 1 : 0, &T));
  }
  PetscCheck(qdim == dim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Point dimension %" PetscInt_FMT " != quadrature dimension %" PetscInt_FMT, dim, qdim);
  {
    const PetscReal *basis    = T->T[0];
    const PetscReal *basisDer = T->T[1];
    PetscReal        detJt;

#if defined(PETSC_USE_DEBUG)
    PetscCheck(Nq == T->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Np %" PetscInt_FMT " != %" PetscInt_FMT, Nq, T->Np);
    PetscCheck(pdim == T->Nb,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nb %" PetscInt_FMT " != %" PetscInt_FMT, pdim, T->Nb);
    PetscCheck(dim == T->Nc,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nc %" PetscInt_FMT " != %" PetscInt_FMT, dim, T->Nc);
    PetscCheck(cdim == T->cdim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "cdim %" PetscInt_FMT " != %" PetscInt_FMT, cdim, T->cdim);
#endif
    if (v) {
      PetscCall(PetscArrayzero(v, Nq*cdim));
      for (q = 0; q < Nq; ++q) {
        PetscInt i, k;

        for (k = 0; k < pdim; ++k) {
          const PetscInt vertex = k/cdim;
          for (i = 0; i < cdim; ++i) {
            v[q*cdim + i] += basis[(q*pdim + k)*cdim + i] * PetscRealPart(coords[vertex*cdim + i]);
          }
        }
        PetscCall(PetscLogFlops(2.0*pdim*cdim));
      }
    }
    if (J) {
      PetscCall(PetscArrayzero(J, Nq*cdim*cdim));
      for (q = 0; q < Nq; ++q) {
        PetscInt i, j, k, c, r;

        /* J = dx_i/d\xi_j = sum[k=0,n-1] dN_k/d\xi_j * x_i(k) */
        for (k = 0; k < pdim; ++k) {
          const PetscInt vertex = k/cdim;
          for (j = 0; j < dim; ++j) {
            for (i = 0; i < cdim; ++i) {
              J[(q*cdim + i)*cdim + j] += basisDer[((q*pdim + k)*cdim + i)*dim + j] * PetscRealPart(coords[vertex*cdim + i]);
            }
          }
        }
        PetscCall(PetscLogFlops(2.0*pdim*dim*cdim));
        if (cdim > dim) {
          for (c = dim; c < cdim; ++c)
            for (r = 0; r < cdim; ++r)
              J[r*cdim+c] = r == c ? 1.0 : 0.0;
        }
        if (!detJ && !invJ) continue;
        detJt = 0.;
        switch (cdim) {
        case 3:
          DMPlex_Det3D_Internal(&detJt, &J[q*cdim*dim]);
          if (invJ) {DMPlex_Invert3D_Internal(&invJ[q*cdim*dim], &J[q*cdim*dim], detJt);}
          break;
        case 2:
          DMPlex_Det2D_Internal(&detJt, &J[q*cdim*dim]);
          if (invJ) {DMPlex_Invert2D_Internal(&invJ[q*cdim*dim], &J[q*cdim*dim], detJt);}
          break;
        case 1:
          detJt = J[q*cdim*dim];
          if (invJ) invJ[q*cdim*dim] = 1.0/detJt;
        }
        if (detJ) detJ[q] = detJt;
      }
    } else PetscCheck(!detJ && !invJ,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Need J to compute invJ or detJ");
  }
  if (feQuad != quad) PetscCall(PetscTabulationDestroy(&T));
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, point, &numCoords, &coords));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeCellGeometryFEM - Compute the Jacobian, inverse Jacobian, and Jacobian determinant at each quadrature point in the given cell

  Collective on dm

  Input Parameters:
+ dm   - the DM
. cell - the cell
- quad - the quadrature containing the points in the reference element where the geometry will be evaluated.  If quad == NULL, geometry will be
         evaluated at the first vertex of the reference element

  Output Parameters:
+ v    - the image of the transformed quadrature points, otherwise the image of the first vertex in the closure of the reference element
. J    - the Jacobian of the transform from the reference element at each quadrature point
. invJ - the inverse of the Jacobian at each quadrature point
- detJ - the Jacobian determinant at each quadrature point

  Level: advanced

  Fortran Notes:
  Since it returns arrays, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: `DMGetCoordinateSection()`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexComputeCellGeometryFEM(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal *v, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  DM             cdm;
  PetscFE        fe = NULL;

  PetscFunctionBegin;
  PetscValidRealPointer(detJ, 7);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  if (cdm) {
    PetscClassId id;
    PetscInt     numFields;
    PetscDS      prob;
    PetscObject  disc;

    PetscCall(DMGetNumFields(cdm, &numFields));
    if (numFields) {
      PetscCall(DMGetDS(cdm, &prob));
      PetscCall(PetscDSGetDiscretization(prob,0,&disc));
      PetscCall(PetscObjectGetClassId(disc,&id));
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
      }
    }
  }
  if (!fe) PetscCall(DMPlexComputeCellGeometryFEM_Implicit(dm, cell, quad, v, J, invJ, detJ));
  else     PetscCall(DMPlexComputeCellGeometryFEM_FE(dm, fe, cell, quad, v, J, invJ, detJ));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeGeometryFVM_0D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection        coordSection;
  Vec                 coordinates;
  const PetscScalar  *coords = NULL;
  PetscInt            d, dof, off;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(VecGetArrayRead(coordinates, &coords));

  /* for a point the centroid is just the coord */
  if (centroid) {
    PetscCall(PetscSectionGetDof(coordSection, cell, &dof));
    PetscCall(PetscSectionGetOffset(coordSection, cell, &off));
    for (d = 0; d < dof; d++){
      centroid[d] = PetscRealPart(coords[off + d]);
    }
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
    for (d = 0; d < dof; d++){
      normal[d] -= PetscRealPart(coords[off + d]);
    }

    /* Determine the sign of the normal based upon its location in the support */
    PetscCall(DMPlexGetCone(dm, support[0], &cones));
    sign = cones[0] == cell ? 1.0 : -1.0;

    norm = DMPlex_NormD_Internal(dim, normal);
    for (d = 0; d < dim; ++d) normal[d] /= (norm*sign);
  }
  if (vol) {
    *vol = 1.0;
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeGeometryFVM_1D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscScalar    tmp[2];
  PetscInt       coordSize, d;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords));
  PetscCall(DMLocalizeCoordinate_Internal(dm, dim, coords, &coords[dim], tmp));
  if (centroid) {
    for (d = 0; d < dim; ++d) centroid[d] = 0.5*PetscRealPart(coords[d] + tmp[d]);
  }
  if (normal) {
    PetscReal norm;

    PetscCheck(dim == 2,PETSC_COMM_SELF, PETSC_ERR_SUP, "We only support 2D edges right now");
    normal[0]  = -PetscRealPart(coords[1] - tmp[1]);
    normal[1]  =  PetscRealPart(coords[0] - tmp[0]);
    norm       = DMPlex_NormD_Internal(dim, normal);
    for (d = 0; d < dim; ++d) normal[d] /= norm;
  }
  if (vol) {
    *vol = 0.0;
    for (d = 0; d < dim; ++d) *vol += PetscSqr(PetscRealPart(coords[d] - tmp[d]));
    *vol = PetscSqrtReal(*vol);
  }
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, cell, &coordSize, &coords));
  PetscFunctionReturn(0);
}

/* Centroid_i = (\sum_n A_n Cn_i) / A */
static PetscErrorCode DMPlexComputeGeometryFVM_2D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  DMPolytopeType ct;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       fv[4] = {0, 1, 2, 3};
  PetscInt       cdim, coordSize, numCorners, p, d;

  PetscFunctionBegin;
  /* Must check for hybrid cells because prisms have a different orientation scheme */
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
    case DM_POLYTOPE_SEG_PRISM_TENSOR: fv[2] = 3; fv[3] = 2;break;
    default: break;
  }
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMPlexGetConeSize(dm, cell, &numCorners));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  {
    PetscReal c[3] = {0., 0., 0.}, n[3] = {0., 0., 0.}, origin[3] = {0., 0., 0.}, norm;

    for (d = 0; d < cdim; d++) origin[d] = PetscRealPart(coords[d]);
    for (p = 0; p < numCorners-2; ++p) {
      PetscReal e0[3] = {0., 0., 0.}, e1[3] = {0., 0., 0.};
      for (d = 0; d < cdim; d++) {
        e0[d] = PetscRealPart(coords[cdim*fv[p+1]+d]) - origin[d];
        e1[d] = PetscRealPart(coords[cdim*fv[p+2]+d]) - origin[d];
      }
      const PetscReal dx = e0[1] * e1[2] - e0[2] * e1[1];
      const PetscReal dy = e0[2] * e1[0] - e0[0] * e1[2];
      const PetscReal dz = e0[0] * e1[1] - e0[1] * e1[0];
      const PetscReal a  = PetscSqrtReal(dx*dx + dy*dy + dz*dz);

      n[0] += dx;
      n[1] += dy;
      n[2] += dz;
      for (d = 0; d < cdim; d++) {
        c[d] += a * PetscRealPart(origin[d] + coords[cdim*fv[p+1]+d] + coords[cdim*fv[p+2]+d]) / 3.;
      }
    }
    norm = PetscSqrtReal(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] /= norm;
    n[1] /= norm;
    n[2] /= norm;
    c[0] /= norm;
    c[1] /= norm;
    c[2] /= norm;
    if (vol) *vol = 0.5*norm;
    if (centroid) for (d = 0; d < cdim; ++d) centroid[d] = c[d];
    if (normal) for (d = 0; d < cdim; ++d) normal[d] = n[d];
  }
  PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, cell, &coordSize, &coords));
  PetscFunctionReturn(0);
}

/* Centroid_i = (\sum_n V_n Cn_i) / V */
static PetscErrorCode DMPlexComputeGeometryFVM_3D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  DMPolytopeType  ct;
  PetscSection    coordSection;
  Vec             coordinates;
  PetscScalar    *coords = NULL;
  PetscReal       vsum = 0.0, vtmp, coordsTmp[3*3], origin[3];
  const PetscInt *faces, *facesO;
  PetscBool       isHybrid = PETSC_FALSE;
  PetscInt        numFaces, f, coordSize, p, d;

  PetscFunctionBegin;
  PetscCheck(dim <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No support for dim %" PetscInt_FMT " > 3",dim);
  /* Must check for hybrid cells because prisms have a different orientation scheme */
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  switch (ct) {
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      isHybrid = PETSC_TRUE;
    default: break;
  }

  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));

  if (centroid) for (d = 0; d < dim; ++d) centroid[d] = 0.0;
  PetscCall(DMPlexGetConeSize(dm, cell, &numFaces));
  PetscCall(DMPlexGetCone(dm, cell, &faces));
  PetscCall(DMPlexGetConeOrientation(dm, cell, &facesO));
  for (f = 0; f < numFaces; ++f) {
    PetscBool      flip = isHybrid && f == 0 ? PETSC_TRUE : PETSC_FALSE; /* The first hybrid face is reversed */
    DMPolytopeType ct;

    PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, faces[f], &coordSize, &coords));
    // If using zero as the origin vertex for each tetrahedron, an element far from the origin will have positive and
    // negative volumes that nearly cancel, thus incurring rounding error. Here we define origin[] as the first vertex
    // so that all tetrahedra have positive volume.
    if (f == 0) for (d = 0; d < dim; d++) origin[d] = PetscRealPart(coords[d]);
    PetscCall(DMPlexGetCellType(dm, faces[f], &ct));
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[0*dim+d]) - origin[d];
        coordsTmp[1*dim+d] = PetscRealPart(coords[1*dim+d]) - origin[d];
        coordsTmp[2*dim+d] = PetscRealPart(coords[2*dim+d]) - origin[d];
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (facesO[f] < 0 || flip) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {           /* Centroid of OABC = (a+b+c)/4 */
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p*dim+d]*vtmp;
        }
      }
      break;
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    {
      PetscInt fv[4] = {0, 1, 2, 3};

      /* Side faces for hybrid cells are are stored as tensor products */
      if (isHybrid && f > 1) {fv[2] = 3; fv[3] = 2;}
      /* DO FOR PYRAMID */
      /* First tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[fv[0]*dim+d]) - origin[d];
        coordsTmp[1*dim+d] = PetscRealPart(coords[fv[1]*dim+d]) - origin[d];
        coordsTmp[2*dim+d] = PetscRealPart(coords[fv[3]*dim+d]) - origin[d];
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (facesO[f] < 0 || flip) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p*dim+d]*vtmp;
        }
      }
      /* Second tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[fv[1]*dim+d]) - origin[d];
        coordsTmp[1*dim+d] = PetscRealPart(coords[fv[2]*dim+d]) - origin[d];
        coordsTmp[2*dim+d] = PetscRealPart(coords[fv[3]*dim+d]) - origin[d];
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (facesO[f] < 0 || flip) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p*dim+d]*vtmp;
        }
      }
      break;
    }
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle face %" PetscInt_FMT " of type %s", faces[f], DMPolytopeTypes[ct]);
    }
    PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, faces[f], &coordSize, &coords));
  }
  if (vol)     *vol = PetscAbsReal(vsum);
  if (normal)   for (d = 0; d < dim; ++d) normal[d]    = 0.0;
  if (centroid) for (d = 0; d < dim; ++d) centroid[d] = centroid[d] / (vsum*4) + origin[d];
;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeCellGeometryFVM - Compute the volume for a given cell

  Collective on dm

  Input Parameters:
+ dm   - the DM
- cell - the cell

  Output Parameters:
+ volume   - the cell volume
. centroid - the cell centroid
- normal - the cell normal, if appropriate

  Level: advanced

  Fortran Notes:
  Since it returns arrays, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: `DMGetCoordinateSection()`, `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexComputeCellGeometryFVM(DM dm, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscInt       depth, dim;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(depth == dim,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated");
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
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeGeometryFEM - Precompute cell geometry for the entire mesh

  Collective on dm

  Input Parameter:
. dm - The DMPlex

  Output Parameter:
. cellgeom - A vector with the cell geometry data for each cell

  Level: beginner

@*/
PetscErrorCode DMPlexComputeGeometryFEM(DM dm, Vec *cellgeom)
{
  DM             dmCell;
  Vec            coordinates;
  PetscSection   coordSection, sectionCell;
  PetscScalar   *cgeom;
  PetscInt       cStart, cEnd, c;

  PetscFunctionBegin;
  PetscCall(DMClone(dm, &dmCell));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMSetCoordinateSection(dmCell, PETSC_DETERMINE, coordSection));
  PetscCall(DMSetCoordinatesLocal(dmCell, coordinates));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionCell));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(PetscSectionSetChart(sectionCell, cStart, cEnd));
  /* TODO This needs to be multiplied by Nq for non-affine */
  for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionCell, c, (PetscInt) PetscCeilReal(((PetscReal) sizeof(PetscFEGeom))/sizeof(PetscScalar))));
  PetscCall(PetscSectionSetUp(sectionCell));
  PetscCall(DMSetLocalSection(dmCell, sectionCell));
  PetscCall(PetscSectionDestroy(&sectionCell));
  PetscCall(DMCreateLocalVector(dmCell, cellgeom));
  PetscCall(VecGetArray(*cellgeom, &cgeom));
  for (c = cStart; c < cEnd; ++c) {
    PetscFEGeom *cg;

    PetscCall(DMPlexPointLocalRef(dmCell, c, cgeom, &cg));
    PetscCall(PetscArrayzero(cg, 1));
    PetscCall(DMPlexComputeCellGeometryFEM(dmCell, c, NULL, cg->v, cg->J, cg->invJ, cg->detJ));
    PetscCheck(*cg->detJ > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double) *cg->detJ, c);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeGeometryFVM - Computes the cell and face geometry for a finite volume method

  Input Parameter:
. dm - The DM

  Output Parameters:
+ cellgeom - A Vec of PetscFVCellGeom data
- facegeom - A Vec of PetscFVFaceGeom data

  Level: developer

.seealso: `PetscFVFaceGeom`, `PetscFVCellGeom`, `DMPlexComputeGeometryFEM()`
@*/
PetscErrorCode DMPlexComputeGeometryFVM(DM dm, Vec *cellgeom, Vec *facegeom)
{
  DM             dmFace, dmCell;
  DMLabel        ghostLabel;
  PetscSection   sectionFace, sectionCell;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *fgeom, *cgeom;
  PetscReal      minradius, gminradius;
  PetscInt       dim, cStart, cEnd, cEndInterior, c, fStart, fEnd, f;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  /* Make cell centroids and volumes */
  PetscCall(DMClone(dm, &dmCell));
  PetscCall(DMSetCoordinateSection(dmCell, PETSC_DETERMINE, coordSection));
  PetscCall(DMSetCoordinatesLocal(dmCell, coordinates));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionCell));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
  PetscCall(PetscSectionSetChart(sectionCell, cStart, cEnd));
  for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionCell, c, (PetscInt) PetscCeilReal(((PetscReal) sizeof(PetscFVCellGeom))/sizeof(PetscScalar))));
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
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionFace));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(PetscSectionSetChart(sectionFace, fStart, fEnd));
  for (f = fStart; f < fEnd; ++f) PetscCall(PetscSectionSetDof(sectionFace, f, (PetscInt) PetscCeilReal(((PetscReal) sizeof(PetscFVFaceGeom))/sizeof(PetscScalar))));
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
    PetscCall(DMPlexGetTreeChildren(dm,f,&numChildren,NULL));
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
      }
      else {
        rcentroid = fg->centroid;
      }
      PetscCall(DMLocalizeCoordinateReal_Internal(dm, dim, fg->centroid, lcentroid, l));
      PetscCall(DMLocalizeCoordinateReal_Internal(dm, dim, fg->centroid, rcentroid, r));
      DMPlex_WaxpyD_Internal(dim, -1, l, r, v);
      if (DMPlex_DotRealD_Internal(dim, fg->normal, v) < 0) {
        for (d = 0; d < dim; ++d) fg->normal[d] = -fg->normal[d];
      }
      if (DMPlex_DotRealD_Internal(dim, fg->normal, v) <= 0) {
        PetscCheck(dim != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %" PetscInt_FMT " could not be fixed, normal (%g,%g) v (%g,%g)", f, (double) fg->normal[0], (double) fg->normal[1], (double) v[0], (double) v[1]);
        PetscCheck(dim != 3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %" PetscInt_FMT " could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", f, (double) fg->normal[0], (double) fg->normal[1], (double) fg->normal[2], (double) v[0], (double) v[1], (double) v[2]);
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %" PetscInt_FMT " could not be fixed", f);
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
  PetscCall(MPIU_Allreduce(&minradius, &gminradius, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm)));
  PetscCall(DMPlexSetMinRadius(dm, gminradius));
  /* Compute centroids of ghost cells */
  for (c = cEndInterior; c < cEnd; ++c) {
    PetscFVFaceGeom *fg;
    const PetscInt  *cone,    *support;
    PetscInt         coneSize, supportSize, s;

    PetscCall(DMPlexGetConeSize(dmCell, c, &coneSize));
    PetscCheck(coneSize == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %" PetscInt_FMT " has cone size %" PetscInt_FMT " != 1", c, coneSize);
    PetscCall(DMPlexGetCone(dmCell, c, &cone));
    PetscCall(DMPlexGetSupportSize(dmCell, cone[0], &supportSize));
    PetscCheck(supportSize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " has support size %" PetscInt_FMT " != 2", cone[0], supportSize);
    PetscCall(DMPlexGetSupport(dmCell, cone[0], &support));
    PetscCall(DMPlexPointLocalRef(dmFace, cone[0], fgeom, &fg));
    for (s = 0; s < 2; ++s) {
      /* Reflect ghost centroid across plane of face */
      if (support[s] == c) {
        PetscFVCellGeom       *ci;
        PetscFVCellGeom       *cg;
        PetscReal              c2f[3], a;

        PetscCall(DMPlexPointLocalRead(dmCell, support[(s+1)%2], cgeom, &ci));
        DMPlex_WaxpyD_Internal(dim, -1, ci->centroid, fg->centroid, c2f); /* cell to face centroid */
        a    = DMPlex_DotRealD_Internal(dim, c2f, fg->normal)/DMPlex_DotRealD_Internal(dim, fg->normal, fg->normal);
        PetscCall(DMPlexPointLocalRef(dmCell, support[s], cgeom, &cg));
        DMPlex_WaxpyD_Internal(dim, 2*a, fg->normal, ci->centroid, cg->centroid);
        cg->volume = ci->volume;
      }
    }
  }
  PetscCall(VecRestoreArray(*facegeom, &fgeom));
  PetscCall(VecRestoreArray(*cellgeom, &cgeom));
  PetscCall(DMDestroy(&dmCell));
  PetscCall(DMDestroy(&dmFace));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetMinRadius - Returns the minimum distance from any cell centroid to a face

  Not collective

  Input Parameter:
. dm - the DM

  Output Parameter:
. minradius - the minimum cell radius

  Level: developer

.seealso: `DMGetCoordinates()`
@*/
PetscErrorCode DMPlexGetMinRadius(DM dm, PetscReal *minradius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidRealPointer(minradius,2);
  *minradius = ((DM_Plex*) dm->data)->minradius;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexSetMinRadius - Sets the minimum distance from the cell centroid to a face

  Logically collective

  Input Parameters:
+ dm - the DM
- minradius - the minimum cell radius

  Level: developer

.seealso: `DMSetCoordinates()`
@*/
PetscErrorCode DMPlexSetMinRadius(DM dm, PetscReal minradius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ((DM_Plex*) dm->data)->minradius = minradius;
  PetscFunctionReturn(0);
}

static PetscErrorCode BuildGradientReconstruction_Internal(DM dm, PetscFV fvm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  DMLabel        ghostLabel;
  PetscScalar   *dx, *grad, **gref;
  PetscInt       dim, cStart, cEnd, c, cEndInterior, maxNumFaces;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
  cEndInterior = cEndInterior < 0 ? cEnd : cEndInterior;
  PetscCall(DMPlexGetMaxSizes(dm, &maxNumFaces, NULL));
  PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(PetscMalloc3(maxNumFaces*dim, &dx, maxNumFaces*dim, &grad, maxNumFaces, &gref));
  for (c = cStart; c < cEndInterior; c++) {
    const PetscInt        *faces;
    PetscInt               numFaces, usedFaces, f, d;
    PetscFVCellGeom        *cg;
    PetscBool              boundary;
    PetscInt               ghost;

    // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
    PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
    if (ghost >= 0) continue;

    PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
    PetscCall(DMPlexGetConeSize(dm, c, &numFaces));
    PetscCall(DMPlexGetCone(dm, c, &faces));
    PetscCheck(numFaces >= dim,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      PetscFVCellGeom       *cg1;
      PetscFVFaceGeom       *fg;
      const PetscInt        *fcells;
      PetscInt               ncell, side;

      PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
      PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
      if ((ghost >= 0) || boundary) continue;
      PetscCall(DMPlexGetSupport(dm, faces[f], &fcells));
      side  = (c != fcells[0]); /* c is on left=0 or right=1 of face */
      ncell = fcells[!side];    /* the neighbor */
      PetscCall(DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg));
      PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
      for (d = 0; d < dim; ++d) dx[usedFaces*dim+d] = cg1->centroid[d] - cg->centroid[d];
      gref[usedFaces++] = fg->grad[side];  /* Gradient reconstruction term will go here */
    }
    PetscCheck(usedFaces,PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
    PetscCall(PetscFVComputeGradient(fvm, usedFaces, dx, grad));
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
      PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
      if ((ghost >= 0) || boundary) continue;
      for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces*dim+d];
      ++usedFaces;
    }
  }
  PetscCall(PetscFree3(dx, grad, gref));
  PetscFunctionReturn(0);
}

static PetscErrorCode BuildGradientReconstruction_Internal_Tree(DM dm, PetscFV fvm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  DMLabel        ghostLabel;
  PetscScalar   *dx, *grad, **gref;
  PetscInt       dim, cStart, cEnd, c, cEndInterior, fStart, fEnd, f, nStart, nEnd, maxNumFaces = 0;
  PetscSection   neighSec;
  PetscInt     (*neighbors)[2];
  PetscInt      *counter;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
  if (cEndInterior < 0) cEndInterior = cEnd;
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm),&neighSec));
  PetscCall(PetscSectionSetChart(neighSec,cStart,cEndInterior));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  for (f = fStart; f < fEnd; f++) {
    const PetscInt        *fcells;
    PetscBool              boundary;
    PetscInt               ghost = -1;
    PetscInt               numChildren, numCells, c;

    if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
    PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
    PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
    if ((ghost >= 0) || boundary || numChildren) continue;
    PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
    if (numCells == 2) {
      PetscCall(DMPlexGetSupport(dm, f, &fcells));
      for (c = 0; c < 2; c++) {
        PetscInt cell = fcells[c];

        if (cell >= cStart && cell < cEndInterior) {
          PetscCall(PetscSectionAddDof(neighSec,cell,1));
        }
      }
    }
  }
  PetscCall(PetscSectionSetUp(neighSec));
  PetscCall(PetscSectionGetMaxDof(neighSec,&maxNumFaces));
  PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
  nStart = 0;
  PetscCall(PetscSectionGetStorageSize(neighSec,&nEnd));
  PetscCall(PetscMalloc1((nEnd-nStart),&neighbors));
  PetscCall(PetscCalloc1((cEndInterior-cStart),&counter));
  for (f = fStart; f < fEnd; f++) {
    const PetscInt        *fcells;
    PetscBool              boundary;
    PetscInt               ghost = -1;
    PetscInt               numChildren, numCells, c;

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
          PetscCall(PetscSectionGetOffset(neighSec,cell,&off));
          off += counter[cell - cStart]++;
          neighbors[off][0] = f;
          neighbors[off][1] = fcells[1 - c];
        }
      }
    }
  }
  PetscCall(PetscFree(counter));
  PetscCall(PetscMalloc3(maxNumFaces*dim, &dx, maxNumFaces*dim, &grad, maxNumFaces, &gref));
  for (c = cStart; c < cEndInterior; c++) {
    PetscInt               numFaces, f, d, off, ghost = -1;
    PetscFVCellGeom        *cg;

    PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
    PetscCall(PetscSectionGetDof(neighSec, c, &numFaces));
    PetscCall(PetscSectionGetOffset(neighSec, c, &off));

    // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
    if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
    if (ghost >= 0) continue;

    PetscCheck(numFaces >= dim,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0; f < numFaces; ++f) {
      PetscFVCellGeom       *cg1;
      PetscFVFaceGeom       *fg;
      const PetscInt        *fcells;
      PetscInt               ncell, side, nface;

      nface = neighbors[off + f][0];
      ncell = neighbors[off + f][1];
      PetscCall(DMPlexGetSupport(dm,nface,&fcells));
      side  = (c != fcells[0]);
      PetscCall(DMPlexPointLocalRef(dmFace, nface, fgeom, &fg));
      PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
      for (d = 0; d < dim; ++d) dx[f*dim+d] = cg1->centroid[d] - cg->centroid[d];
      gref[f] = fg->grad[side];  /* Gradient reconstruction term will go here */
    }
    PetscCall(PetscFVComputeGradient(fvm, numFaces, dx, grad));
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) gref[f][d] = grad[f*dim+d];
    }
  }
  PetscCall(PetscFree3(dx, grad, gref));
  PetscCall(PetscSectionDestroy(&neighSec));
  PetscCall(PetscFree(neighbors));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeGradientFVM - Compute geometric factors for gradient reconstruction, which are stored in the geometry data, and compute layout for gradient data

  Collective on dm

  Input Parameters:
+ dm  - The DM
. fvm - The PetscFV
- cellGeometry - The face geometry from DMPlexComputeCellGeometryFVM()

  Input/Output Parameter:
. faceGeometry - The face geometry from DMPlexComputeFaceGeometryFVM(); on output
                 the geometric factors for gradient calculation are inserted

  Output Parameter:
. dmGrad - The DM describing the layout of gradient data

  Level: developer

.seealso: `DMPlexGetFaceGeometryFVM()`, `DMPlexGetCellGeometryFVM()`
@*/
PetscErrorCode DMPlexComputeGradientFVM(DM dm, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM *dmGrad)
{
  DM             dmFace, dmCell;
  PetscScalar   *fgeom, *cgeom;
  PetscSection   sectionGrad, parentSection;
  PetscInt       dim, pdim, cStart, cEnd, cEndInterior, c;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFVGetNumComponents(fvm, &pdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
  /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
  PetscCall(VecGetDM(faceGeometry, &dmFace));
  PetscCall(VecGetDM(cellGeometry, &dmCell));
  PetscCall(VecGetArray(faceGeometry, &fgeom));
  PetscCall(VecGetArray(cellGeometry, &cgeom));
  PetscCall(DMPlexGetTree(dm,&parentSection,NULL,NULL,NULL,NULL));
  if (!parentSection) {
    PetscCall(BuildGradientReconstruction_Internal(dm, fvm, dmFace, fgeom, dmCell, cgeom));
  } else {
    PetscCall(BuildGradientReconstruction_Internal_Tree(dm, fvm, dmFace, fgeom, dmCell, cgeom));
  }
  PetscCall(VecRestoreArray(faceGeometry, &fgeom));
  PetscCall(VecRestoreArray(cellGeometry, &cgeom));
  /* Create storage for gradients */
  PetscCall(DMClone(dm, dmGrad));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionGrad));
  PetscCall(PetscSectionSetChart(sectionGrad, cStart, cEnd));
  for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionGrad, c, pdim*dim));
  PetscCall(PetscSectionSetUp(sectionGrad));
  PetscCall(DMSetLocalSection(*dmGrad, sectionGrad));
  PetscCall(PetscSectionDestroy(&sectionGrad));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetDataFVM - Retrieve precomputed cell geometry

  Collective on dm

  Input Parameters:
+ dm  - The DM
- fv  - The PetscFV

  Output Parameters:
+ cellGeometry - The cell geometry
. faceGeometry - The face geometry
- gradDM       - The gradient matrices

  Level: developer

.seealso: `DMPlexComputeGeometryFVM()`
@*/
PetscErrorCode DMPlexGetDataFVM(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM)
{
  PetscObject    cellgeomobj, facegeomobj;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj));
  if (!cellgeomobj) {
    Vec cellgeomInt, facegeomInt;

    PetscCall(DMPlexComputeGeometryFVM(dm, &cellgeomInt, &facegeomInt));
    PetscCall(PetscObjectCompose((PetscObject) dm, "DMPlex_cellgeom_fvm",(PetscObject)cellgeomInt));
    PetscCall(PetscObjectCompose((PetscObject) dm, "DMPlex_facegeom_fvm",(PetscObject)facegeomInt));
    PetscCall(VecDestroy(&cellgeomInt));
    PetscCall(VecDestroy(&facegeomInt));
    PetscCall(PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj));
  }
  PetscCall(PetscObjectQuery((PetscObject) dm, "DMPlex_facegeom_fvm", &facegeomobj));
  if (cellgeom) *cellgeom = (Vec) cellgeomobj;
  if (facegeom) *facegeom = (Vec) facegeomobj;
  if (gradDM) {
    PetscObject gradobj;
    PetscBool   computeGradients;

    PetscCall(PetscFVGetComputeGradients(fv,&computeGradients));
    if (!computeGradients) {
      *gradDM = NULL;
      PetscFunctionReturn(0);
    }
    PetscCall(PetscObjectQuery((PetscObject) dm, "DMPlex_dmgrad_fvm", &gradobj));
    if (!gradobj) {
      DM dmGradInt;

      PetscCall(DMPlexComputeGradientFVM(dm,fv,(Vec) facegeomobj,(Vec) cellgeomobj,&dmGradInt));
      PetscCall(PetscObjectCompose((PetscObject) dm, "DMPlex_dmgrad_fvm", (PetscObject)dmGradInt));
      PetscCall(DMDestroy(&dmGradInt));
      PetscCall(PetscObjectQuery((PetscObject) dm, "DMPlex_dmgrad_fvm", &gradobj));
    }
    *gradDM = (DM) gradobj;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCoordinatesToReference_NewtonUpdate(PetscInt dimC, PetscInt dimR, PetscScalar *J, PetscScalar *invJ, PetscScalar *work,  PetscReal *resNeg, PetscReal *guess)
{
  PetscInt l, m;

  PetscFunctionBeginHot;
  if (dimC == dimR && dimR <= 3) {
    /* invert Jacobian, multiply */
    PetscScalar det, idet;

    switch (dimR) {
    case 1:
      invJ[0] = 1./ J[0];
      break;
    case 2:
      det = J[0] * J[3] - J[1] * J[2];
      idet = 1./det;
      invJ[0] =  J[3] * idet;
      invJ[1] = -J[1] * idet;
      invJ[2] = -J[2] * idet;
      invJ[3] =  J[0] * idet;
      break;
    case 3:
      {
        invJ[0] = J[4] * J[8] - J[5] * J[7];
        invJ[1] = J[2] * J[7] - J[1] * J[8];
        invJ[2] = J[1] * J[5] - J[2] * J[4];
        det = invJ[0] * J[0] + invJ[1] * J[3] + invJ[2] * J[6];
        idet = 1./det;
        invJ[0] *= idet;
        invJ[1] *= idet;
        invJ[2] *= idet;
        invJ[3]  = idet * (J[5] * J[6] - J[3] * J[8]);
        invJ[4]  = idet * (J[0] * J[8] - J[2] * J[6]);
        invJ[5]  = idet * (J[2] * J[3] - J[0] * J[5]);
        invJ[6]  = idet * (J[3] * J[7] - J[4] * J[6]);
        invJ[7]  = idet * (J[1] * J[6] - J[0] * J[7]);
        invJ[8]  = idet * (J[0] * J[4] - J[1] * J[3]);
      }
      break;
    }
    for (l = 0; l < dimR; l++) {
      for (m = 0; m < dimC; m++) {
        guess[l] += PetscRealPart(invJ[l * dimC + m]) * resNeg[m];
      }
    }
  } else {
#if defined(PETSC_USE_COMPLEX)
    char transpose = 'C';
#else
    char transpose = 'T';
#endif
    PetscBLASInt m = dimR;
    PetscBLASInt n = dimC;
    PetscBLASInt one = 1;
    PetscBLASInt worksize = dimR * dimC, info;

    for (l = 0; l < dimC; l++) {invJ[l] = resNeg[l];}

    PetscStackCallBLAS("LAPACKgels",LAPACKgels_(&transpose,&m,&n,&one,J,&m,invJ,&n,work,&worksize, &info));
    PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELS");

    for (l = 0; l < dimR; l++) {guess[l] += PetscRealPart(invJ[l]);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCoordinatesToReference_Tensor(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[], Vec coords, PetscInt dimC, PetscInt dimR)
{
  PetscInt       coordSize, i, j, k, l, m, maxIts = 7, numV = (1 << dimR);
  PetscScalar    *coordsScalar = NULL;
  PetscReal      *cellData, *cellCoords, *cellCoeffs, *extJ, *resNeg;
  PetscScalar    *J, *invJ, *work;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscCheck(coordSize >= dimC * numV,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expecting at least %" PetscInt_FMT " coordinates, got %" PetscInt_FMT,dimC * (1 << dimR), coordSize);
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

      for (j = 0; j < dimC; j++) {
        cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
      }
    }
  } else if (dimR == 3) {
    const PetscInt zToPlex[8] = {0, 3, 1, 2, 4, 5, 7, 6};

    for (i = 0; i < 8; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) {
        cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
      }
    }
  } else {
    for (i = 0; i < coordSize; i++) {cellCoords[i] = PetscRealPart(coordsScalar[i]);}
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
      swap = cellCoeffs;
      cellCoeffs = cellCoords;
      cellCoords = swap;
    }
  }
  PetscCall(PetscArrayzero(refCoords,numPoints * dimR));
  for (j = 0; j < numPoints; j++) {
    for (i = 0; i < maxIts; i++) {
      PetscReal *guess = &refCoords[dimR * j];

      /* compute -residual and Jacobian */
      for (k = 0; k < dimC; k++) {resNeg[k] = realCoords[dimC * j + k];}
      for (k = 0; k < dimC * dimR; k++) {J[k] = 0.;}
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
          for (m = 0; m < dimR; m++) {
            J[dimR * l + m] += coeff * extJ[m];
          }
        }
      }
      if (0 && PetscDefined(USE_DEBUG)) {
        PetscReal maxAbs = 0.;

        for (l = 0; l < dimC; l++) {
          maxAbs = PetscMax(maxAbs,PetscAbsReal(resNeg[l]));
        }
        PetscCall(PetscInfo(dm,"cell %" PetscInt_FMT ", point %" PetscInt_FMT ", iter %" PetscInt_FMT ": res %g\n",cell,j,i,(double) maxAbs));
      }

      PetscCall(DMPlexCoordinatesToReference_NewtonUpdate(dimC,dimR,J,invJ,work,resNeg,guess));
    }
  }
  PetscCall(DMRestoreWorkArray(dm, 3 * dimR * dimC, MPIU_SCALAR, &J));
  PetscCall(DMRestoreWorkArray(dm, 2 * coordSize + dimR + dimC, MPIU_REAL, &cellData));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceToCoordinates_Tensor(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[], Vec coords, PetscInt dimC, PetscInt dimR)
{
  PetscInt       coordSize, i, j, k, l, numV = (1 << dimR);
  PetscScalar    *coordsScalar = NULL;
  PetscReal      *cellData, *cellCoords, *cellCoeffs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar));
  PetscCheck(coordSize >= dimC * numV,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expecting at least %" PetscInt_FMT " coordinates, got %" PetscInt_FMT,dimC * (1 << dimR), coordSize);
  PetscCall(DMGetWorkArray(dm, 2 * coordSize, MPIU_REAL, &cellData));
  cellCoords = &cellData[0];
  cellCoeffs = &cellData[coordSize];
  if (dimR == 2) {
    const PetscInt zToPlex[4] = {0, 1, 3, 2};

    for (i = 0; i < 4; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) {
        cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
      }
    }
  } else if (dimR == 3) {
    const PetscInt zToPlex[8] = {0, 3, 1, 2, 4, 5, 7, 6};

    for (i = 0; i < 8; i++) {
      PetscInt plexI = zToPlex[i];

      for (j = 0; j < dimC; j++) {
        cellCoords[dimC * i + j] = PetscRealPart(coordsScalar[dimC * plexI + j]);
      }
    }
  } else {
    for (i = 0; i < coordSize; i++) {cellCoords[i] = PetscRealPart(coordsScalar[i]);}
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
      swap = cellCoeffs;
      cellCoeffs = cellCoords;
      cellCoords = swap;
    }
  }
  PetscCall(PetscArrayzero(realCoords,numPoints * dimC));
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
  PetscFunctionReturn(0);
}

/* TODO: TOBY please fix this for Nc > 1 */
static PetscErrorCode DMPlexCoordinatesToReference_FE(DM dm, PetscFE fe, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[], Vec coords, PetscInt Nc, PetscInt dimR)
{
  PetscInt       numComp, pdim, i, j, k, l, m, maxIter = 7, coordSize;
  PetscScalar    *nodes = NULL;
  PetscReal      *invV, *modes;
  PetscReal      *B, *D, *resNeg;
  PetscScalar    *J, *invJ, *work;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDimension(fe, &pdim));
  PetscCall(PetscFEGetNumComponents(fe, &numComp));
  PetscCheck(numComp == Nc,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"coordinate discretization must have as many components (%" PetscInt_FMT ") as embedding dimension (!= %" PetscInt_FMT ")",numComp,Nc);
  PetscCall(DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &nodes));
  /* convert nodes to values in the stable evaluation basis */
  PetscCall(DMGetWorkArray(dm,pdim,MPIU_REAL,&modes));
  invV = fe->invV;
  for (i = 0; i < pdim; ++i) {
    modes[i] = 0.;
    for (j = 0; j < pdim; ++j) {
      modes[i] += invV[i * pdim + j] * PetscRealPart(nodes[j]);
    }
  }
  PetscCall(DMGetWorkArray(dm,pdim * Nc + pdim * Nc * dimR + Nc,MPIU_REAL,&B));
  D      = &B[pdim*Nc];
  resNeg = &D[pdim*Nc * dimR];
  PetscCall(DMGetWorkArray(dm,3 * Nc * dimR,MPIU_SCALAR,&J));
  invJ = &J[Nc * dimR];
  work = &invJ[Nc * dimR];
  for (i = 0; i < numPoints * dimR; i++) {refCoords[i] = 0.;}
  for (j = 0; j < numPoints; j++) {
      for (i = 0; i < maxIter; i++) { /* we could batch this so that we're not making big B and D arrays all the time */
      PetscReal *guess = &refCoords[j * dimR];
      PetscCall(PetscSpaceEvaluate(fe->basisSpace, 1, guess, B, D, NULL));
      for (k = 0; k < Nc; k++) {resNeg[k] = realCoords[j * Nc + k];}
      for (k = 0; k < Nc * dimR; k++) {J[k] = 0.;}
      for (k = 0; k < pdim; k++) {
        for (l = 0; l < Nc; l++) {
          resNeg[l] -= modes[k] * B[k * Nc + l];
          for (m = 0; m < dimR; m++) {
            J[l * dimR + m] += modes[k] * D[(k * Nc + l) * dimR + m];
          }
        }
      }
      if (0 && PetscDefined(USE_DEBUG)) {
        PetscReal maxAbs = 0.;

        for (l = 0; l < Nc; l++) {
          maxAbs = PetscMax(maxAbs,PetscAbsReal(resNeg[l]));
        }
        PetscCall(PetscInfo(dm,"cell %" PetscInt_FMT ", point %" PetscInt_FMT ", iter %" PetscInt_FMT ": res %g\n",cell,j,i,(double) maxAbs));
      }
      PetscCall(DMPlexCoordinatesToReference_NewtonUpdate(Nc,dimR,J,invJ,work,resNeg,guess));
    }
  }
  PetscCall(DMRestoreWorkArray(dm,3 * Nc * dimR,MPIU_SCALAR,&J));
  PetscCall(DMRestoreWorkArray(dm,pdim * Nc + pdim * Nc * dimR + Nc,MPIU_REAL,&B));
  PetscCall(DMRestoreWorkArray(dm,pdim,MPIU_REAL,&modes));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &nodes));
  PetscFunctionReturn(0);
}

/* TODO: TOBY please fix this for Nc > 1 */
static PetscErrorCode DMPlexReferenceToCoordinates_FE(DM dm, PetscFE fe, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[], Vec coords, PetscInt Nc, PetscInt dimR)
{
  PetscInt       numComp, pdim, i, j, k, l, coordSize;
  PetscScalar    *nodes = NULL;
  PetscReal      *invV, *modes;
  PetscReal      *B;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDimension(fe, &pdim));
  PetscCall(PetscFEGetNumComponents(fe, &numComp));
  PetscCheck(numComp == Nc,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"coordinate discretization must have as many components (%" PetscInt_FMT ") as embedding dimension (!= %" PetscInt_FMT ")",numComp,Nc);
  PetscCall(DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &nodes));
  /* convert nodes to values in the stable evaluation basis */
  PetscCall(DMGetWorkArray(dm,pdim,MPIU_REAL,&modes));
  invV = fe->invV;
  for (i = 0; i < pdim; ++i) {
    modes[i] = 0.;
    for (j = 0; j < pdim; ++j) {
      modes[i] += invV[i * pdim + j] * PetscRealPart(nodes[j]);
    }
  }
  PetscCall(DMGetWorkArray(dm,numPoints * pdim * Nc,MPIU_REAL,&B));
  PetscCall(PetscSpaceEvaluate(fe->basisSpace, numPoints, refCoords, B, NULL, NULL));
  for (i = 0; i < numPoints * Nc; i++) {realCoords[i] = 0.;}
  for (j = 0; j < numPoints; j++) {
    PetscReal *mapped = &realCoords[j * Nc];

    for (k = 0; k < pdim; k++) {
      for (l = 0; l < Nc; l++) {
        mapped[l] += modes[k] * B[(j * pdim + k) * Nc + l];
      }
    }
  }
  PetscCall(DMRestoreWorkArray(dm,numPoints * pdim * Nc,MPIU_REAL,&B));
  PetscCall(DMRestoreWorkArray(dm,pdim,MPIU_REAL,&modes));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &nodes));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCoordinatesToReference - Pull coordinates back from the mesh to the reference element using a single element
  map.  This inversion will be accurate inside the reference element, but may be inaccurate for mappings that do not
  extend uniquely outside the reference cell (e.g, most non-affine maps)

  Not collective

  Input Parameters:
+ dm         - The mesh, with coordinate maps defined either by a PetscDS for the coordinate DM (see DMGetCoordinateDM()) or
               implicitly by the coordinates of the corner vertices of the cell: as an affine map for simplicial elements, or
               as a multilinear map for tensor-product elements
. cell       - the cell whose map is used.
. numPoints  - the number of points to locate
- realCoords - (numPoints x coordinate dimension) array of coordinates (see DMGetCoordinateDim())

  Output Parameters:
. refCoords  - (numPoints x dimension) array of reference coordinates (see DMGetDimension())

  Level: intermediate

.seealso: `DMPlexReferenceToCoordinates()`
@*/
PetscErrorCode DMPlexCoordinatesToReference(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[])
{
  PetscInt       dimC, dimR, depth, cStart, cEnd, i;
  DM             coordDM = NULL;
  Vec            coords;
  PetscFE        fe = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDimension(dm,&dimR));
  PetscCall(DMGetCoordinateDim(dm,&dimC));
  if (dimR <= 0 || dimC <= 0 || numPoints <= 0) PetscFunctionReturn(0);
  PetscCall(DMPlexGetDepth(dm,&depth));
  PetscCall(DMGetCoordinatesLocal(dm,&coords));
  PetscCall(DMGetCoordinateDM(dm,&coordDM));
  if (coordDM) {
    PetscInt coordFields;

    PetscCall(DMGetNumFields(coordDM,&coordFields));
    if (coordFields) {
      PetscClassId id;
      PetscObject  disc;

      PetscCall(DMGetField(coordDM,0,NULL,&disc));
      PetscCall(PetscObjectGetClassId(disc,&id));
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
      }
    }
  }
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCheck(cell >= cStart && cell < cEnd,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"point %" PetscInt_FMT " not in cell range [%" PetscInt_FMT ",%" PetscInt_FMT ")",cell,cStart,cEnd);
  if (!fe) { /* implicit discretization: affine or multilinear */
    PetscInt  coneSize;
    PetscBool isSimplex, isTensor;

    PetscCall(DMPlexGetConeSize(dm,cell,&coneSize));
    isSimplex = (coneSize == (dimR + 1)) ? PETSC_TRUE : PETSC_FALSE;
    isTensor  = (coneSize == ((depth == 1) ? (1 << dimR) : (2 * dimR))) ? PETSC_TRUE : PETSC_FALSE;
    if (isSimplex) {
      PetscReal detJ, *v0, *J, *invJ;

      PetscCall(DMGetWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
      J    = &v0[dimC];
      invJ = &J[dimC * dimC];
      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, cell, v0, J, invJ, &detJ));
      for (i = 0; i < numPoints; i++) { /* Apply the inverse affine transformation for each point */
        const PetscReal x0[3] = {-1.,-1.,-1.};

        CoordinatesRealToRef(dimC, dimR, x0, v0, invJ, &realCoords[dimC * i], &refCoords[dimR * i]);
      }
      PetscCall(DMRestoreWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
    } else if (isTensor) {
      PetscCall(DMPlexCoordinatesToReference_Tensor(coordDM, cell, numPoints, realCoords, refCoords, coords, dimC, dimR));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unrecognized cone size %" PetscInt_FMT,coneSize);
  } else {
    PetscCall(DMPlexCoordinatesToReference_FE(coordDM, fe, cell, numPoints, realCoords, refCoords, coords, dimC, dimR));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexReferenceToCoordinates - Map references coordinates to coordinates in the the mesh for a single element map.

  Not collective

  Input Parameters:
+ dm         - The mesh, with coordinate maps defined either by a PetscDS for the coordinate DM (see DMGetCoordinateDM()) or
               implicitly by the coordinates of the corner vertices of the cell: as an affine map for simplicial elements, or
               as a multilinear map for tensor-product elements
. cell       - the cell whose map is used.
. numPoints  - the number of points to locate
- refCoords  - (numPoints x dimension) array of reference coordinates (see DMGetDimension())

  Output Parameters:
. realCoords - (numPoints x coordinate dimension) array of coordinates (see DMGetCoordinateDim())

   Level: intermediate

.seealso: `DMPlexCoordinatesToReference()`
@*/
PetscErrorCode DMPlexReferenceToCoordinates(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[])
{
  PetscInt       dimC, dimR, depth, cStart, cEnd, i;
  DM             coordDM = NULL;
  Vec            coords;
  PetscFE        fe = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDimension(dm,&dimR));
  PetscCall(DMGetCoordinateDim(dm,&dimC));
  if (dimR <= 0 || dimC <= 0 || numPoints <= 0) PetscFunctionReturn(0);
  PetscCall(DMPlexGetDepth(dm,&depth));
  PetscCall(DMGetCoordinatesLocal(dm,&coords));
  PetscCall(DMGetCoordinateDM(dm,&coordDM));
  if (coordDM) {
    PetscInt coordFields;

    PetscCall(DMGetNumFields(coordDM,&coordFields));
    if (coordFields) {
      PetscClassId id;
      PetscObject  disc;

      PetscCall(DMGetField(coordDM,0,NULL,&disc));
      PetscCall(PetscObjectGetClassId(disc,&id));
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
      }
    }
  }
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCheck(cell >= cStart && cell < cEnd,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"point %" PetscInt_FMT " not in cell range [%" PetscInt_FMT ",%" PetscInt_FMT ")",cell,cStart,cEnd);
  if (!fe) { /* implicit discretization: affine or multilinear */
    PetscInt  coneSize;
    PetscBool isSimplex, isTensor;

    PetscCall(DMPlexGetConeSize(dm,cell,&coneSize));
    isSimplex = (coneSize == (dimR + 1)) ? PETSC_TRUE : PETSC_FALSE;
    isTensor  = (coneSize == ((depth == 1) ? (1 << dimR) : (2 * dimR))) ? PETSC_TRUE : PETSC_FALSE;
    if (isSimplex) {
      PetscReal detJ, *v0, *J;

      PetscCall(DMGetWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
      J    = &v0[dimC];
      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, cell, v0, J, NULL, &detJ));
      for (i = 0; i < numPoints; i++) { /* Apply the affine transformation for each point */
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        CoordinatesRefToReal(dimC, dimR, xi0, v0, J, &refCoords[dimR * i], &realCoords[dimC * i]);
      }
      PetscCall(DMRestoreWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0));
    } else if (isTensor) {
      PetscCall(DMPlexReferenceToCoordinates_Tensor(coordDM, cell, numPoints, refCoords, realCoords, coords, dimC, dimR));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unrecognized cone size %" PetscInt_FMT,coneSize);
  } else {
    PetscCall(DMPlexReferenceToCoordinates_FE(coordDM, fe, cell, numPoints, refCoords, realCoords, coords, dimC, dimR));
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRemapGeometry - This function maps the original DM coordinates to new coordinates.

  Not collective

  Input Parameters:
+ dm      - The DM
. time    - The time
- func    - The function transforming current coordinates to new coordaintes

   Calling sequence of func:
$    func(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

+  dim          - The spatial dimension
.  Nf           - The number of input fields (here 1)
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of the coordinates in u[] (here 0)
.  uOff_x       - The offset of the coordinates in u_x[] (here 0)
.  u            - The coordinate values at this point in space
.  u_t          - The coordinate time derivative at this point in space (here NULL)
.  u_x          - The coordinate derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point (here not used)
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The new coordinates at this point in space

  Level: intermediate

.seealso: `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCoordinateDM()`, `DMProjectFieldLocal()`, `DMProjectFieldLabelLocal()`
@*/
PetscErrorCode DMPlexRemapGeometry(DM dm, PetscReal time,
                                   void (*func)(PetscInt, PetscInt, PetscInt,
                                                const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  DM             cdm;
  DMField        cf;
  Vec            lCoords, tmpCoords;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &lCoords));
  PetscCall(DMGetLocalVector(cdm, &tmpCoords));
  PetscCall(VecCopy(lCoords, tmpCoords));
  /* We have to do the coordinate field manually right now since the coordinate DM will not have its own */
  PetscCall(DMGetCoordinateField(dm, &cf));
  cdm->coordinateField = cf;
  PetscCall(DMProjectFieldLocal(cdm, time, tmpCoords, &func, INSERT_VALUES, lCoords));
  cdm->coordinateField = NULL;
  PetscCall(DMRestoreLocalVector(cdm, &tmpCoords));
  PetscCall(DMSetCoordinatesLocal(dm, lCoords));
  PetscFunctionReturn(0);
}

/* Shear applies the transformation, assuming we fix z,
  / 1  0  m_0 \
  | 0  1  m_1 |
  \ 0  0   1  /
*/
static void f0_shear(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar coords[])
{
  const PetscInt Nc = uOff[1]-uOff[0];
  const PetscInt ax = (PetscInt) PetscRealPart(constants[0]);
  PetscInt       c;

  for (c = 0; c < Nc; ++c) {
    coords[c] = u[c] + constants[c+1]*u[ax];
  }
}

/*@C
  DMPlexShearGeometry - This shears the domain, meaning adds a multiple of the shear coordinate to all other coordinates.

  Not collective

  Input Parameters:
+ dm          - The DM
. direction   - The shear coordinate direction, e.g. 0 is the x-axis
- multipliers - The multiplier m for each direction which is not the shear direction

  Level: intermediate

.seealso: `DMPlexRemapGeometry()`
@*/
PetscErrorCode DMPlexShearGeometry(DM dm, DMDirection direction, PetscReal multipliers[])
{
  DM             cdm;
  PetscDS        cds;
  PetscObject    obj;
  PetscClassId   id;
  PetscScalar   *moduli;
  const PetscInt dir = (PetscInt) direction;
  PetscInt       dE, d, e;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(PetscMalloc1(dE+1, &moduli));
  moduli[0] = dir;
  for (d = 0, e = 0; d < dE; ++d) moduli[d+1] = d == dir ? 0.0 : (multipliers ? multipliers[e++] : 1.0);
  PetscCall(DMGetDS(cdm, &cds));
  PetscCall(PetscDSGetDiscretization(cds, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  if (id != PETSCFE_CLASSID) {
    Vec           lCoords;
    PetscSection  cSection;
    PetscScalar  *coords;
    PetscInt      vStart, vEnd, v;

    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(DMGetCoordinateSection(dm, &cSection));
    PetscCall(DMGetCoordinatesLocal(dm, &lCoords));
    PetscCall(VecGetArray(lCoords, &coords));
    for (v = vStart; v < vEnd; ++v) {
      PetscReal ds;
      PetscInt  off, c;

      PetscCall(PetscSectionGetOffset(cSection, v, &off));
      ds   = PetscRealPart(coords[off+dir]);
      for (c = 0; c < dE; ++c) coords[off+c] += moduli[c]*ds;
    }
    PetscCall(VecRestoreArray(lCoords, &coords));
  } else {
    PetscCall(PetscDSSetConstants(cds, dE+1, moduli));
    PetscCall(DMPlexRemapGeometry(dm, 0.0, f0_shear));
  }
  PetscCall(PetscFree(moduli));
  PetscFunctionReturn(0);
}

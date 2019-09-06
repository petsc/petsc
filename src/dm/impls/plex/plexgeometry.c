#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/petscfeimpl.h>  /*I      "petscfe.h"       I*/
#include <petscblaslapack.h>
#include <petsctime.h>

/*@
  DMPlexFindVertices - Try to find DAG points based on their coordinates.

  Not Collective (provided DMGetCoordinatesLocalSetUp() has been called already)

  Input Parameters:
+ dm - The DMPlex object
. npoints - The number of sought points
. coords - The array of coordinates of the sought points
- eps - The tolerance or PETSC_DEFAULT

  Output Parameters:
. dagPoints - The array of found DAG points, or -1 if not found

  Level: intermediate

  Notes:
  The length of the array coords must be npoints * dim where dim is the spatial dimension returned by DMGetDimension().

  The output array dagPoints is NOT newly allocated; the user must pass an array of length npoints.

  Each rank does the search independently; a nonnegative value is returned only if this rank's local DMPlex portion contains the point.

  The tolerance is interpreted as the maximum Euclidean (L2) distance of the sought point from the specified coordinates.

  Complexity of this function is currently O(mn) with m number of vertices to find and n number of vertices in the local mesh. This could probably be improved.

.seealso: DMPlexCreate(), DMGetCoordinates()
@*/
PetscErrorCode DMPlexFindVertices(DM dm, PetscInt npoints, const PetscReal coord[], PetscReal eps, PetscInt dagPoints[])
{
  PetscInt          c, dim, i, j, o, p, vStart, vEnd;
  Vec               allCoordsVec;
  const PetscScalar *allCoords;
  PetscReal         norm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (eps < 0) eps = PETSC_SQRT_MACHINE_EPSILON;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &allCoordsVec);CHKERRQ(ierr);
  ierr = VecGetArrayRead(allCoordsVec, &allCoords);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  /* check coordinate section is consistent with DM dimension */
  {
    PetscSection      cs;
    PetscInt          ndof;

    ierr = DMGetCoordinateSection(dm, &cs);CHKERRQ(ierr);
    for (p = vStart; p < vEnd; p++) {
      ierr = PetscSectionGetDof(cs, p, &ndof);CHKERRQ(ierr);
      if (PetscUnlikely(ndof != dim)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "point %D: ndof = %D != %D = dim", p, ndof, dim);
    }
  }
#endif
  if (eps == 0.0) {
    for (i=0,j=0; i < npoints; i++,j+=dim) {
      dagPoints[i] = -1;
      for (p = vStart,o=0; p < vEnd; p++,o+=dim) {
        for (c = 0; c < dim; c++) {
          if (coord[j+c] != PetscRealPart(allCoords[o+c])) break;
        }
        if (c == dim) {
          dagPoints[i] = p;
          break;
        }
      }
    }
    ierr = VecRestoreArrayRead(allCoordsVec, &allCoords);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i=0,j=0; i < npoints; i++,j+=dim) {
    dagPoints[i] = -1;
    for (p = vStart,o=0; p < vEnd; p++,o+=dim) {
      norm = 0.0;
      for (c = 0; c < dim; c++) {
        norm += PetscSqr(coord[j+c] - PetscRealPart(allCoords[o+c]));
      }
      norm = PetscSqrtReal(norm);
      if (norm <= eps) {
        dagPoints[i] = p;
        break;
      }
    }
  }
  ierr = VecRestoreArrayRead(allCoordsVec, &allCoords);CHKERRQ(ierr);
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

static PetscErrorCode DMPlexLocatePoint_Simplex_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscInt  embedDim = 2;
  const PetscReal eps      = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal       x        = PetscRealPart(point[0]);
  PetscReal       y        = PetscRealPart(point[1]);
  PetscReal       v0[2], J[4], invJ[4], detJ;
  PetscReal       xi, eta;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
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

static PetscErrorCode DMPlexLocatePoint_General_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  PetscSection       coordSection;
  Vec             coordsLocal;
  PetscScalar    *coords = NULL;
  const PetscInt  faces[8]  = {0, 1, 1, 2, 2, 3, 3, 0};
  PetscReal       x         = PetscRealPart(point[0]);
  PetscReal       y         = PetscRealPart(point[1]);
  PetscInt        crossings = 0, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, NULL, &coords);CHKERRQ(ierr);
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
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLocatePoint_Simplex_3D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscInt embedDim = 3;
  PetscReal      v0[3], J[9], invJ[9], detJ;
  PetscReal      x = PetscRealPart(point[0]);
  PetscReal      y = PetscRealPart(point[1]);
  PetscReal      z = PetscRealPart(point[2]);
  PetscReal      xi, eta, zeta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
  xi   = invJ[0*embedDim+0]*(x - v0[0]) + invJ[0*embedDim+1]*(y - v0[1]) + invJ[0*embedDim+2]*(z - v0[2]);
  eta  = invJ[1*embedDim+0]*(x - v0[0]) + invJ[1*embedDim+1]*(y - v0[1]) + invJ[1*embedDim+2]*(z - v0[2]);
  zeta = invJ[2*embedDim+0]*(x - v0[0]) + invJ[2*embedDim+1]*(y - v0[1]) + invJ[2*embedDim+2]*(z - v0[2]);

  if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 2.0)) *cell = c;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, NULL, &coords);CHKERRQ(ierr);
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
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &coords);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(1, box);CHKERRQ(ierr);
  ierr = PetscGridHashInitialize_Internal(*box, dim, point);CHKERRQ(ierr);
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

.seealso: PetscGridHashCreate()
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

.seealso: PetscGridHashCreate()
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
      if (dbox < 0 || dbox >= n[d]) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input point %d (%g, %g, %g) is outside of our bounding box",
                                             p, (double) PetscRealPart(points[p*dim+0]), dim > 1 ? (double) PetscRealPart(points[p*dim+1]) : 0.0, dim > 2 ? (double) PetscRealPart(points[p*dim+2]) : 0.0);
      dboxes[p*dim+d] = dbox;
    }
    if (boxes) for (d = 1, boxes[p] = dboxes[p*dim]; d < dim; ++d) boxes[p] += dboxes[p*dim+d]*n[d-1];
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

.seealso: PetscGridHashGetEnclosingBox()
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
    if (boxes) for (d = 1, boxes[p] = dboxes[p*dim]; d < dim; ++d) boxes[p] += dboxes[p*dim+d]*n[d-1];
  }
  *found = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGridHashDestroy(PetscGridHash *box)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*box) {
    ierr = PetscSectionDestroy(&(*box)->cellSection);CHKERRQ(ierr);
    ierr = ISDestroy(&(*box)->cells);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&(*box)->cellsSparse);CHKERRQ(ierr);
  }
  ierr = PetscFree(*box);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLocatePoint_Internal(DM dm, PetscInt dim, const PetscScalar point[], PetscInt cellStart, PetscInt *cell)
{
  PetscInt       coneSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (dim) {
  case 2:
    ierr = DMPlexGetConeSize(dm, cellStart, &coneSize);CHKERRQ(ierr);
    switch (coneSize) {
    case 3:
      ierr = DMPlexLocatePoint_Simplex_2D_Internal(dm, point, cellStart, cell);CHKERRQ(ierr);
      break;
    case 4:
      ierr = DMPlexLocatePoint_General_2D_Internal(dm, point, cellStart, cell);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for cell with cone size %D", coneSize);
    }
    break;
  case 3:
    ierr = DMPlexGetConeSize(dm, cellStart, &coneSize);CHKERRQ(ierr);
    switch (coneSize) {
    case 4:
      ierr = DMPlexLocatePoint_Simplex_3D_Internal(dm, point, cellStart, cell);CHKERRQ(ierr);
      break;
    case 6:
      ierr = DMPlexLocatePoint_General_3D_Internal(dm, point, cellStart, cell);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for cell with cone size %D", coneSize);
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for mesh dimension %D", dim);
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexClosestPoint_Internal - Returns the closest point in the cell to the given point
*/
PetscErrorCode DMPlexClosestPoint_Internal(DM dm, PetscInt dim, const PetscScalar point[], PetscInt cell, PetscReal cpoint[])
{
  PetscInt       coneSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (dim) {
  case 2:
    ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
    switch (coneSize) {
    case 3:
      ierr = DMPlexClosestPoint_Simplex_2D_Internal(dm, point, cell, cpoint);CHKERRQ(ierr);
      break;
#if 0
    case 4:
      ierr = DMPlexClosestPoint_General_2D_Internal(dm, point, cell, cpoint);CHKERRQ(ierr);
      break;
#endif
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No closest point location for cell with cone size %D", coneSize);
    }
    break;
#if 0
  case 3:
    ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
    switch (coneSize) {
    case 4:
      ierr = DMPlexClosestPoint_Simplex_3D_Internal(dm, point, cell, cpoint);CHKERRQ(ierr);
      break;
    case 6:
      ierr = DMPlexClosestPoint_General_3D_Internal(dm, point, cell, cpoint);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No closest point location for cell with cone size %D", coneSize);
    }
    break;
#endif
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No closest point location for mesh dimension %D", dim);
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

.seealso: PetscGridHashCreate(), PetscGridHashGetEnclosingBox()
*/
PetscErrorCode DMPlexComputeGridHash_Internal(DM dm, PetscGridHash *localBox)
{
  MPI_Comm           comm;
  PetscGridHash      lbox;
  Vec                coordinates;
  PetscSection       coordSection;
  Vec                coordsLocal;
  const PetscScalar *coords;
  PetscInt          *dboxes, *boxes;
  PetscInt           n[3] = {10, 10, 10};
  PetscInt           dim, N, cStart, cEnd, cMax, c, i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(comm, PETSC_ERR_SUP, "I have only coded this for 2D");
  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscGridHashCreate(comm, dim, coords, &lbox);CHKERRQ(ierr);
  for (i = 0; i < N; i += dim) {ierr = PetscGridHashEnlarge(lbox, &coords[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dm_plex_hash_box_nijk",&n[0],NULL);CHKERRQ(ierr);
  n[1] = n[0];
  n[2] = n[0];
  ierr = PetscGridHashSetGrid(lbox, n, NULL);CHKERRQ(ierr);
#if 0
  /* Could define a custom reduction to merge these */
  ierr = MPIU_Allreduce(lbox->lower, gbox->lower, 3, MPIU_REAL, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(lbox->upper, gbox->upper, 3, MPIU_REAL, MPI_MAX, comm);CHKERRQ(ierr);
#endif
  /* Is there a reason to snap the local bounding box to a division of the global box? */
  /* Should we compute all overlaps of local boxes? We could do this with a rendevouz scheme partitioning the global box */
  /* Create label */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  if (cMax >= 0) cEnd = PetscMin(cEnd, cMax);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "cells", &lbox->cellsSparse);CHKERRQ(ierr);
  ierr = DMLabelCreateIndex(lbox->cellsSparse, cStart, cEnd);CHKERRQ(ierr);
  /* Compute boxes which overlap each cell: https://stackoverflow.com/questions/13790208/triangle-square-intersection-test-in-2d */
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscCalloc2(16 * dim, &dboxes, 16, &boxes);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscReal *h       = lbox->h;
    PetscScalar     *ccoords = NULL;
    PetscInt         csize   = 0;
    PetscScalar      point[3];
    PetscInt         dlim[6], d, e, i, j, k;

    /* Find boxes enclosing each vertex */
    ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, &csize, &ccoords);CHKERRQ(ierr);
    ierr = PetscGridHashGetEnclosingBox(lbox, csize/dim, ccoords, dboxes, boxes);CHKERRQ(ierr);
    /* Mark cells containing the vertices */
    for (e = 0; e < csize/dim; ++e) {ierr = DMLabelSetValue(lbox->cellsSparse, c, boxes[e]);CHKERRQ(ierr);}
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

          /* Check whether cell contains any vertex of these subboxes TODO vectorize this */
          for (kk = 0, cpoint[2] = point[2]; kk < (dim > 2 ? 2 : 1); ++kk, cpoint[2] += h[2]) {
            for (jj = 0, cpoint[1] = point[1]; jj < (dim > 1 ? 2 : 1); ++jj, cpoint[1] += h[1]) {
              for (ii = 0, cpoint[0] = point[0]; ii < 2; ++ii, cpoint[0] += h[0]) {

                ierr = DMPlexLocatePoint_Internal(dm, dim, cpoint, c, &cell);CHKERRQ(ierr);
                if (cell >= 0) { ierr = DMLabelSetValue(lbox->cellsSparse, c, box);CHKERRQ(ierr); ii = jj = kk = 2;}
              }
            }
          }
          /* Check whether cell edge intersects any edge of these subboxes TODO vectorize this */
          for (edge = 0; edge < dim+1; ++edge) {
            PetscReal segA[6], segB[6];

            if (PetscUnlikely(dim > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected dim %d > 3",dim);
            for (d = 0; d < dim; ++d) {segA[d] = PetscRealPart(ccoords[edge*dim+d]); segA[dim+d] = PetscRealPart(ccoords[((edge+1)%(dim+1))*dim+d]);}
            for (kk = 0; kk < (dim > 2 ? 2 : 1); ++kk) {
              if (dim > 2) {segB[2]     = PetscRealPart(point[2]);
                            segB[dim+2] = PetscRealPart(point[2]) + kk*h[2];}
              for (jj = 0; jj < (dim > 1 ? 2 : 1); ++jj) {
                if (dim > 1) {segB[1]     = PetscRealPart(point[1]);
                              segB[dim+1] = PetscRealPart(point[1]) + jj*h[1];}
                for (ii = 0; ii < 2; ++ii) {
                  PetscBool intersects;

                  segB[0]     = PetscRealPart(point[0]);
                  segB[dim+0] = PetscRealPart(point[0]) + ii*h[0];
                  ierr = DMPlexGetLineIntersection_2D_Internal(segA, segB, NULL, &intersects);CHKERRQ(ierr);
                  if (intersects) { ierr = DMLabelSetValue(lbox->cellsSparse, c, box);CHKERRQ(ierr); edge = ii = jj = kk = dim+1;}
                }
              }
            }
          }
        }
      }
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &ccoords);CHKERRQ(ierr);
  }
  ierr = PetscFree2(dboxes, boxes);CHKERRQ(ierr);
  ierr = DMLabelConvertToSection(lbox->cellsSparse, &lbox->cellSection, &lbox->cells);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&lbox->cellsSparse);CHKERRQ(ierr);
  *localBox = lbox;
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocatePoints_Plex(DM dm, Vec v, DMPointLocationType ltype, PetscSF cellSF)
{
  DM_Plex        *mesh = (DM_Plex *) dm->data;
  PetscBool       hash = mesh->useHashLocation, reuse = PETSC_FALSE;
  PetscInt        bs, numPoints, p, numFound, *found = NULL;
  PetscInt        dim, cStart, cEnd, cMax, numCells, c, d;
  const PetscInt *boxCells;
  PetscSFNode    *cells;
  PetscScalar    *a;
  PetscMPIInt     result;
  PetscLogDouble  t0,t1;
  PetscReal       gmin[3],gmax[3];
  PetscInt        terminating_query_type[] = { 0, 0, 0 };
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTime(&t0);CHKERRQ(ierr);
  if (ltype == DM_POINTLOCATION_NEAREST && !hash) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Nearest point location only supported with grid hashing. Use -dm_plex_hash_location to enable it.");
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = VecGetBlockSize(v, &bs);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)cellSF),PETSC_COMM_SELF,&result);CHKERRQ(ierr);
  if (result != MPI_IDENT && result != MPI_CONGRUENT) SETERRQ(PetscObjectComm((PetscObject)cellSF),PETSC_ERR_SUP, "Trying parallel point location: only local point location supported");
  if (bs != dim) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Block size for point vector %D must be the mesh coordinate dimension %D", bs, dim);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  if (cMax >= 0) cEnd = PetscMin(cEnd, cMax);
  ierr = VecGetLocalSize(v, &numPoints);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  numPoints /= bs;
  {
    const PetscSFNode *sf_cells;

    ierr = PetscSFGetGraph(cellSF,NULL,NULL,NULL,&sf_cells);CHKERRQ(ierr);
    if (sf_cells) {
      ierr = PetscInfo(dm,"[DMLocatePoints_Plex] Re-using existing StarForest node list\n");CHKERRQ(ierr);
      cells = (PetscSFNode*)sf_cells;
      reuse = PETSC_TRUE;
    } else {
      ierr = PetscInfo(dm,"[DMLocatePoints_Plex] Creating and initializing new StarForest node list\n");CHKERRQ(ierr);
      ierr = PetscMalloc1(numPoints, &cells);CHKERRQ(ierr);
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

    ierr = DMGetCoordinates(dm,&coorglobal);CHKERRQ(ierr);
    ierr = VecStrideMaxAll(coorglobal,NULL,gmax);CHKERRQ(ierr);
    ierr = VecStrideMinAll(coorglobal,NULL,gmin);CHKERRQ(ierr);
  }
  if (hash) {
    if (!mesh->lbox) {ierr = PetscInfo(dm, "Initializing grid hashing");CHKERRQ(ierr);ierr = DMPlexComputeGridHash_Internal(dm, &mesh->lbox);CHKERRQ(ierr);}
    /* Designate the local box for each point */
    /* Send points to correct process */
    /* Search cells that lie in each subbox */
    /*   Should we bin points before doing search? */
    ierr = ISGetIndices(mesh->lbox->cells, &boxCells);CHKERRQ(ierr);
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
      ierr = DMPlexLocatePoint_Internal(dm, dim, point, c, &cell);CHKERRQ(ierr);
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

      /* allow for case that point is outside box - abort early */
      ierr = PetscGridHashGetEnclosingBoxQuery(mesh->lbox, 1, point, dbin, &bin,&found_box);CHKERRQ(ierr);
      if (found_box) {
        /* TODO Lay an interface over this so we can switch between Section (dense) and Label (sparse) */
        ierr = PetscSectionGetDof(mesh->lbox->cellSection, bin, &numCells);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->lbox->cellSection, bin, &cellOffset);CHKERRQ(ierr);
        for (c = cellOffset; c < cellOffset + numCells; ++c) {
          ierr = DMPlexLocatePoint_Internal(dm, dim, point, boxCells[c], &cell);CHKERRQ(ierr);
          if (cell >= 0) {
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
        ierr = DMPlexLocatePoint_Internal(dm, dim, point, c, &cell);CHKERRQ(ierr);
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
  if (hash) {ierr = ISRestoreIndices(mesh->lbox->cells, &boxCells);CHKERRQ(ierr);}
  if (ltype == DM_POINTLOCATION_NEAREST && hash && numFound < numPoints) {
    for (p = 0; p < numPoints; p++) {
      const PetscScalar *point = &a[p*bs];
      PetscReal          cpoint[3], diff[3], dist, distMax = PETSC_MAX_REAL;
      PetscInt           dbin[3] = {-1,-1,-1}, bin, cellOffset, d;

      if (cells[p].index < 0) {
        ++numFound;
        ierr = PetscGridHashGetEnclosingBox(mesh->lbox, 1, point, dbin, &bin);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(mesh->lbox->cellSection, bin, &numCells);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->lbox->cellSection, bin, &cellOffset);CHKERRQ(ierr);
        for (c = cellOffset; c < cellOffset + numCells; ++c) {
          ierr = DMPlexClosestPoint_Internal(dm, dim, point, boxCells[c], cpoint);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) diff[d] = cpoint[d] - PetscRealPart(point[d]);
          dist = DMPlex_NormD_Internal(dim, diff);
          if (dist < distMax) {
            for (d = 0; d < dim; ++d) a[p*bs+d] = cpoint[d];
            cells[p].rank  = 0;
            cells[p].index = boxCells[c];
            distMax = dist;
            break;
          }
        }
      }
    }
  }
  /* This code is only be relevant when interfaced to parallel point location */
  /* Check for highest numbered proc that claims a point (do we care?) */
  if (ltype == DM_POINTLOCATION_REMOVE && numFound < numPoints) {
    ierr = PetscMalloc1(numFound,&found);CHKERRQ(ierr);
    for (p = 0, numFound = 0; p < numPoints; p++) {
      if (cells[p].rank >= 0 && cells[p].index >= 0) {
        if (numFound < p) {
          cells[numFound] = cells[p];
        }
        found[numFound++] = p;
      }
    }
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  if (!reuse) {
    ierr = PetscSFSetGraph(cellSF, cEnd - cStart, numFound, found, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  ierr = PetscTime(&t1);CHKERRQ(ierr);
  if (hash) {
    ierr = PetscInfo3(dm,"[DMLocatePoints_Plex] terminating_query_type : %D [outside domain] : %D [inside intial cell] : %D [hash]\n",terminating_query_type[0],terminating_query_type[1],terminating_query_type[2]);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo3(dm,"[DMLocatePoints_Plex] terminating_query_type : %D [outside domain] : %D [inside intial cell] : %D [brute-force]\n",terminating_query_type[0],terminating_query_type[1],terminating_query_type[2]);CHKERRQ(ierr);
  }
  ierr = PetscInfo3(dm,"[DMLocatePoints_Plex] npoints %D : time(rank0) %1.2e (sec): points/sec %1.4e\n",numPoints,t1-t0,(double)((double)numPoints/(t1-t0)));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeProjection2Dto1D - Rewrite coordinates to be the 1D projection of the 2D coordinates

  Not collective

  Input Parameter:
. coords - The coordinates of a segment

  Output Parameters:
+ coords - The new y-coordinate, and 0 for x
- R - The rotation which accomplishes the projection

  Level: developer

.seealso: DMPlexComputeProjection3Dto1D(), DMPlexComputeProjection3Dto2D()
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

  Input Parameter:
. coords - The coordinates of a segment

  Output Parameters:
+ coords - The new y-coordinate, and 0 for x and z
- R - The rotation which accomplishes the projection

  Note: This uses the basis completion described by Frisvad in http://www.imm.dtu.dk/~jerf/papers/abstracts/onb.html, DOI:10.1080/2165347X.2012.689606

  Level: developer

.seealso: DMPlexComputeProjection2Dto1D(), DMPlexComputeProjection3Dto2D()
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
  DMPlexComputeProjection3Dto2D - Rewrite coordinates to be the 2D projection of the 3D coordinates

  Not collective

  Input Parameter:
. coords - The coordinates of a segment

  Output Parameters:
+ coords - The new y- and z-coordinates, and 0 for x
- R - The rotation which accomplishes the projection

  Level: developer

.seealso: DMPlexComputeProjection2Dto1D(), DMPlexComputeProjection3Dto1D()
@*/
PetscErrorCode DMPlexComputeProjection3Dto2D(PetscInt coordSize, PetscScalar coords[], PetscReal R[])
{
  PetscReal      x1[3],  x2[3], n[3], norm;
  PetscReal      x1p[3], x2p[3], xnp[3];
  PetscReal      sqrtz, alpha;
  const PetscInt dim = 3;
  PetscInt       d, e, p;

  PetscFunctionBegin;
  /* 0) Calculate normal vector */
  for (d = 0; d < dim; ++d) {
    x1[d] = PetscRealPart(coords[1*dim+d] - coords[0*dim+d]);
    x2[d] = PetscRealPart(coords[2*dim+d] - coords[0*dim+d]);
  }
  n[0] = x1[1]*x2[2] - x1[2]*x2[1];
  n[1] = x1[2]*x2[0] - x1[0]*x2[2];
  n[2] = x1[0]*x2[1] - x1[1]*x2[0];
  norm = PetscSqrtReal(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  n[0] /= norm;
  n[1] /= norm;
  n[2] /= norm;
  /* 1) Take the normal vector and rotate until it is \hat z

    Let the normal vector be <nx, ny, nz> and alpha = 1/sqrt(1 - nz^2), then

    R = /  alpha nx nz  alpha ny nz -1/alpha \
        | -alpha ny     alpha nx        0    |
        \     nx            ny         nz    /

    will rotate the normal vector to \hat z
  */
  sqrtz = PetscSqrtReal(1.0 - n[2]*n[2]);
  /* Check for n = z */
  if (sqrtz < 1.0e-10) {
    const PetscInt s = PetscSign(n[2]);
    /* If nz < 0, rotate 180 degrees around x-axis */
    for (p = 3; p < coordSize/3; ++p) {
      coords[p*2+0] = PetscRealPart(coords[p*dim+0] - coords[0*dim+0]);
      coords[p*2+1] = (PetscRealPart(coords[p*dim+1] - coords[0*dim+1])) * s;
    }
    coords[0] = 0.0;
    coords[1] = 0.0;
    coords[2] = x1[0];
    coords[3] = x1[1] * s;
    coords[4] = x2[0];
    coords[5] = x2[1] * s;
    R[0] = 1.0;     R[1] = 0.0;     R[2] = 0.0;
    R[3] = 0.0;     R[4] = 1.0 * s; R[5] = 0.0;
    R[6] = 0.0;     R[7] = 0.0;     R[8] = 1.0 * s;
    PetscFunctionReturn(0);
  }
  alpha = 1.0/sqrtz;
  R[0] =  alpha*n[0]*n[2]; R[1] = alpha*n[1]*n[2]; R[2] = -sqrtz;
  R[3] = -alpha*n[1];      R[4] = alpha*n[0];      R[5] = 0.0;
  R[6] =  n[0];            R[7] = n[1];            R[8] = n[2];
  for (d = 0; d < dim; ++d) {
    x1p[d] = 0.0;
    x2p[d] = 0.0;
    for (e = 0; e < dim; ++e) {
      x1p[d] += R[d*dim+e]*x1[e];
      x2p[d] += R[d*dim+e]*x2[e];
    }
  }
  if (PetscAbsReal(x1p[2]) > 10. * PETSC_SMALL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid rotation calculated");
  if (PetscAbsReal(x2p[2]) > 10. * PETSC_SMALL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid rotation calculated");
  /* 2) Project to (x, y) */
  for (p = 3; p < coordSize/3; ++p) {
    for (d = 0; d < dim; ++d) {
      xnp[d] = 0.0;
      for (e = 0; e < dim; ++e) {
        xnp[d] += R[d*dim+e]*PetscRealPart(coords[p*dim+e] - coords[0*dim+e]);
      }
      if (d < dim-1) coords[p*2+d] = xnp[d];
    }
  }
  coords[0] = 0.0;
  coords[1] = 0.0;
  coords[2] = x1p[0];
  coords[3] = x1p[1];
  coords[4] = x2p[0];
  coords[5] = x2p[1];
  /* Output R^T which rotates \hat z to the input normal */
  for (d = 0; d < dim; ++d) {
    for (e = d+1; e < dim; ++e) {
      PetscReal tmp;

      tmp        = R[d*dim+e];
      R[d*dim+e] = R[e*dim+d];
      R[e*dim+d] = tmp;
    }
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED
PETSC_STATIC_INLINE void Volume_Triangle_Internal(PetscReal *vol, PetscReal coords[])
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

PETSC_STATIC_INLINE void Volume_Triangle_Origin_Internal(PetscReal *vol, PetscReal coords[])
{
  DMPlex_Det2D_Internal(vol, coords);
  *vol *= 0.5;
}

PETSC_UNUSED
PETSC_STATIC_INLINE void Volume_Tetrahedron_Internal(PetscReal *vol, PetscReal coords[])
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

PETSC_STATIC_INLINE void Volume_Tetrahedron_Origin_Internal(PetscReal *vol, PetscReal coords[])
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(coordSection,e,&dim);CHKERRQ(ierr);
  if (!dim) PetscFunctionReturn(0);
  ierr = PetscSectionGetOffset(coordSection,e,&off);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates,&coords);CHKERRQ(ierr);
  if (v0) {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[off + d]);}
  ierr = VecRestoreArrayRead(coordinates,&coords);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection,&pStart,&pEnd);CHKERRQ(ierr);
  if (e >= pStart && e < pEnd) {ierr = PetscSectionGetDof(coordSection,e,&numSelfCoords);CHKERRQ(ierr);}
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  numCoords = numSelfCoords ? numSelfCoords : numCoords;
  if (invJ && !J) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In order to compute invJ, J must not be NULL");
  *detJ = 0.0;
  if (numCoords == 6) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    ierr = DMPlexComputeProjection3Dto1D(coords, R);CHKERRQ(ierr);
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
    ierr = DMPlexComputeProjection2Dto1D(coords, R);CHKERRQ(ierr);
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
      ierr = PetscLogFlops(2.0);CHKERRQ(ierr);
      if (invJ) {invJ[0] = 1.0/J[0]; ierr = PetscLogFlops(1.0);CHKERRQ(ierr);}
    }
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this segment is %D != 2", numCoords);
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeTriangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, numSelfCoords = 0, d, f, g, pStart, pEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection,&pStart,&pEnd);CHKERRQ(ierr);
  if (e >= pStart && e < pEnd) {ierr = PetscSectionGetDof(coordSection,e,&numSelfCoords);CHKERRQ(ierr);}
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  numCoords = numSelfCoords ? numSelfCoords : numCoords;
  *detJ = 0.0;
  if (numCoords == 9) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    ierr = DMPlexComputeProjection3Dto2D(numCoords, coords, R);CHKERRQ(ierr);
    if (J)    {
      const PetscInt pdim = 2;

      for (d = 0; d < pdim; d++) {
        for (f = 0; f < pdim; f++) {
          J0[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*pdim+d]) - PetscRealPart(coords[0*pdim+d]));
        }
      }
      ierr = PetscLogFlops(8.0);CHKERRQ(ierr);
      DMPlex_Det3D_Internal(detJ, J0);
      for (d = 0; d < dim; d++) {
        for (f = 0; f < dim; f++) {
          J[d*dim+f] = 0.0;
          for (g = 0; g < dim; g++) {
            J[d*dim+f] += R[d*dim+g]*J0[g*dim+f];
          }
        }
      }
      ierr = PetscLogFlops(18.0);CHKERRQ(ierr);
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
      ierr = PetscLogFlops(8.0);CHKERRQ(ierr);
      DMPlex_Det2D_Internal(detJ, J);
    }
    if (invJ) {DMPlex_Invert2D_Internal(invJ, J, *detJ);}
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this triangle is %D != 6 or 9", numCoords);
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeRectangleGeometry_Internal(DM dm, PetscInt e, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, numSelfCoords = 0, d, f, g, pStart, pEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection,&pStart,&pEnd);CHKERRQ(ierr);
  if (e >= pStart && e < pEnd) {ierr = PetscSectionGetDof(coordSection,e,&numSelfCoords);CHKERRQ(ierr);}
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  numCoords = numSelfCoords ? numSelfCoords : numCoords;
  if (!Nq) {
    *detJ = 0.0;
    if (numCoords == 12) {
      const PetscInt dim = 3;
      PetscReal      R[9], J0[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};

      if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
      ierr = DMPlexComputeProjection3Dto2D(numCoords, coords, R);CHKERRQ(ierr);
      if (J)    {
        const PetscInt pdim = 2;

        for (d = 0; d < pdim; d++) {
          J0[d*dim+0] = 0.5*(PetscRealPart(coords[1*pdim+d]) - PetscRealPart(coords[0*pdim+d]));
          J0[d*dim+1] = 0.5*(PetscRealPart(coords[2*pdim+d]) - PetscRealPart(coords[1*pdim+d]));
        }
        ierr = PetscLogFlops(8.0);CHKERRQ(ierr);
        DMPlex_Det3D_Internal(detJ, J0);
        for (d = 0; d < dim; d++) {
          for (f = 0; f < dim; f++) {
            J[d*dim+f] = 0.0;
            for (g = 0; g < dim; g++) {
              J[d*dim+f] += R[d*dim+g]*J0[g*dim+f];
            }
          }
        }
        ierr = PetscLogFlops(18.0);CHKERRQ(ierr);
      }
      if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
    } else if (numCoords == 8) {
      const PetscInt dim = 2;

      if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
      if (J)    {
        for (d = 0; d < dim; d++) {
          J[d*dim+0] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
          J[d*dim+1] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
        }
        ierr = PetscLogFlops(8.0);CHKERRQ(ierr);
        DMPlex_Det2D_Internal(detJ, J);
      }
      if (invJ) {DMPlex_Invert2D_Internal(invJ, J, *detJ);}
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %D != 8 or 12", numCoords);
  } else {
    const PetscInt Nv = 4;
    const PetscInt dimR = 2;
    const PetscInt zToPlex[4] = {0, 1, 3, 2};
    PetscReal zOrder[12];
    PetscReal zCoeff[12];
    PetscInt  i, j, k, l, dim;

    if (numCoords == 12) {
      dim = 3;
    } else if (numCoords == 8) {
      dim = 2;
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %D != 8 or 12", numCoords);
    for (i = 0; i < Nv; i++) {
      PetscInt zi = zToPlex[i];

      for (j = 0; j < dim; j++) {
        zOrder[dim * i + j] = PetscRealPart(coords[dim * zi + j]);
      }
    }
    for (j = 0; j < dim; j++) {
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
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeTetrahedronGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  const PetscInt dim = 3;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  *detJ = 0.0;
  if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
  if (J)    {
    for (d = 0; d < dim; d++) {
      /* I orient with outward face normals */
      J[d*dim+0] = 0.5*(PetscRealPart(coords[2*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+1] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+2] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
    }
    ierr = PetscLogFlops(18.0);CHKERRQ(ierr);
    DMPlex_Det3D_Internal(detJ, J);
  }
  if (invJ) {DMPlex_Invert3D_Internal(invJ, J, *detJ);}
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeHexahedronGeometry_Internal(DM dm, PetscInt e, PetscInt Nq, const PetscReal points[], PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  const PetscInt dim = 3;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  if (!Nq) {
    *detJ = 0.0;
    if (v)   {for (d = 0; d < dim; d++) v[d] = PetscRealPart(coords[d]);}
    if (J)    {
      for (d = 0; d < dim; d++) {
        J[d*dim+0] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+1] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+2] = 0.5*(PetscRealPart(coords[4*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
      ierr = PetscLogFlops(18.0);CHKERRQ(ierr);
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
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeCellGeometryFEM_Implicit(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal *v, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  PetscInt        depth, dim, coordDim, coneSize, i;
  PetscInt        Nq = 0;
  const PetscReal *points = NULL;
  DMLabel         depthLabel;
  PetscReal       xi0[3] = {-1.,-1.,-1.}, v0[3], J0[9], detJ0;
  PetscBool       isAffine = PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetValue(depthLabel, cell, &dim);CHKERRQ(ierr);
  if (depth == 1 && dim == 1) {
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  }
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  if (coordDim > 3) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported coordinate dimension %D > 3", coordDim);
  if (quad) {ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &points, NULL);CHKERRQ(ierr);}
  switch (dim) {
  case 0:
    ierr = DMPlexComputePointGeometry_Internal(dm, cell, v, J, invJ, detJ);CHKERRQ(ierr);
    isAffine = PETSC_FALSE;
    break;
  case 1:
    if (Nq) {
      ierr = DMPlexComputeLineGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0);CHKERRQ(ierr);
    } else {
      ierr = DMPlexComputeLineGeometry_Internal(dm, cell, v, J, invJ, detJ);CHKERRQ(ierr);
    }
    break;
  case 2:
    switch (coneSize) {
    case 3:
      if (Nq) {
        ierr = DMPlexComputeTriangleGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0);CHKERRQ(ierr);
      } else {
        ierr = DMPlexComputeTriangleGeometry_Internal(dm, cell, v, J, invJ, detJ);CHKERRQ(ierr);
      }
      break;
    case 4:
      ierr = DMPlexComputeRectangleGeometry_Internal(dm, cell, Nq, points, v, J, invJ, detJ);CHKERRQ(ierr);
      isAffine = PETSC_FALSE;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported number of faces %D in cell %D for element geometry computation", coneSize, cell);
    }
    break;
  case 3:
    switch (coneSize) {
    case 4:
      if (Nq) {
        ierr = DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v0, J0, NULL, &detJ0);CHKERRQ(ierr);
      } else {
        ierr = DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v, J, invJ, detJ);CHKERRQ(ierr);
      }
      break;
    case 6: /* Faces */
    case 8: /* Vertices */
      ierr = DMPlexComputeHexahedronGeometry_Internal(dm, cell, Nq, points, v, J, invJ, detJ);CHKERRQ(ierr);
      isAffine = PETSC_FALSE;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported number of faces %D in cell %D for element geometry computation", coneSize, cell);
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %D for element geometry computation", dim);
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

  Input Arguments:
+ dm   - the DM
- cell - the cell

  Output Arguments:
+ v0   - the translation part of this affine transform
. J    - the Jacobian of the transform from the reference element
. invJ - the inverse of the Jacobian
- detJ - the Jacobian determinant

  Level: advanced

  Fortran Notes:
  Since it returns arrays, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMPlexComputeCellGeometryFEM(), DMGetCoordinateSection(), DMGetCoordinates()
@*/
PetscErrorCode DMPlexComputeCellGeometryAffineFEM(DM dm, PetscInt cell, PetscReal *v0, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFEM_Implicit(dm,cell,NULL,v0,J,invJ,detJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeCellGeometryFEM_FE(DM dm, PetscFE fe, PetscInt point, PetscQuadrature quad, PetscReal v[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscQuadrature  feQuad;
  PetscSection     coordSection;
  Vec              coordinates;
  PetscScalar     *coords = NULL;
  const PetscReal *quadPoints;
  PetscReal       *basisDer, *basis, detJt;
  PetscInt         dim, cdim, pdim, qdim, Nq, numCoords, q;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, point, &numCoords, &coords);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  if (!quad) { /* use the first point of the first functional of the dual space */
    PetscDualSpace dsp;

    ierr = PetscFEGetDualSpace(fe, &dsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetFunctional(dsp, 0, &quad);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad, &qdim, NULL, &Nq, &quadPoints, NULL);CHKERRQ(ierr);
    Nq = 1;
  } else {
    ierr = PetscQuadratureGetData(quad, &qdim, NULL, &Nq, &quadPoints, NULL);CHKERRQ(ierr);
  }
  ierr = PetscFEGetDimension(fe, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &feQuad);CHKERRQ(ierr);
  if (feQuad == quad) {
    ierr = PetscFEGetDefaultTabulation(fe, &basis, J ? &basisDer : NULL, NULL);CHKERRQ(ierr);
    if (numCoords != pdim*cdim) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "There are %d coordinates for point %d != %d*%d", numCoords, point, pdim, cdim);
  } else {
    ierr = PetscFEGetTabulation(fe, Nq, quadPoints, &basis, J ? &basisDer : NULL, NULL);CHKERRQ(ierr);
  }
  if (qdim != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Point dimension %d != quadrature dimension %d", dim, qdim);
  if (v) {
    ierr = PetscArrayzero(v, Nq*cdim);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscInt i, k;

      for (k = 0; k < pdim; ++k)
        for (i = 0; i < cdim; ++i)
          v[q*cdim + i] += basis[q*pdim + k] * PetscRealPart(coords[k*cdim + i]);
      ierr = PetscLogFlops(2.0*pdim*cdim);CHKERRQ(ierr);
    }
  }
  if (J) {
    ierr = PetscArrayzero(J, Nq*cdim*cdim);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscInt i, j, k, c, r;

      /* J = dx_i/d\xi_j = sum[k=0,n-1] dN_k/d\xi_j * x_i(k) */
      for (k = 0; k < pdim; ++k)
        for (j = 0; j < dim; ++j)
          for (i = 0; i < cdim; ++i)
            J[(q*cdim + i)*cdim + j] += basisDer[(q*pdim + k)*dim + j] * PetscRealPart(coords[k*cdim + i]);
      ierr = PetscLogFlops(2.0*pdim*dim*cdim);CHKERRQ(ierr);
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
  }
  else if (detJ || invJ) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Need J to compute invJ or detJ");
  if (feQuad != quad) {
    ierr = PetscFERestoreTabulation(fe, Nq, quadPoints, &basis, J ? &basisDer : NULL, NULL);CHKERRQ(ierr);
  }
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, point, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeCellGeometryFEM - Compute the Jacobian, inverse Jacobian, and Jacobian determinant at each quadrature point in the given cell

  Collective on dm

  Input Arguments:
+ dm   - the DM
. cell - the cell
- quad - the quadrature containing the points in the reference element where the geometry will be evaluated.  If quad == NULL, geometry will be
         evaluated at the first vertex of the reference element

  Output Arguments:
+ v    - the image of the transformed quadrature points, otherwise the image of the first vertex in the closure of the reference element
. J    - the Jacobian of the transform from the reference element at each quadrature point
. invJ - the inverse of the Jacobian at each quadrature point
- detJ - the Jacobian determinant at each quadrature point

  Level: advanced

  Fortran Notes:
  Since it returns arrays, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMGetCoordinateSection(), DMGetCoordinates()
@*/
PetscErrorCode DMPlexComputeCellGeometryFEM(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal *v, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  DM             cdm;
  PetscFE        fe = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(detJ, 7);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  if (cdm) {
    PetscClassId id;
    PetscInt     numFields;
    PetscDS      prob;
    PetscObject  disc;

    ierr = DMGetNumFields(cdm, &numFields);CHKERRQ(ierr);
    if (numFields) {
      ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
      ierr = PetscDSGetDiscretization(prob,0,&disc);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
      }
    }
  }
  if (!fe) {ierr = DMPlexComputeCellGeometryFEM_Implicit(dm, cell, quad, v, J, invJ, detJ);CHKERRQ(ierr);}
  else     {ierr = DMPlexComputeCellGeometryFEM_FE(dm, fe, cell, quad, v, J, invJ, detJ);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeGeometryFVM_1D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscScalar    tmp[2];
  PetscInt       coordSize, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  ierr = DMLocalizeCoordinate_Internal(dm, dim, coords, &coords[dim], tmp);CHKERRQ(ierr);
  if (centroid) {
    for (d = 0; d < dim; ++d) centroid[d] = 0.5*PetscRealPart(coords[d] + tmp[d]);
  }
  if (normal) {
    PetscReal norm;

    if (dim != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "We only support 2D edges right now");
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
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Centroid_i = (\sum_n A_n Cn_i ) / A */
static PetscErrorCode DMPlexComputeGeometryFVM_2D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  DMLabel        depth;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscReal      vsum = 0.0, csum[3] = {0.0, 0.0, 0.0}, vtmp, ctmp[4], v0[3], R[9];
  PetscBool      isHybrid = PETSC_FALSE;
  PetscInt       fv[4] = {0, 1, 2, 3};
  PetscInt       pEndInterior[4], cdepth, tdim = 2, coordSize, numCorners, p, d, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Must check for hybrid cells because prisms have a different orientation scheme */
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMLabelGetValue(depth, cell, &cdepth);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pEndInterior[3], &pEndInterior[2], &pEndInterior[1], &pEndInterior[0]);CHKERRQ(ierr);
  if ((pEndInterior[cdepth]) >=0 && (cell >= pEndInterior[cdepth])) isHybrid = PETSC_TRUE;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cell, &numCorners);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  /* Side faces for hybrid cells are are stored as tensor products */
  if (isHybrid && numCorners == 4) {fv[2] = 3; fv[3] = 2;}

  if (dim > 2 && centroid) {
    v0[0] = PetscRealPart(coords[0]);
    v0[1] = PetscRealPart(coords[1]);
    v0[2] = PetscRealPart(coords[2]);
  }
  if (normal) {
    if (dim > 2) {
      const PetscReal x0 = PetscRealPart(coords[dim*fv[1]+0] - coords[0]), x1 = PetscRealPart(coords[dim*fv[2]+0] - coords[0]);
      const PetscReal y0 = PetscRealPart(coords[dim*fv[1]+1] - coords[1]), y1 = PetscRealPart(coords[dim*fv[2]+1] - coords[1]);
      const PetscReal z0 = PetscRealPart(coords[dim*fv[1]+2] - coords[2]), z1 = PetscRealPart(coords[dim*fv[2]+2] - coords[2]);
      PetscReal       norm;

      normal[0] = y0*z1 - z0*y1;
      normal[1] = z0*x1 - x0*z1;
      normal[2] = x0*y1 - y0*x1;
      norm = PetscSqrtReal(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
      normal[0] /= norm;
      normal[1] /= norm;
      normal[2] /= norm;
    } else {
      for (d = 0; d < dim; ++d) normal[d] = 0.0;
    }
  }
  if (dim == 3) {ierr = DMPlexComputeProjection3Dto2D(coordSize, coords, R);CHKERRQ(ierr);}
  for (p = 0; p < numCorners; ++p) {
    const PetscInt pi  = p < 4 ? fv[p] : p;
    const PetscInt pin = p < 3 ? fv[(p+1)%numCorners] : (p+1)%numCorners;
    /* Need to do this copy to get types right */
    for (d = 0; d < tdim; ++d) {
      ctmp[d]      = PetscRealPart(coords[pi*tdim+d]);
      ctmp[tdim+d] = PetscRealPart(coords[pin*tdim+d]);
    }
    Volume_Triangle_Origin_Internal(&vtmp, ctmp);
    vsum += vtmp;
    for (d = 0; d < tdim; ++d) {
      csum[d] += (ctmp[d] + ctmp[tdim+d])*vtmp;
    }
  }
  for (d = 0; d < tdim; ++d) {
    csum[d] /= (tdim+1)*vsum;
  }
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  if (vol) *vol = PetscAbsReal(vsum);
  if (centroid) {
    if (dim > 2) {
      for (d = 0; d < dim; ++d) {
        centroid[d] = v0[d];
        for (e = 0; e < dim; ++e) {
          centroid[d] += R[d*dim+e]*csum[e];
        }
      }
    } else for (d = 0; d < dim; ++d) centroid[d] = csum[d];
  }
  PetscFunctionReturn(0);
}

/* Centroid_i = (\sum_n V_n Cn_i ) / V */
static PetscErrorCode DMPlexComputeGeometryFVM_3D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  DMLabel         depth;
  PetscSection    coordSection;
  Vec             coordinates;
  PetscScalar    *coords = NULL;
  PetscReal       vsum = 0.0, vtmp, coordsTmp[3*3];
  const PetscInt *faces, *facesO;
  PetscBool       isHybrid = PETSC_FALSE;
  PetscInt        pEndInterior[4], cdepth, numFaces, f, coordSize, numCorners, p, d;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (PetscUnlikely(dim > 3)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No support for dim %D > 3",dim);
  /* Must check for hybrid cells because prisms have a different orientation scheme */
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMLabelGetValue(depth, cell, &cdepth);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pEndInterior[3], &pEndInterior[2], &pEndInterior[1], &pEndInterior[0]);CHKERRQ(ierr);
  if ((pEndInterior[cdepth]) >=0 && (cell >= pEndInterior[cdepth])) isHybrid = PETSC_TRUE;

  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);

  if (centroid) for (d = 0; d < dim; ++d) centroid[d] = 0.0;
  ierr = DMPlexGetConeSize(dm, cell, &numFaces);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, cell, &facesO);CHKERRQ(ierr);
  for (f = 0; f < numFaces; ++f) {
    PetscBool flip = isHybrid && f == 0 ? PETSC_TRUE : PETSC_FALSE; /* The first hybrid face is reversed */

    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, faces[f], &coordSize, &coords);CHKERRQ(ierr);
    numCorners = coordSize/dim;
    switch (numCorners) {
    case 3:
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[0*dim+d]);
        coordsTmp[1*dim+d] = PetscRealPart(coords[1*dim+d]);
        coordsTmp[2*dim+d] = PetscRealPart(coords[2*dim+d]);
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
    case 4:
    {
      PetscInt fv[4] = {0, 1, 2, 3};

      /* Side faces for hybrid cells are are stored as tensor products */
      if (isHybrid && f > 1) {fv[2] = 3; fv[3] = 2;}
      /* DO FOR PYRAMID */
      /* First tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[fv[0]*dim+d]);
        coordsTmp[1*dim+d] = PetscRealPart(coords[fv[1]*dim+d]);
        coordsTmp[2*dim+d] = PetscRealPart(coords[fv[3]*dim+d]);
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
        coordsTmp[0*dim+d] = PetscRealPart(coords[fv[1]*dim+d]);
        coordsTmp[1*dim+d] = PetscRealPart(coords[fv[2]*dim+d]);
        coordsTmp[2*dim+d] = PetscRealPart(coords[fv[3]*dim+d]);
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
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle faces with %D vertices", numCorners);
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, faces[f], &coordSize, &coords);CHKERRQ(ierr);
  }
  if (vol)     *vol = PetscAbsReal(vsum);
  if (normal)   for (d = 0; d < dim; ++d) normal[d]    = 0.0;
  if (centroid) for (d = 0; d < dim; ++d) centroid[d] /= (vsum*4);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeCellGeometryFVM - Compute the volume for a given cell

  Collective on dm

  Input Arguments:
+ dm   - the DM
- cell - the cell

  Output Arguments:
+ volume   - the cell volume
. centroid - the cell centroid
- normal - the cell normal, if appropriate

  Level: advanced

  Fortran Notes:
  Since it returns arrays, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMGetCoordinateSection(), DMGetCoordinates()
@*/
PetscErrorCode DMPlexComputeCellGeometryFVM(DM dm, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscInt       depth, dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (depth != dim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated");
  /* We need to keep a pointer to the depth label */
  ierr = DMGetLabelValue(dm, "depth", cell, &depth);CHKERRQ(ierr);
  /* Cone size is now the number of faces */
  switch (depth) {
  case 1:
    ierr = DMPlexComputeGeometryFVM_1D_Internal(dm, dim, cell, vol, centroid, normal);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMPlexComputeGeometryFVM_2D_Internal(dm, dim, cell, vol, centroid, normal);CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMPlexComputeGeometryFVM_3D_Internal(dm, dim, cell, vol, centroid, normal);CHKERRQ(ierr);
    break;
  default:
    SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %D (depth %D) for element geometry computation", dim, depth);
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
  PetscInt       cStart, cEnd, cMax, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone(dm, &dmCell);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dmCell, PETSC_DETERMINE, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmCell, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cMax < 0 ? cEnd : cMax;
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  /* TODO This needs to be multiplied by Nq for non-affine */
  for (c = cStart; c < cEnd; ++c) {ierr = PetscSectionSetDof(sectionCell, c, (PetscInt) PetscCeilReal(((PetscReal) sizeof(PetscFEGeom))/sizeof(PetscScalar)));CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmCell, cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscFEGeom *cg;

    ierr = DMPlexPointLocalRef(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = PetscArrayzero(cg, 1);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFEM(dmCell, c, NULL, cg->v, cg->J, cg->invJ, cg->detJ);CHKERRQ(ierr);
    if (*cg->detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D", (double) *cg->detJ, c);
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

.seealso: PetscFVFaceGeom, PetscFVCellGeom, DMPlexComputeGeometryFEM()
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  /* Make cell centroids and volumes */
  ierr = DMClone(dm, &dmCell);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dmCell, PETSC_DETERMINE, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmCell, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {ierr = PetscSectionSetDof(sectionCell, c, (PetscInt) PetscCeilReal(((PetscReal) sizeof(PetscFVCellGeom))/sizeof(PetscScalar)));CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmCell, cellgeom);CHKERRQ(ierr);
  if (cEndInterior < 0) cEndInterior = cEnd;
  ierr = VecGetArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; ++c) {
    PetscFVCellGeom *cg;

    ierr = DMPlexPointLocalRef(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = PetscArrayzero(cg, 1);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dmCell, c, &cg->volume, cg->centroid, NULL);CHKERRQ(ierr);
  }
  /* Compute face normals and minimum cell radius */
  ierr = DMClone(dm, &dmFace);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionFace);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionFace, fStart, fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {ierr = PetscSectionSetDof(sectionFace, f, (PetscInt) PetscCeilReal(((PetscReal) sizeof(PetscFVFaceGeom))/sizeof(PetscScalar)));CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionFace);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dmFace, sectionFace);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionFace);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmFace, facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(*facegeom, &fgeom);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  minradius = PETSC_MAX_REAL;
  for (f = fStart; f < fEnd; ++f) {
    PetscFVFaceGeom *fg;
    PetscReal        area;
    PetscInt         ghost = -1, d, numChildren;

    if (ghostLabel) {ierr = DMLabelGetValue(ghostLabel, f, &ghost);CHKERRQ(ierr);}
    ierr = DMPlexGetTreeChildren(dm,f,&numChildren,NULL);CHKERRQ(ierr);
    if (ghost >= 0 || numChildren) continue;
    ierr = DMPlexPointLocalRef(dmFace, f, fgeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dm, f, &area, fg->centroid, fg->normal);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) fg->normal[d] *= area;
    /* Flip face orientation if necessary to match ordering in support, and Update minimum radius */
    {
      PetscFVCellGeom *cL, *cR;
      PetscInt         ncells;
      const PetscInt  *cells;
      PetscReal       *lcentroid, *rcentroid;
      PetscReal        l[3], r[3], v[3];

      ierr = DMPlexGetSupport(dm, f, &cells);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &ncells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[0], cgeom, &cL);CHKERRQ(ierr);
      lcentroid = cells[0] >= cEndInterior ? fg->centroid : cL->centroid;
      if (ncells > 1) {
        ierr = DMPlexPointLocalRead(dmCell, cells[1], cgeom, &cR);CHKERRQ(ierr);
        rcentroid = cells[1] >= cEndInterior ? fg->centroid : cR->centroid;
      }
      else {
        rcentroid = fg->centroid;
      }
      ierr = DMLocalizeCoordinateReal_Internal(dm, dim, fg->centroid, lcentroid, l);CHKERRQ(ierr);
      ierr = DMLocalizeCoordinateReal_Internal(dm, dim, fg->centroid, rcentroid, r);CHKERRQ(ierr);
      DMPlex_WaxpyD_Internal(dim, -1, l, r, v);
      if (DMPlex_DotRealD_Internal(dim, fg->normal, v) < 0) {
        for (d = 0; d < dim; ++d) fg->normal[d] = -fg->normal[d];
      }
      if (DMPlex_DotRealD_Internal(dim, fg->normal, v) <= 0) {
        if (dim == 2) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g) v (%g,%g)", f, (double) fg->normal[0], (double) fg->normal[1], (double) v[0], (double) v[1]);
        if (dim == 3) SETERRQ7(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", f, (double) fg->normal[0], (double) fg->normal[1], (double) fg->normal[2], (double) v[0], (double) v[1], (double) v[2]);
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed", f);
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
  ierr = MPIU_Allreduce(&minradius, &gminradius, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = DMPlexSetMinRadius(dm, gminradius);CHKERRQ(ierr);
  /* Compute centroids of ghost cells */
  for (c = cEndInterior; c < cEnd; ++c) {
    PetscFVFaceGeom *fg;
    const PetscInt  *cone,    *support;
    PetscInt         coneSize, supportSize, s;

    ierr = DMPlexGetConeSize(dmCell, c, &coneSize);CHKERRQ(ierr);
    if (coneSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %d has cone size %d != 1", c, coneSize);
    ierr = DMPlexGetCone(dmCell, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmCell, cone[0], &supportSize);CHKERRQ(ierr);
    if (supportSize != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d has support size %d != 2", cone[0], supportSize);
    ierr = DMPlexGetSupport(dmCell, cone[0], &support);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dmFace, cone[0], fgeom, &fg);CHKERRQ(ierr);
    for (s = 0; s < 2; ++s) {
      /* Reflect ghost centroid across plane of face */
      if (support[s] == c) {
        PetscFVCellGeom       *ci;
        PetscFVCellGeom       *cg;
        PetscReal              c2f[3], a;

        ierr = DMPlexPointLocalRead(dmCell, support[(s+1)%2], cgeom, &ci);CHKERRQ(ierr);
        DMPlex_WaxpyD_Internal(dim, -1, ci->centroid, fg->centroid, c2f); /* cell to face centroid */
        a    = DMPlex_DotRealD_Internal(dim, c2f, fg->normal)/DMPlex_DotRealD_Internal(dim, fg->normal, fg->normal);
        ierr = DMPlexPointLocalRef(dmCell, support[s], cgeom, &cg);CHKERRQ(ierr);
        DMPlex_WaxpyD_Internal(dim, 2*a, fg->normal, ci->centroid, cg->centroid);
        cg->volume = ci->volume;
      }
    }
  }
  ierr = VecRestoreArray(*facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  ierr = DMDestroy(&dmFace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetMinRadius - Returns the minimum distance from any cell centroid to a face

  Not collective

  Input Argument:
. dm - the DM

  Output Argument:
. minradius - the minium cell radius

  Level: developer

.seealso: DMGetCoordinates()
@*/
PetscErrorCode DMPlexGetMinRadius(DM dm, PetscReal *minradius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(minradius,2);
  *minradius = ((DM_Plex*) dm->data)->minradius;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexSetMinRadius - Sets the minimum distance from the cell centroid to a face

  Logically collective

  Input Arguments:
+ dm - the DM
- minradius - the minium cell radius

  Level: developer

.seealso: DMSetCoordinates()
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxNumFaces, NULL);CHKERRQ(ierr);
  ierr = PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = PetscMalloc3(maxNumFaces*dim, &dx, maxNumFaces*dim, &grad, maxNumFaces, &gref);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; c++) {
    const PetscInt        *faces;
    PetscInt               numFaces, usedFaces, f, d;
    PetscFVCellGeom        *cg;
    PetscBool              boundary;
    PetscInt               ghost;

    ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, c, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);CHKERRQ(ierr);
    if (numFaces < dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D has only %D faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      PetscFVCellGeom       *cg1;
      PetscFVFaceGeom       *fg;
      const PetscInt        *fcells;
      PetscInt               ncell, side;

      ierr = DMLabelGetValue(ghostLabel, faces[f], &ghost);CHKERRQ(ierr);
      ierr = DMIsBoundaryPoint(dm, faces[f], &boundary);CHKERRQ(ierr);
      if ((ghost >= 0) || boundary) continue;
      ierr  = DMPlexGetSupport(dm, faces[f], &fcells);CHKERRQ(ierr);
      side  = (c != fcells[0]); /* c is on left=0 or right=1 of face */
      ncell = fcells[!side];    /* the neighbor */
      ierr  = DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg);CHKERRQ(ierr);
      ierr  = DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) dx[usedFaces*dim+d] = cg1->centroid[d] - cg->centroid[d];
      gref[usedFaces++] = fg->grad[side];  /* Gradient reconstruction term will go here */
    }
    if (!usedFaces) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
    ierr = PetscFVComputeGradient(fvm, usedFaces, dx, grad);CHKERRQ(ierr);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      ierr = DMLabelGetValue(ghostLabel, faces[f], &ghost);CHKERRQ(ierr);
      ierr = DMIsBoundaryPoint(dm, faces[f], &boundary);CHKERRQ(ierr);
      if ((ghost >= 0) || boundary) continue;
      for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces*dim+d];
      ++usedFaces;
    }
  }
  ierr = PetscFree3(dx, grad, gref);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
  if (cEndInterior < 0) cEndInterior = cEnd;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&neighSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(neighSec,cStart,cEndInterior);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; f++) {
    const PetscInt        *fcells;
    PetscBool              boundary;
    PetscInt               ghost = -1;
    PetscInt               numChildren, numCells, c;

    if (ghostLabel) {ierr = DMLabelGetValue(ghostLabel, f, &ghost);CHKERRQ(ierr);}
    ierr = DMIsBoundaryPoint(dm, f, &boundary);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, f, &numChildren, NULL);CHKERRQ(ierr);
    if ((ghost >= 0) || boundary || numChildren) continue;
    ierr = DMPlexGetSupportSize(dm, f, &numCells);CHKERRQ(ierr);
    if (numCells == 2) {
      ierr = DMPlexGetSupport(dm, f, &fcells);CHKERRQ(ierr);
      for (c = 0; c < 2; c++) {
        PetscInt cell = fcells[c];

        if (cell >= cStart && cell < cEndInterior) {
          ierr = PetscSectionAddDof(neighSec,cell,1);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscSectionSetUp(neighSec);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(neighSec,&maxNumFaces);CHKERRQ(ierr);
  ierr = PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces);CHKERRQ(ierr);
  nStart = 0;
  ierr = PetscSectionGetStorageSize(neighSec,&nEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1((nEnd-nStart),&neighbors);CHKERRQ(ierr);
  ierr = PetscCalloc1((cEndInterior-cStart),&counter);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; f++) {
    const PetscInt        *fcells;
    PetscBool              boundary;
    PetscInt               ghost = -1;
    PetscInt               numChildren, numCells, c;

    if (ghostLabel) {ierr = DMLabelGetValue(ghostLabel, f, &ghost);CHKERRQ(ierr);}
    ierr = DMIsBoundaryPoint(dm, f, &boundary);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, f, &numChildren, NULL);CHKERRQ(ierr);
    if ((ghost >= 0) || boundary || numChildren) continue;
    ierr = DMPlexGetSupportSize(dm, f, &numCells);CHKERRQ(ierr);
    if (numCells == 2) {
      ierr  = DMPlexGetSupport(dm, f, &fcells);CHKERRQ(ierr);
      for (c = 0; c < 2; c++) {
        PetscInt cell = fcells[c], off;

        if (cell >= cStart && cell < cEndInterior) {
          ierr = PetscSectionGetOffset(neighSec,cell,&off);CHKERRQ(ierr);
          off += counter[cell - cStart]++;
          neighbors[off][0] = f;
          neighbors[off][1] = fcells[1 - c];
        }
      }
    }
  }
  ierr = PetscFree(counter);CHKERRQ(ierr);
  ierr = PetscMalloc3(maxNumFaces*dim, &dx, maxNumFaces*dim, &grad, maxNumFaces, &gref);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; c++) {
    PetscInt               numFaces, f, d, off, ghost = -1;
    PetscFVCellGeom        *cg;

    ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(neighSec, c, &numFaces);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(neighSec, c, &off);CHKERRQ(ierr);
    if (ghostLabel) {ierr = DMLabelGetValue(ghostLabel, c, &ghost);CHKERRQ(ierr);}
    if (ghost < 0 && numFaces < dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D has only %D faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0; f < numFaces; ++f) {
      PetscFVCellGeom       *cg1;
      PetscFVFaceGeom       *fg;
      const PetscInt        *fcells;
      PetscInt               ncell, side, nface;

      nface = neighbors[off + f][0];
      ncell = neighbors[off + f][1];
      ierr  = DMPlexGetSupport(dm,nface,&fcells);CHKERRQ(ierr);
      side  = (c != fcells[0]);
      ierr  = DMPlexPointLocalRef(dmFace, nface, fgeom, &fg);CHKERRQ(ierr);
      ierr  = DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) dx[f*dim+d] = cg1->centroid[d] - cg->centroid[d];
      gref[f] = fg->grad[side];  /* Gradient reconstruction term will go here */
    }
    ierr = PetscFVComputeGradient(fvm, numFaces, dx, grad);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      for (d = 0; d < dim; ++d) gref[f][d] = grad[f*dim+d];
    }
  }
  ierr = PetscFree3(dx, grad, gref);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&neighSec);CHKERRQ(ierr);
  ierr = PetscFree(neighbors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeGradientFVM - Compute geometric factors for gradient reconstruction, which are stored in the geometry data, and compute layout for gradient data

  Collective on dm

  Input Arguments:
+ dm  - The DM
. fvm - The PetscFV
. faceGeometry - The face geometry from DMPlexComputeFaceGeometryFVM()
- cellGeometry - The face geometry from DMPlexComputeCellGeometryFVM()

  Output Parameters:
+ faceGeometry - The geometric factors for gradient calculation are inserted
- dmGrad - The DM describing the layout of gradient data

  Level: developer

.seealso: DMPlexGetFaceGeometryFVM(), DMPlexGetCellGeometryFVM()
@*/
PetscErrorCode DMPlexComputeGradientFVM(DM dm, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM *dmGrad)
{
  DM             dmFace, dmCell;
  PetscScalar   *fgeom, *cgeom;
  PetscSection   sectionGrad, parentSection;
  PetscInt       dim, pdim, cStart, cEnd, cEndInterior, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
  /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArray(faceGeometry, &fgeom);CHKERRQ(ierr);
  ierr = VecGetArray(cellGeometry, &cgeom);CHKERRQ(ierr);
  ierr = DMPlexGetTree(dm,&parentSection,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!parentSection) {
    ierr = BuildGradientReconstruction_Internal(dm, fvm, dmFace, fgeom, dmCell, cgeom);CHKERRQ(ierr);
  } else {
    ierr = BuildGradientReconstruction_Internal_Tree(dm, fvm, dmFace, fgeom, dmCell, cgeom);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(faceGeometry, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(cellGeometry, &cgeom);CHKERRQ(ierr);
  /* Create storage for gradients */
  ierr = DMClone(dm, dmGrad);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionGrad);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionGrad, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {ierr = PetscSectionSetDof(sectionGrad, c, pdim*dim);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionGrad);CHKERRQ(ierr);
  ierr = DMSetLocalSection(*dmGrad, sectionGrad);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetDataFVM - Retrieve precomputed cell geometry

  Collective on dm

  Input Arguments:
+ dm  - The DM
- fvm - The PetscFV

  Output Parameters:
+ cellGeometry - The cell geometry
. faceGeometry - The face geometry
- dmGrad       - The gradient matrices

  Level: developer

.seealso: DMPlexComputeGeometryFVM()
@*/
PetscErrorCode DMPlexGetDataFVM(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM)
{
  PetscObject    cellgeomobj, facegeomobj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj);CHKERRQ(ierr);
  if (!cellgeomobj) {
    Vec cellgeomInt, facegeomInt;

    ierr = DMPlexComputeGeometryFVM(dm, &cellgeomInt, &facegeomInt);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dm, "DMPlex_cellgeom_fvm",(PetscObject)cellgeomInt);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dm, "DMPlex_facegeom_fvm",(PetscObject)facegeomInt);CHKERRQ(ierr);
    ierr = VecDestroy(&cellgeomInt);CHKERRQ(ierr);
    ierr = VecDestroy(&facegeomInt);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj);CHKERRQ(ierr);
  }
  ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_facegeom_fvm", &facegeomobj);CHKERRQ(ierr);
  if (cellgeom) *cellgeom = (Vec) cellgeomobj;
  if (facegeom) *facegeom = (Vec) facegeomobj;
  if (gradDM) {
    PetscObject gradobj;
    PetscBool   computeGradients;

    ierr = PetscFVGetComputeGradients(fv,&computeGradients);CHKERRQ(ierr);
    if (!computeGradients) {
      *gradDM = NULL;
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_dmgrad_fvm", &gradobj);CHKERRQ(ierr);
    if (!gradobj) {
      DM dmGradInt;

      ierr = DMPlexComputeGradientFVM(dm,fv,(Vec) facegeomobj,(Vec) cellgeomobj,&dmGradInt);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "DMPlex_dmgrad_fvm", (PetscObject)dmGradInt);CHKERRQ(ierr);
      ierr = DMDestroy(&dmGradInt);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_dmgrad_fvm", &gradobj);CHKERRQ(ierr);
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
    if (info != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELS");

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar);CHKERRQ(ierr);
  if (coordSize < dimC * numV) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expecting at least %D coordinates, got %D",dimC * (1 << dimR), coordSize);
  ierr = DMGetWorkArray(dm, 2 * coordSize + dimR + dimC, MPIU_REAL, &cellData);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, 3 * dimR * dimC, MPIU_SCALAR, &J);CHKERRQ(ierr);
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
  ierr = PetscArrayzero(refCoords,numPoints * dimR);CHKERRQ(ierr);
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
#if 0 && defined(PETSC_USE_DEBUG)
      {
        PetscReal maxAbs = 0.;

        for (l = 0; l < dimC; l++) {
          maxAbs = PetscMax(maxAbs,PetscAbsReal(resNeg[l]));
        }
        ierr = PetscInfo4(dm,"cell %D, point %D, iter %D: res %g\n",cell,j,i,(double) maxAbs);CHKERRQ(ierr);
      }
#endif

      ierr = DMPlexCoordinatesToReference_NewtonUpdate(dimC,dimR,J,invJ,work,resNeg,guess);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreWorkArray(dm, 3 * dimR * dimC, MPIU_SCALAR, &J);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 2 * coordSize + dimR + dimC, MPIU_REAL, &cellData);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceToCoordinates_Tensor(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[], Vec coords, PetscInt dimC, PetscInt dimR)
{
  PetscInt       coordSize, i, j, k, l, numV = (1 << dimR);
  PetscScalar    *coordsScalar = NULL;
  PetscReal      *cellData, *cellCoords, *cellCoeffs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar);CHKERRQ(ierr);
  if (coordSize < dimC * numV) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expecting at least %D coordinates, got %D",dimC * (1 << dimR), coordSize);
  ierr = DMGetWorkArray(dm, 2 * coordSize, MPIU_REAL, &cellData);CHKERRQ(ierr);
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
  ierr = PetscArrayzero(realCoords,numPoints * dimC);CHKERRQ(ierr);
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
  ierr = DMRestoreWorkArray(dm, 2 * coordSize, MPIU_REAL, &cellData);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &coordsScalar);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetDimension(fe, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &numComp);CHKERRQ(ierr);
  if (numComp != Nc) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"coordinate discretization must have as many components (%D) as embedding dimension (!= %D)",numComp,Nc);
  ierr = DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &nodes);CHKERRQ(ierr);
  /* convert nodes to values in the stable evaluation basis */
  ierr = DMGetWorkArray(dm,pdim,MPIU_REAL,&modes);CHKERRQ(ierr);
  invV = fe->invV;
  for (i = 0; i < pdim; ++i) {
    modes[i] = 0.;
    for (j = 0; j < pdim; ++j) {
      modes[i] += invV[i * pdim + j] * PetscRealPart(nodes[j]);
    }
  }
  ierr   = DMGetWorkArray(dm,pdim * Nc + pdim * Nc * dimR + Nc,MPIU_REAL,&B);CHKERRQ(ierr);
  D      = &B[pdim*Nc];
  resNeg = &D[pdim*Nc * dimR];
  ierr = DMGetWorkArray(dm,3 * Nc * dimR,MPIU_SCALAR,&J);CHKERRQ(ierr);
  invJ = &J[Nc * dimR];
  work = &invJ[Nc * dimR];
  for (i = 0; i < numPoints * dimR; i++) {refCoords[i] = 0.;}
  for (j = 0; j < numPoints; j++) {
      for (i = 0; i < maxIter; i++) { /* we could batch this so that we're not making big B and D arrays all the time */
      PetscReal *guess = &refCoords[j * dimR];
      ierr = PetscSpaceEvaluate(fe->basisSpace, 1, guess, B, D, NULL);CHKERRQ(ierr);
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
#if 0 && defined(PETSC_USE_DEBUG)
      {
        PetscReal maxAbs = 0.;

        for (l = 0; l < Nc; l++) {
          maxAbs = PetscMax(maxAbs,PetscAbsReal(resNeg[l]));
        }
        ierr = PetscInfo4(dm,"cell %D, point %D, iter %D: res %g\n",cell,j,i,(double) maxAbs);CHKERRQ(ierr);
      }
#endif
      ierr = DMPlexCoordinatesToReference_NewtonUpdate(Nc,dimR,J,invJ,work,resNeg,guess);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreWorkArray(dm,3 * Nc * dimR,MPIU_SCALAR,&J);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,pdim * Nc + pdim * Nc * dimR + Nc,MPIU_REAL,&B);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,pdim,MPIU_REAL,&modes);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &nodes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO: TOBY please fix this for Nc > 1 */
static PetscErrorCode DMPlexReferenceToCoordinates_FE(DM dm, PetscFE fe, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[], Vec coords, PetscInt Nc, PetscInt dimR)
{
  PetscInt       numComp, pdim, i, j, k, l, coordSize;
  PetscScalar    *nodes = NULL;
  PetscReal      *invV, *modes;
  PetscReal      *B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetDimension(fe, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &numComp);CHKERRQ(ierr);
  if (numComp != Nc) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"coordinate discretization must have as many components (%D) as embedding dimension (!= %D)",numComp,Nc);
  ierr = DMPlexVecGetClosure(dm, NULL, coords, cell, &coordSize, &nodes);CHKERRQ(ierr);
  /* convert nodes to values in the stable evaluation basis */
  ierr = DMGetWorkArray(dm,pdim,MPIU_REAL,&modes);CHKERRQ(ierr);
  invV = fe->invV;
  for (i = 0; i < pdim; ++i) {
    modes[i] = 0.;
    for (j = 0; j < pdim; ++j) {
      modes[i] += invV[i * pdim + j] * PetscRealPart(nodes[j]);
    }
  }
  ierr = DMGetWorkArray(dm,numPoints * pdim * Nc,MPIU_REAL,&B);CHKERRQ(ierr);
  ierr = PetscSpaceEvaluate(fe->basisSpace, numPoints, refCoords, B, NULL, NULL);CHKERRQ(ierr);
  for (i = 0; i < numPoints * Nc; i++) {realCoords[i] = 0.;}
  for (j = 0; j < numPoints; j++) {
    PetscReal *mapped = &realCoords[j * Nc];

    for (k = 0; k < pdim; k++) {
      for (l = 0; l < Nc; l++) {
        mapped[l] += modes[k] * B[(j * pdim + k) * Nc + l];
      }
    }
  }
  ierr = DMRestoreWorkArray(dm,numPoints * pdim * Nc,MPIU_REAL,&B);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,pdim,MPIU_REAL,&modes);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, coords, cell, &coordSize, &nodes);CHKERRQ(ierr);
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

.seealso: DMPlexReferenceToCoordinates()
@*/
PetscErrorCode DMPlexCoordinatesToReference(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal realCoords[], PetscReal refCoords[])
{
  PetscInt       dimC, dimR, depth, cStart, cEnd, i;
  DM             coordDM = NULL;
  Vec            coords;
  PetscFE        fe = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDimension(dm,&dimR);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm,&dimC);CHKERRQ(ierr);
  if (dimR <= 0 || dimC <= 0 || numPoints <= 0) PetscFunctionReturn(0);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&coordDM);CHKERRQ(ierr);
  if (coordDM) {
    PetscInt coordFields;

    ierr = DMGetNumFields(coordDM,&coordFields);CHKERRQ(ierr);
    if (coordFields) {
      PetscClassId id;
      PetscObject  disc;

      ierr = DMGetField(coordDM,0,NULL,&disc);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
      }
    }
  }
  ierr = DMPlexGetInteriorCellStratum(dm,&cStart,&cEnd);CHKERRQ(ierr);
  if (cell < cStart || cell >= cEnd) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"point %D not in cell range [%D,%D)",cell,cStart,cEnd);
  if (!fe) { /* implicit discretization: affine or multilinear */
    PetscInt  coneSize;
    PetscBool isSimplex, isTensor;

    ierr = DMPlexGetConeSize(dm,cell,&coneSize);CHKERRQ(ierr);
    isSimplex = (coneSize == (dimR + 1)) ? PETSC_TRUE : PETSC_FALSE;
    isTensor  = (coneSize == ((depth == 1) ? (1 << dimR) : (2 * dimR))) ? PETSC_TRUE : PETSC_FALSE;
    if (isSimplex) {
      PetscReal detJ, *v0, *J, *invJ;

      ierr = DMGetWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0);CHKERRQ(ierr);
      J    = &v0[dimC];
      invJ = &J[dimC * dimC];
      ierr = DMPlexComputeCellGeometryAffineFEM(dm, cell, v0, J, invJ, &detJ);CHKERRQ(ierr);
      for (i = 0; i < numPoints; i++) { /* Apply the inverse affine transformation for each point */
        const PetscReal x0[3] = {-1.,-1.,-1.};

        CoordinatesRealToRef(dimC, dimR, x0, v0, invJ, &realCoords[dimC * i], &refCoords[dimR * i]);
      }
      ierr = DMRestoreWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0);CHKERRQ(ierr);
    } else if (isTensor) {
      ierr = DMPlexCoordinatesToReference_Tensor(coordDM, cell, numPoints, realCoords, refCoords, coords, dimC, dimR);CHKERRQ(ierr);
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unrecognized cone size %D",coneSize);
  } else {
    ierr = DMPlexCoordinatesToReference_FE(coordDM, fe, cell, numPoints, realCoords, refCoords, coords, dimC, dimR);CHKERRQ(ierr);
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

.seealso: DMPlexCoordinatesToReference()
@*/
PetscErrorCode DMPlexReferenceToCoordinates(DM dm, PetscInt cell, PetscInt numPoints, const PetscReal refCoords[], PetscReal realCoords[])
{
  PetscInt       dimC, dimR, depth, cStart, cEnd, i;
  DM             coordDM = NULL;
  Vec            coords;
  PetscFE        fe = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDimension(dm,&dimR);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm,&dimC);CHKERRQ(ierr);
  if (dimR <= 0 || dimC <= 0 || numPoints <= 0) PetscFunctionReturn(0);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&coordDM);CHKERRQ(ierr);
  if (coordDM) {
    PetscInt coordFields;

    ierr = DMGetNumFields(coordDM,&coordFields);CHKERRQ(ierr);
    if (coordFields) {
      PetscClassId id;
      PetscObject  disc;

      ierr = DMGetField(coordDM,0,NULL,&disc);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
      }
    }
  }
  ierr = DMPlexGetInteriorCellStratum(dm,&cStart,&cEnd);CHKERRQ(ierr);
  if (cell < cStart || cell >= cEnd) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"point %D not in cell range [%D,%D)",cell,cStart,cEnd);
  if (!fe) { /* implicit discretization: affine or multilinear */
    PetscInt  coneSize;
    PetscBool isSimplex, isTensor;

    ierr = DMPlexGetConeSize(dm,cell,&coneSize);CHKERRQ(ierr);
    isSimplex = (coneSize == (dimR + 1)) ? PETSC_TRUE : PETSC_FALSE;
    isTensor  = (coneSize == ((depth == 1) ? (1 << dimR) : (2 * dimR))) ? PETSC_TRUE : PETSC_FALSE;
    if (isSimplex) {
      PetscReal detJ, *v0, *J;

      ierr = DMGetWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0);CHKERRQ(ierr);
      J    = &v0[dimC];
      ierr = DMPlexComputeCellGeometryAffineFEM(dm, cell, v0, J, NULL, &detJ);CHKERRQ(ierr);
      for (i = 0; i < numPoints; i++) { /* Apply the affine transformation for each point */
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        CoordinatesRefToReal(dimC, dimR, xi0, v0, J, &refCoords[dimR * i], &realCoords[dimC * i]);
      }
      ierr = DMRestoreWorkArray(dm,dimC + 2 * dimC * dimC, MPIU_REAL, &v0);CHKERRQ(ierr);
    } else if (isTensor) {
      ierr = DMPlexReferenceToCoordinates_Tensor(coordDM, cell, numPoints, refCoords, realCoords, coords, dimC, dimR);CHKERRQ(ierr);
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unrecognized cone size %D",coneSize);
  } else {
    ierr = DMPlexReferenceToCoordinates_FE(coordDM, fe, cell, numPoints, refCoords, realCoords, coords, dimC, dimR);CHKERRQ(ierr);
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

.seealso: DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMProjectFieldLocal(), DMProjectFieldLabelLocal()
@*/
PetscErrorCode DMPlexRemapGeometry(DM dm, PetscReal time,
                                   void (*func)(PetscInt, PetscInt, PetscInt,
                                                const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
{
  DM             cdm;
  Vec            lCoords, tmpCoords;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &lCoords);CHKERRQ(ierr);
  ierr = DMGetLocalVector(cdm, &tmpCoords);CHKERRQ(ierr);
  ierr = VecCopy(lCoords, tmpCoords);CHKERRQ(ierr);
  ierr = DMProjectFieldLocal(cdm, time, tmpCoords, &func, INSERT_VALUES, lCoords);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(cdm, &tmpCoords);CHKERRQ(ierr);
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

.seealso: DMPlexRemapGeometry()
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
  ierr = PetscMalloc1(dE+1, &moduli);CHKERRQ(ierr);
  moduli[0] = dir;
  for (d = 0, e = 0; d < dE; ++d) moduli[d] = d == dir ? 0.0 : (multipliers ? multipliers[e++] : 1.0);
  ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(cds, 0, &obj);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
  if (id != PETSCFE_CLASSID) {
    Vec           lCoords;
    PetscSection  cSection;
    PetscScalar  *coords;
    PetscInt      vStart, vEnd, v;

    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &cSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &lCoords);CHKERRQ(ierr);
    ierr = VecGetArray(lCoords, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscReal ds;
      PetscInt  off, c;

      ierr = PetscSectionGetOffset(cSection, v, &off);CHKERRQ(ierr);
      ds   = PetscRealPart(coords[off+dir]);
      for (c = 0; c < dE; ++c) coords[off+c] += moduli[c]*ds;
    }
    ierr = VecRestoreArray(lCoords, &coords);CHKERRQ(ierr);
  } else {
    ierr = PetscDSSetConstants(cds, dE+1, moduli);CHKERRQ(ierr);
    ierr = DMPlexRemapGeometry(dm, 0.0, f0_shear);CHKERRQ(ierr);
  }
  ierr = PetscFree(moduli);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexLocatePoint_Simplex_2D_Internal"
static PetscErrorCode DMPlexLocatePoint_Simplex_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  const PetscInt embedDim = 2;
  PetscReal      x        = PetscRealPart(point[0]);
  PetscReal      y        = PetscRealPart(point[1]);
  PetscReal      v0[2], J[4], invJ[4], detJ;
  PetscReal      xi, eta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
  xi  = invJ[0*embedDim+0]*(x - v0[0]) + invJ[0*embedDim+1]*(y - v0[1]);
  eta = invJ[1*embedDim+0]*(x - v0[0]) + invJ[1*embedDim+1]*(y - v0[1]);

  if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 2.0)) *cell = c;
  else *cell = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexLocatePoint_General_2D_Internal"
static PetscErrorCode DMPlexLocatePoint_General_2D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  PetscSection       coordSection;
  Vec             coordsLocal;
  PetscScalar    *coords;
  const PetscInt  faces[8]  = {0, 1, 1, 2, 2, 3, 3, 0};
  PetscReal       x         = PetscRealPart(point[0]);
  PetscReal       y         = PetscRealPart(point[1]);
  PetscInt        crossings = 0, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
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
  else *cell = -1;
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexLocatePoint_Simplex_3D_Internal"
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
  ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
  xi   = invJ[0*embedDim+0]*(x - v0[0]) + invJ[0*embedDim+1]*(y - v0[1]) + invJ[0*embedDim+2]*(z - v0[2]);
  eta  = invJ[1*embedDim+0]*(x - v0[0]) + invJ[1*embedDim+1]*(y - v0[1]) + invJ[1*embedDim+2]*(z - v0[2]);
  zeta = invJ[2*embedDim+0]*(x - v0[0]) + invJ[2*embedDim+1]*(y - v0[1]) + invJ[2*embedDim+2]*(z - v0[2]);

  if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 2.0)) *cell = c;
  else *cell = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexLocatePoint_General_3D_Internal"
static PetscErrorCode DMPlexLocatePoint_General_3D_Internal(DM dm, const PetscScalar point[], PetscInt c, PetscInt *cell)
{
  PetscSection       coordSection;
  Vec            coordsLocal;
  PetscScalar   *coords;
  const PetscInt faces[24] = {0, 1, 2, 3,  5, 4, 7, 6,  1, 0, 4, 5,
                              3, 2, 6, 7,  1, 5, 6, 2,  0, 3, 7, 4};
  PetscBool      found = PETSC_TRUE;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
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
  else *cell = -1;
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocatePoints_Plex"
/*
 Need to implement using the guess
*/
PetscErrorCode DMLocatePoints_Plex(DM dm, Vec v, IS *cellIS)
{
  PetscInt       cell = -1 /*, guess = -1*/;
  PetscInt       bs, numPoints, p;
  PetscInt       dim, cStart, cEnd, cMax, c, coneSize;
  PetscInt      *cells;
  PetscScalar   *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  if (cMax >= 0) cEnd = PetscMin(cEnd, cMax);
  ierr = VecGetLocalSize(v, &numPoints);CHKERRQ(ierr);
  ierr = VecGetBlockSize(v, &bs);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  if (bs != dim) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Block size for point vector %d must be the mesh coordinate dimension %d", bs, dim);
  numPoints /= bs;
  ierr       = PetscMalloc(numPoints * sizeof(PetscInt), &cells);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) {
    const PetscScalar *point = &a[p*bs];

    switch (dim) {
    case 2:
      for (c = cStart; c < cEnd; ++c) {
        ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
        switch (coneSize) {
        case 3:
          ierr = DMPlexLocatePoint_Simplex_2D_Internal(dm, point, c, &cell);CHKERRQ(ierr);
          break;
        case 4:
          ierr = DMPlexLocatePoint_General_2D_Internal(dm, point, c, &cell);CHKERRQ(ierr);
          break;
        default:
          SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for cell with cone size %d", coneSize);
        }
        if (cell >= 0) break;
      }
      break;
    case 3:
      for (c = cStart; c < cEnd; ++c) {
        ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
        switch (coneSize) {
        case 4:
          ierr = DMPlexLocatePoint_Simplex_3D_Internal(dm, point, c, &cell);CHKERRQ(ierr);
          break;
        case 8:
          ierr = DMPlexLocatePoint_General_3D_Internal(dm, point, c, &cell);CHKERRQ(ierr);
          break;
        default:
          SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for cell with cone size %d", coneSize);
        }
        if (cell >= 0) break;
      }
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No point location for mesh dimension %d", dim);
    }
    cells[p] = cell;
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints, cells, PETSC_OWN_POINTER, cellIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeProjection3Dto2D_Internal"
/*
  DMPlexComputeProjection3Dto2D_Internal - Rewrite coordinates to be the 2D projection of the 3D
*/
static PetscErrorCode DMPlexComputeProjection3Dto2D_Internal(PetscScalar coords[])
{
  PetscScalar    x1[3], x2[3], n[3], norm;
  PetscScalar    R[9], x1p[3], x2p[3];
  PetscReal      sqrtz, alpha;
  const PetscInt dim = 3;
  PetscInt       d, e;

  PetscFunctionBegin;
  /* 0) Calculate normal vector */
  for (d = 0; d < dim; ++d) {
    x1[d] = coords[1*dim+d] - coords[0*dim+d];
    x2[d] = coords[2*dim+d] - coords[0*dim+d];
  }
  n[0] = x1[1]*x2[2] - x1[2]*x2[1];
  n[1] = x1[2]*x2[0] - x1[0]*x2[2];
  n[2] = x1[0]*x2[1] - x1[1]*x2[0];
  norm = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
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
  sqrtz = sqrt(1.0 - PetscAbsScalar(n[2]*n[2]));
  /* Check for n = z */
  if (sqrtz < 1.0e-10) {
    coords[0] = 0.0;
    coords[1] = 0.0;
    if (PetscRealPart(n[2]) < 0.0) {
      coords[2] = x2[0];
      coords[3] = x2[1];
      coords[4] = x1[0];
      coords[5] = x1[1];
    } else {
      coords[2] = x1[0];
      coords[3] = x1[1];
      coords[4] = x2[0];
      coords[5] = x2[1];
    }
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
  if (PetscAbsScalar(x1p[2]) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid rotation calculated");
  if (PetscAbsScalar(x2p[2]) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid rotation calculated");
  /* 2) Project to (x, y) */
  coords[0] = 0.0;
  coords[1] = 0.0;
  coords[2] = x1p[0];
  coords[3] = x1p[1];
  coords[4] = x2p[0];
  coords[5] = x2p[1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeTriangleGeometry_Internal"
static PetscErrorCode DMPlexComputeTriangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  const PetscInt dim = 2;
  PetscInt       numCoords, d, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  if (numCoords == 9) {
    ierr = DMPlexComputeProjection3Dto2D_Internal(coords);CHKERRQ(ierr);
  } else if (numCoords != 6) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this triangle is %d != 6", numCoords);
  if (v0) {
    for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      for (f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
    }
    *detJ = J[0]*J[3] - J[1]*J[2];
#if 0
    if (detJ < 0.0) {
      const PetscReal xLength = mesh->periodicity[0];

      if (xLength != 0.0) {
        PetscReal v0x = coords[0*dim+0];

        if (v0x == 0.0) v0x = v0[0] = xLength;
        for (f = 0; f < dim; f++) {
          const PetscReal px = coords[(f+1)*dim+0] == 0.0 ? xLength : coords[(f+1)*dim+0];

          J[0*dim+f] = 0.5*(px - v0x);
        }
      }
      detJ = J[0]*J[3] - J[1]*J[2];
    }
#endif
    PetscLogFlops(8.0 + 3.0);
  } else {
    *detJ = 0.0;
  }
  if (invJ) {
    const PetscReal invDet = 1.0/(*detJ);

    invJ[0] =  invDet*J[3];
    invJ[1] = -invDet*J[1];
    invJ[2] = -invDet*J[2];
    invJ[3] =  invDet*J[0];
    PetscLogFlops(5.0);
  }
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeRectangleGeometry_Internal"
static PetscErrorCode DMPlexComputeRectangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  const PetscInt dim = 2;
  PetscInt       d, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      for (f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f*2+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
    }
    *detJ = J[0]*J[3] - J[1]*J[2];
    PetscLogFlops(8.0 + 3.0);
  } else {
    *detJ = 0.0;
  }
  if (invJ) {
    const PetscReal invDet = 1.0/(*detJ);

    invJ[0] =  invDet*J[3];
    invJ[1] = -invDet*J[1];
    invJ[2] = -invDet*J[2];
    invJ[3] =  invDet*J[0];
    PetscLogFlops(5.0);
  }
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeTetrahedronGeometry_Internal"
static PetscErrorCode DMPlexComputeTetrahedronGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  const PetscInt dim = 3;
  PetscInt       d, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      for (f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
    }
    *detJ = (J[0*3+0]*(J[1*3+2]*J[2*3+1] - J[1*3+1]*J[2*3+2]) +
             J[0*3+1]*(J[1*3+0]*J[2*3+2] - J[1*3+2]*J[2*3+0]) +
             J[0*3+2]*(J[1*3+1]*J[2*3+0] - J[1*3+0]*J[2*3+1]));
    PetscLogFlops(18.0 + 12.0);
  }
  if (invJ) {
    const PetscReal invDet = 1.0/(*detJ);

    invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
    invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
    invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
    invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
    invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
    invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
    invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
    invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
    invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
    PetscLogFlops(37.0);
  }
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeHexahedronGeometry_Internal"
static PetscErrorCode DMPlexComputeHexahedronGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  const PetscInt dim = 3;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  if (v0) {
    for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);
  }
  if (J) {
    for (d = 0; d < dim; d++) {
      J[d*dim+0] = 0.5*(PetscRealPart(coords[(0+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+1] = 0.5*(PetscRealPart(coords[(1+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+2] = 0.5*(PetscRealPart(coords[(3+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
    }
    *detJ = (J[0*3+0]*(J[1*3+2]*J[2*3+1] - J[1*3+1]*J[2*3+2]) +
             J[0*3+1]*(J[1*3+0]*J[2*3+2] - J[1*3+2]*J[2*3+0]) +
             J[0*3+2]*(J[1*3+1]*J[2*3+0] - J[1*3+0]*J[2*3+1]));
    PetscLogFlops(18.0 + 12.0);
  } else {
    *detJ = 0.0;
  }
  if (invJ) {
    const PetscReal invDet = -1.0/(*detJ);

    invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
    invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
    invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
    invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
    invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
    invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
    invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
    invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
    invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
    PetscLogFlops(37.0);
  }
  *detJ *= 8.0;
  ierr   = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeCellGeometry"
/*@C
  DMPlexComputeCellGeometry - Compute the Jacobian, inverse Jacobian, and Jacobian determinant for a given cell

  Collective on DM

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

.seealso: DMPlexGetCoordinateSection(), DMPlexGetCoordinateVec()
@*/
PetscErrorCode DMPlexComputeCellGeometry(DM dm, PetscInt cell, PetscReal *v0, PetscReal *J, PetscReal *invJ, PetscReal *detJ)
{
  PetscInt       depth, dim, coneSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  if (depth == 1) {
    ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      switch (coneSize) {
      case 3:
        ierr = DMPlexComputeTriangleGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      case 4:
        ierr = DMPlexComputeRectangleGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported number of vertices %D in cell %D for element geometry computation", coneSize, cell);
      }
      break;
    case 3:
      switch (coneSize) {
      case 4:
        ierr = DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      case 8:
        ierr = DMPlexComputeHexahedronGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported number of vertices %D in cell %D for element geometry computation", coneSize, cell);
      }
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %D for element geometry computation", dim);
    }
  } else {
    /* We need to keep a pointer to the depth label */
    ierr = DMPlexGetLabelValue(dm, "depth", cell, &dim);CHKERRQ(ierr);
    /* Cone size is now the number of faces */
    switch (dim) {
    case 2:
      switch (coneSize) {
      case 3:
        ierr = DMPlexComputeTriangleGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      case 4:
        ierr = DMPlexComputeRectangleGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported number of vertices %D in cell %D for element geometry computation", coneSize, cell);
      }
      break;
    case 3:
      switch (coneSize) {
      case 4:
        ierr = DMPlexComputeTetrahedronGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      case 6:
        ierr = DMPlexComputeHexahedronGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported number of vertices %D in cell %D for element geometry computation", coneSize, cell);
      }
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %D for element geometry computation", dim);
    }
  }
  PetscFunctionReturn(0);
}

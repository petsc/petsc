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
  PetscScalar    *coords = NULL;
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
#define __FUNCT__ "DMPlexComputeProjection2Dto1D_Internal"
/*
  DMPlexComputeProjection2Dto1D_Internal - Rewrite coordinates to be the 1D projection of the 2D
*/
static PetscErrorCode DMPlexComputeProjection2Dto1D_Internal(PetscScalar coords[], PetscReal R[])
{
  const PetscReal x = PetscRealPart(coords[2] - coords[0]);
  const PetscReal y = PetscRealPart(coords[3] - coords[1]);
  const PetscReal r = PetscSqrtReal(x*x + y*y), c = x/r, s = y/r;

  PetscFunctionBegin;
  R[0] =  c; R[1] = s;
  R[2] = -s; R[3] = c;
  coords[0] = 0.0;
  coords[1] = r;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeProjection3Dto2D_Internal"
/*
  DMPlexComputeProjection3Dto2D_Internal - Rewrite coordinates to be the 2D projection of the 3D
*/
static PetscErrorCode DMPlexComputeProjection3Dto2D_Internal(PetscInt coordSize, PetscScalar coords[], PetscReal R[])
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
    if (n[2] < 0.0) {
      if (coordSize > 9) {
        coords[2] = PetscRealPart(coords[3*dim+0] - coords[0*dim+0]);
        coords[3] = PetscRealPart(coords[3*dim+0] - coords[0*dim+0]);
        coords[4] = x2[0];
        coords[5] = x2[1];
        coords[6] = x1[0];
        coords[7] = x1[1];
      } else {
        coords[2] = x2[0];
        coords[3] = x2[1];
        coords[4] = x1[0];
        coords[5] = x1[1];
      }
      R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
      R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
      R[6] = 0.0; R[7] = 0.0; R[8] = -1.0;
    } else {
      for (p = 3; p < coordSize/3; ++p) {
        coords[p*2+0] = PetscRealPart(coords[p*dim+0] - coords[0*dim+0]);
        coords[p*2+1] = PetscRealPart(coords[p*dim+1] - coords[0*dim+1]);
      }
      coords[2] = x1[0];
      coords[3] = x1[1];
      coords[4] = x2[0];
      coords[5] = x2[1];
      R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
      R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
      R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
    }
    coords[0] = 0.0;
    coords[1] = 0.0;
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
  if (PetscAbsReal(x1p[2]) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid rotation calculated");
  if (PetscAbsReal(x2p[2]) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid rotation calculated");
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

#undef __FUNCT__
#define __FUNCT__ "Invert2D_Internal"
PETSC_STATIC_INLINE void Invert2D_Internal(PetscReal invJ[], PetscReal J[], PetscReal detJ)
{
  const PetscReal invDet = 1.0/detJ;

  invJ[0] =  invDet*J[3];
  invJ[1] = -invDet*J[1];
  invJ[2] = -invDet*J[2];
  invJ[3] =  invDet*J[0];
  PetscLogFlops(5.0);
}

#undef __FUNCT__
#define __FUNCT__ "Invert3D_Internal"
PETSC_STATIC_INLINE void Invert3D_Internal(PetscReal invJ[], PetscReal J[], PetscReal detJ)
{
  const PetscReal invDet = 1.0/detJ;

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

#undef __FUNCT__
#define __FUNCT__ "Det2D_Internal"
PETSC_STATIC_INLINE void Det2D_Internal(PetscReal *detJ, PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
  PetscLogFlops(3.0);
}

#undef __FUNCT__
#define __FUNCT__ "Det3D_Internal"
PETSC_STATIC_INLINE void Det3D_Internal(PetscReal *detJ, PetscReal J[])
{
  *detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
           J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
           J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
  PetscLogFlops(12.0);
}

#undef __FUNCT__
#define __FUNCT__ "Volume_Triangle_Internal"
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
  Det2D_Internal(&detM, M);
  *vol = 0.5*detM;
  PetscLogFlops(5.0);
}

#undef __FUNCT__
#define __FUNCT__ "Volume_Triangle_Origin_Internal"
PETSC_STATIC_INLINE void Volume_Triangle_Origin_Internal(PetscReal *vol, PetscReal coords[])
{
  Det2D_Internal(vol, coords);
  *vol *= 0.5;
}

#undef __FUNCT__
#define __FUNCT__ "Volume_Tetrahedron_Internal"
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
  PetscReal       M[9], detM;
  M[0] = x1; M[1] = x2; M[2] = x3;
  M[3] = y1; M[4] = y2; M[5] = y3;
  M[6] = z1; M[7] = z2; M[8] = z3;
  Det3D_Internal(&detM, M);
  *vol = -0.16666666666666666666666*detM;
  PetscLogFlops(10.0);
}

#undef __FUNCT__
#define __FUNCT__ "Volume_Tetrahedron_Origin_Internal"
PETSC_STATIC_INLINE void Volume_Tetrahedron_Origin_Internal(PetscReal *vol, PetscReal coords[])
{
  Det3D_Internal(vol, coords);
  *vol *= -0.16666666666666666666666;
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeLineGeometry_Internal"
static PetscErrorCode DMPlexComputeLineGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  *detJ = 0.0;
  if (numCoords == 4) {
    const PetscInt dim = 2;
    PetscReal      R[4], J0;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    ierr = DMPlexComputeProjection2Dto1D_Internal(coords, R);CHKERRQ(ierr);
    if (J)    {
      J0   = 0.5*PetscRealPart(coords[1]);
      J[0] = R[0]*J0; J[1] = R[1];
      J[2] = R[2]*J0; J[3] = R[3];
      Det2D_Internal(detJ, J);
    }
    if (invJ) {Invert2D_Internal(invJ, J, *detJ);}
  } else if (numCoords == 2) {
    const PetscInt dim = 1;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    if (J)    {
      J[0]  = 0.5*(PetscRealPart(coords[1]) - PetscRealPart(coords[0]));
      *detJ = J[0];
      PetscLogFlops(2.0);
    }
    if (invJ) {invJ[0] = 1.0/J[0]; PetscLogFlops(1.0);}
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this segment is %d != 2", numCoords);
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeTriangleGeometry_Internal"
static PetscErrorCode DMPlexComputeTriangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, d, f, g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  *detJ = 0.0;
  if (numCoords == 9) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    ierr = DMPlexComputeProjection3Dto2D_Internal(numCoords, coords, R);CHKERRQ(ierr);
    if (J)    {
      const PetscInt pdim = 2;

      for (d = 0; d < pdim; d++) {
        for (f = 0; f < pdim; f++) {
          J0[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*pdim+d]) - PetscRealPart(coords[0*pdim+d]));
        }
      }
      PetscLogFlops(8.0);
      Det3D_Internal(detJ, J0);
      for (d = 0; d < dim; d++) {
        for (f = 0; f < dim; f++) {
          J[d*dim+f] = 0.0;
          for (g = 0; g < dim; g++) {
            J[d*dim+f] += R[d*dim+g]*J0[g*dim+f];
          }
        }
      }
      PetscLogFlops(18.0);
    }
    if (invJ) {Invert3D_Internal(invJ, J, *detJ);}
  } else if (numCoords == 6) {
    const PetscInt dim = 2;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    if (J)    {
      for (d = 0; d < dim; d++) {
        for (f = 0; f < dim; f++) {
          J[d*dim+f] = 0.5*(PetscRealPart(coords[(f+1)*dim+d]) - PetscRealPart(coords[0*dim+d]));
        }
      }
      PetscLogFlops(8.0);
      Det2D_Internal(detJ, J);
    }
    if (invJ) {Invert2D_Internal(invJ, J, *detJ);}
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this triangle is %d != 6", numCoords);
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeRectangleGeometry_Internal"
static PetscErrorCode DMPlexComputeRectangleGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       numCoords, d, f, g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  *detJ = 0.0;
  if (numCoords == 12) {
    const PetscInt dim = 3;
    PetscReal      R[9], J0[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    ierr = DMPlexComputeProjection3Dto2D_Internal(numCoords, coords, R);CHKERRQ(ierr);
    if (J)    {
      const PetscInt pdim = 2;

      for (d = 0; d < pdim; d++) {
        J0[d*dim+0] = 0.5*(PetscRealPart(coords[1*pdim+d]) - PetscRealPart(coords[0*pdim+d]));
        J0[d*dim+1] = 0.5*(PetscRealPart(coords[3*pdim+d]) - PetscRealPart(coords[0*pdim+d]));
      }
      PetscLogFlops(8.0);
      Det3D_Internal(detJ, J0);
      for (d = 0; d < dim; d++) {
        for (f = 0; f < dim; f++) {
          J[d*dim+f] = 0.0;
          for (g = 0; g < dim; g++) {
            J[d*dim+f] += R[d*dim+g]*J0[g*dim+f];
          }
        }
      }
      PetscLogFlops(18.0);
    }
    if (invJ) {Invert3D_Internal(invJ, J, *detJ);}
  } else if (numCoords == 8) {
    const PetscInt dim = 2;

    if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
    if (J)    {
      for (d = 0; d < dim; d++) {
        J[d*dim+0] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
        J[d*dim+1] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
      }
      PetscLogFlops(8.0);
      Det2D_Internal(detJ, J);
    }
    if (invJ) {Invert2D_Internal(invJ, J, *detJ);}
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for this quadrilateral is %d != 6", numCoords);
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, &numCoords, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeTetrahedronGeometry_Internal"
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
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
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
    PetscLogFlops(18.0);
    Det3D_Internal(detJ, J);
  }
  if (invJ) {Invert3D_Internal(invJ, J, *detJ);}
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeHexahedronGeometry_Internal"
static PetscErrorCode DMPlexComputeHexahedronGeometry_Internal(DM dm, PetscInt e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  const PetscInt dim = 3;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
  *detJ = 0.0;
  if (v0)   {for (d = 0; d < dim; d++) v0[d] = PetscRealPart(coords[d]);}
  if (J)    {
    for (d = 0; d < dim; d++) {
      J[d*dim+0] = 0.5*(PetscRealPart(coords[3*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+1] = 0.5*(PetscRealPart(coords[1*dim+d]) - PetscRealPart(coords[0*dim+d]));
      J[d*dim+2] = 0.5*(PetscRealPart(coords[4*dim+d]) - PetscRealPart(coords[0*dim+d]));
    }
    PetscLogFlops(18.0);
    Det3D_Internal(detJ, J);
  }
  if (invJ) {Invert3D_Internal(invJ, J, *detJ);}
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, e, NULL, &coords);CHKERRQ(ierr);
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
    case 1:
      ierr = DMPlexComputeLineGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
      break;
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
    case 1:
      ierr = DMPlexComputeLineGeometry_Internal(dm, cell, v0, J, invJ, detJ);CHKERRQ(ierr);
      break;
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeGeometryFVM_1D_Internal"
static PetscErrorCode DMPlexComputeGeometryFVM_1D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscInt       coordSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "We only support 2D edges right now");
  if (centroid) {
    centroid[0] = 0.5*PetscRealPart(coords[0] + coords[dim+0]);
    centroid[1] = 0.5*PetscRealPart(coords[1] + coords[dim+1]);
  }
  if (normal) {
    normal[0] =  PetscRealPart(coords[1] - coords[dim+1]);
    normal[1] = -PetscRealPart(coords[0] - coords[dim+0]);
  }
  if (vol) {
    *vol = PetscSqrtReal(PetscSqr(PetscRealPart(coords[0] - coords[dim+0])) + PetscSqr(PetscRealPart(coords[1] - coords[dim+1])));
  }
  ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeGeometryFVM_2D_Internal"
/* Centroid_i = (\sum_n A_n Cn_i ) / A */
static PetscErrorCode DMPlexComputeGeometryFVM_2D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords = NULL;
  PetscReal      vsum = 0.0, csum[3] = {0.0, 0.0, 0.0}, vtmp, ctmp[4], v0[3], R[9];
  PetscInt       tdim = 2, coordSize, numCorners, p, d, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cell, &numCorners);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
  dim  = coordSize/numCorners;
  if (normal) {
    if (dim > 2) {
      const PetscReal x0 = PetscRealPart(coords[dim+0] - coords[0]), x1 = PetscRealPart(coords[dim*2+0] - coords[0]);
      const PetscReal y0 = PetscRealPart(coords[dim+1] - coords[1]), y1 = PetscRealPart(coords[dim*2+1] - coords[1]);
      const PetscReal z0 = PetscRealPart(coords[dim+2] - coords[2]), z1 = PetscRealPart(coords[dim*2+2] - coords[2]);
      PetscReal       norm;

      v0[0]     = PetscRealPart(coords[0]);
      v0[1]     = PetscRealPart(coords[1]);
      v0[2]     = PetscRealPart(coords[2]);
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
  if (dim == 3) {ierr = DMPlexComputeProjection3Dto2D_Internal(coordSize, coords, R);CHKERRQ(ierr);}
  for (p = 0; p < numCorners; ++p) {
    /* Need to do this copy to get types right */
    for (d = 0; d < tdim; ++d) {
      ctmp[d]      = PetscRealPart(coords[p*tdim+d]);
      ctmp[tdim+d] = PetscRealPart(coords[((p+1)%numCorners)*tdim+d]);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeGeometryFVM_3D_Internal"
/* Centroid_i = (\sum_n V_n Cn_i ) / V */
static PetscErrorCode DMPlexComputeGeometryFVM_3D_Internal(DM dm, PetscInt dim, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscSection    coordSection;
  Vec             coordinates;
  PetscScalar    *coords = NULL;
  PetscReal       vsum = 0.0, vtmp, coordsTmp[3*3];
  const PetscInt *faces, *facesO;
  PetscInt        numFaces, f, coordSize, numCorners, p, d;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);

  if (centroid) for (d = 0; d < dim; ++d) centroid[d] = 0.0;
  ierr = DMPlexGetConeSize(dm, cell, &numFaces);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, cell, &facesO);CHKERRQ(ierr);
  for (f = 0; f < numFaces; ++f) {
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
      if (facesO[f] < 0) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {           /* Centroid of OABC = (a+b+c)/4 */
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p*dim+d]*vtmp;
        }
      }
      break;
    case 4:
      /* DO FOR PYRAMID */
      /* First tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[0*dim+d]);
        coordsTmp[1*dim+d] = PetscRealPart(coords[1*dim+d]);
        coordsTmp[2*dim+d] = PetscRealPart(coords[3*dim+d]);
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (facesO[f] < 0) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p*dim+d]*vtmp;
        }
      }
      /* Second tet */
      for (d = 0; d < dim; ++d) {
        coordsTmp[0*dim+d] = PetscRealPart(coords[1*dim+d]);
        coordsTmp[1*dim+d] = PetscRealPart(coords[2*dim+d]);
        coordsTmp[2*dim+d] = PetscRealPart(coords[3*dim+d]);
      }
      Volume_Tetrahedron_Origin_Internal(&vtmp, coordsTmp);
      if (facesO[f] < 0) vtmp = -vtmp;
      vsum += vtmp;
      if (centroid) {
        for (d = 0; d < dim; ++d) {
          for (p = 0; p < 3; ++p) centroid[d] += coordsTmp[p*dim+d]*vtmp;
        }
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle faces with %d vertices", numCorners);
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, faces[f], &coordSize, &coords);CHKERRQ(ierr);
  }
  if (vol)     *vol = PetscAbsReal(vsum);
  if (normal)   for (d = 0; d < dim; ++d) normal[d]    = 0.0;
  if (centroid) for (d = 0; d < dim; ++d) centroid[d] /= (vsum*4);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeCellGeometryFVM"
/*@C
  DMPlexComputeCellGeometryFVM - Compute the volume for a given cell

  Collective on DM

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

.seealso: DMPlexGetCoordinateSection(), DMPlexGetCoordinateVec()
@*/
PetscErrorCode DMPlexComputeCellGeometryFVM(DM dm, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[])
{
  PetscInt       depth, dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (depth != dim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated");
  /* We need to keep a pointer to the depth label */
  ierr = DMPlexGetLabelValue(dm, "depth", cell, &depth);CHKERRQ(ierr);
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
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %D for element geometry computation", dim);
  }
  PetscFunctionReturn(0);
}

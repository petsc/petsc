#include <petsc-private/daimpl.h>     /*I  "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "FillClosureArray_Private"
PETSC_STATIC_INLINE PetscErrorCode FillClosureArray_Private(DM dm, PetscSection section, PetscInt nP, const PetscInt points[], PetscScalar *vArray, const PetscScalar **array)
{
  PetscScalar   *a;
  PetscInt       size = 0, dof, off, d, k, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(i = 0; i < nP; ++i) {
    ierr = PetscSectionGetDof(section, points[i], &dof);CHKERRQ(ierr);
    size += dof;
  }
  ierr = DMGetWorkArray(dm, 2*size, &a);CHKERRQ(ierr);
  for(i = 0, k = 0; i < nP; ++i) {
    ierr = PetscSectionGetDof(section, points[i], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, points[i], &off);CHKERRQ(ierr);

    for(d = 0; d < dof; ++d, ++k) {
      a[k] = vArray[off+d];
    }
  }
  *array = a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FillClosureVec_Private"
PETSC_STATIC_INLINE PetscErrorCode FillClosureVec_Private(DM dm, PetscSection section, PetscInt nP, const PetscInt points[], PetscScalar *vArray, const PetscScalar *array, InsertMode mode)
{
  PetscInt       dof, off, d, k, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((mode == INSERT_VALUES) || (mode == INSERT_ALL_VALUES)) {
    for(i = 0, k = 0; i < nP; ++i) {
      ierr = PetscSectionGetDof(section, points[i], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, points[i], &off);CHKERRQ(ierr);

      for(d = 0; d < dof; ++d, ++k) {
        vArray[off+d] = array[k];
      }
    }
  } else {
    for(i = 0, k = 0; i < nP; ++i) {
      ierr = PetscSectionGetDof(section, points[i], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, points[i], &off);CHKERRQ(ierr);

      for(d = 0; d < dof; ++d, ++k) {
        vArray[off+d] += array[k];
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecGetClosure"
PetscErrorCode DMDAVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt p, const PetscScalar **values)
{
  DM_DA         *da  = (DM_DA *) dm->data;
  PetscInt       dim = da->dim;
  PetscScalar   *vArray;
  PetscInt       nVx, nVy, nxF, nXF, nyF, nYF, nzF, nZF;
  PetscInt       pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart, yfEnd, zfStart, zfEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  PetscValidPointer(values, 5);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!section) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "This DM has not default PetscSection");
  ierr = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 1,   &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, &nVx, &nVy, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  xfStart = fStart; xfEnd = xfStart+nXF;
  yfStart = xfEnd;  yfEnd = yfStart+nYF;
  zfStart = yfEnd;  zfEnd = zfStart+nZF;
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid point %d should be in [%d, %d)", p, pStart, pEnd);
  ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
  if ((p >= cStart) || (p < cEnd)) {
    /* Cell */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    } else if (dim == 2) {
      /* 4 faces, 4 vertices
         Bottom-left vertex follows same order as cells
         Bottom y-face same order as cells
         Left x-face follows same order as cells
         We number the quad:

           8--3--7
           |     |
           4  0  2
           |     |
           5--1--6
      */
      PetscInt c         = p - cStart, cx = c % (nVx-1), cy = c / (nVx-1);
      PetscInt v         = cy*nVx + cx +  vStart;
      PetscInt xf        = cy*nxF + cx + xfStart;
      PetscInt yf        = c + yfStart;
      PetscInt points[9] = {p, yf, xf+1, yf+nyF, xf+0, v+0, v+1, v+nVx+1, v+nVx+0};

      ierr = FillClosureArray_Private(dm, section, 9, points, vArray, values);CHKERRQ(ierr);
    } else {
      /* 6 faces, 8 vertices
         Bottom-left-back vertex follows same order as cells
         Back z-face follows same order as cells
         Bottom y-face follows same order as cells
         Left x-face follows same order as cells

              14-----13
              /|    /|
             / | 2 / |
            / 5|  /  |
          10-----9  4|
           |  11-|---12
           |6 /  |  /
           | /1 3| /
           |/    |/
           7-----8
      */
      PetscInt c          = p - cStart;
      PetscInt points[15] = {p, c+zfStart, c+zfStart+nzF, c+yfStart, c+xfStart+nxF, c+yfStart+nyF, c+xfStart,
                             c+vStart+0, c+vStart+1, c+vStart+nVx+1, c+vStart+nVx+0, c+vStart+nVx*nVy+0, c+vStart+nVx*nVy+1, c+vStart+nVx*nVy+nVx+1, c+vStart+nVx*nVy+nVx+0};

      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Broken");
      ierr = FillClosureArray_Private(dm, section, 15, points, vArray, values);CHKERRQ(ierr);
    }
  } else if ((p >= vStart) || (p < vEnd)) {
    /* Vertex */
    ierr = FillClosureArray_Private(dm, section, 1, &p, vArray, values);CHKERRQ(ierr);
  } else if ((p >= fStart) || (p < fStart + nXF)) {
    /* X Face */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no faces in 1D");
    } else if (dim == 2) {
      /* 2 vertices: The bottom vertex has the same numbering as the face */
      PetscInt f         = p - xfStart;
      PetscInt points[3] = {p, f, f+nVx};

      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Broken");
      ierr = FillClosureArray_Private(dm, section, 3, points, vArray, values);CHKERRQ(ierr);
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    }
  } else if ((p >= fStart + nXF) || (p < fStart + nXF + nYF)) {
    /* Y Face */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no faces in 1D");
    } else if (dim == 2) {
      /* 2 vertices: The left vertex has the same numbering as the face */
      PetscInt f         = p - yfStart;
      PetscInt points[3] = {p, f, f+1};

      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Broken");
      ierr = FillClosureArray_Private(dm, section, 3, points, vArray, values);CHKERRQ(ierr);
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    }
  } else {
    /* Z Face */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no faces in 1D");
    } else if (dim == 2) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no z-faces in 2D");
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    }
  }
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecSetClosure"
PetscErrorCode DMDAVecSetClosure(DM dm, PetscSection section, Vec v, PetscInt p, const PetscScalar *values, InsertMode mode)
{
  DM_DA         *da  = (DM_DA *) dm->data;
  PetscInt       dim = da->dim;
  PetscScalar   *vArray;
  PetscInt       nVx, nVy, nxF, nXF, nyF, nYF, nzF, nZF;
  PetscInt       pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart, yfEnd, zfStart, zfEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  PetscValidPointer(values, 5);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!section) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "This DM has not default PetscSection");
  ierr = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 1,   &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, &nVx, &nVy, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  xfStart = fStart; xfEnd = xfStart+nXF;
  yfStart = xfEnd;  yfEnd = yfStart+nYF;
  zfStart = yfEnd;  zfEnd = zfStart+nZF;
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid point %d should be in [%d, %d)", p, pStart, pEnd);
  ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
  if ((p >= cStart) || (p < cEnd)) {
    /* Cell */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    } else if (dim == 2) {
      /* 4 faces, 4 vertices
         Bottom-left vertex follows same order as cells
         Bottom y-face same order as cells
         Left x-face follows same order as cells
         We number the quad:

           8--3--7
           |     |
           4  0  2
           |     |
           5--1--6
      */
      PetscInt c         = p - cStart;
      PetscInt points[9] = {p, c+yfStart, c+xfStart+1, c+yfStart+nyF, c+xfStart+0, c+vStart+0, c+vStart+1, c+vStart+nVx+1, c+vStart+nVx+0};

      ierr = FillClosureVec_Private(dm, section, 9, points, vArray, values, mode);CHKERRQ(ierr);
    } else {
      /* 6 faces, 8 vertices
         Bottom-left-back vertex follows same order as cells
         Back z-face follows same order as cells
         Bottom y-face follows same order as cells
         Left x-face follows same order as cells

              14-----13
              /|    /|
             / | 2 / |
            / 5|  /  |
          10-----9  4|
           |  11-|---12
           |6 /  |  /
           | /1 3| /
           |/    |/
           7-----8
      */
      PetscInt c          = p - cStart;
      PetscInt points[15] = {p, c+zfStart, c+zfStart+nzF, c+yfStart, c+xfStart+nxF, c+yfStart+nyF, c+xfStart,
                             c+vStart+0, c+vStart+1, c+vStart+nVx+1, c+vStart+nVx+0, c+vStart+nVx*nVy+0, c+vStart+nVx*nVy+1, c+vStart+nVx*nVy+nVx+1, c+vStart+nVx*nVy+nVx+0};

      ierr = FillClosureVec_Private(dm, section, 15, points, vArray, values, mode);CHKERRQ(ierr);
    }
  } else if ((p >= vStart) || (p < vEnd)) {
    /* Vertex */
    ierr = FillClosureVec_Private(dm, section, 1, &p, vArray, values, mode);CHKERRQ(ierr);
  } else if ((p >= fStart) || (p < fStart + nXF)) {
    /* X Face */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no faces in 1D");
    } else if (dim == 2) {
      /* 2 vertices: The bottom vertex has the same numbering as the face */
      PetscInt f         = p - xfStart;
      PetscInt points[3] = {p, f, f+nVx};

      ierr = FillClosureVec_Private(dm, section, 3, points, vArray, values, mode);CHKERRQ(ierr);
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    }
  } else if ((p >= fStart + nXF) || (p < fStart + nXF + nYF)) {
    /* Y Face */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no faces in 1D");
    } else if (dim == 2) {
      /* 2 vertices: The left vertex has the same numbering as the face */
      PetscInt f         = p - yfStart;
      PetscInt points[3] = {p, f, f+1};

      ierr = FillClosureVec_Private(dm, section, 3, points, vArray, values, mode);CHKERRQ(ierr);
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    }
  } else {
    /* Z Face */
    if (dim == 1) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no faces in 1D");
    } else if (dim == 2) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "There are no z-faces in 2D");
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
    }
  }
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeCellGeometry_2D"
PetscErrorCode DMDAComputeCellGeometry_2D(DM dm, const PetscScalar vertices[], const PetscReal refPoint[], PetscReal J[], PetscReal invJ[], PetscScalar *detJ)
{
  const PetscScalar x0   = vertices[0];
  const PetscScalar y0   = vertices[1];
  const PetscScalar x1   = vertices[2];
  const PetscScalar y1   = vertices[3];
  const PetscScalar x2   = vertices[4];
  const PetscScalar y2   = vertices[5];
  const PetscScalar x3   = vertices[6];
  const PetscScalar y3   = vertices[7];
  const PetscScalar f_01 = x2 - x1 - x3 + x0;
  const PetscScalar g_01 = y2 - y1 - y3 + y0;
  const PetscScalar x    = refPoint[0];
  const PetscScalar y    = refPoint[1];
  PetscReal         invDet;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell (%g,%g)--(%g,%g)--(%g,%g)--(%g,%g)\n", x0, y0, x1, y1, x2, y2, x3, y3);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Ref Point (%g,%g)\n", x, y);CHKERRQ(ierr);
  J[0] = (x1 - x0 + f_01*y) * 0.5; J[1] = (x3 - x0 + f_01*x) * 0.5;
  J[2] = (y1 - y0 + g_01*y) * 0.5; J[3] = (y3 - y0 + g_01*x) * 0.5;
  *detJ   = J[0]*J[3] - J[1]*J[2];
  invDet  = 1.0/(*detJ);
  invJ[0] =  invDet*J[3]; invJ[1] = -invDet*J[1];
  invJ[2] = -invDet*J[2]; invJ[3] =  invDet*J[0];
  ierr = PetscLogFlops(30);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeCellGeometry"
PetscErrorCode DMDAComputeCellGeometry(DM dm, PetscInt cell, PetscQuadrature *quad, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal detJ[])
{
  DM                 cdm;
  Vec                coordinates;
  const PetscScalar *vertices;
  PetscInt           dim, d, q;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMDAGetInfo(dm, &dim, 0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMDAGetCoordinateDA(dm, &cdm);CHKERRQ(ierr);
  ierr = DMDAVecGetClosure(cdm, PETSC_NULL, coordinates, cell, &vertices);CHKERRQ(ierr);
  for(d = 0; d < dim; ++d) {
    v0[d] = vertices[d];
  }
  switch(dim) {
  case 2:
    for(q = 0; q < quad->numQuadPoints; ++q) {
      ierr = DMDAComputeCellGeometry_2D(dm, vertices, &quad->quadPoints[q*dim], J, invJ, detJ);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Dimension %d not supported", dim);
  }
  PetscFunctionReturn(0);
}

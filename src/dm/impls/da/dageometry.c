#include <petsc/private/dmdaimpl.h>     /*I  "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "FillClosureArray_Static"
PETSC_STATIC_INLINE PetscErrorCode FillClosureArray_Static(DM dm, PetscSection section, PetscInt nP, const PetscInt points[], PetscScalar *vArray, PetscInt *csize, PetscScalar **array)
{
  PetscScalar    *a;
  PetscInt       pStart, pEnd, size = 0, dof, off, d, k, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (i = 0; i < nP; ++i) {
    const PetscInt p = points[i];

    if ((p < pStart) || (p >= pEnd)) continue;
    ierr  = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    size += dof;
  }
  if (csize) *csize = size;
  if (array) {
    ierr = DMGetWorkArray(dm, size, PETSC_SCALAR, &a);CHKERRQ(ierr);
    for (i = 0, k = 0; i < nP; ++i) {
      const PetscInt p = points[i];

      if ((p < pStart) || (p >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);

      for (d = 0; d < dof; ++d, ++k) a[k] = vArray[off+d];
    }
    *array = a;
  }
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
    for (i = 0, k = 0; i < nP; ++i) {
      ierr = PetscSectionGetDof(section, points[i], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, points[i], &off);CHKERRQ(ierr);

      for (d = 0; d < dof; ++d, ++k) vArray[off+d] = array[k];
    }
  } else {
    for (i = 0, k = 0; i < nP; ++i) {
      ierr = PetscSectionGetDof(section, points[i], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, points[i], &off);CHKERRQ(ierr);

      for (d = 0; d < dof; ++d, ++k) vArray[off+d] += array[k];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetPointArray_Private"
PETSC_STATIC_INLINE PetscErrorCode GetPointArray_Private(DM dm,PetscInt n,PetscInt *points,PetscInt *rn,const PetscInt **rpoints)
{
  PetscErrorCode ierr;
  PetscInt       *work;

  PetscFunctionBegin;
  if (rn) *rn = n;
  if (rpoints) {
    ierr     = DMGetWorkArray(dm,n,PETSC_INT,&work);CHKERRQ(ierr);
    ierr     = PetscMemcpy(work,points,n*sizeof(PetscInt));CHKERRQ(ierr);
    *rpoints = work;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RestorePointArray_Private"
PETSC_STATIC_INLINE PetscErrorCode RestorePointArray_Private(DM dm,PetscInt *rn,const PetscInt **rpoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rn) *rn = 0;
  if (rpoints) {
    ierr = DMRestoreWorkArray(dm,*rn, PETSC_INT, (void*) rpoints);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetTransitiveClosure"
PetscErrorCode DMDAGetTransitiveClosure(DM dm, PetscInt p, PetscBool useClosure, PetscInt *closureSize, PetscInt **closure)
{
  PetscInt       dim = dm->dim;
  PetscInt       nVx, nVy, nVz, nxF, nXF, nyF, nYF, nzF, nZF;
  PetscInt       pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!useClosure) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Star operation is not yet supported");
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm,  0,  &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm,  1,  &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, &nVx, &nVy, &nVz, NULL);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  xfStart = fStart; xfEnd = xfStart+nXF;
  yfStart = xfEnd;
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid point %d should be in [%d, %d)", p, pStart, pEnd);
  if ((p >= cStart) || (p < cEnd)) {
    /* Cell */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    else if (dim == 2) {
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
      PetscInt c  = p - cStart, cx = c % (nVx-1), cy = c / (nVx-1);
      PetscInt v  = cy*nVx + cx +  vStart;
      PetscInt xf = cy*nxF + cx + xfStart;
      PetscInt yf = c + yfStart;

      if (closureSize) {PetscValidPointer(closureSize, 4); *closureSize = 9;}
      if (!(*closure)) {ierr = DMGetWorkArray(dm, *closureSize, PETSC_INT, closure);CHKERRQ(ierr);}
      (*closure)[0] = p; (*closure)[1] = yf; (*closure)[2] = xf+1; (*closure)[3] = yf+nyF; (*closure)[4] = xf+0; (*closure)[5] = v+0; (*closure)[6]= v+1; (*closure)[7] = v+nVx+1; (*closure)[8] = v+nVx+0;
    } else {
      /* 6 faces, 12 edges, 8 vertices
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
#if 0
      PetscInt c  = p - cStart, cx = c % (nVx-1), cy = c / (nVx-1), cz = c / ((nVx-1)*(nVy-1));
      PetscInt v  = cy*nVx + cx +  vStart;
      PetscInt xf = cy*nxF + cx + xfStart;
      PetscInt yf = c + yfStart;

      if (closureSize) {PetscValidPointer(closureSize, 4); *closureSize = 26;}
      if (!(*closure)) {ierr = DMGetWorkArray(dm, *closureSize, PETSC_INT, closure);CHKERRQ(ierr);}
#endif
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  } else if ((p >= vStart) || (p < vEnd)) {
    /* Vertex */
    if (closureSize) {PetscValidPointer(closureSize, 4); *closureSize = 1;}
    if (!(*closure)) {ierr = DMGetWorkArray(dm, *closureSize, PETSC_INT, closure);CHKERRQ(ierr);}
    (*closure)[0] = p;
  } else if ((p >= fStart) || (p < fStart + nXF)) {
    /* X Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The bottom vertex has the same numbering as the face */
      PetscInt f = p - xfStart;

      if (closureSize) {PetscValidPointer(closureSize, 4); *closureSize = 3;}
      if (!(*closure)) {ierr = DMGetWorkArray(dm, *closureSize, PETSC_INT, closure);CHKERRQ(ierr);}
      (*closure)[0] = p; (*closure)[1] = f; (*closure)[2] = f+nVx;
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  } else if ((p >= fStart + nXF) || (p < fStart + nXF + nYF)) {
    /* Y Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The left vertex has the same numbering as the face */
      PetscInt f = p - yfStart;

      if (closureSize) {PetscValidPointer(closureSize, 4); *closureSize = 3;}
      if (!(*closure)) {ierr = DMGetWorkArray(dm, *closureSize, PETSC_INT, closure);CHKERRQ(ierr);}
      (*closure)[0] = p; (*closure)[1] = f; (*closure)[2]= f+1;
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  } else {
    /* Z Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no z-faces in 2D");
    else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreTransitiveClosure"
PetscErrorCode DMDARestoreTransitiveClosure(DM dm, PetscInt p, PetscBool useClosure, PetscInt *closureSize, PetscInt **closure)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, closure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetClosure"
PetscErrorCode DMDAGetClosure(DM dm, PetscSection section, PetscInt p,PetscInt *n,const PetscInt **closure)
{
  PetscInt       dim = dm->dim;
  PetscInt       nVx, nVy, nxF, nXF, nyF, nYF, nzF, nZF;
  PetscInt       pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (n) PetscValidIntPointer(n,4);
  if (closure) PetscValidPointer(closure, 5);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!section) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This DM has not default PetscSection");
  ierr    = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 0,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 1,   &fStart, &fEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  ierr    = DMDAGetNumVertices(dm, &nVx, &nVy, NULL, NULL);CHKERRQ(ierr);
  ierr    = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  xfStart = fStart; xfEnd = xfStart+nXF;
  yfStart = xfEnd;
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid point %d should be in [%d, %d)", p, pStart, pEnd);
  if ((p >= cStart) || (p < cEnd)) {
    /* Cell */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    else if (dim == 2) {
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
      PetscInt c  = p - cStart, cx = c % (nVx-1), cy = c / (nVx-1);
      PetscInt v  = cy*nVx + cx +  vStart;
      PetscInt xf = cy*nxF + cx + xfStart;
      PetscInt yf = c + yfStart;
      PetscInt points[9];

      /* Note: initializer list is not computable at compile time, hence we fill the array manually */
      points[0] = p; points[1] = yf; points[2] = xf+1; points[3] = yf+nyF; points[4] = xf+0; points[5] = v+0; points[6]= v+1; points[7] = v+nVx+1; points[8] = v+nVx+0;

      ierr = GetPointArray_Private(dm,9,points,n,closure);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  } else if ((p >= vStart) || (p < vEnd)) {
    /* Vertex */
    ierr = GetPointArray_Private(dm,1,&p,n,closure);CHKERRQ(ierr);
  } else if ((p >= fStart) || (p < fStart + nXF)) {
    /* X Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The bottom vertex has the same numbering as the face */
      PetscInt f = p - xfStart;
      PetscInt points[3];

      points[0] = p; points[1] = f; points[2] = f+nVx;
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Broken");
      ierr = GetPointArray_Private(dm,3,points,n,closure);CHKERRQ(ierr);
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  } else if ((p >= fStart + nXF) || (p < fStart + nXF + nYF)) {
    /* Y Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The left vertex has the same numbering as the face */
      PetscInt f = p - yfStart;
      PetscInt points[3];

      points[0] = p; points[1] = f; points[2]= f+1;
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Broken");
      ierr = GetPointArray_Private(dm, 3, points, n, closure);CHKERRQ(ierr);
    } else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  } else {
    /* Z Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no z-faces in 2D");
    else if (dim == 3) {
      /* 4 vertices */
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreClosure"
/* Zeros n and closure. */
PetscErrorCode DMDARestoreClosure(DM dm, PetscSection section, PetscInt p,PetscInt *n,const PetscInt **closure)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (n) PetscValidIntPointer(n,4);
  if (closure) PetscValidPointer(closure, 5);
  ierr = RestorePointArray_Private(dm,n,closure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetClosureScalar"
PetscErrorCode DMDAGetClosureScalar(DM dm, PetscSection section, PetscInt p, PetscScalar *vArray, PetscInt *csize, PetscScalar **values)
{
  PetscInt       dim = dm->dim;
  PetscInt       nVx, nVy, nxF, nXF, nyF, nYF, nzF, nZF;
  PetscInt       pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart, yfEnd, zfStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(vArray, 4);
  if (values) PetscValidPointer(values, 6);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!section) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This DM has not default PetscSection");
  ierr    = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 0,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 1,   &fStart, &fEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  ierr    = DMDAGetNumVertices(dm, &nVx, &nVy, NULL, NULL);CHKERRQ(ierr);
  ierr    = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  xfStart = fStart; xfEnd = xfStart+nXF;
  yfStart = xfEnd;  yfEnd = yfStart+nYF;
  zfStart = yfEnd;
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid point %d should be in [%d, %d)", p, pStart, pEnd);
  if ((p >= cStart) || (p < cEnd)) {
    /* Cell */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    else if (dim == 2) {
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
      PetscInt c  = p - cStart, cx = c % (nVx-1), cy = c / (nVx-1);
      PetscInt v  = cy*nVx + cx +  vStart;
      PetscInt xf = cx*nxF + cy + xfStart;
      PetscInt yf = c + yfStart;
      PetscInt points[9];

      points[0] = p;
      points[1] = yf;  points[2] = xf+nxF; points[3] = yf+nyF;  points[4] = xf;
      points[5] = v+0; points[6] = v+1;    points[7] = v+nVx+1; points[8] = v+nVx+0;
      ierr = FillClosureArray_Static(dm, section, 9, points, vArray, csize, values);CHKERRQ(ierr);
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
      PetscInt c = p - cStart;
      PetscInt points[15];

      points[0]  = p; points[1] = c+zfStart; points[2] = c+zfStart+nzF; points[3] = c+yfStart; points[4] = c+xfStart+nxF; points[5] = c+yfStart+nyF; points[6] = c+xfStart;
      points[7]  = c+vStart+0; points[8] = c+vStart+1; points[9] = c+vStart+nVx+1; points[10] = c+vStart+nVx+0; points[11] = c+vStart+nVx*nVy+0;
      points[12] = c+vStart+nVx*nVy+1; points[13] = c+vStart+nVx*nVy+nVx+1; points[14] = c+vStart+nVx*nVy+nVx+0;

      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Broken");
      ierr = FillClosureArray_Static(dm, section, 15, points, vArray, csize, values);CHKERRQ(ierr);
    }
  } else if ((p >= vStart) || (p < vEnd)) {
    /* Vertex */
    ierr = FillClosureArray_Static(dm, section, 1, &p, vArray, csize, values);CHKERRQ(ierr);
  } else if ((p >= fStart) || (p < fStart + nXF)) {
    /* X Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The bottom vertex has the same numbering as the face */
      PetscInt f = p - xfStart;
      PetscInt points[3];

      points[0] = p; points[1] = f; points[2] = f+nVx;
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Broken");
      ierr = FillClosureArray_Static(dm, section, 3, points, vArray, csize, values);CHKERRQ(ierr);
    } else if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  } else if ((p >= fStart + nXF) || (p < fStart + nXF + nYF)) {
    /* Y Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The left vertex has the same numbering as the face */
      PetscInt f = p - yfStart;
      PetscInt points[3];

      points[0] = p; points[1] = f; points[2] = f+1;
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Broken");
      ierr = FillClosureArray_Static(dm, section, 3, points, vArray, csize, values);CHKERRQ(ierr);
    } else if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  } else {
    /* Z Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no z-faces in 2D");
    else if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecGetClosure"
PetscErrorCode DMDAVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt p, PetscInt *csize, PetscScalar **values)
{
  PetscScalar    *vArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  if (values) PetscValidPointer(values, 6);
  ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
  ierr = DMDAGetClosureScalar(dm, section, p, vArray, csize, values);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreClosureScalar"
PetscErrorCode DMDARestoreClosureScalar(DM dm, PetscSection section, PetscInt p, PetscScalar *vArray, PetscInt *csize, PetscScalar **values)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(values, 6);
  ierr  = DMRestoreWorkArray(dm, csize ? *csize : 0, PETSC_SCALAR, (void*) values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecRestoreClosure"
PetscErrorCode DMDAVecRestoreClosure(DM dm, PetscSection section, Vec v, PetscInt p, PetscInt *csize, PetscScalar **values)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  PetscValidPointer(values, 5);
  ierr = DMDARestoreClosureScalar(dm, section, p, NULL, csize, values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetClosureScalar"
PetscErrorCode DMDASetClosureScalar(DM dm, PetscSection section, PetscInt p,PetscScalar *vArray, const PetscScalar *values, InsertMode mode)
{
  PetscInt       dim = dm->dim;
  PetscInt       nVx, nVy, nxF, nXF, nyF, nYF, nzF, nZF, nCx, nCy;
  PetscInt       pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart, yfEnd, zfStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(values, 4);
  PetscValidPointer(values, 5);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!section) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This DM has not default PetscSection");
  ierr    = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 0,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 1,   &fStart, &fEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  ierr    = DMDAGetNumCells(dm, &nCx, &nCy, NULL, NULL);CHKERRQ(ierr);
  ierr    = DMDAGetNumVertices(dm, &nVx, &nVy, NULL, NULL);CHKERRQ(ierr);
  ierr    = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  xfStart = fStart; xfEnd = xfStart+nXF;
  yfStart = xfEnd;  yfEnd = yfStart+nYF;
  zfStart = yfEnd;
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid point %d should be in [%d, %d)", p, pStart, pEnd);
  if ((p >= cStart) || (p < cEnd)) {
    /* Cell */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
    else if (dim == 2) {
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
      PetscInt c = p - cStart, cx = c % (nVx-1), cy = c / (nVx-1);
      PetscInt v  = cy*nVx + cx +  vStart;
      PetscInt xf = cx*nxF + cy + xfStart;
      PetscInt yf = c + yfStart;
      PetscInt points[9];

      points[0] = p;
      points[1] = yf;  points[2] = xf+nxF; points[3] = yf+nyF;  points[4] = xf;
      points[5] = v+0; points[6] = v+1;    points[7] = v+nVx+1; points[8] = v+nVx+0;
      ierr      = FillClosureVec_Private(dm, section, 9, points, vArray, values, mode);CHKERRQ(ierr);
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
      PetscInt c = p - cStart;
      PetscInt points[15];

      points[0]  = p; points[1] = c+zfStart; points[2] = c+zfStart+nzF; points[3] = c+yfStart; points[4] = c+xfStart+nxF; points[5] = c+yfStart+nyF; points[6] = c+xfStart;
      points[7]  = c+vStart+0; points[8] = c+vStart+1; points[9] = c+vStart+nVx+1; points[10] = c+vStart+nVx+0; points[11] = c+vStart+nVx*nVy+0; points[12] = c+vStart+nVx*nVy+1;
      points[13] = c+vStart+nVx*nVy+nVx+1; points[14] = c+vStart+nVx*nVy+nVx+0;
      ierr       = FillClosureVec_Private(dm, section, 15, points, vArray, values, mode);CHKERRQ(ierr);
    }
  } else if ((p >= vStart) || (p < vEnd)) {
    /* Vertex */
    ierr = FillClosureVec_Private(dm, section, 1, &p, vArray, values, mode);CHKERRQ(ierr);
  } else if ((p >= fStart) || (p < fStart + nXF)) {
    /* X Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The bottom vertex has the same numbering as the face */
      PetscInt f = p - xfStart;
      PetscInt points[3];

      points[0] = p; points[1] = f; points[2] = f+nVx;
      ierr      = FillClosureVec_Private(dm, section, 3, points, vArray, values, mode);CHKERRQ(ierr);
    } else if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  } else if ((p >= fStart + nXF) || (p < fStart + nXF + nYF)) {
    /* Y Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) {
      /* 2 vertices: The left vertex has the same numbering as the face */
      PetscInt f = p - yfStart;
      PetscInt points[3];

      points[0] = p; points[1] = f; points[2] = f+1;
      ierr      = FillClosureVec_Private(dm, section, 3, points, vArray, values, mode);CHKERRQ(ierr);
    } else if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  } else {
    /* Z Face */
    if (dim == 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no faces in 1D");
    else if (dim == 2) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "There are no z-faces in 2D");
    else if (dim == 3) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecSetClosure"
PetscErrorCode DMDAVecSetClosure(DM dm, PetscSection section, Vec v, PetscInt p, const PetscScalar *values, InsertMode mode)
{
  PetscScalar    *vArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  PetscValidPointer(values, 5);
  ierr = VecGetArray(v,&vArray);CHKERRQ(ierr);
  ierr = DMDASetClosureScalar(dm,section,p,vArray,values,mode);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&vArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAConvertToCell"
/*@
  DMDAConvertToCell - Convert (i,j,k) to local cell number

  Not Collective

  Input Parameter:
+ da - the distributed array
- s - A MatStencil giving (i,j,k)

  Output Parameter:
. cell - the local cell number

  Level: developer

.seealso: DMDAVecGetClosure()
@*/
PetscErrorCode DMDAConvertToCell(DM dm, MatStencil s, PetscInt *cell)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys /*, mz = da->Ze - da->Zs*/;
  const PetscInt il  = s.i - da->Xs/da->w, jl = dim > 1 ? s.j - da->Ys : 0, kl = dim > 2 ? s.k - da->Zs : 0;

  PetscFunctionBegin;
  *cell = -1;
  if ((s.i < da->Xs/da->w) || (s.i >= da->Xe/da->w))    SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil i %d should be in [%d, %d)", s.i, da->Xs/da->w, da->Xe/da->w);
  if ((dim > 1) && ((s.j < da->Ys) || (s.j >= da->Ye))) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil j %d should be in [%d, %d)", s.j, da->Ys, da->Ye);
  if ((dim > 2) && ((s.k < da->Zs) || (s.k >= da->Ze))) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil k %d should be in [%d, %d)", s.k, da->Zs, da->Ze);
  *cell = (kl*my + jl)*mx + il;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeCellGeometry_2D"
PetscErrorCode DMDAComputeCellGeometry_2D(DM dm, const PetscScalar vertices[], const PetscReal refPoint[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
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
#if 0
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell (%g,%g)--(%g,%g)--(%g,%g)--(%g,%g)\n",
                     PetscRealPart(x0),PetscRealPart(y0),PetscRealPart(x1),PetscRealPart(y1),PetscRealPart(x2),PetscRealPart(y2),PetscRealPart(x3),PetscRealPart(y3));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Ref Point (%g,%g)\n", PetscRealPart(x), PetscRealPart(y));CHKERRQ(ierr);
#endif
  J[0]    = PetscRealPart(x1 - x0 + f_01*y) * 0.5; J[1] = PetscRealPart(x3 - x0 + f_01*x) * 0.5;
  J[2]    = PetscRealPart(y1 - y0 + g_01*y) * 0.5; J[3] = PetscRealPart(y3 - y0 + g_01*x) * 0.5;
  *detJ   = J[0]*J[3] - J[1]*J[2];
  invDet  = 1.0/(*detJ);
  if (invJ) {
    invJ[0] =  invDet*J[3]; invJ[1] = -invDet*J[1];
    invJ[2] = -invDet*J[2]; invJ[3] =  invDet*J[0];
  }
  ierr    = PetscLogFlops(30);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAComputeCellGeometryFEM"
PetscErrorCode DMDAComputeCellGeometryFEM(DM dm, PetscInt cell, PetscQuadrature quad, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal detJ[])
{
  DM               cdm;
  Vec              coordinates;
  const PetscReal *quadPoints;
  PetscScalar     *vertices = NULL;
  PetscInt         numQuadPoints, csize, dim, d, q;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMDAGetInfo(dm, &dim, 0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMDAVecGetClosure(cdm, NULL, coordinates, cell, &csize, &vertices);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) v0[d] = PetscRealPart(vertices[d]);
  switch (dim) {
  case 2:
    ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, NULL);CHKERRQ(ierr);
    for (q = 0; q < numQuadPoints; ++q) {
      ierr = DMDAComputeCellGeometry_2D(dm, vertices, &quadPoints[q*dim], J, invJ, detJ);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Dimension %d not supported", dim);
  }
  ierr = DMDAVecRestoreClosure(cdm, NULL, coordinates, cell, &csize, &vertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

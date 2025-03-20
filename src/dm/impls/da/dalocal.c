/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/
#include <petscbt.h>
#include <petscsf.h>
#include <petscds.h>
#include <petscfe.h>

/*
   This allows the DMDA vectors to properly tell MATLAB their dimensions
*/
#if defined(PETSC_HAVE_MATLAB)
  #include <engine.h> /* MATLAB include file */
  #include <mex.h>    /* MATLAB include file */
static PetscErrorCode VecMatlabEnginePut_DA2d(PetscObject obj, void *mengine)
{
  PetscInt     n, m;
  Vec          vec = (Vec)obj;
  PetscScalar *array;
  mxArray     *mat;
  DM           da;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vec, &da));
  PetscCheck(da, PetscObjectComm((PetscObject)vec), PETSC_ERR_ARG_WRONGSTATE, "Vector not associated with a DMDA");
  PetscCall(DMDAGetGhostCorners(da, 0, 0, 0, &m, &n, 0));

  PetscCall(VecGetArray(vec, &array));
  #if !defined(PETSC_USE_COMPLEX)
  mat = mxCreateDoubleMatrix(m, n, mxREAL);
  #else
  mat = mxCreateDoubleMatrix(m, n, mxCOMPLEX);
  #endif
  PetscCall(PetscArraycpy(mxGetPr(mat), array, n * m));
  PetscCall(PetscObjectName(obj));
  engPutVariable((Engine *)mengine, obj->name, mat);

  PetscCall(VecRestoreArray(vec, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode DMCreateLocalVector_DA(DM da, Vec *g)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscAssertPointer(g, 2);
  PetscCall(VecCreate(PETSC_COMM_SELF, g));
  PetscCall(VecSetSizes(*g, dd->nlocal, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(*g, dd->w));
  PetscCall(VecSetType(*g, da->vectype));
  if (dd->nlocal < da->bind_below) {
    PetscCall(VecSetBindingPropagates(*g, PETSC_TRUE));
    PetscCall(VecBindToCPU(*g, PETSC_TRUE));
  }
  PetscCall(VecSetDM(*g, da));
#if defined(PETSC_HAVE_MATLAB)
  if (dd->w == 1 && da->dim == 2) PetscCall(PetscObjectComposeFunction((PetscObject)*g, "PetscMatlabEnginePut_C", VecMatlabEnginePut_DA2d));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetNumCells - Get the number of cells (or vertices) in the local piece of the `DMDA`. This includes ghost cells.

  Input Parameter:
. dm - The `DMDA` object

  Output Parameters:
+ numCellsX - The number of local cells in the x-direction
. numCellsY - The number of local cells in the y-direction
. numCellsZ - The number of local cells in the z-direction
- numCells  - The number of local cells

  Level: developer

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetCellPoint()`
@*/
PetscErrorCode DMDAGetNumCells(DM dm, PeOp PetscInt *numCellsX, PeOp PetscInt *numCellsY, PeOp PetscInt *numCellsZ, PeOp PetscInt *numCells)
{
  DM_DA         *da  = (DM_DA *)dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx = (da->Xe - da->Xs) / da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nC = (mx) * (dim > 1 ? (my) * (dim > 2 ? (mz) : 1) : 1);

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  if (numCellsX) {
    PetscAssertPointer(numCellsX, 2);
    *numCellsX = mx;
  }
  if (numCellsY) {
    PetscAssertPointer(numCellsY, 3);
    *numCellsY = my;
  }
  if (numCellsZ) {
    PetscAssertPointer(numCellsZ, 4);
    *numCellsZ = mz;
  }
  if (numCells) {
    PetscAssertPointer(numCells, 5);
    *numCells = nC;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetCellPoint - Get the `DM` point corresponding to the tuple (i, j, k) in the `DMDA`

  Input Parameters:
+ dm - The `DMDA` object
. i  - The global x index for the cell
. j  - The global y index for the cell
- k  - The global z index for the cell

  Output Parameter:
. point - The local `DM` point

  Level: developer

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetNumCells()`
@*/
PetscErrorCode DMDAGetCellPoint(DM dm, PetscInt i, PetscInt j, PetscInt k, PetscInt *point)
{
  const PetscInt dim = dm->dim;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  PetscAssertPointer(point, 5);
  PetscCall(DMDAGetLocalInfo(dm, &info));
  if (dim > 0) PetscCheck(!(i < info.gxs) && !(i >= info.gxs + info.gxm), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "X index %" PetscInt_FMT " not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", i, info.gxs, info.gxs + info.gxm);
  if (dim > 1) PetscCheck(!(j < info.gys) && !(j >= info.gys + info.gym), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Y index %" PetscInt_FMT " not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", j, info.gys, info.gys + info.gym);
  if (dim > 2) PetscCheck(!(k < info.gzs) && !(k >= info.gzs + info.gzm), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Z index %" PetscInt_FMT " not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", k, info.gzs, info.gzs + info.gzm);
  *point = i + (dim > 1 ? (j + (dim > 2 ? k * info.gym : 0)) * info.gxm : 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMDAGetNumVertices(DM dm, PetscInt *numVerticesX, PetscInt *numVerticesY, PetscInt *numVerticesZ, PetscInt *numVertices)
{
  DM_DA         *da  = (DM_DA *)dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx = (da->Xe - da->Xs) / da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nVx = mx + 1;
  const PetscInt nVy = dim > 1 ? (my + 1) : 1;
  const PetscInt nVz = dim > 2 ? (mz + 1) : 1;
  const PetscInt nV  = nVx * nVy * nVz;

  PetscFunctionBegin;
  if (numVerticesX) {
    PetscAssertPointer(numVerticesX, 2);
    *numVerticesX = nVx;
  }
  if (numVerticesY) {
    PetscAssertPointer(numVerticesY, 3);
    *numVerticesY = nVy;
  }
  if (numVerticesZ) {
    PetscAssertPointer(numVerticesZ, 4);
    *numVerticesZ = nVz;
  }
  if (numVertices) {
    PetscAssertPointer(numVertices, 5);
    *numVertices = nV;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMDAGetNumFaces(DM dm, PetscInt *numXFacesX, PetscInt *numXFaces, PetscInt *numYFacesY, PetscInt *numYFaces, PetscInt *numZFacesZ, PetscInt *numZFaces)
{
  DM_DA         *da  = (DM_DA *)dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx = (da->Xe - da->Xs) / da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nxF = (dim > 1 ? (my) * (dim > 2 ? (mz) : 1) : 1);
  const PetscInt nXF = (mx + 1) * nxF;
  const PetscInt nyF = mx * (dim > 2 ? mz : 1);
  const PetscInt nYF = dim > 1 ? (my + 1) * nyF : 0;
  const PetscInt nzF = mx * (dim > 1 ? my : 0);
  const PetscInt nZF = dim > 2 ? (mz + 1) * nzF : 0;

  PetscFunctionBegin;
  if (numXFacesX) {
    PetscAssertPointer(numXFacesX, 2);
    *numXFacesX = nxF;
  }
  if (numXFaces) {
    PetscAssertPointer(numXFaces, 3);
    *numXFaces = nXF;
  }
  if (numYFacesY) {
    PetscAssertPointer(numYFacesY, 4);
    *numYFacesY = nyF;
  }
  if (numYFaces) {
    PetscAssertPointer(numYFaces, 5);
    *numYFaces = nYF;
  }
  if (numZFacesZ) {
    PetscAssertPointer(numZFacesZ, 6);
    *numZFacesZ = nzF;
  }
  if (numZFaces) {
    PetscAssertPointer(numZFaces, 7);
    *numZFaces = nZF;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetHeightStratum - Get the bounds [`start`, `end`) for all points at a certain height.

  Not Collective

  Input Parameters:
+ dm     - The `DMDA` object
- height - The requested height

  Output Parameters:
+ pStart - The first point at this `height`
- pEnd   - One beyond the last point at this `height`

  Level: developer

  Note:
  See `DMPlexGetHeightStratum()` for the meaning of these values

.seealso: [](ch_unstructured), `DM`, `DMDA`,  `DMPlexGetDepthStratum()`, `DMPlexGetHeightStratum()`, `DMPlexGetCellTypeStratum()`, `DMPlexGetDepth()`,
          `DMPlexGetDepthLabel()`, `DMPlexGetPointDepth()`, `DMPlexSymmetrize()`, `DMPlexInterpolate()`, `DMDAGetDepthStratum()`
@*/
PetscErrorCode DMDAGetHeightStratum(DM dm, PetscInt height, PeOp PetscInt *pStart, PeOp PetscInt *pEnd)
{
  const PetscInt dim = dm->dim;
  PetscInt       nC, nV, nXF, nYF, nZF;

  PetscFunctionBegin;
  if (pStart) PetscAssertPointer(pStart, 3);
  if (pEnd) PetscAssertPointer(pEnd, 4);
  PetscCall(DMDAGetNumCells(dm, NULL, NULL, NULL, &nC));
  PetscCall(DMDAGetNumVertices(dm, NULL, NULL, NULL, &nV));
  PetscCall(DMDAGetNumFaces(dm, NULL, &nXF, NULL, &nYF, NULL, &nZF));
  if (height == 0) {
    /* Cells */
    if (pStart) *pStart = 0;
    if (pEnd) *pEnd = nC;
  } else if (height == 1) {
    /* Faces */
    if (pStart) *pStart = nC + nV;
    if (pEnd) *pEnd = nC + nV + nXF + nYF + nZF;
  } else if (height == dim) {
    /* Vertices */
    if (pStart) *pStart = nC;
    if (pEnd) *pEnd = nC + nV;
  } else if (height < 0) {
    /* All points */
    if (pStart) *pStart = 0;
    if (pEnd) *pEnd = nC + nV + nXF + nYF + nZF;
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No points of height %" PetscInt_FMT " in the DA", height);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetDepthStratum - Get the bounds [`start`, `end`) for all points at a certain depth.

  Not Collective

  Input Parameters:
+ dm    - The `DMDA` object
- depth - The requested depth

  Output Parameters:
+ pStart - The first point at this `depth`
- pEnd   - One beyond the last point at this `depth`

  Level: developer

  Note:
  See `DMPlexGetDepthStratum()` for the meaning of these values

.seealso: [](ch_unstructured), `DM`, `DMDA`,  `DMPlexGetDepthStratum()`, `DMPlexGetHeightStratum()`, `DMPlexGetCellTypeStratum()`, `DMPlexGetDepth()`,
          `DMPlexGetDepthLabel()`, `DMPlexGetPointDepth()`, `DMPlexSymmetrize()`, `DMPlexInterpolate()`, `DMDAGetHeightStratum()`
@*/
PetscErrorCode DMDAGetDepthStratum(DM dm, PetscInt depth, PeOp PetscInt *pStart, PeOp PetscInt *pEnd)
{
  const PetscInt dim = dm->dim;
  PetscInt       nC, nV, nXF, nYF, nZF;

  PetscFunctionBegin;
  if (pStart) PetscAssertPointer(pStart, 3);
  if (pEnd) PetscAssertPointer(pEnd, 4);
  PetscCall(DMDAGetNumCells(dm, NULL, NULL, NULL, &nC));
  PetscCall(DMDAGetNumVertices(dm, NULL, NULL, NULL, &nV));
  PetscCall(DMDAGetNumFaces(dm, NULL, &nXF, NULL, &nYF, NULL, &nZF));
  if (depth == dim) {
    /* Cells */
    if (pStart) *pStart = 0;
    if (pEnd) *pEnd = nC;
  } else if (depth == dim - 1) {
    /* Faces */
    if (pStart) *pStart = nC + nV;
    if (pEnd) *pEnd = nC + nV + nXF + nYF + nZF;
  } else if (depth == 0) {
    /* Vertices */
    if (pStart) *pStart = nC;
    if (pEnd) *pEnd = nC + nV;
  } else if (depth < 0) {
    /* All points */
    if (pStart) *pStart = 0;
    if (pEnd) *pEnd = nC + nV + nXF + nYF + nZF;
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No points of depth %" PetscInt_FMT " in the DA", depth);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDASetVertexCoordinates - Sets the lower and upper coordinates for a `DMDA`

  Logically Collective

  Input Parameters:
+ dm - The `DMDA` object
. xl - the lower x coordinate
. xu - the upper x coordinate
. yl - the lower y coordinate
. yu - the upper y coordinate
. zl - the lower z coordinate
- zu - the upper z coordinate

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMDA`
@*/
PetscErrorCode DMDASetVertexCoordinates(DM dm, PetscReal xl, PetscReal xu, PetscReal yl, PetscReal yu, PetscReal zl, PetscReal zu)
{
  DM_DA       *da = (DM_DA *)dm->data;
  Vec          coordinates;
  PetscSection section;
  PetscScalar *coords;
  PetscReal    h[3];
  PetscInt     dim, size, M, N, P, nVx, nVy, nVz, nV, vStart, vEnd, v, i, j, k;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  PetscCall(DMDAGetInfo(dm, &dim, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCheck(dim <= 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "The following code only works for dim <= 3");
  h[0] = (xu - xl) / M;
  h[1] = (yu - yl) / N;
  h[2] = (zu - zl) / P;
  PetscCall(DMDAGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMDAGetNumVertices(dm, &nVx, &nVy, &nVz, &nV));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(PetscSectionSetNumFields(section, 1));
  PetscCall(PetscSectionSetFieldComponents(section, 0, dim));
  PetscCall(PetscSectionSetChart(section, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) PetscCall(PetscSectionSetDof(section, v, dim));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(PetscSectionGetStorageSize(section, &size));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, size, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
  PetscCall(VecGetArray(coordinates, &coords));
  for (k = 0; k < nVz; ++k) {
    PetscInt ind[3], d, off;

    ind[0] = 0;
    ind[1] = 0;
    ind[2] = k + da->zs;
    for (j = 0; j < nVy; ++j) {
      ind[1] = j + da->ys;
      for (i = 0; i < nVx; ++i) {
        const PetscInt vertex = (k * nVy + j) * nVx + i + vStart;

        PetscCall(PetscSectionGetOffset(section, vertex, &off));
        ind[0] = i + da->xs;
        for (d = 0; d < dim; ++d) coords[off + d] = h[d] * ind[d];
      }
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinateSection(dm, PETSC_DETERMINE, section));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(VecDestroy(&coordinates));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDAGetArray - Gets a work array for a `DMDA`

  Input Parameters:
+ da      - a `DMDA`
- ghosted - do you want arrays for the ghosted or nonghosted patch

  Output Parameter:
. vptr - array data structured

  Level: advanced

  Notes:
  The vector values are NOT initialized and may have garbage in them, so you may need
  to zero them.

  Use `DMDARestoreArray()` to return the array

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDARestoreArray()`
@*/
PetscErrorCode DMDAGetArray(DM da, PetscBool ghosted, void *vptr)
{
  PetscInt j, i, xs, ys, xm, ym, zs, zm;
  char    *iarray_start;
  void   **iptr = (void **)vptr;
  DM_DA   *dd   = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  if (ghosted) {
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayghostedin[i]) {
        *iptr                 = dd->arrayghostedin[i];
        iarray_start          = (char *)dd->startghostedin[i];
        dd->arrayghostedin[i] = NULL;
        dd->startghostedin[i] = NULL;

        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    zs = dd->Zs;
    xm = dd->Xe - dd->Xs;
    ym = dd->Ye - dd->Ys;
    zm = dd->Ze - dd->Zs;
  } else {
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayin[i]) {
        *iptr          = dd->arrayin[i];
        iarray_start   = (char *)dd->startin[i];
        dd->arrayin[i] = NULL;
        dd->startin[i] = NULL;

        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    zs = dd->zs;
    xm = dd->xe - dd->xs;
    ym = dd->ye - dd->ys;
    zm = dd->ze - dd->zs;
  }

  switch (da->dim) {
  case 1: {
    void *ptr;

    PetscCall(PetscMalloc(xm * sizeof(PetscScalar), &iarray_start));

    ptr   = (void *)((PetscScalar *)iarray_start - xs);
    *iptr = ptr;
    break;
  }
  case 2: {
    void **ptr;

    PetscCall(PetscMalloc((ym + 1) * sizeof(void *) + xm * ym * sizeof(PetscScalar), &iarray_start));

    ptr = (void **)(iarray_start + xm * ym * sizeof(PetscScalar) - ys * sizeof(void *));
    for (j = ys; j < ys + ym; j++) ptr[j] = iarray_start + sizeof(PetscScalar) * (xm * (j - ys) - xs);
    *iptr = (void *)ptr;
    break;
  }
  case 3: {
    void ***ptr, **bptr;

    PetscCall(PetscMalloc((zm + 1) * sizeof(void **) + (ym * zm + 1) * sizeof(void *) + xm * ym * zm * sizeof(PetscScalar), &iarray_start));

    ptr  = (void ***)(iarray_start + xm * ym * zm * sizeof(PetscScalar) - zs * sizeof(void *));
    bptr = (void **)(iarray_start + xm * ym * zm * sizeof(PetscScalar) + zm * sizeof(void **));
    for (i = zs; i < zs + zm; i++) ptr[i] = bptr + ((i - zs) * ym - ys);
    for (i = zs; i < zs + zm; i++) {
      for (j = ys; j < ys + ym; j++) ptr[i][j] = iarray_start + sizeof(PetscScalar) * (xm * ym * (i - zs) + xm * (j - ys) - xs);
    }
    *iptr = (void *)ptr;
    break;
  }
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Dimension %" PetscInt_FMT " not supported", da->dim);
  }

done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayghostedout[i]) {
        dd->arrayghostedout[i] = *iptr;
        dd->startghostedout[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayout[i]) {
        dd->arrayout[i] = *iptr;
        dd->startout[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDARestoreArray - Restores an array for a `DMDA` obtained with  `DMDAGetArray()`

  Input Parameters:
+ da      - information about my local patch
. ghosted - do you want arrays for the ghosted or nonghosted patch
- vptr    - array data structured

  Level: advanced

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetArray()`
@*/
PetscErrorCode DMDARestoreArray(DM da, PetscBool ghosted, void *vptr)
{
  PetscInt i;
  void   **iptr = (void **)vptr, *iarray_start = NULL;
  DM_DA   *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  if (ghosted) {
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayghostedout[i] == *iptr) {
        iarray_start           = dd->startghostedout[i];
        dd->arrayghostedout[i] = NULL;
        dd->startghostedout[i] = NULL;
        break;
      }
    }
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayghostedin[i]) {
        dd->arrayghostedin[i] = *iptr;
        dd->startghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayout[i] == *iptr) {
        iarray_start    = dd->startout[i];
        dd->arrayout[i] = NULL;
        dd->startout[i] = NULL;
        break;
      }
    }
    for (i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayin[i]) {
        dd->arrayin[i] = *iptr;
        dd->startin[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/dmpleximpl.h>    /*I      "petscdmplex.h"    I*/
#include <petsc/private/dmlabelimpl.h>   /*I      "petscdmlabel.h"   I*/
#include <petscsf.h>

static PetscErrorCode DMPlexCellIsHybrid_Internal(DM dm, PetscInt p, PetscBool *isHybrid)
{
  DMPolytopeType ct;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
  switch (ct) {
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      *isHybrid = PETSC_TRUE;
    default: *isHybrid = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetTensorPrismBounds_Internal(DM dm, PetscInt dim, PetscInt *cStart, PetscInt *cEnd)
{
  DMLabel        ctLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cStart) *cStart = -1;
  if (cEnd)   *cEnd   = -1;
  ierr = DMPlexGetCellTypeLabel(dm, &ctLabel);CHKERRQ(ierr);
  switch (dim) {
    case 1: ierr = DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_POINT_PRISM_TENSOR, cStart, cEnd);CHKERRQ(ierr);break;
    case 2: ierr = DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_SEG_PRISM_TENSOR, cStart, cEnd);CHKERRQ(ierr);break;
    case 3:
      ierr = DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_TRI_PRISM_TENSOR, cStart, cEnd);CHKERRQ(ierr);
      if (*cStart < 0) {ierr = DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_QUAD_PRISM_TENSOR, cStart, cEnd);CHKERRQ(ierr);}
      break;
    default: PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkBoundaryFaces_Internal(DM dm, PetscInt val, PetscInt cellHeight, DMLabel label)
{
  PetscInt       fStart, fEnd, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, &fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {
    PetscInt supportSize;

    ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
    if (supportSize == 1) {
      if (val < 0) {
        PetscInt *closure = NULL;
        PetscInt  clSize, cl, cval;

        ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
        for (cl = 0; cl < clSize*2; cl += 2) {
          ierr = DMLabelGetValue(label, closure[cl], &cval);CHKERRQ(ierr);
          if (cval < 0) continue;
          ierr = DMLabelSetValue(label, f, cval);CHKERRQ(ierr);
          break;
        }
        if (cl == clSize*2) {ierr = DMLabelSetValue(label, f, 1);CHKERRQ(ierr);}
        ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
      } else {
        ierr = DMLabelSetValue(label, f, val);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexMarkBoundaryFaces - Mark all faces on the boundary

  Not Collective

  Input Parameter:
+ dm - The original DM
- val - The marker value, or PETSC_DETERMINE to use some value in the closure (or 1 if none are found)

  Output Parameter:
. label - The DMLabel marking boundary faces with the given value

  Level: developer

.seealso: DMLabelCreate(), DMCreateLabel()
@*/
PetscErrorCode DMPlexMarkBoundaryFaces(DM dm, PetscInt val, DMLabel label)
{
  DMPlexInterpolatedFlag  flg;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexIsInterpolated(dm, &flg);CHKERRQ(ierr);
  if (flg != DMPLEX_INTERPOLATED_FULL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not fully interpolated on this rank");
  ierr = DMPlexMarkBoundaryFaces_Internal(dm, val, 0, label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLabelComplete_Internal(DM dm, DMLabel label, PetscBool completeCells)
{
  IS              valueIS;
  PetscSF         sfPoint;
  const PetscInt *values;
  PetscInt        numValues, v, cStart, cEnd, nroots;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    ierr = DMLabelGetStratumSize(label, values[v], &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt  q = points[p];
      PetscInt *closure = NULL;
      PetscInt  closureSize, c;

      if (cStart <= q && q < cEnd && !completeCells) { /* skip cells */
        continue;
      }
      ierr = DMPlexGetTransitiveClosure(dm, q, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        ierr = DMLabelSetValue(label, closure[c], values[v]);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, q, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    DMLabel         lblRoots, lblLeaves;
    IS              valueIS, pointIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscErrorCode  ierr;

    /* Pull point contributions from remote leaves into local roots */
    ierr = DMLabelGather(label, sfPoint, &lblLeaves);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(lblLeaves, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      const PetscInt value = values[v];

      ierr = DMLabelGetStratumIS(lblLeaves, value, &pointIS);CHKERRQ(ierr);
      ierr = DMLabelInsertIS(label, pointIS, value);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&lblLeaves);CHKERRQ(ierr);
    /* Push point contributions from roots into remote leaves */
    ierr = DMLabelDistribute(label, sfPoint, &lblRoots);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(lblRoots, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      const PetscInt value = values[v];

      ierr = DMLabelGetStratumIS(lblRoots, value, &pointIS);CHKERRQ(ierr);
      ierr = DMLabelInsertIS(label, pointIS, value);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&lblRoots);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelComplete - Starting with a label marking points on a surface, we add the transitive closure to the surface

  Input Parameters:
+ dm - The DM
- label - A DMLabel marking the surface points

  Output Parameter:
. label - A DMLabel marking all surface points in the transitive closure

  Level: developer

.seealso: DMPlexLabelCohesiveComplete()
@*/
PetscErrorCode DMPlexLabelComplete(DM dm, DMLabel label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexLabelComplete_Internal(dm, label, PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelAddCells - Starting with a label marking points on a surface, we add a cell for each point

  Input Parameters:
+ dm - The DM
- label - A DMLabel marking the surface points

  Output Parameter:
. label - A DMLabel incorporating cells

  Level: developer

  Note: The cells allow FEM boundary conditions to be applied using the cell geometry

.seealso: DMPlexLabelAddFaceCells(), DMPlexLabelComplete(), DMPlexLabelCohesiveComplete()
@*/
PetscErrorCode DMPlexLabelAddCells(DM dm, DMLabel label)
{
  IS              valueIS;
  const PetscInt *values;
  PetscInt        numValues, v, cStart, cEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    ierr = DMLabelGetStratumSize(label, values[v], &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = closureSize-1; cl > 0; --cl) {
        const PetscInt cell = closure[cl*2];
        if ((cell >= cStart) && (cell < cEnd)) {ierr = DMLabelSetValue(label, cell, values[v]);CHKERRQ(ierr); break;}
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelAddFaceCells - Starting with a label marking faces on a surface, we add a cell for each face

  Input Parameters:
+ dm - The DM
- label - A DMLabel marking the surface points

  Output Parameter:
. label - A DMLabel incorporating cells

  Level: developer

  Note: The cells allow FEM boundary conditions to be applied using the cell geometry

.seealso: DMPlexLabelAddCells(), DMPlexLabelComplete(), DMPlexLabelCohesiveComplete()
@*/
PetscErrorCode DMPlexLabelAddFaceCells(DM dm, DMLabel label)
{
  IS              valueIS;
  const PetscInt *values;
  PetscInt        numValues, v, cStart, cEnd, fStart, fEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    ierr = DMLabelGetStratumSize(label, values[v], &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      const PetscInt face = points[p];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, cl;

      if ((face < fStart) || (face >= fEnd)) continue;
      ierr = DMPlexGetTransitiveClosure(dm, face, PETSC_FALSE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = closureSize-1; cl > 0; --cl) {
        const PetscInt cell = closure[cl*2];
        if ((cell >= cStart) && (cell < cEnd)) {ierr = DMLabelSetValue(label, cell, values[v]);CHKERRQ(ierr); break;}
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, face, PETSC_FALSE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelClearCells - Remove cells from a label

  Input Parameters:
+ dm - The DM
- label - A DMLabel marking surface points and their adjacent cells

  Output Parameter:
. label - A DMLabel without cells

  Level: developer

  Note: This undoes DMPlexLabelAddCells() or DMPlexLabelAddFaceCells()

.seealso: DMPlexLabelComplete(), DMPlexLabelCohesiveComplete(), DMPlexLabelAddCells()
@*/
PetscErrorCode DMPlexLabelClearCells(DM dm, DMLabel label)
{
  IS              valueIS;
  const PetscInt *values;
  PetscInt        numValues, v, cStart, cEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    ierr = DMLabelGetStratumSize(label, values[v], &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt point = points[p];

      if (point >= cStart && point < cEnd) {
        ierr = DMLabelClearValue(label,point,values[v]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* take (oldEnd, added) pairs, ordered by height and convert them to (oldstart, newstart) pairs, ordered by ascending
 * index (skipping first, which is (0,0)) */
PETSC_STATIC_INLINE PetscErrorCode DMPlexShiftPointSetUp_Internal(PetscInt depth, PetscInt depthShift[])
{
  PetscInt d, off = 0;

  PetscFunctionBegin;
  /* sort by (oldend): yes this is an O(n^2) sort, we expect depth <= 3 */
  for (d = 0; d < depth; d++) {
    PetscInt firstd = d;
    PetscInt firstStart = depthShift[2*d];
    PetscInt e;

    for (e = d+1; e <= depth; e++) {
      if (depthShift[2*e] < firstStart) {
        firstd = e;
        firstStart = depthShift[2*d];
      }
    }
    if (firstd != d) {
      PetscInt swap[2];

      e = firstd;
      swap[0] = depthShift[2*d];
      swap[1] = depthShift[2*d+1];
      depthShift[2*d]   = depthShift[2*e];
      depthShift[2*d+1] = depthShift[2*e+1];
      depthShift[2*e]   = swap[0];
      depthShift[2*e+1] = swap[1];
    }
  }
  /* convert (oldstart, added) to (oldstart, newstart) */
  for (d = 0; d <= depth; d++) {
    off += depthShift[2*d+1];
    depthShift[2*d+1] = depthShift[2*d] + off;
  }
  PetscFunctionReturn(0);
}

/* depthShift is a list of (old, new) pairs */
PETSC_STATIC_INLINE PetscInt DMPlexShiftPoint_Internal(PetscInt p, PetscInt depth, PetscInt depthShift[])
{
  PetscInt d;
  PetscInt newOff = 0;

  for (d = 0; d <= depth; d++) {
    if (p < depthShift[2*d]) return p + newOff;
    else newOff = depthShift[2*d+1] - depthShift[2*d];
  }
  return p + newOff;
}

/* depthShift is a list of (old, new) pairs */
PETSC_STATIC_INLINE PetscInt DMPlexShiftPointInverse_Internal(PetscInt p, PetscInt depth, PetscInt depthShift[])
{
  PetscInt d;
  PetscInt newOff = 0;

  for (d = 0; d <= depth; d++) {
    if (p < depthShift[2*d+1]) return p + newOff;
    else newOff = depthShift[2*d] - depthShift[2*d+1];
  }
  return p + newOff;
}

static PetscErrorCode DMPlexShiftSizes_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt       depth = 0, d, pStart, pEnd, p;
  DMLabel        depthLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth < 0) PetscFunctionReturn(0);
  /* Step 1: Expand chart */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  pEnd = DMPlexShiftPoint_Internal(pEnd,depth,depthShift);
  ierr = DMPlexSetChart(dmNew, pStart, pEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(dmNew,"depth");CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dmNew,&depthLabel);CHKERRQ(ierr);
  ierr = DMCreateLabel(dmNew, "celltype");CHKERRQ(ierr);
  /* Step 2: Set cone and support sizes */
  for (d = 0; d <= depth; ++d) {
    PetscInt pStartNew, pEndNew;
    IS pIS;

    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    pStartNew = DMPlexShiftPoint_Internal(pStart, depth, depthShift);
    pEndNew = DMPlexShiftPoint_Internal(pEnd, depth, depthShift);
    ierr = ISCreateStride(PETSC_COMM_SELF, pEndNew - pStartNew, pStartNew, 1, &pIS);CHKERRQ(ierr);
    ierr = DMLabelSetStratumIS(depthLabel, d, pIS);CHKERRQ(ierr);
    ierr = ISDestroy(&pIS);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt       newp = DMPlexShiftPoint_Internal(p, depth, depthShift);
      PetscInt       size;
      DMPolytopeType ct;

      ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(dmNew, newp, size);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(dmNew, newp, size);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
      ierr = DMPlexSetCellType(dmNew, newp, ct);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftPoints_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt      *newpoints;
  PetscInt       depth = 0, maxConeSize, maxSupportSize, maxConeSizeNew, maxSupportSizeNew, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth < 0) PetscFunctionReturn(0);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dmNew, &maxConeSizeNew, &maxSupportSizeNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(PetscMax(maxConeSize, maxSupportSize), PetscMax(maxConeSizeNew, maxSupportSizeNew)),&newpoints);CHKERRQ(ierr);
  /* Step 5: Set cones and supports */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *points = NULL, *orientations = NULL;
    PetscInt        size,sizeNew, i, newp = DMPlexShiftPoint_Internal(p, depth, depthShift);

    ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &points);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &orientations);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthShift);
    }
    ierr = DMPlexSetCone(dmNew, newp, newpoints);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dmNew, newp, orientations);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmNew, newp, &sizeNew);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &points);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthShift);
    }
    for (i = size; i < sizeNew; ++i) newpoints[i] = 0;
    ierr = DMPlexSetSupport(dmNew, newp, newpoints);CHKERRQ(ierr);
  }
  ierr = PetscFree(newpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftCoordinates_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscSection   coordSection, newCoordSection;
  Vec            coordinates, newCoordinates;
  PetscScalar   *coords, *newCoords;
  PetscInt       coordSize, sStart, sEnd;
  PetscInt       dim, depth = 0, cStart, cEnd, cStartNew, cEndNew, c, vStart, vEnd, vStartNew, vEndNew, v;
  PetscBool      hasCells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dmNew, dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* Step 8: Convert coordinates */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmNew, 0, &vStartNew, &vEndNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmNew, 0, &cStartNew, &cEndNew);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(newCoordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(newCoordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection, &sStart, &sEnd);CHKERRQ(ierr);
  hasCells = sStart == cStart ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscSectionSetChart(newCoordSection, hasCells ? cStartNew : vStartNew, vEndNew);CHKERRQ(ierr);
  if (hasCells) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt cNew = DMPlexShiftPoint_Internal(c, depth, depthShift), dof;

      ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(newCoordSection, cNew, dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(newCoordSection, cNew, 0, dof);CHKERRQ(ierr);
    }
  }
  for (v = vStartNew; v < vEndNew; ++v) {
    ierr = PetscSectionSetDof(newCoordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(newCoordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(newCoordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dmNew, PETSC_DETERMINE, newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newCoordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &newCoordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newCoordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(newCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(newCoordinates, dim);CHKERRQ(ierr);
  ierr = VecSetType(newCoordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmNew, newCoordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  if (hasCells) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt cNew = DMPlexShiftPoint_Internal(c, depth, depthShift), dof, off, noff, d;

      ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, c, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(newCoordSection, cNew, &noff);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) newCoords[noff+d] = coords[off+d];
    }
  }
  for (v = vStart; v < vEnd; ++v) {
    PetscInt dof, off, noff, d;

    ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(newCoordSection, DMPlexShiftPoint_Internal(v, depth, depthShift), &noff);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) newCoords[noff+d] = coords[off+d];
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&newCoordSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftSF_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt           depth = 0;
  PetscSF            sfPoint, sfPointNew;
  const PetscSFNode *remotePoints;
  PetscSFNode       *gremotePoints;
  const PetscInt    *localPoints;
  PetscInt          *glocalPoints, *newLocation, *newRemoteLocation;
  PetscInt           numRoots, numLeaves, l, pStart, pEnd, totShift = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* Step 9: Convert pointSF */
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMGetPointSF(dmNew, &sfPointNew);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  totShift = DMPlexShiftPoint_Internal(pEnd,depth,depthShift) - pEnd;
  if (numRoots >= 0) {
    ierr = PetscMalloc2(numRoots,&newLocation,pEnd-pStart,&newRemoteLocation);CHKERRQ(ierr);
    for (l=0; l<numRoots; l++) newLocation[l] = DMPlexShiftPoint_Internal(l, depth, depthShift);
    ierr = PetscSFBcastBegin(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeaves,    &glocalPoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeaves, &gremotePoints);CHKERRQ(ierr);
    for (l = 0; l < numLeaves; ++l) {
      glocalPoints[l]        = DMPlexShiftPoint_Internal(localPoints[l], depth, depthShift);
      gremotePoints[l].rank  = remotePoints[l].rank;
      gremotePoints[l].index = newRemoteLocation[localPoints[l]];
    }
    ierr = PetscFree2(newLocation,newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfPointNew, numRoots + totShift, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftLabels_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscSF            sfPoint;
  DMLabel            vtkLabel, ghostLabel;
  const PetscSFNode *leafRemote;
  const PetscInt    *leafLocal;
  PetscInt           depth = 0, numLeaves, numLabels, l, cStart, cEnd, c, fStart, fEnd, f;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* Step 10: Convert labels */
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, newlabel;
    const char     *lname;
    PetscBool       isDepth, isDim;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, val;

    ierr = DMGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = PetscStrcmp(lname, "dim", &isDim);CHKERRQ(ierr);
    if (isDim) continue;
    ierr = DMCreateLabel(dmNew, lname);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, lname, &label);CHKERRQ(ierr);
    ierr = DMGetLabel(dmNew, lname, &newlabel);CHKERRQ(ierr);
    ierr = DMLabelGetDefaultValue(label,&val);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(newlabel,val);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    for (val = 0; val < numValues; ++val) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        numPoints, p;

      ierr = DMLabelGetStratumIS(label, values[val], &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints; ++p) {
        const PetscInt newpoint = DMPlexShiftPoint_Internal(points[p], depth, depthShift);

        ierr = DMLabelSetValue(newlabel, newpoint, values[val]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  /* Step 11: Make label for output (vtk) and to mark ghost points (ghost) */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &numLeaves, &leafLocal, &leafRemote);CHKERRQ(ierr);
  ierr = DMCreateLabel(dmNew, "vtk");CHKERRQ(ierr);
  ierr = DMCreateLabel(dmNew, "ghost");CHKERRQ(ierr);
  ierr = DMGetLabel(dmNew, "vtk", &vtkLabel);CHKERRQ(ierr);
  ierr = DMGetLabel(dmNew, "ghost", &ghostLabel);CHKERRQ(ierr);
  for (l = 0, c = cStart; l < numLeaves && c < cEnd; ++l, ++c) {
    for (; c < leafLocal[l] && c < cEnd; ++c) {
      ierr = DMLabelSetValue(vtkLabel, c, 1);CHKERRQ(ierr);
    }
    if (leafLocal[l] >= cEnd) break;
    if (leafRemote[l].rank == rank) {
      ierr = DMLabelSetValue(vtkLabel, c, 1);CHKERRQ(ierr);
    } else {
      ierr = DMLabelSetValue(ghostLabel, c, 2);CHKERRQ(ierr);
    }
  }
  for (; c < cEnd; ++c) {
    ierr = DMLabelSetValue(vtkLabel, c, 1);CHKERRQ(ierr);
  }
  ierr = DMPlexGetHeightStratum(dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {
    PetscInt numCells;

    ierr = DMPlexGetSupportSize(dmNew, f, &numCells);CHKERRQ(ierr);
    if (numCells < 2) {
      ierr = DMLabelSetValue(ghostLabel, f, 1);CHKERRQ(ierr);
    } else {
      const PetscInt *cells = NULL;
      PetscInt        vA, vB;

      ierr = DMPlexGetSupport(dmNew, f, &cells);CHKERRQ(ierr);
      ierr = DMLabelGetValue(vtkLabel, cells[0], &vA);CHKERRQ(ierr);
      ierr = DMLabelGetValue(vtkLabel, cells[1], &vB);CHKERRQ(ierr);
      if (vA != 1 && vB != 1) {ierr = DMLabelSetValue(ghostLabel, f, 1);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftTree_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  DM             refTree;
  PetscSection   pSec;
  PetscInt       *parents, *childIDs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  ierr = DMPlexSetReferenceTree(dmNew,refTree);CHKERRQ(ierr);
  ierr = DMPlexGetTree(dm,&pSec,&parents,&childIDs,NULL,NULL);CHKERRQ(ierr);
  if (pSec) {
    PetscInt p, pStart, pEnd, *parentsShifted, pStartShifted, pEndShifted, depth;
    PetscInt *childIDsShifted;
    PetscSection pSecShifted;

    ierr = PetscSectionGetChart(pSec,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
    pStartShifted = DMPlexShiftPoint_Internal(pStart,depth,depthShift);
    pEndShifted   = DMPlexShiftPoint_Internal(pEnd,depth,depthShift);
    ierr = PetscMalloc2(pEndShifted - pStartShifted,&parentsShifted,pEndShifted-pStartShifted,&childIDsShifted);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dmNew),&pSecShifted);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(pSecShifted,pStartShifted,pEndShifted);CHKERRQ(ierr);
    for (p = pStartShifted; p < pEndShifted; p++) {
      /* start off assuming no children */
      ierr = PetscSectionSetDof(pSecShifted,p,0);CHKERRQ(ierr);
    }
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof;
      PetscInt pNew = DMPlexShiftPoint_Internal(p,depth,depthShift);

      ierr = PetscSectionGetDof(pSec,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(pSecShifted,pNew,dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(pSecShifted);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof;
      PetscInt pNew = DMPlexShiftPoint_Internal(p,depth,depthShift);

      ierr = PetscSectionGetDof(pSec,p,&dof);CHKERRQ(ierr);
      if (dof) {
        PetscInt off, offNew;

        ierr = PetscSectionGetOffset(pSec,p,&off);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(pSecShifted,pNew,&offNew);CHKERRQ(ierr);
        parentsShifted[offNew] = DMPlexShiftPoint_Internal(parents[off],depth,depthShift);
        childIDsShifted[offNew] = childIDs[off];
      }
    }
    ierr = DMPlexSetTree(dmNew,pSecShifted,parentsShifted,childIDsShifted);CHKERRQ(ierr);
    ierr = PetscFree2(parentsShifted,childIDsShifted);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&pSecShifted);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexConstructGhostCells_Internal(DM dm, DMLabel label, PetscInt *numGhostCells, DM gdm)
{
  PetscSF               sf;
  IS                    valueIS;
  const PetscInt       *values, *leaves;
  PetscInt             *depthShift;
  PetscInt              d, depth = 0, nleaves, loc, Ng, numFS, fs, fStart, fEnd, ghostCell, cEnd, c;
  PetscBool             isper;
  const PetscReal      *maxCell, *L;
  const DMBoundaryType *bd;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL);CHKERRQ(ierr);
  nleaves = PetscMax(0, nleaves);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  /* Count ghost cells */
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(valueIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  Ng   = 0;
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f, numBdFaces = 0;

    ierr = DMLabelGetStratumIS(label, values[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      PetscInt numChildren;

      ierr = PetscFindInt(faces[f], nleaves, leaves, &loc);CHKERRQ(ierr);
      ierr = DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL);CHKERRQ(ierr);
      /* non-local and ancestors points don't get to register ghosts */
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] >= fStart) && (faces[f] < fEnd)) ++numBdFaces;
    }
    Ng += numBdFaces;
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*(depth+1), &depthShift);CHKERRQ(ierr);
  for (d = 0; d <= depth; d++) {
    PetscInt dEnd;

    ierr = DMPlexGetDepthStratum(dm,d,NULL,&dEnd);CHKERRQ(ierr);
    depthShift[2*d]   = dEnd;
    depthShift[2*d+1] = 0;
  }
  if (depth >= 0) depthShift[2*depth+1] = Ng;
  ierr = DMPlexShiftPointSetUp_Internal(depth,depthShift);CHKERRQ(ierr);
  ierr = DMPlexShiftSizes_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  /* Step 3: Set cone/support sizes for new points */
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  for (c = cEnd; c < cEnd + Ng; ++c) {
    ierr = DMPlexSetConeSize(gdm, c, 1);CHKERRQ(ierr);
  }
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    ierr = DMLabelGetStratumIS(label, values[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      PetscInt size, numChildren;

      ierr = PetscFindInt(faces[f], nleaves, leaves, &loc);CHKERRQ(ierr);
      ierr = DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL);CHKERRQ(ierr);
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] < fStart) || (faces[f] >= fEnd)) continue;
      ierr = DMPlexGetSupportSize(dm, faces[f], &size);CHKERRQ(ierr);
      if (size != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM has boundary face %d with %d support cells", faces[f], size);
      ierr = DMPlexSetSupportSize(gdm, faces[f] + Ng, 2);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  /* Step 4: Setup ghosted DM */
  ierr = DMSetUp(gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftPoints_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  /* Step 6: Set cones and supports for new points */
  ghostCell = cEnd;
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    ierr = DMLabelGetStratumIS(label, values[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      PetscInt newFace = faces[f] + Ng, numChildren;

      ierr = PetscFindInt(faces[f], nleaves, leaves, &loc);CHKERRQ(ierr);
      ierr = DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL);CHKERRQ(ierr);
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] < fStart) || (faces[f] >= fEnd)) continue;
      ierr = DMPlexSetCone(gdm, ghostCell, &newFace);CHKERRQ(ierr);
      ierr = DMPlexInsertSupport(gdm, newFace, 1, ghostCell);CHKERRQ(ierr);
      ++ghostCell;
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  ierr = DMPlexShiftCoordinates_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftSF_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftLabels_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftTree_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = PetscFree(depthShift);CHKERRQ(ierr);
  for (c = cEnd; c < cEnd + Ng; ++c) {
    ierr = DMPlexSetCellType(gdm, c, DM_POLYTOPE_FV_GHOST);CHKERRQ(ierr);
  }
  /* Step 7: Periodicity */
  ierr = DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(gdm, isper, maxCell,  L,  bd);CHKERRQ(ierr);
  if (numGhostCells) *numGhostCells = Ng;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexConstructGhostCells - Construct ghost cells which connect to every boundary face

  Collective on dm

  Input Parameters:
+ dm - The original DM
- labelName - The label specifying the boundary faces, or "Face Sets" if this is NULL

  Output Parameters:
+ numGhostCells - The number of ghost cells added to the DM
- dmGhosted - The new DM

  Note: If no label exists of that name, one will be created marking all boundary faces

  Level: developer

.seealso: DMCreate()
@*/
PetscErrorCode DMPlexConstructGhostCells(DM dm, const char labelName[], PetscInt *numGhostCells, DM *dmGhosted)
{
  DM             gdm;
  DMLabel        label;
  const char    *name = labelName ? labelName : "Face Sets";
  PetscInt       dim, Ng = 0;
  PetscBool      useCone, useClosure;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numGhostCells) PetscValidPointer(numGhostCells, 3);
  PetscValidPointer(dmGhosted, 4);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &gdm);CHKERRQ(ierr);
  ierr = DMSetType(gdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(gdm, dim);CHKERRQ(ierr);
  ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(gdm, useCone,  useClosure);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) {
    /* Get label for boundary faces */
    ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
  }
  ierr = DMPlexConstructGhostCells_Internal(dm, label, &Ng, gdm);CHKERRQ(ierr);
  ierr = DMCopyBoundary(dm, gdm);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, gdm);CHKERRQ(ierr);
  gdm->setfromoptionscalled = dm->setfromoptionscalled;
  if (numGhostCells) *numGhostCells = Ng;
  *dmGhosted = gdm;
  PetscFunctionReturn(0);
}

/*
  We are adding three kinds of points here:
    Replicated:     Copies of points which exist in the mesh, such as vertices identified across a fault
    Non-replicated: Points which exist on the fault, but are not replicated
    Hybrid:         Entirely new points, such as cohesive cells

  When creating subsequent cohesive cells, we shift the old hybrid cells to the end of the numbering at
  each depth so that the new split/hybrid points can be inserted as a block.
*/
static PetscErrorCode DMPlexConstructCohesiveCells_Internal(DM dm, DMLabel label, DMLabel splitLabel, DM sdm)
{
  MPI_Comm         comm;
  IS               valueIS;
  PetscInt         numSP = 0;       /* The number of depths for which we have replicated points */
  const PetscInt  *values;          /* List of depths for which we have replicated points */
  IS              *splitIS;
  IS              *unsplitIS;
  PetscInt        *numSplitPoints;     /* The number of replicated points at each depth */
  PetscInt        *numUnsplitPoints;   /* The number of non-replicated points at each depth which still give rise to hybrid points */
  PetscInt        *numHybridPoints;    /* The number of new hybrid points at each depth */
  PetscInt        *numHybridPointsOld; /* The number of existing hybrid points at each depth */
  const PetscInt **splitPoints;        /* Replicated points for each depth */
  const PetscInt **unsplitPoints;      /* Non-replicated points for each depth */
  PetscSection     coordSection;
  Vec              coordinates;
  PetscScalar     *coords;
  PetscInt        *depthMax;           /* The first hybrid point at each depth in the original mesh */
  PetscInt        *depthEnd;           /* The point limit at each depth in the original mesh */
  PetscInt        *depthShift;         /* Number of replicated+hybrid points at each depth */
  PetscInt        *pMaxNew;            /* The first replicated point at each depth in the new mesh, hybrids come after this */
  PetscInt        *coneNew, *coneONew, *supportNew;
  PetscInt         shift = 100, shift2 = 200, depth = 0, dep, dim, d, sp, maxConeSize, maxSupportSize, maxConeSizeNew, maxSupportSizeNew, numLabels, vStart, vEnd, pEnd, p, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  /* We do not want this label automatically computed, instead we compute it here */
  ierr = DMCreateLabel(sdm, "celltype");CHKERRQ(ierr);
  /* Count split points and add cohesive cells */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = PetscMalloc5(depth+1,&depthMax,depth+1,&depthEnd,2*(depth+1),&depthShift,depth+1,&pMaxNew,depth+1,&numHybridPointsOld);CHKERRQ(ierr);
  ierr = PetscMalloc7(depth+1,&splitIS,depth+1,&unsplitIS,depth+1,&numSplitPoints,depth+1,&numUnsplitPoints,depth+1,&numHybridPoints,depth+1,&splitPoints,depth+1,&unsplitPoints);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, NULL, &pMaxNew[d]);CHKERRQ(ierr);
    ierr = DMPlexGetTensorPrismBounds_Internal(dm, d, &depthMax[d], NULL);CHKERRQ(ierr);
    depthEnd[d]           = pMaxNew[d];
    depthMax[d]           = depthMax[d] < 0 ? depthEnd[d] : depthMax[d];
    numSplitPoints[d]     = 0;
    numUnsplitPoints[d]   = 0;
    numHybridPoints[d]    = 0;
    numHybridPointsOld[d] = depthMax[d] < 0 ? 0 : depthEnd[d] - depthMax[d];
    splitPoints[d]        = NULL;
    unsplitPoints[d]      = NULL;
    splitIS[d]            = NULL;
    unsplitIS[d]          = NULL;
    /* we are shifting the existing hybrid points with the stratum behind them, so
     * the split comes at the end of the normal points, i.e., at depthMax[d] */
    depthShift[2*d]       = depthMax[d];
    depthShift[2*d+1]     = 0;
  }
  if (label) {
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numSP);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    ierr = DMLabelGetStratumIS(label, dep, &splitIS[dep]);CHKERRQ(ierr);
    if (splitIS[dep]) {
      ierr = ISGetLocalSize(splitIS[dep], &numSplitPoints[dep]);CHKERRQ(ierr);
      ierr = ISGetIndices(splitIS[dep], &splitPoints[dep]);CHKERRQ(ierr);
    }
    ierr = DMLabelGetStratumIS(label, shift2+dep, &unsplitIS[dep]);CHKERRQ(ierr);
    if (unsplitIS[dep]) {
      ierr = ISGetLocalSize(unsplitIS[dep], &numUnsplitPoints[dep]);CHKERRQ(ierr);
      ierr = ISGetIndices(unsplitIS[dep], &unsplitPoints[dep]);CHKERRQ(ierr);
    }
  }
  /* Calculate number of hybrid points */
  for (d = 1; d <= depth; ++d) numHybridPoints[d]     = numSplitPoints[d-1] + numUnsplitPoints[d-1]; /* There is a hybrid cell/face/edge for every split face/edge/vertex   */
  for (d = 0; d <= depth; ++d) depthShift[2*d+1]      = numSplitPoints[d] + numHybridPoints[d];
  ierr = DMPlexShiftPointSetUp_Internal(depth,depthShift);CHKERRQ(ierr);
  /* the end of the points in this stratum that come before the new points:
   * shifting pMaxNew[d] gets the new start of the next stratum, then count back the old hybrid points and the newly
   * added points */
  for (d = 0; d <= depth; ++d) pMaxNew[d]             = DMPlexShiftPoint_Internal(pMaxNew[d],depth,depthShift) - (numHybridPointsOld[d] + numSplitPoints[d] + numHybridPoints[d]);
  ierr = DMPlexShiftSizes_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  /* Step 3: Set cone/support sizes for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt  splitp = p    + pMaxNew[dep];
      const PetscInt *support;
      PetscInt        coneSize, supportSize, qf, qn, qp, e;

      ierr = DMPlexGetConeSize(dm, oldp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, splitp, coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, oldp, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(sdm, splitp, supportSize);CHKERRQ(ierr);
      if (dep == depth-1) {
        const PetscInt hybcell = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Add cohesive cells, they are prisms */
        ierr = DMPlexSetConeSize(sdm, hybcell, 2 + coneSize);CHKERRQ(ierr);
        switch (coneSize) {
          case 2: ierr = DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_SEG_PRISM_TENSOR);CHKERRQ(ierr);break;
          case 3: ierr = DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_TRI_PRISM_TENSOR);CHKERRQ(ierr);break;
          case 4: ierr = DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_QUAD_PRISM_TENSOR);CHKERRQ(ierr);break;
        }
      } else if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
        for (e = 0, qn = 0, qp = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == 1) ++qf;
          if ((val == 1) || (val ==  (shift + 1))) ++qn;
          if ((val == 1) || (val == -(shift + 1))) ++qp;
        }
        /* Split old vertex: Edges into original vertex and new cohesive edge */
        ierr = DMPlexSetSupportSize(sdm, newp, qn+1);CHKERRQ(ierr);
        /* Split new vertex: Edges into split vertex and new cohesive edge */
        ierr = DMPlexSetSupportSize(sdm, splitp, qp+1);CHKERRQ(ierr);
        /* Add hybrid edge */
        ierr = DMPlexSetConeSize(sdm, hybedge, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(sdm, hybedge, qf);CHKERRQ(ierr);
        ierr = DMPlexSetCellType(sdm, hybedge, DM_POLYTOPE_POINT_PRISM_TENSOR);CHKERRQ(ierr);
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
        for (e = 0, qn = 0, qp = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) ++qf;
          if ((val == dim-1) || (val ==  (shift + dim-1))) ++qn;
          if ((val == dim-1) || (val == -(shift + dim-1))) ++qp;
        }
        /* Split old edge: Faces into original edge and cohesive face (positive side?) */
        ierr = DMPlexSetSupportSize(sdm, newp, qn+1);CHKERRQ(ierr);
        /* Split new edge: Faces into split edge and cohesive face (negative side?) */
        ierr = DMPlexSetSupportSize(sdm, splitp, qp+1);CHKERRQ(ierr);
        /* Add hybrid face */
        ierr = DMPlexSetConeSize(sdm, hybface, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(sdm, hybface, qf);CHKERRQ(ierr);
        ierr = DMPlexSetCellType(sdm, hybface, DM_POLYTOPE_SEG_PRISM_TENSOR);CHKERRQ(ierr);
      }
    }
  }
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numUnsplitPoints[dep]; ++p) {
      const PetscInt  oldp   = unsplitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt *support;
      PetscInt        coneSize, supportSize, qf, e, s;

      ierr = DMPlexGetConeSize(dm, oldp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, oldp, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
      if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit vertex: Edges into original vertex, split edges, and new cohesive edge twice */
        for (s = 0, qf = 0; s < supportSize; ++s, ++qf) {
          ierr = PetscFindInt(support[s], numSplitPoints[dep+1], splitPoints[dep+1], &e);CHKERRQ(ierr);
          if (e >= 0) ++qf;
        }
        ierr = DMPlexSetSupportSize(sdm, newp, qf+2);CHKERRQ(ierr);
        /* Add hybrid edge */
        ierr = DMPlexSetConeSize(sdm, hybedge, 2);CHKERRQ(ierr);
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          /* Split and unsplit edges produce hybrid faces */
          if (val == 1) ++qf;
          if (val == (shift2 + 1)) ++qf;
        }
        ierr = DMPlexSetSupportSize(sdm, hybedge, qf);CHKERRQ(ierr);
        ierr = DMPlexSetCellType(sdm, hybedge, DM_POLYTOPE_POINT_PRISM_TENSOR);CHKERRQ(ierr);
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];
        PetscInt       val;

        for (e = 0, qf = 0; e < supportSize; ++e) {
          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) qf += 2;
          else              ++qf;
        }
        /* Unsplit edge: Faces into original edge, split face, and cohesive face twice */
        ierr = DMPlexSetSupportSize(sdm, newp, qf+2);CHKERRQ(ierr);
        /* Add hybrid face */
        for (e = 0, qf = 0; e < supportSize; ++e) {
          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) ++qf;
        }
        ierr = DMPlexSetConeSize(sdm, hybface, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(sdm, hybface, qf);CHKERRQ(ierr);
        ierr = DMPlexSetCellType(sdm, hybface, DM_POLYTOPE_SEG_PRISM_TENSOR);CHKERRQ(ierr);
      }
    }
  }
  /* Step 4: Setup split DM */
  ierr = DMSetUp(sdm);CHKERRQ(ierr);
  ierr = DMPlexShiftPoints_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(sdm, &maxConeSizeNew, &maxSupportSizeNew);CHKERRQ(ierr);
  ierr = PetscMalloc3(PetscMax(maxConeSize, maxConeSizeNew)*3,&coneNew,PetscMax(maxConeSize, maxConeSizeNew)*3,&coneONew,PetscMax(maxSupportSize, maxSupportSizeNew),&supportNew);CHKERRQ(ierr);
  /* Step 6: Set cones and supports for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt  splitp = p    + pMaxNew[dep];
      const PetscInt *cone, *support, *ornt;
      PetscInt        coneSize, supportSize, q, qf, qn, qp, v, e, s;

      ierr = DMPlexGetConeSize(dm, oldp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, oldp, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, oldp, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, oldp, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
      if (dep == depth-1) {
        PetscBool       hasUnsplit = PETSC_FALSE;
        const PetscInt  hybcell    = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        const PetscInt *supportF;

        /* Split face:       copy in old face to new face to start */
        ierr = DMPlexGetSupport(sdm, newp,  &supportF);CHKERRQ(ierr);
        ierr = DMPlexSetSupport(sdm, splitp, supportF);CHKERRQ(ierr);
        /* Split old face:   old vertices/edges in cone so no change */
        /* Split new face:   new vertices/edges in cone */
        for (q = 0; q < coneSize; ++q) {
          ierr = PetscFindInt(cone[q], numSplitPoints[dep-1], splitPoints[dep-1], &v);CHKERRQ(ierr);
          if (v < 0) {
            ierr = PetscFindInt(cone[q], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v);CHKERRQ(ierr);
            if (v < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[q], dep-1);
            coneNew[2+q] = DMPlexShiftPoint_Internal(cone[q], depth, depthShift) /*cone[q] + depthOffset[dep-1]*/;
            hasUnsplit   = PETSC_TRUE;
          } else {
            coneNew[2+q] = v + pMaxNew[dep-1];
            if (dep > 1) {
              const PetscInt *econe;
              PetscInt        econeSize, r, vs, vu;

              ierr = DMPlexGetConeSize(dm, cone[q], &econeSize);CHKERRQ(ierr);
              ierr = DMPlexGetCone(dm, cone[q], &econe);CHKERRQ(ierr);
              for (r = 0; r < econeSize; ++r) {
                ierr = PetscFindInt(econe[r], numSplitPoints[dep-2],   splitPoints[dep-2],   &vs);CHKERRQ(ierr);
                ierr = PetscFindInt(econe[r], numUnsplitPoints[dep-2], unsplitPoints[dep-2], &vu);CHKERRQ(ierr);
                if (vs >= 0) continue;
                if (vu < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", econe[r], dep-2);
                hasUnsplit   = PETSC_TRUE;
              }
            }
          }
        }
        ierr = DMPlexSetCone(sdm, splitp, &coneNew[2]);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(sdm, splitp, ornt);CHKERRQ(ierr);
        /* Face support */
        for (s = 0; s < supportSize; ++s) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[s], &val);CHKERRQ(ierr);
          if (val < 0) {
            /* Split old face:   Replace negative side cell with cohesive cell */
             ierr = DMPlexInsertSupport(sdm, newp, s, hybcell);CHKERRQ(ierr);
          } else {
            /* Split new face:   Replace positive side cell with cohesive cell */
            ierr = DMPlexInsertSupport(sdm, splitp, s, hybcell);CHKERRQ(ierr);
            /* Get orientation for cohesive face */
            {
              const PetscInt *ncone, *nconeO;
              PetscInt        nconeSize, nc;

              ierr = DMPlexGetConeSize(dm, support[s], &nconeSize);CHKERRQ(ierr);
              ierr = DMPlexGetCone(dm, support[s], &ncone);CHKERRQ(ierr);
              ierr = DMPlexGetConeOrientation(dm, support[s], &nconeO);CHKERRQ(ierr);
              for (nc = 0; nc < nconeSize; ++nc) {
                if (ncone[nc] == oldp) {
                  coneONew[0] = nconeO[nc];
                  break;
                }
              }
              if (nc >= nconeSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate face %d in neighboring cell %d", oldp, support[s]);
            }
          }
        }
        /* Cohesive cell:    Old and new split face, then new cohesive faces */
        coneNew[0]  = newp;   /* Extracted negative side orientation above */
        coneNew[1]  = splitp;
        coneONew[1] = coneONew[0];
        for (q = 0; q < coneSize; ++q) {
          /* Hybrid faces must follow order from oriented end face */
          const PetscInt o  = coneONew[0];
          const PetscInt qo = o < 0 ? (-(o+1)+coneSize-q)%coneSize : (q+o)%coneSize;

          ierr = PetscFindInt(cone[qo], numSplitPoints[dep-1], splitPoints[dep-1], &v);CHKERRQ(ierr);
          if (v < 0) {
            ierr = PetscFindInt(cone[qo], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v);CHKERRQ(ierr);
            coneNew[2+q]  = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
          } else {
            coneNew[2+q]  = v + pMaxNew[dep] + numSplitPoints[dep];
          }
          coneONew[2+q] = ((o < 0) + (ornt[qo] < 0))%2 ? -1 : 0;
        }
        ierr = DMPlexSetCone(sdm, hybcell, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(sdm, hybcell, coneONew);CHKERRQ(ierr);
        /* Label the hybrid cells on the boundary of the split */
        if (hasUnsplit) {ierr = DMLabelSetValue(label, -hybcell, dim);CHKERRQ(ierr);}
      } else if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Split old vertex: Edges in old split faces and new cohesive edge */
        for (e = 0, qn = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == 1) || (val == (shift + 1))) {
            supportNew[qn++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qn] = hybedge;
        ierr = DMPlexSetSupport(sdm, newp, supportNew);CHKERRQ(ierr);
        /* Split new vertex: Edges in new split faces and new cohesive edge */
        for (e = 0, qp = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == 1) {
            ierr = PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge);CHKERRQ(ierr);
            if (edge < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qp++] = edge + pMaxNew[dep+1];
          } else if (val == -(shift + 1)) {
            supportNew[qp++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qp] = hybedge;
        ierr = DMPlexSetSupport(sdm, splitp, supportNew);CHKERRQ(ierr);
        /* Hybrid edge:    Old and new split vertex */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        ierr = DMPlexSetCone(sdm, hybedge, coneNew);CHKERRQ(ierr);
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == 1) {
            ierr = PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge);CHKERRQ(ierr);
            if (edge < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        ierr = DMPlexSetSupport(sdm, hybedge, supportNew);CHKERRQ(ierr);
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Split old edge:   old vertices in cone so no change */
        /* Split new edge:   new vertices in cone */
        for (q = 0; q < coneSize; ++q) {
          ierr = PetscFindInt(cone[q], numSplitPoints[dep-1], splitPoints[dep-1], &v);CHKERRQ(ierr);
          if (v < 0) {
            ierr = PetscFindInt(cone[q], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v);CHKERRQ(ierr);
            if (v < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[q], dep-1);
            coneNew[q] = DMPlexShiftPoint_Internal(cone[q], depth, depthShift) /*cone[q] + depthOffset[dep-1]*/;
          } else {
            coneNew[q] = v + pMaxNew[dep-1];
          }
        }
        ierr = DMPlexSetCone(sdm, splitp, coneNew);CHKERRQ(ierr);
        /* Split old edge: Faces in positive side cells and old split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          } else if (val == (shift + dim-1)) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[q++] = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        ierr = DMPlexSetSupport(sdm, newp, supportNew);CHKERRQ(ierr);
        /* Split new edge: Faces in negative side cells and new split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val, face;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) {
            ierr = PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &face);CHKERRQ(ierr);
            if (face < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[q++] = face + pMaxNew[dep+1];
          } else if (val == -(shift + dim-1)) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[q++] = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        ierr = DMPlexSetSupport(sdm, splitp, supportNew);CHKERRQ(ierr);
        /* Hybrid face */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        for (v = 0; v < coneSize; ++v) {
          PetscInt vertex;
          ierr = PetscFindInt(cone[v], numSplitPoints[dep-1], splitPoints[dep-1], &vertex);CHKERRQ(ierr);
          if (vertex < 0) {
            ierr = PetscFindInt(cone[v], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &vertex);CHKERRQ(ierr);
            if (vertex < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[v], dep-1);
            coneNew[2+v] = vertex + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
          } else {
            coneNew[2+v] = vertex + pMaxNew[dep] + numSplitPoints[dep];
          }
        }
        ierr = DMPlexSetCone(sdm, hybface, coneNew);CHKERRQ(ierr);
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, face;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) {
            ierr = PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &face);CHKERRQ(ierr);
            if (face < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[qf++] = face + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        ierr = DMPlexSetSupport(sdm, hybface, supportNew);CHKERRQ(ierr);
      }
    }
  }
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numUnsplitPoints[dep]; ++p) {
      const PetscInt  oldp   = unsplitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt *cone, *support, *ornt;
      PetscInt        coneSize, supportSize, supportSizeNew, q, qf, e, f, s;

      ierr = DMPlexGetConeSize(dm, oldp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, oldp, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, oldp, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, oldp, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
      if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit vertex */
        ierr = DMPlexGetSupportSize(sdm, newp, &supportSizeNew);CHKERRQ(ierr);
        for (s = 0, q = 0; s < supportSize; ++s) {
          supportNew[q++] = DMPlexShiftPoint_Internal(support[s], depth, depthShift) /*support[s] + depthOffset[dep+1]*/;
          ierr = PetscFindInt(support[s], numSplitPoints[dep+1], splitPoints[dep+1], &e);CHKERRQ(ierr);
          if (e >= 0) {
            supportNew[q++] = e + pMaxNew[dep+1];
          }
        }
        supportNew[q++] = hybedge;
        supportNew[q++] = hybedge;
        if (q != supportSizeNew) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Support size %d != %d for vertex %d", q, supportSizeNew, newp);
        ierr = DMPlexSetSupport(sdm, newp, supportNew);CHKERRQ(ierr);
        /* Hybrid edge */
        coneNew[0] = newp;
        coneNew[1] = newp;
        ierr = DMPlexSetCone(sdm, hybedge, coneNew);CHKERRQ(ierr);
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == 1) {
            ierr = PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge);CHKERRQ(ierr);
            if (edge < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2];
          } else if  (val ==  (shift2 + 1)) {
            ierr = PetscFindInt(support[e], numUnsplitPoints[dep+1], unsplitPoints[dep+1], &edge);CHKERRQ(ierr);
            if (edge < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a unsplit edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2] + numSplitPoints[dep+1];
          }
        }
        ierr = DMPlexSetSupport(sdm, hybedge, supportNew);CHKERRQ(ierr);
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit edge: Faces into original edge, split face, and hybrid face twice */
        for (f = 0, qf = 0; f < supportSize; ++f) {
          PetscInt val, face;

          ierr = DMLabelGetValue(label, support[f], &val);CHKERRQ(ierr);
          if (val == dim-1) {
            ierr = PetscFindInt(support[f], numSplitPoints[dep+1], splitPoints[dep+1], &face);CHKERRQ(ierr);
            if (face < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[f]);
            supportNew[qf++] = DMPlexShiftPoint_Internal(support[f], depth, depthShift) /*support[f] + depthOffset[dep+1]*/;
            supportNew[qf++] = face + pMaxNew[dep+1];
          } else {
            supportNew[qf++] = DMPlexShiftPoint_Internal(support[f], depth, depthShift) /*support[f] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qf++] = hybface;
        supportNew[qf++] = hybface;
        ierr = DMPlexGetSupportSize(sdm, newp, &supportSizeNew);CHKERRQ(ierr);
        if (qf != supportSizeNew) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support size for unsplit edge %d is %d != %d\n", newp, qf, supportSizeNew);
        ierr = DMPlexSetSupport(sdm, newp, supportNew);CHKERRQ(ierr);
        /* Add hybrid face */
        coneNew[0] = newp;
        coneNew[1] = newp;
        ierr = PetscFindInt(cone[0], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v);CHKERRQ(ierr);
        if (v < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Vertex %d is not an unsplit vertex", cone[0]);
        coneNew[2] = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
        ierr = PetscFindInt(cone[1], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v);CHKERRQ(ierr);
        if (v < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Vertex %d is not an unsplit vertex", cone[1]);
        coneNew[3] = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
        ierr = DMPlexSetCone(sdm, hybface, coneNew);CHKERRQ(ierr);
        for (f = 0, qf = 0; f < supportSize; ++f) {
          PetscInt val, face;

          ierr = DMLabelGetValue(label, support[f], &val);CHKERRQ(ierr);
          if (val == dim-1) {
            ierr = PetscFindInt(support[f], numSplitPoints[dep+1], splitPoints[dep+1], &face);CHKERRQ(ierr);
            supportNew[qf++] = face + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        ierr = DMPlexGetSupportSize(sdm, hybface, &supportSizeNew);CHKERRQ(ierr);
        if (qf != supportSizeNew) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support size for hybrid face %d is %d != %d\n", hybface, qf, supportSizeNew);
        ierr = DMPlexSetSupport(sdm, hybface, supportNew);CHKERRQ(ierr);
      }
    }
  }
  /* Step 6b: Replace split points in negative side cones */
  for (sp = 0; sp < numSP; ++sp) {
    PetscInt        dep = values[sp];
    IS              pIS;
    PetscInt        numPoints;
    const PetscInt *points;

    if (dep >= 0) continue;
    ierr = DMLabelGetStratumIS(label, dep, &pIS);CHKERRQ(ierr);
    if (!pIS) continue;
    dep  = -dep - shift;
    ierr = ISGetLocalSize(pIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(pIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      const PetscInt  oldp = points[p];
      const PetscInt  newp = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*depthOffset[dep] + oldp*/;
      const PetscInt *cone;
      PetscInt        coneSize, c;
      /* PetscBool       replaced = PETSC_FALSE; */

      /* Negative edge: replace split vertex */
      /* Negative cell: replace split face */
      ierr = DMPlexGetConeSize(sdm, newp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(sdm, newp, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        const PetscInt coldp = DMPlexShiftPointInverse_Internal(cone[c],depth,depthShift);
        PetscInt       csplitp, cp, val;

        ierr = DMLabelGetValue(label, coldp, &val);CHKERRQ(ierr);
        if (val == dep-1) {
          ierr = PetscFindInt(coldp, numSplitPoints[dep-1], splitPoints[dep-1], &cp);CHKERRQ(ierr);
          if (cp < 0) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "Point %d is not a split point of dimension %d", oldp, dep-1);
          csplitp  = pMaxNew[dep-1] + cp;
          ierr     = DMPlexInsertCone(sdm, newp, c, csplitp);CHKERRQ(ierr);
          /* replaced = PETSC_TRUE; */
        }
      }
      /* Cells with only a vertex or edge on the submesh have no replacement */
      /* if (!replaced) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "The cone of point %d does not contain split points", oldp); */
    }
    ierr = ISRestoreIndices(pIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pIS);CHKERRQ(ierr);
  }
  /* Step 7: Coordinates */
  ierr = DMPlexShiftCoordinates_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(sdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(sdm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < (numSplitPoints ? numSplitPoints[0] : 0); ++v) {
    const PetscInt newp   = DMPlexShiftPoint_Internal(splitPoints[0][v], depth, depthShift) /*depthOffset[0] + splitPoints[0][v]*/;
    const PetscInt splitp = pMaxNew[0] + v;
    PetscInt       dof, off, soff, d;

    ierr = PetscSectionGetDof(coordSection, newp, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, newp, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, splitp, &soff);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) coords[soff+d] = coords[off+d];
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  /* Step 8: SF, if I can figure this out we can split the mesh in parallel */
  ierr = DMPlexShiftSF_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  /* Step 9: Labels */
  ierr = DMPlexShiftLabels_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  ierr = DMGetNumLabels(sdm, &numLabels);CHKERRQ(ierr);
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt newp   = DMPlexShiftPoint_Internal(splitPoints[dep][p], depth, depthShift) /*depthOffset[dep] + splitPoints[dep][p]*/;
      const PetscInt splitp = pMaxNew[dep] + p;
      PetscInt       l;

      if (splitLabel) {
        const PetscInt val = 100 + dep;

        ierr = DMLabelSetValue(splitLabel, newp,    val);CHKERRQ(ierr);
        ierr = DMLabelSetValue(splitLabel, splitp, -val);CHKERRQ(ierr);
      }
      for (l = 0; l < numLabels; ++l) {
        DMLabel     mlabel;
        const char *lname;
        PetscInt    val;
        PetscBool   isDepth;

        ierr = DMGetLabelName(sdm, l, &lname);CHKERRQ(ierr);
        ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = DMGetLabel(sdm, lname, &mlabel);CHKERRQ(ierr);
        ierr = DMLabelGetValue(mlabel, newp, &val);CHKERRQ(ierr);
        if (val >= 0) {
          ierr = DMLabelSetValue(mlabel, splitp, val);CHKERRQ(ierr);
        }
      }
    }
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    if (splitIS[dep]) {ierr = ISRestoreIndices(splitIS[dep], &splitPoints[dep]);CHKERRQ(ierr);}
    ierr = ISDestroy(&splitIS[dep]);CHKERRQ(ierr);
    if (unsplitIS[dep]) {ierr = ISRestoreIndices(unsplitIS[dep], &unsplitPoints[dep]);CHKERRQ(ierr);}
    ierr = ISDestroy(&unsplitIS[dep]);CHKERRQ(ierr);
  }
  if (label) {
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(sdm, d, NULL, &pEnd);CHKERRQ(ierr);
    pMaxNew[d] = pEnd - numHybridPoints[d] - numHybridPointsOld[d];
  }
  ierr = PetscFree3(coneNew, coneONew, supportNew);CHKERRQ(ierr);
  ierr = PetscFree5(depthMax, depthEnd, depthShift, pMaxNew, numHybridPointsOld);CHKERRQ(ierr);
  ierr = PetscFree7(splitIS, unsplitIS, numSplitPoints, numUnsplitPoints, numHybridPoints, splitPoints, unsplitPoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexConstructCohesiveCells - Construct cohesive cells which split the face along an internal interface

  Collective on dm

  Input Parameters:
+ dm - The original DM
- label - The label specifying the boundary faces (this could be auto-generated)

  Output Parameters:
+ splitLabel - The label containing the split points, or NULL if no output is desired
- dmSplit - The new DM

  Level: developer

.seealso: DMCreate(), DMPlexLabelCohesiveComplete()
@*/
PetscErrorCode DMPlexConstructCohesiveCells(DM dm, DMLabel label, DMLabel splitLabel, DM *dmSplit)
{
  DM             sdm;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmSplit, 3);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(sdm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
  case 3:
    ierr = DMPlexConstructCohesiveCells_Internal(dm, label, splitLabel, sdm);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct cohesive cells for dimension %d", dim);
  }
  *dmSplit = sdm;
  PetscFunctionReturn(0);
}

/* Returns the side of the surface for a given cell with a face on the surface */
static PetscErrorCode GetSurfaceSide_Static(DM dm, DM subdm, PetscInt numSubpoints, const PetscInt *subpoints, PetscInt cell, PetscInt face, PetscBool *pos)
{
  const PetscInt *cone, *ornt;
  PetscInt        dim, coneSize, c;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *pos = PETSC_TRUE;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, cell, &ornt);CHKERRQ(ierr);
  for (c = 0; c < coneSize; ++c) {
    if (cone[c] == face) {
      PetscInt o = ornt[c];

      if (subdm) {
        const PetscInt *subcone, *subornt;
        PetscInt        subpoint, subface, subconeSize, sc;

        ierr = PetscFindInt(cell, numSubpoints, subpoints, &subpoint);CHKERRQ(ierr);
        ierr = PetscFindInt(face, numSubpoints, subpoints, &subface);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(subdm, subpoint, &subconeSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(subdm, subpoint, &subcone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(subdm, subpoint, &subornt);CHKERRQ(ierr);
        for (sc = 0; sc < subconeSize; ++sc) {
          if (subcone[sc] == subface) {
            o = subornt[0];
            break;
          }
        }
        if (sc >= subconeSize) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find subpoint %d (%d) in cone for subpoint %d (%d)", subface, face, subpoint, cell);
      }
      if (o >= 0) *pos = PETSC_TRUE;
      else        *pos = PETSC_FALSE;
      break;
    }
  }
  if (c == coneSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Cell %d in split face %d support does not have it in the cone", cell, face);
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelCohesiveComplete - Starting with a label marking points on an internal surface, we add all other mesh pieces
  to complete the surface

  Input Parameters:
+ dm     - The DM
. label  - A DMLabel marking the surface
. blabel - A DMLabel marking the vertices on the boundary which will not be duplicated, or NULL to find them automatically
. flip   - Flag to flip the submesh normal and replace points on the other side
- subdm  - The subDM associated with the label, or NULL

  Output Parameter:
. label - A DMLabel marking all surface points

  Note: The vertices in blabel are called "unsplit" in the terminology from hybrid cell creation.

  Level: developer

.seealso: DMPlexConstructCohesiveCells(), DMPlexLabelComplete()
@*/
PetscErrorCode DMPlexLabelCohesiveComplete(DM dm, DMLabel label, DMLabel blabel, PetscBool flip, DM subdm)
{
  DMLabel         depthLabel;
  IS              dimIS, subpointIS = NULL, facePosIS, faceNegIS, crossEdgeIS = NULL;
  const PetscInt *points, *subpoints;
  const PetscInt  rev   = flip ? -1 : 1;
  PetscInt        shift = 100, shift2 = 200, dim, depth, dep, cStart, cEnd, vStart, vEnd, numPoints, numSubpoints, p, val;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  if (subdm) {
    ierr = DMPlexGetSubpointIS(subdm, &subpointIS);CHKERRQ(ierr);
    if (subpointIS) {
      ierr = ISGetLocalSize(subpointIS, &numSubpoints);CHKERRQ(ierr);
      ierr = ISGetIndices(subpointIS, &subpoints);CHKERRQ(ierr);
    }
  }
  /* Mark cell on the fault, and its faces which touch the fault: cell orientation for face gives the side of the fault */
  ierr = DMLabelGetStratumIS(label, dim-1, &dimIS);CHKERRQ(ierr);
  if (!dimIS) PetscFunctionReturn(0);
  ierr = ISGetLocalSize(dimIS, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(dimIS, &points);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) { /* Loop over fault faces */
    const PetscInt *support;
    PetscInt        supportSize, s;

    ierr = DMPlexGetSupportSize(dm, points[p], &supportSize);CHKERRQ(ierr);
#if 0
    if (supportSize != 2) {
      const PetscInt *lp;
      PetscInt        Nlp, pind;

      /* Check that for a cell with a single support face, that face is in the SF */
      /*   THis check only works for the remote side. We would need root side information */
      ierr = PetscSFGetGraph(dm->sf, NULL, &Nlp, &lp, NULL);CHKERRQ(ierr);
      ierr = PetscFindInt(points[p], Nlp, lp, &pind);CHKERRQ(ierr);
      if (pind < 0) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Split face %d has %d != 2 supports, and the face is not shared with another process", points[p], supportSize);
    }
#endif
    ierr = DMPlexGetSupport(dm, points[p], &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;
      PetscBool       pos;

      ierr = GetSurfaceSide_Static(dm, subdm, numSubpoints, subpoints, support[s], points[p], &pos);CHKERRQ(ierr);
      if (pos) {ierr = DMLabelSetValue(label, support[s],  rev*(shift+dim));CHKERRQ(ierr);}
      else     {ierr = DMLabelSetValue(label, support[s], -rev*(shift+dim));CHKERRQ(ierr);}
      if (rev < 0) pos = !pos ? PETSC_TRUE : PETSC_FALSE;
      /* Put faces touching the fault in the label */
      ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];

        ierr = DMLabelGetValue(label, point, &val);CHKERRQ(ierr);
        if (val == -1) {
          PetscInt *closure = NULL;
          PetscInt  closureSize, cl;

          ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          for (cl = 0; cl < closureSize*2; cl += 2) {
            const PetscInt clp  = closure[cl];
            PetscInt       bval = -1;

            ierr = DMLabelGetValue(label, clp, &val);CHKERRQ(ierr);
            if (blabel) {ierr = DMLabelGetValue(blabel, clp, &bval);CHKERRQ(ierr);}
            if ((val >= 0) && (val < dim-1) && (bval < 0)) {
              ierr = DMLabelSetValue(label, point, pos == PETSC_TRUE ? shift+dim-1 : -(shift+dim-1));CHKERRQ(ierr);
              break;
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = ISRestoreIndices(dimIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&dimIS);CHKERRQ(ierr);
  if (subpointIS) {ierr = ISRestoreIndices(subpointIS, &subpoints);CHKERRQ(ierr);}
  /* Mark boundary points as unsplit */
  if (blabel) {
    ierr = DMLabelGetStratumIS(blabel, 1, &dimIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(dimIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(dimIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscInt       val, bval;

      ierr = DMLabelGetValue(blabel, point, &bval);CHKERRQ(ierr);
      if (bval >= 0) {
        ierr = DMLabelGetValue(label, point, &val);CHKERRQ(ierr);
        if ((val < 0) || (val > dim)) {
          /* This could be a point added from splitting a vertex on an adjacent fault, otherwise its just wrong */
          ierr = DMLabelClearValue(blabel, point, bval);CHKERRQ(ierr);
        }
      }
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscInt       val, bval;

      ierr = DMLabelGetValue(blabel, point, &bval);CHKERRQ(ierr);
      if (bval >= 0) {
        const PetscInt *cone,    *support;
        PetscInt        coneSize, supportSize, s, valA, valB, valE;

        /* Mark as unsplit */
        ierr = DMLabelGetValue(label, point, &val);CHKERRQ(ierr);
        if ((val < 0) || (val > dim)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %d has label value %d, should be part of the fault", point, val);
        ierr = DMLabelClearValue(label, point, val);CHKERRQ(ierr);
        ierr = DMLabelSetValue(label, point, shift2+val);CHKERRQ(ierr);
        /* Check for cross-edge
             A cross-edge has endpoints which are both on the boundary of the surface, but the edge itself is not. */
        if (val != 0) continue;
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          if (coneSize != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %D has %D vertices != 2", support[s], coneSize);
          ierr = DMLabelGetValue(blabel, cone[0], &valA);CHKERRQ(ierr);
          ierr = DMLabelGetValue(blabel, cone[1], &valB);CHKERRQ(ierr);
          ierr = DMLabelGetValue(blabel, support[s], &valE);CHKERRQ(ierr);
          if ((valE < 0) && (valA >= 0) && (valB >= 0) && (cone[0] != cone[1])) {ierr = DMLabelSetValue(blabel, support[s], 2);CHKERRQ(ierr);}
        }
      }
    }
    ierr = ISRestoreIndices(dimIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&dimIS);CHKERRQ(ierr);
  }
  /* Search for other cells/faces/edges connected to the fault by a vertex */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, 0, &dimIS);CHKERRQ(ierr);
  if (blabel) {ierr = DMLabelGetStratumIS(blabel, 2, &crossEdgeIS);CHKERRQ(ierr);}
  if (dimIS && crossEdgeIS) {
    IS vertIS = dimIS;

    ierr = ISExpand(vertIS, crossEdgeIS, &dimIS);CHKERRQ(ierr);
    ierr = ISDestroy(&crossEdgeIS);CHKERRQ(ierr);
    ierr = ISDestroy(&vertIS);CHKERRQ(ierr);
  }
  if (!dimIS) {
    PetscFunctionReturn(0);
  }
  ierr = ISGetLocalSize(dimIS, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(dimIS, &points);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) { /* Loop over fault vertices */
    PetscInt *star = NULL;
    PetscInt  starSize, s;
    PetscInt  again = 1;  /* 0: Finished 1: Keep iterating after a change 2: No change */

    /* All points connected to the fault are inside a cell, so at the top level we will only check cells */
    ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    while (again) {
      if (again > 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Could not classify all cells connected to the fault");
      again = 0;
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt  point = star[s];
        const PetscInt *cone;
        PetscInt        coneSize, c;

        if ((point < cStart) || (point >= cEnd)) continue;
        ierr = DMLabelGetValue(label, point, &val);CHKERRQ(ierr);
        if (val != -1) continue;
        again = again == 1 ? 1 : 2;
        ierr  = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
        ierr  = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
        for (c = 0; c < coneSize; ++c) {
          ierr = DMLabelGetValue(label, cone[c], &val);CHKERRQ(ierr);
          if (val != -1) {
            const PetscInt *ccone;
            PetscInt        cconeSize, cc, side;

            if (PetscAbs(val) < shift) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d on cell %d has an invalid label %d", cone[c], point, val);
            if (val > 0) side =  1;
            else         side = -1;
            ierr = DMLabelSetValue(label, point, side*(shift+dim));CHKERRQ(ierr);
            /* Mark cell faces which touch the fault */
            ierr = DMPlexGetConeSize(dm, point, &cconeSize);CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, point, &ccone);CHKERRQ(ierr);
            for (cc = 0; cc < cconeSize; ++cc) {
              PetscInt *closure = NULL;
              PetscInt  closureSize, cl;

              ierr = DMLabelGetValue(label, ccone[cc], &val);CHKERRQ(ierr);
              if (val != -1) continue;
              ierr = DMPlexGetTransitiveClosure(dm, ccone[cc], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
              for (cl = 0; cl < closureSize*2; cl += 2) {
                const PetscInt clp = closure[cl];

                ierr = DMLabelGetValue(label, clp, &val);CHKERRQ(ierr);
                if (val == -1) continue;
                ierr = DMLabelSetValue(label, ccone[cc], side*(shift+dim-1));CHKERRQ(ierr);
                break;
              }
              ierr = DMPlexRestoreTransitiveClosure(dm, ccone[cc], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
            }
            again = 1;
            break;
          }
        }
      }
    }
    /* Classify the rest by cell membership */
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];

      ierr = DMLabelGetValue(label, point, &val);CHKERRQ(ierr);
      if (val == -1) {
        PetscInt      *sstar = NULL;
        PetscInt       sstarSize, ss;
        PetscBool      marked = PETSC_FALSE, isHybrid;

        ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &sstarSize, &sstar);CHKERRQ(ierr);
        for (ss = 0; ss < sstarSize*2; ss += 2) {
          const PetscInt spoint = sstar[ss];

          if ((spoint < cStart) || (spoint >= cEnd)) continue;
          ierr = DMLabelGetValue(label, spoint, &val);CHKERRQ(ierr);
          if (val == -1) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Cell %d in star of %d does not have a valid label", spoint, point);
          ierr = DMLabelGetValue(depthLabel, point, &dep);CHKERRQ(ierr);
          if (val > 0) {
            ierr = DMLabelSetValue(label, point,   shift+dep);CHKERRQ(ierr);
          } else {
            ierr = DMLabelSetValue(label, point, -(shift+dep));CHKERRQ(ierr);
          }
          marked = PETSC_TRUE;
          break;
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &sstarSize, &sstar);CHKERRQ(ierr);
        ierr = DMPlexCellIsHybrid_Internal(dm, point, &isHybrid);CHKERRQ(ierr);
        if (!isHybrid && !marked) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d could not be classified", point);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(dimIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&dimIS);CHKERRQ(ierr);
  /* If any faces touching the fault divide cells on either side, split them */
  ierr = DMLabelGetStratumIS(label,   shift+dim-1,  &facePosIS);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, -(shift+dim-1), &faceNegIS);CHKERRQ(ierr);
  ierr = ISExpand(facePosIS, faceNegIS, &dimIS);CHKERRQ(ierr);
  ierr = ISDestroy(&facePosIS);CHKERRQ(ierr);
  ierr = ISDestroy(&faceNegIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(dimIS, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(dimIS, &points);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) {
    const PetscInt  point = points[p];
    const PetscInt *support;
    PetscInt        supportSize, valA, valB;

    ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
    if (supportSize != 2) continue;
    ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label, support[0], &valA);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label, support[1], &valB);CHKERRQ(ierr);
    if ((valA == -1) || (valB == -1)) continue;
    if (valA*valB > 0) continue;
    /* Split the face */
    ierr = DMLabelGetValue(label, point, &valA);CHKERRQ(ierr);
    ierr = DMLabelClearValue(label, point, valA);CHKERRQ(ierr);
    ierr = DMLabelSetValue(label, point, dim-1);CHKERRQ(ierr);
    /* Label its closure:
      unmarked: label as unsplit
      incident: relabel as split
      split:    do nothing
    */
    {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        ierr = DMLabelGetValue(label, closure[cl], &valA);CHKERRQ(ierr);
        if (valA == -1) { /* Mark as unsplit */
          ierr = DMLabelGetValue(depthLabel, closure[cl], &dep);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label, closure[cl], shift2+dep);CHKERRQ(ierr);
        } else if (((valA >= shift) && (valA < shift2)) || ((valA <= -shift) && (valA > -shift2))) {
          ierr = DMLabelGetValue(depthLabel, closure[cl], &dep);CHKERRQ(ierr);
          ierr = DMLabelClearValue(label, closure[cl], valA);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label, closure[cl], dep);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(dimIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&dimIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Check that no cell have all vertices on the fault */
PetscErrorCode DMPlexCheckValidSubmesh_Private(DM dm, DMLabel label, DM subdm)
{
  IS              subpointIS;
  const PetscInt *dmpoints;
  PetscInt        defaultValue, cStart, cEnd, c, vStart, vEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetDefaultValue(label, &defaultValue);CHKERRQ(ierr);
  ierr = DMPlexGetSubpointIS(subdm, &subpointIS);CHKERRQ(ierr);
  if (!subpointIS) PetscFunctionReturn(0);
  ierr = DMPlexGetHeightStratum(subdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(subpointIS, &dmpoints);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscBool invalidCell = PETSC_TRUE;
    PetscInt *closure     = NULL;
    PetscInt  closureSize, cl;

    ierr = DMPlexGetTransitiveClosure(dm, dmpoints[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      PetscInt value = 0;

      if ((closure[cl] < vStart) || (closure[cl] >= vEnd)) continue;
      ierr = DMLabelGetValue(label, closure[cl], &value);CHKERRQ(ierr);
      if (value == defaultValue) {invalidCell = PETSC_FALSE; break;}
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, dmpoints[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (invalidCell) {
      ierr = ISRestoreIndices(subpointIS, &dmpoints);CHKERRQ(ierr);
      ierr = ISDestroy(&subpointIS);CHKERRQ(ierr);
      ierr = DMDestroy(&subdm);CHKERRQ(ierr);
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ambiguous submesh. Cell %D has all of its vertices on the submesh.", dmpoints[c]);
    }
  }
  ierr = ISRestoreIndices(subpointIS, &dmpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
  DMPlexCreateHybridMesh - Create a mesh with hybrid cells along an internal interface

  Collective on dm

  Input Parameters:
+ dm - The original DM
. label - The label specifying the interface vertices
- bdlabel - The optional label specifying the interface boundary vertices

  Output Parameters:
+ hybridLabel - The label fully marking the interface, or NULL if no output is desired
. splitLabel - The label containing the split points, or NULL if no output is desired
. dmInterface - The new interface DM, or NULL
- dmHybrid - The new DM with cohesive cells

  Note: The hybridLabel indicates what parts of the original mesh impinged on the on division surface. For points
  directly on the division surface, they are labeled with their dimension, so an edge 7 on the division surface would be
  7 (1) in hybridLabel. For points that impinge from the positive side, they are labeled with 100+dim, so an edge 6 with
  one vertex 3 on the surface would be 6 (101) and 3 (0) in hybridLabel. If an edge 9 from the negative side of the
  surface also hits vertex 3, it would be 9 (-101) in hybridLabel.

  The splitLabel indicates what points in the new hybrid mesh were the result of splitting points in the original
  mesh. The label value is +=100+dim for each point. For example, if two edges 10 and 14 in the hybrid resulting from
  splitting an edge in the original mesh, you would have 10 (101) and 14 (-101) in the splitLabel.

  The dmInterface is a DM built from the original division surface. It has a label which can be retrieved using
  DMPlexGetSubpointMap() which maps each point back to the point in the surface of the original mesh.

  Level: developer

.seealso: DMPlexConstructCohesiveCells(), DMPlexLabelCohesiveComplete(), DMPlexGetSubpointMap(), DMCreate()
@*/
PetscErrorCode DMPlexCreateHybridMesh(DM dm, DMLabel label, DMLabel bdlabel, DMLabel *hybridLabel, DMLabel *splitLabel, DM *dmInterface, DM *dmHybrid)
{
  DM             idm;
  DMLabel        subpointMap, hlabel, slabel = NULL;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (bdlabel) PetscValidPointer(bdlabel, 3);
  if (hybridLabel) PetscValidPointer(hybridLabel, 4);
  if (splitLabel)  PetscValidPointer(splitLabel, 5);
  if (dmInterface) PetscValidPointer(dmInterface, 6);
  PetscValidPointer(dmHybrid, 7);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexCreateSubmesh(dm, label, 1, PETSC_FALSE, &idm);CHKERRQ(ierr);
  ierr = DMPlexCheckValidSubmesh_Private(dm, label, idm);CHKERRQ(ierr);
  ierr = DMPlexOrient(idm);CHKERRQ(ierr);
  ierr = DMPlexGetSubpointMap(idm, &subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDuplicate(subpointMap, &hlabel);CHKERRQ(ierr);
  ierr = DMLabelClearStratum(hlabel, dim);CHKERRQ(ierr);
  if (splitLabel) {
    const char *name;
    char        sname[PETSC_MAX_PATH_LEN];

    ierr = PetscObjectGetName((PetscObject) hlabel, &name);CHKERRQ(ierr);
    ierr = PetscStrncpy(sname, name, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscStrcat(sname, " split");CHKERRQ(ierr);
    ierr = DMLabelCreate(PETSC_COMM_SELF, sname, &slabel);CHKERRQ(ierr);
  }
  ierr = DMPlexLabelCohesiveComplete(dm, hlabel, bdlabel, PETSC_FALSE, idm);CHKERRQ(ierr);
  if (dmInterface) {*dmInterface = idm;}
  else             {ierr = DMDestroy(&idm);CHKERRQ(ierr);}
  ierr = DMPlexConstructCohesiveCells(dm, hlabel, slabel, dmHybrid);CHKERRQ(ierr);
  if (hybridLabel) *hybridLabel = hlabel;
  else             {ierr = DMLabelDestroy(&hlabel);CHKERRQ(ierr);}
  if (splitLabel)  *splitLabel  = slabel;
  PetscFunctionReturn(0);
}

/* Here we need the explicit assumption that:

     For any marked cell, the marked vertices constitute a single face
*/
static PetscErrorCode DMPlexMarkSubmesh_Uninterpolated(DM dm, DMLabel vertexLabel, PetscInt value, DMLabel subpointMap, PetscInt *numFaces, PetscInt *nFV, DM subdm)
{
  IS               subvertexIS = NULL;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd, pSize;
  PetscInt         depth, dim, d, numSubVerticesInitial = 0, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV      = 0;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  pSize = PetscMax(depth, dim) + 1;
  ierr = PetscMalloc2(pSize, &pStart, pSize, &pEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetSimplexOrBoxCells(dm, depth-d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  if (vertexLabel) {ierr = DMLabelGetStratumIS(vertexLabel, value, &subvertexIS);CHKERRQ(ierr);}
  if (subvertexIS) {
    ierr = ISGetSize(subvertexIS, &numSubVerticesInitial);CHKERRQ(ierr);
    ierr = ISGetIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numCells = 0, c;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      if ((point >= pStart[depth]) && (point < pEnd[depth])) star[numCells++] = point;
    }
    for (c = 0; c < numCells; ++c) {
      const PetscInt cell    = star[c];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, cl;
      PetscInt       cellLoc, numCorners = 0, faceSize = 0;

      ierr = DMLabelGetValue(subpointMap, cell, &cellLoc);CHKERRQ(ierr);
      if (cellLoc == 2) continue;
      if (cellLoc >= 0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Cell %d has dimension %d in the surface label", cell, cellLoc);
      ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt point = closure[cl];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          ++numCorners;
          ierr = DMLabelGetValue(vertexLabel, point, &vertexLoc);CHKERRQ(ierr);
          if (vertexLoc == value) closure[faceSize++] = point;
        }
      }
      if (!(*nFV)) {ierr = DMPlexGetNumFaceVertices(dm, dim, numCorners, nFV);CHKERRQ(ierr);}
      if (faceSize > *nFV) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
      if (faceSize == *nFV) {
        const PetscInt *cells = NULL;
        PetscInt        numCells, nc;

        ++(*numFaces);
        for (cl = 0; cl < faceSize; ++cl) {
          ierr = DMLabelSetValue(subpointMap, closure[cl], 0);CHKERRQ(ierr);
        }
        ierr = DMPlexGetJoin(dm, faceSize, closure, &numCells, &cells);CHKERRQ(ierr);
        for (nc = 0; nc < numCells; ++nc) {
          ierr = DMLabelSetValue(subpointMap, cells[nc], 2);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreJoin(dm, faceSize, closure, &numCells, &cells);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  }
  if (subvertexIS) {
    ierr = ISRestoreIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  ierr = PetscFree2(pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DMLabel subpointMap, DM subdm)
{
  IS               subvertexIS = NULL;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd;
  PetscInt         dim, d, numSubVerticesInitial = 0, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim+1, &pStart, dim+1, &pEnd);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMPlexGetSimplexOrBoxCells(dm, dim-d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  if (vertexLabel) {
    ierr = DMLabelGetStratumIS(vertexLabel, value, &subvertexIS);CHKERRQ(ierr);
    if (subvertexIS) {
      ierr = ISGetSize(subvertexIS, &numSubVerticesInitial);CHKERRQ(ierr);
      ierr = ISGetIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
    }
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numFaces = 0, f;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      PetscInt       faceLoc;

      if ((point >= pStart[dim-1]) && (point < pEnd[dim-1])) {
        if (markedFaces) {
          ierr = DMLabelGetValue(vertexLabel, point, &faceLoc);CHKERRQ(ierr);
          if (faceLoc < 0) continue;
        }
        star[numFaces++] = point;
      }
    }
    for (f = 0; f < numFaces; ++f) {
      const PetscInt face    = star[f];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, c;
      PetscInt       faceLoc;

      ierr = DMLabelGetValue(subpointMap, face, &faceLoc);CHKERRQ(ierr);
      if (faceLoc == dim-1) continue;
      if (faceLoc >= 0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d has dimension %d in the surface label", face, faceLoc);
      ierr = DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        const PetscInt point = closure[c];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          ierr = DMLabelGetValue(vertexLabel, point, &vertexLoc);CHKERRQ(ierr);
          if (vertexLoc != value) break;
        }
      }
      if (c == closureSize*2) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        for (c = 0; c < closureSize*2; c += 2) {
          const PetscInt point = closure[c];

          for (d = 0; d < dim; ++d) {
            if ((point >= pStart[d]) && (point < pEnd[d])) {
              ierr = DMLabelSetValue(subpointMap, point, d);CHKERRQ(ierr);
              break;
            }
          }
        }
        ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMLabelSetValue(subpointMap, support[s], dim);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  }
  if (subvertexIS) {ierr = ISRestoreIndices(subvertexIS, &subvertices);CHKERRQ(ierr);}
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  ierr = PetscFree2(pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkCohesiveSubmesh_Uninterpolated(DM dm, PetscBool hasLagrange, const char labelname[], PetscInt value, DMLabel subpointMap, PetscInt *numFaces, PetscInt *nFV, PetscInt *subCells[], DM subdm)
{
  DMLabel         label = NULL;
  const PetscInt *cone;
  PetscInt        dim, cMax, cEnd, c, subc = 0, p, coneSize = -1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV = 0;
  if (labelname) {ierr = DMGetLabel(dm, labelname, &label);CHKERRQ(ierr);}
  *subCells = NULL;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetTensorPrismBounds_Internal(dm, dim, &cMax, &cEnd);CHKERRQ(ierr);
  if (cMax < 0) PetscFunctionReturn(0);
  if (label) {
    for (c = cMax; c < cEnd; ++c) {
      PetscInt val;

      ierr = DMLabelGetValue(label, c, &val);CHKERRQ(ierr);
      if (val == value) {
        ++(*numFaces);
        ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
      }
    }
  } else {
    *numFaces = cEnd - cMax;
    ierr = DMPlexGetConeSize(dm, cMax, &coneSize);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(*numFaces *2, subCells);CHKERRQ(ierr);
  if (!(*numFaces)) PetscFunctionReturn(0);
  *nFV = hasLagrange ? coneSize/3 : coneSize/2;
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cells;
    PetscInt        numCells;

    if (label) {
      PetscInt val;

      ierr = DMLabelGetValue(label, c, &val);CHKERRQ(ierr);
      if (val != value) continue;
    }
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    for (p = 0; p < *nFV; ++p) {
      ierr = DMLabelSetValue(subpointMap, cone[p], 0);CHKERRQ(ierr);
    }
    /* Negative face */
    ierr = DMPlexGetJoin(dm, *nFV, cone, &numCells, &cells);CHKERRQ(ierr);
    /* Not true in parallel
    if (numCells != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells"); */
    for (p = 0; p < numCells; ++p) {
      ierr = DMLabelSetValue(subpointMap, cells[p], 2);CHKERRQ(ierr);
      (*subCells)[subc++] = cells[p];
    }
    ierr = DMPlexRestoreJoin(dm, *nFV, cone, &numCells, &cells);CHKERRQ(ierr);
    /* Positive face is not included */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkCohesiveSubmesh_Interpolated(DM dm, DMLabel label, PetscInt value, DMLabel subpointMap, DM subdm)
{
  PetscInt      *pStart, *pEnd;
  PetscInt       dim, cMax, cEnd, c, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetTensorPrismBounds_Internal(dm, dim, &cMax, &cEnd);CHKERRQ(ierr);
  if (cMax < 0) PetscFunctionReturn(0);
  ierr = PetscMalloc2(dim+1,&pStart,dim+1,&pEnd);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);}
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cone;
    PetscInt       *closure = NULL;
    PetscInt        fconeSize, coneSize, closureSize, cl, val;

    if (label) {
      ierr = DMLabelGetValue(label, c, &val);CHKERRQ(ierr);
      if (val != value) continue;
    }
    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cone[0], &fconeSize);CHKERRQ(ierr);
    if (coneSize != (fconeSize ? fconeSize : 1) + 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells");
    /* Negative face */
    ierr = DMPlexGetTransitiveClosure(dm, cone[0], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];

      for (d = 0; d <= dim; ++d) {
        if ((point >= pStart[d]) && (point < pEnd[d])) {
          ierr = DMLabelSetValue(subpointMap, point, d);CHKERRQ(ierr);
          break;
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cone[0], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    /* Cells -- positive face is not included */
    for (cl = 0; cl < 1; ++cl) {
      const PetscInt *support;
      PetscInt        supportSize, s;

      ierr = DMPlexGetSupportSize(dm, cone[cl], &supportSize);CHKERRQ(ierr);
      /* if (supportSize != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive faces should separate two cells"); */
      ierr = DMPlexGetSupport(dm, cone[cl], &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        ierr = DMLabelSetValue(subpointMap, support[s], dim);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree2(pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFaceOrientation(DM dm, PetscInt cell, PetscInt numCorners, PetscInt indices[], PetscInt oppositeVertex, PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  MPI_Comm       comm;
  PetscBool      posOrient = PETSC_FALSE;
  const PetscInt debug     = 0;
  PetscInt       cellDim, faceSize, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &cellDim);CHKERRQ(ierr);
  if (debug) {ierr = PetscPrintf(comm, "cellDim: %d numCorners: %d\n", cellDim, numCorners);CHKERRQ(ierr);}

  if (cellDim == 1 && numCorners == 2) {
    /* Triangle */
    faceSize  = numCorners-1;
    posOrient = !(oppositeVertex%2) ? PETSC_TRUE : PETSC_FALSE;
  } else if (cellDim == 2 && numCorners == 3) {
    /* Triangle */
    faceSize  = numCorners-1;
    posOrient = !(oppositeVertex%2) ? PETSC_TRUE : PETSC_FALSE;
  } else if (cellDim == 3 && numCorners == 4) {
    /* Tetrahedron */
    faceSize  = numCorners-1;
    posOrient = (oppositeVertex%2) ? PETSC_TRUE : PETSC_FALSE;
  } else if (cellDim == 1 && numCorners == 3) {
    /* Quadratic line */
    faceSize  = 1;
    posOrient = PETSC_TRUE;
  } else if (cellDim == 2 && numCorners == 4) {
    /* Quads */
    faceSize = 2;
    if ((indices[1] > indices[0]) && (indices[1] - indices[0] == 1)) {
      posOrient = PETSC_TRUE;
    } else if ((indices[0] == 3) && (indices[1] == 0)) {
      posOrient = PETSC_TRUE;
    } else {
      if (((indices[0] > indices[1]) && (indices[0] - indices[1] == 1)) || ((indices[0] == 0) && (indices[1] == 3))) {
        posOrient = PETSC_FALSE;
      } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossedge");
    }
  } else if (cellDim == 2 && numCorners == 6) {
    /* Quadratic triangle (I hate this) */
    /* Edges are determined by the first 2 vertices (corners of edges) */
    const PetscInt faceSizeTri = 3;
    PetscInt       sortedIndices[3], i, iFace;
    PetscBool      found                    = PETSC_FALSE;
    PetscInt       faceVerticesTriSorted[9] = {
      0, 3,  4, /* bottom */
      1, 4,  5, /* right */
      2, 3,  5, /* left */
    };
    PetscInt       faceVerticesTri[9] = {
      0, 3,  4, /* bottom */
      1, 4,  5, /* right */
      2, 5,  3, /* left */
    };

    for (i = 0; i < faceSizeTri; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTri, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 3; ++iFace) {
      const PetscInt ii = iFace*faceSizeTri;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTriSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTriSorted[ii+1])) {
        for (fVertex = 0; fVertex < faceSizeTri; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeTri; ++cVertex) {
            if (indices[cVertex] == faceVerticesTri[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid tri crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 2 && numCorners == 9) {
    /* Quadratic quad (I hate this) */
    /* Edges are determined by the first 2 vertices (corners of edges) */
    const PetscInt faceSizeQuad = 3;
    PetscInt       sortedIndices[3], i, iFace;
    PetscBool      found                      = PETSC_FALSE;
    PetscInt       faceVerticesQuadSorted[12] = {
      0, 1,  4, /* bottom */
      1, 2,  5, /* right */
      2, 3,  6, /* top */
      0, 3,  7, /* left */
    };
    PetscInt       faceVerticesQuad[12] = {
      0, 1,  4, /* bottom */
      1, 2,  5, /* right */
      2, 3,  6, /* top */
      3, 0,  7, /* left */
    };

    for (i = 0; i < faceSizeQuad; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuad, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuad;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadSorted[ii+1])) {
        for (fVertex = 0; fVertex < faceSizeQuad; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeQuad; ++cVertex) {
            if (indices[cVertex] == faceVerticesQuad[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 8) {
    /* Hexes
       A hex is two oriented quads with the normal of the first
       pointing up at the second.

          7---6
         /|  /|
        4---5 |
        | 1-|-2
        |/  |/
        0---3

        Faces are determined by the first 4 vertices (corners of faces) */
    const PetscInt faceSizeHex = 4;
    PetscInt       sortedIndices[4], i, iFace;
    PetscBool      found                     = PETSC_FALSE;
    PetscInt       faceVerticesHexSorted[24] = {
      0, 1, 2, 3,  /* bottom */
      4, 5, 6, 7,  /* top */
      0, 3, 4, 5,  /* front */
      2, 3, 5, 6,  /* right */
      1, 2, 6, 7,  /* back */
      0, 1, 4, 7,  /* left */
    };
    PetscInt       faceVerticesHex[24] = {
      1, 2, 3, 0,  /* bottom */
      4, 5, 6, 7,  /* top */
      0, 3, 5, 4,  /* front */
      3, 2, 6, 5,  /* right */
      2, 1, 7, 6,  /* back */
      1, 0, 4, 7,  /* left */
    };

    for (i = 0; i < faceSizeHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeHex, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesHexSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeHex; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeHex; ++cVertex) {
            if (indices[cVertex] == faceVerticesHex[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 10) {
    /* Quadratic tet */
    /* Faces are determined by the first 3 vertices (corners of faces) */
    const PetscInt faceSizeTet = 6;
    PetscInt       sortedIndices[6], i, iFace;
    PetscBool      found                     = PETSC_FALSE;
    PetscInt       faceVerticesTetSorted[24] = {
      0, 1, 2,  6, 7, 8, /* bottom */
      0, 3, 4,  6, 7, 9,  /* front */
      1, 4, 5,  7, 8, 9,  /* right */
      2, 3, 5,  6, 8, 9,  /* left */
    };
    PetscInt       faceVerticesTet[24] = {
      0, 1, 2,  6, 7, 8, /* bottom */
      0, 4, 3,  6, 7, 9,  /* front */
      1, 5, 4,  7, 8, 9,  /* right */
      2, 3, 5,  8, 6, 9,  /* left */
    };

    for (i = 0; i < faceSizeTet; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTet, sortedIndices);CHKERRQ(ierr);
    for (iFace=0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeTet;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTetSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTetSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesTetSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesTetSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeTet; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeTet; ++cVertex) {
            if (indices[cVertex] == faceVerticesTet[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid tet crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 27) {
    /* Quadratic hexes (I hate this)
       A hex is two oriented quads with the normal of the first
       pointing up at the second.

         7---6
        /|  /|
       4---5 |
       | 3-|-2
       |/  |/
       0---1

       Faces are determined by the first 4 vertices (corners of faces) */
    const PetscInt faceSizeQuadHex = 9;
    PetscInt       sortedIndices[9], i, iFace;
    PetscBool      found                         = PETSC_FALSE;
    PetscInt       faceVerticesQuadHexSorted[54] = {
      0, 1, 2, 3,  8, 9, 10, 11,  24, /* bottom */
      4, 5, 6, 7,  12, 13, 14, 15,  25, /* top */
      0, 1, 4, 5,  8, 12, 16, 17,  22, /* front */
      1, 2, 5, 6,  9, 13, 17, 18,  21, /* right */
      2, 3, 6, 7,  10, 14, 18, 19,  23, /* back */
      0, 3, 4, 7,  11, 15, 16, 19,  20, /* left */
    };
    PetscInt       faceVerticesQuadHex[54] = {
      3, 2, 1, 0,  10, 9, 8, 11,  24, /* bottom */
      4, 5, 6, 7,  12, 13, 14, 15,  25, /* top */
      0, 1, 5, 4,  8, 17, 12, 16,  22, /* front */
      1, 2, 6, 5,  9, 18, 13, 17,  21, /* right */
      2, 3, 7, 6,  10, 19, 14, 18,  23, /* back */
      3, 0, 4, 7,  11, 16, 15, 19,  20 /* left */
    };

    for (i = 0; i < faceSizeQuadHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuadHex, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuadHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesQuadHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesQuadHexSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeQuadHex; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeQuadHex; ++cVertex) {
            if (indices[cVertex] == faceVerticesQuadHex[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown cell type for faceOrientation().");
  if (!posOrient) {
    if (debug) {ierr = PetscPrintf(comm, "  Reversing initial face orientation\n");CHKERRQ(ierr);}
    for (f = 0; f < faceSize; ++f) faceVertices[f] = origVertices[faceSize-1 - f];
  } else {
    if (debug) {ierr = PetscPrintf(comm, "  Keeping initial face orientation\n");CHKERRQ(ierr);}
    for (f = 0; f < faceSize; ++f) faceVertices[f] = origVertices[f];
  }
  if (posOriented) *posOriented = posOrient;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetOrientedFace - Given a cell and a face, as a set of vertices, return the oriented face, as a set of vertices,
  in faceVertices. The orientation is such that the face normal points out of the cell

  Not collective

  Input Parameters:
+ dm           - The original mesh
. cell         - The cell mesh point
. faceSize     - The number of vertices on the face
. face         - The face vertices
. numCorners   - The number of vertices on the cell
. indices      - Local numbering of face vertices in cell cone
- origVertices - Original face vertices

  Output Parameter:
+ faceVertices - The face vertices properly oriented
- posOriented  - PETSC_TRUE if the face was oriented with outward normal

  Level: developer

.seealso: DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetOrientedFace(DM dm, PetscInt cell, PetscInt faceSize, const PetscInt face[], PetscInt numCorners, PetscInt indices[], PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  const PetscInt *cone = NULL;
  PetscInt        coneSize, v, f, v2;
  PetscInt        oppositeVertex = -1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  for (v = 0, v2 = 0; v < coneSize; ++v) {
    PetscBool found = PETSC_FALSE;

    for (f = 0; f < faceSize; ++f) {
      if (face[f] == cone[v]) {
        found = PETSC_TRUE; break;
      }
    }
    if (found) {
      indices[v2]      = v;
      origVertices[v2] = cone[v];
      ++v2;
    } else {
      oppositeVertex = v;
    }
  }
  ierr = DMPlexGetFaceOrientation(dm, cell, numCorners, indices, oppositeVertex, origVertices, faceVertices, posOriented);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexInsertFace_Internal - Puts a face into the mesh

  Not collective

  Input Parameters:
  + dm              - The DMPlex
  . numFaceVertex   - The number of vertices in the face
  . faceVertices    - The vertices in the face for dm
  . subfaceVertices - The vertices in the face for subdm
  . numCorners      - The number of vertices in the cell
  . cell            - A cell in dm containing the face
  . subcell         - A cell in subdm containing the face
  . firstFace       - First face in the mesh
  - newFacePoint    - Next face in the mesh

  Output Parameters:
  . newFacePoint - Contains next face point number on input, updated on output

  Level: developer
*/
static PetscErrorCode DMPlexInsertFace_Internal(DM dm, DM subdm, PetscInt numFaceVertices, const PetscInt faceVertices[], const PetscInt subfaceVertices[], PetscInt numCorners, PetscInt cell, PetscInt subcell, PetscInt firstFace, PetscInt *newFacePoint)
{
  MPI_Comm        comm;
  DM_Plex        *submesh = (DM_Plex*) subdm->data;
  const PetscInt *faces;
  PetscInt        numFaces, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(subdm, subcell, &coneSize);CHKERRQ(ierr);
  if (coneSize != 1) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size of cell %d is %d != 1", cell, coneSize);
#if 0
  /* Cannot use this because support() has not been constructed yet */
  ierr = DMPlexGetJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
#else
  {
    PetscInt f;

    numFaces = 0;
    ierr     = DMGetWorkArray(subdm, 1, MPIU_INT, (void **) &faces);CHKERRQ(ierr);
    for (f = firstFace; f < *newFacePoint; ++f) {
      PetscInt dof, off, d;

      ierr = PetscSectionGetDof(submesh->coneSection, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(submesh->coneSection, f, &off);CHKERRQ(ierr);
      /* Yes, I know this is quadratic, but I expect the sizes to be <5 */
      for (d = 0; d < dof; ++d) {
        const PetscInt p = submesh->cones[off+d];
        PetscInt       v;

        for (v = 0; v < numFaceVertices; ++v) {
          if (subfaceVertices[v] == p) break;
        }
        if (v == numFaceVertices) break;
      }
      if (d == dof) {
        numFaces               = 1;
        ((PetscInt*) faces)[0] = f;
      }
    }
  }
#endif
  if (numFaces > 1) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Vertex set had %d faces, not one", numFaces);
  else if (numFaces == 1) {
    /* Add the other cell neighbor for this face */
    ierr = DMPlexSetCone(subdm, subcell, faces);CHKERRQ(ierr);
  } else {
    PetscInt *indices, *origVertices, *orientedVertices, *orientedSubVertices, v, ov;
    PetscBool posOriented;

    ierr                = DMGetWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), MPIU_INT, &orientedVertices);CHKERRQ(ierr);
    origVertices        = &orientedVertices[numFaceVertices];
    indices             = &orientedVertices[numFaceVertices*2];
    orientedSubVertices = &orientedVertices[numFaceVertices*3];
    ierr                = DMPlexGetOrientedFace(dm, cell, numFaceVertices, faceVertices, numCorners, indices, origVertices, orientedVertices, &posOriented);CHKERRQ(ierr);
    /* TODO: I know that routine should return a permutation, not the indices */
    for (v = 0; v < numFaceVertices; ++v) {
      const PetscInt vertex = faceVertices[v], subvertex = subfaceVertices[v];
      for (ov = 0; ov < numFaceVertices; ++ov) {
        if (orientedVertices[ov] == vertex) {
          orientedSubVertices[ov] = subvertex;
          break;
        }
      }
      if (ov == numFaceVertices) SETERRQ1(comm, PETSC_ERR_PLIB, "Could not find face vertex %d in orientated set", vertex);
    }
    ierr = DMPlexSetCone(subdm, *newFacePoint, orientedSubVertices);CHKERRQ(ierr);
    ierr = DMPlexSetCone(subdm, subcell, newFacePoint);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), MPIU_INT, &orientedVertices);CHKERRQ(ierr);
    ++(*newFacePoint);
  }
#if 0
  ierr = DMPlexRestoreJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
#else
  ierr = DMRestoreWorkArray(subdm, 1, MPIU_INT, (void **) &faces);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubmesh_Uninterpolated(DM dm, DMLabel vertexLabel, PetscInt value, DM subdm)
{
  MPI_Comm        comm;
  DMLabel         subpointMap;
  IS              subvertexIS,  subcellIS;
  const PetscInt *subVertices, *subCells;
  PetscInt        numSubVertices, firstSubVertex, numSubCells;
  PetscInt       *subface, maxConeSize, numSubFaces = 0, firstSubFace, newFacePoint, nFV = 0;
  PetscInt        vStart, vEnd, c, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  if (vertexLabel) {ierr = DMPlexMarkSubmesh_Uninterpolated(dm, vertexLabel, value, subpointMap, &numSubFaces, &nFV, subdm);CHKERRQ(ierr);}
  /* Setup chart */
  ierr = DMLabelGetStratumSize(subpointMap, 0, &numSubVertices);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(subpointMap, 2, &numSubCells);CHKERRQ(ierr);
  ierr = DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, 1);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  ierr = DMLabelGetStratumIS(subpointMap, 0, &subvertexIS);CHKERRQ(ierr);
  if (subvertexIS) {ierr = ISGetIndices(subvertexIS, &subVertices);CHKERRQ(ierr);}
  ierr = DMLabelGetStratumIS(subpointMap, 2, &subcellIS);CHKERRQ(ierr);
  if (subcellIS) {ierr = ISGetIndices(subcellIS, &subCells);CHKERRQ(ierr);}
  for (c = 0; c < numSubCells; ++c) {
    ierr = DMPlexSetConeSize(subdm, c, 1);CHKERRQ(ierr);
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    ierr = DMPlexSetConeSize(subdm, f, nFV);CHKERRQ(ierr);
  }
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Create face cones */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface);CHKERRQ(ierr);
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt cell    = subCells[c];
    const PetscInt subcell = c;
    PetscInt      *closure = NULL;
    PetscInt       closureSize, cl, numCorners = 0, faceSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];
      PetscInt       subVertex;

      if ((point >= vStart) && (point < vEnd)) {
        ++numCorners;
        ierr = PetscFindInt(point, numSubVertices, subVertices, &subVertex);CHKERRQ(ierr);
        if (subVertex >= 0) {
          closure[faceSize] = point;
          subface[faceSize] = firstSubVertex+subVertex;
          ++faceSize;
        }
      }
    }
    if (faceSize > nFV) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
    if (faceSize == nFV) {
      ierr = DMPlexInsertFace_Internal(dm, subdm, faceSize, closure, subface, numCorners, cell, subcell, firstSubFace, &newFacePoint);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize, v;
    const char  *name;

    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(subCoordSection, 1);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(coordSection, 0, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(subCoordSection, 0, numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices);CHKERRQ(ierr);
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(subCoordSection, subvertex, dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(subCoordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &subCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)coordinates,&name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)subCoordinates,name);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetType(subCoordinates,VECSTANDARD);CHKERRQ(ierr);
    if (coordSize) {
      ierr = VecGetArray(coordinates,    &coords);CHKERRQ(ierr);
      ierr = VecGetArray(subCoordinates, &subCoords);CHKERRQ(ierr);
      for (v = 0; v < numSubVertices; ++v) {
        const PetscInt vertex    = subVertices[v];
        const PetscInt subvertex = firstSubVertex+v;
        PetscInt       dof, off, sdof, soff, d;

        ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, vertex, &off);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(subCoordSection, subvertex, &sdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(subCoordSection, subvertex, &soff);CHKERRQ(ierr);
        if (dof != sdof) SETERRQ4(comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
        for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
      }
      ierr = VecRestoreArray(coordinates,    &coords);CHKERRQ(ierr);
      ierr = VecRestoreArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    }
    ierr = DMSetCoordinatesLocal(subdm, subCoordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&subCoordinates);CHKERRQ(ierr);
  }
  /* Cleanup */
  if (subvertexIS) {ierr = ISRestoreIndices(subvertexIS, &subVertices);CHKERRQ(ierr);}
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  if (subcellIS) {ierr = ISRestoreIndices(subcellIS, &subCells);CHKERRQ(ierr);}
  ierr = ISDestroy(&subcellIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscInt DMPlexFilterPoint_Internal(PetscInt point, PetscInt firstSubPoint, PetscInt numSubPoints, const PetscInt subPoints[])
{
  PetscInt       subPoint;
  PetscErrorCode ierr;

  ierr = PetscFindInt(point, numSubPoints, subPoints, &subPoint); if (ierr < 0) return ierr;
  return subPoint < 0 ? subPoint : firstSubPoint+subPoint;
}

static PetscErrorCode DMPlexCreateSubmeshGeneric_Interpolated(DM dm, DMLabel label, PetscInt value, PetscBool markedFaces, PetscBool isCohesive, PetscInt cellHeight, DM subdm)
{
  MPI_Comm         comm;
  DMLabel          subpointMap;
  IS              *subpointIS;
  const PetscInt **subpoints;
  PetscInt        *numSubPoints, *firstSubPoint, *coneNew, *orntNew;
  PetscInt         totSubPoints = 0, maxConeSize, dim, p, d, v;
  PetscMPIInt      rank;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  if (cellHeight) {
    if (isCohesive) {ierr = DMPlexMarkCohesiveSubmesh_Interpolated(dm, label, value, subpointMap, subdm);CHKERRQ(ierr);}
    else            {ierr = DMPlexMarkSubmesh_Interpolated(dm, label, value, markedFaces, subpointMap, subdm);CHKERRQ(ierr);}
  } else {
    DMLabel         depth;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints;

    ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(label, value, &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, value, &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, c, pdim;

      ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        ierr = DMLabelGetValue(depth, closure[c], &pdim);CHKERRQ(ierr);
        ierr = DMLabelSetValue(subpointMap, closure[c], pdim);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  /* Setup chart */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim+1,&numSubPoints,dim+1,&firstSubPoint,dim+1,&subpointIS,dim+1,&subpoints);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumSize(subpointMap, d, &numSubPoints[d]);CHKERRQ(ierr);
    totSubPoints += numSubPoints[d];
  }
  ierr = DMPlexSetChart(subdm, 0, totSubPoints);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, cellHeight);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubPoint[dim] = 0;
  firstSubPoint[0]   = firstSubPoint[dim] + numSubPoints[dim];
  if (dim > 1) {firstSubPoint[dim-1] = firstSubPoint[0]     + numSubPoints[0];}
  if (dim > 2) {firstSubPoint[dim-2] = firstSubPoint[dim-1] + numSubPoints[dim-1];}
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumIS(subpointMap, d, &subpointIS[d]);CHKERRQ(ierr);
    if (subpointIS[d]) {ierr = ISGetIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);}
  }
  /* We do not want this label automatically computed, instead we compute it here */
  ierr = DMCreateLabel(subdm, "celltype");CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, coneSizeNew, c, val;
      DMPolytopeType  ct;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(subdm, subpoint, coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(dm, point, &ct);CHKERRQ(ierr);
      ierr = DMPlexSetCellType(subdm, subpoint, ct);CHKERRQ(ierr);
      if (cellHeight && (d == dim)) {
        ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
        for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
          ierr = DMLabelGetValue(subpointMap, cone[c], &val);CHKERRQ(ierr);
          if (val >= 0) coneSizeNew++;
        }
        ierr = DMPlexSetConeSize(subdm, subpoint, coneSizeNew);CHKERRQ(ierr);
        ierr = DMPlexSetCellType(subdm, subpoint, DM_POLYTOPE_FV_GHOST);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Set cones */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxConeSize,&coneNew,maxConeSize,&orntNew);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, subconeSize, coneSizeNew, c, subc, fornt = 0;

      if (d == dim-1) {
        const PetscInt *support, *cone, *ornt;
        PetscInt        supportSize, coneSize, s, subc;

        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          PetscBool isHybrid;

          ierr = DMPlexCellIsHybrid_Internal(dm, support[s], &isHybrid);CHKERRQ(ierr);
          if (!isHybrid) continue;
          ierr = PetscFindInt(support[s], numSubPoints[d+1], subpoints[d+1], &subc);CHKERRQ(ierr);
          if (subc >= 0) {
            const PetscInt ccell = subpoints[d+1][subc];

            ierr = DMPlexGetCone(dm, ccell, &cone);CHKERRQ(ierr);
            ierr = DMPlexGetConeSize(dm, ccell, &coneSize);CHKERRQ(ierr);
            ierr = DMPlexGetConeOrientation(dm, ccell, &ornt);CHKERRQ(ierr);
            for (c = 0; c < coneSize; ++c) {
              if (cone[c] == point) {
                fornt = ornt[c];
                break;
              }
            }
            break;
          }
        }
      }
      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(subdm, subpoint, &subconeSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, point, &ornt);CHKERRQ(ierr);
      for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
        ierr = PetscFindInt(cone[c], numSubPoints[d-1], subpoints[d-1], &subc);CHKERRQ(ierr);
        if (subc >= 0) {
          coneNew[coneSizeNew] = firstSubPoint[d-1] + subc;
          orntNew[coneSizeNew] = ornt[c];
          ++coneSizeNew;
        }
      }
      if (coneSizeNew != subconeSize) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of cone points located %d does not match subcone size %d", coneSizeNew, subconeSize);
      if (fornt < 0) {
        /* This should be replaced by a call to DMPlexReverseCell() */
#if 0
        ierr = DMPlexReverseCell(subdm, subpoint);CHKERRQ(ierr);
#else
        for (c = 0; c < coneSizeNew/2 + coneSizeNew%2; ++c) {
          PetscInt faceSize, tmp;

          tmp        = coneNew[c];
          coneNew[c] = coneNew[coneSizeNew-1-c];
          coneNew[coneSizeNew-1-c] = tmp;
          ierr = DMPlexGetConeSize(dm, cone[c], &faceSize);CHKERRQ(ierr);
          tmp        = orntNew[c] >= 0 ? -(faceSize-orntNew[c]) : faceSize+orntNew[c];
          orntNew[c] = orntNew[coneSizeNew-1-c] >= 0 ? -(faceSize-orntNew[coneSizeNew-1-c]) : faceSize+orntNew[coneSizeNew-1-c];
          orntNew[coneSizeNew-1-c] = tmp;
        }
      }
      ierr = DMPlexSetCone(subdm, subpoint, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(subdm, subpoint, orntNew);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscFree2(coneNew,orntNew);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     cdim, numComp, coordSize;
    const char  *name;

    ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(subCoordSection, 1);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(coordSection, 0, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(subCoordSection, 0, numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(subCoordSection, firstSubPoint[0], firstSubPoint[0]+numSubPoints[0]);CHKERRQ(ierr);
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt       dof;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(subCoordSection, subvertex, dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(subCoordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &subCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)coordinates,&name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)subCoordinates,name);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(subCoordinates, cdim);CHKERRQ(ierr);
    ierr = VecSetType(subCoordinates,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecGetArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt dof, off, sdof, soff, d;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, vertex, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(subCoordSection, subvertex, &sdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(subCoordSection, subvertex, &soff);CHKERRQ(ierr);
      if (dof != sdof) SETERRQ4(comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    ierr = VecRestoreArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(subdm, subCoordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&subCoordinates);CHKERRQ(ierr);
  }
  /* Build SF: We need this complexity because subpoints might not be selected on the owning process */
  {
    PetscSF            sfPoint, sfPointSub;
    IS                 subpIS;
    const PetscSFNode *remotePoints;
    PetscSFNode       *sremotePoints, *newLocalPoints, *newOwners;
    const PetscInt    *localPoints, *subpoints;
    PetscInt          *slocalPoints;
    PetscInt           numRoots, numLeaves, numSubpoints = 0, numSubroots, numSubleaves = 0, l, sl, ll, pStart, pEnd, p;
    PetscMPIInt        rank;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(subdm, &sfPointSub);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetChart(subdm, NULL, &numSubroots);CHKERRQ(ierr);
    ierr = DMPlexGetSubpointIS(subdm, &subpIS);CHKERRQ(ierr);
    if (subpIS) {
      ierr = ISGetIndices(subpIS, &subpoints);CHKERRQ(ierr);
      ierr = ISGetLocalSize(subpIS, &numSubpoints);CHKERRQ(ierr);
    }
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    if (numRoots >= 0) {
      ierr = PetscMalloc2(pEnd-pStart,&newLocalPoints,numRoots,&newOwners);CHKERRQ(ierr);
      for (p = 0; p < pEnd-pStart; ++p) {
        newLocalPoints[p].rank  = -2;
        newLocalPoints[p].index = -2;
      }
      /* Set subleaves */
      for (l = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subpoint = DMPlexFilterPoint_Internal(point, 0, numSubpoints, subpoints);

        if (subpoint < 0) continue;
        newLocalPoints[point-pStart].rank  = rank;
        newLocalPoints[point-pStart].index = subpoint;
        ++numSubleaves;
      }
      /* Must put in owned subpoints */
      for (p = pStart; p < pEnd; ++p) {
        const PetscInt subpoint = DMPlexFilterPoint_Internal(p, 0, numSubpoints, subpoints);

        if (subpoint < 0) {
          newOwners[p-pStart].rank  = -3;
          newOwners[p-pStart].index = -3;
        } else {
          newOwners[p-pStart].rank  = rank;
          newOwners[p-pStart].index = subpoint;
        }
      }
      ierr = PetscSFReduceBegin(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sfPoint, MPIU_2INT, newOwners, newLocalPoints);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfPoint, MPIU_2INT, newOwners, newLocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(numSubleaves, &slocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(numSubleaves, &sremotePoints);CHKERRQ(ierr);
      for (l = 0, sl = 0, ll = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subpoint = DMPlexFilterPoint_Internal(point, 0, numSubpoints, subpoints);

        if (subpoint < 0) continue;
        if (newLocalPoints[point].rank == rank) {++ll; continue;}
        slocalPoints[sl]        = subpoint;
        sremotePoints[sl].rank  = newLocalPoints[point].rank;
        sremotePoints[sl].index = newLocalPoints[point].index;
        if (sremotePoints[sl].rank  < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote rank for local point %d", point);
        if (sremotePoints[sl].index < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote subpoint for local point %d", point);
        ++sl;
      }
      if (sl + ll != numSubleaves) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatch in number of subleaves %d + %d != %d", sl, ll, numSubleaves);
      ierr = PetscFree2(newLocalPoints,newOwners);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(sfPointSub, numSubroots, sl, slocalPoints, PETSC_OWN_POINTER, sremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
    if (subpIS) {
      ierr = ISRestoreIndices(subpIS, &subpoints);CHKERRQ(ierr);
    }
  }
  /* Cleanup */
  for (d = 0; d <= dim; ++d) {
    if (subpointIS[d]) {ierr = ISRestoreIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);}
    ierr = ISDestroy(&subpointIS[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(numSubPoints,firstSubPoint,subpointIS,subpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DM subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateSubmeshGeneric_Interpolated(dm, vertexLabel, value, markedFaces, PETSC_FALSE, 1, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateSubmesh - Extract a hypersurface from the mesh using vertices defined by a label

  Input Parameters:
+ dm           - The original mesh
. vertexLabel  - The DMLabel marking points contained in the surface
. value        - The label value to use
- markedFaces  - PETSC_TRUE if surface faces are marked in addition to vertices, PETSC_FALSE if only vertices are marked

  Output Parameter:
. subdm - The surface mesh

  Note: This function produces a DMLabel mapping original points in the submesh to their depth. This can be obtained using DMPlexGetSubpointMap().

  Level: developer

.seealso: DMPlexGetSubpointMap(), DMGetLabel(), DMLabelSetValue()
@*/
PetscErrorCode DMPlexCreateSubmesh(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DM *subdm)
{
  DMPlexInterpolatedFlag interpolated;
  PetscInt       dim, cdim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 3);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*subdm, dim-1);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*subdm, cdim);CHKERRQ(ierr);
  ierr = DMPlexIsInterpolated(dm, &interpolated);CHKERRQ(ierr);
  if (interpolated == DMPLEX_INTERPOLATED_PARTIAL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated) {
    ierr = DMPlexCreateSubmesh_Interpolated(dm, vertexLabel, value, markedFaces, *subdm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateSubmesh_Uninterpolated(dm, vertexLabel, value, *subdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCohesiveSubmesh_Uninterpolated(DM dm, PetscBool hasLagrange, const char label[], PetscInt value, DM subdm)
{
  MPI_Comm        comm;
  DMLabel         subpointMap;
  IS              subvertexIS;
  const PetscInt *subVertices;
  PetscInt        numSubVertices, firstSubVertex, numSubCells, *subCells = NULL;
  PetscInt       *subface, maxConeSize, numSubFaces, firstSubFace, newFacePoint, nFV;
  PetscInt        c, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  ierr = DMPlexMarkCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, subpointMap, &numSubFaces, &nFV, &subCells, subdm);CHKERRQ(ierr);
  /* Setup chart */
  ierr = DMLabelGetStratumSize(subpointMap, 0, &numSubVertices);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(subpointMap, 2, &numSubCells);CHKERRQ(ierr);
  ierr = DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, 1);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  ierr = DMLabelGetStratumIS(subpointMap, 0, &subvertexIS);CHKERRQ(ierr);
  if (subvertexIS) {ierr = ISGetIndices(subvertexIS, &subVertices);CHKERRQ(ierr);}
  for (c = 0; c < numSubCells; ++c) {
    ierr = DMPlexSetConeSize(subdm, c, 1);CHKERRQ(ierr);
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    ierr = DMPlexSetConeSize(subdm, f, nFV);CHKERRQ(ierr);
  }
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Create face cones */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface);CHKERRQ(ierr);
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt  cell    = subCells[c];
    const PetscInt  subcell = c;
    const PetscInt *cone, *cells;
    PetscBool       isHybrid;
    PetscInt        numCells, subVertex, p, v;

    ierr = DMPlexCellIsHybrid_Internal(dm, cell, &isHybrid);CHKERRQ(ierr);
    if (!isHybrid) continue;
    ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
    for (v = 0; v < nFV; ++v) {
      ierr = PetscFindInt(cone[v], numSubVertices, subVertices, &subVertex);CHKERRQ(ierr);
      subface[v] = firstSubVertex+subVertex;
    }
    ierr = DMPlexSetCone(subdm, newFacePoint, subface);CHKERRQ(ierr);
    ierr = DMPlexSetCone(subdm, subcell, &newFacePoint);CHKERRQ(ierr);
    ierr = DMPlexGetJoin(dm, nFV, cone, &numCells, &cells);CHKERRQ(ierr);
    /* Not true in parallel
    if (numCells != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells"); */
    for (p = 0; p < numCells; ++p) {
      PetscInt  negsubcell;
      PetscBool isHybrid;

      ierr = DMPlexCellIsHybrid_Internal(dm, cells[p], &isHybrid);CHKERRQ(ierr);
      if (isHybrid) continue;
      /* I know this is a crap search */
      for (negsubcell = 0; negsubcell < numSubCells; ++negsubcell) {
        if (subCells[negsubcell] == cells[p]) break;
      }
      if (negsubcell == numSubCells) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find negative face neighbor for cohesive cell %d", cell);
      ierr = DMPlexSetCone(subdm, negsubcell, &newFacePoint);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreJoin(dm, nFV, cone, &numCells, &cells);CHKERRQ(ierr);
    ++newFacePoint;
  }
  ierr = DMRestoreWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     cdim, numComp, coordSize, v;
    const char  *name;

    ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(subCoordSection, 1);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(coordSection, 0, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(subCoordSection, 0, numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices);CHKERRQ(ierr);
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(subCoordSection, subvertex, dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(subCoordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &subCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)coordinates,&name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)subCoordinates,name);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(subCoordinates, cdim);CHKERRQ(ierr);
    ierr = VecSetType(subCoordinates,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecGetArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof, off, sdof, soff, d;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, vertex, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(subCoordSection, subvertex, &sdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(subCoordSection, subvertex, &soff);CHKERRQ(ierr);
      if (dof != sdof) SETERRQ4(comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    ierr = VecRestoreArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(subdm, subCoordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&subCoordinates);CHKERRQ(ierr);
  }
  /* Build SF */
  CHKMEMQ;
  {
    PetscSF            sfPoint, sfPointSub;
    const PetscSFNode *remotePoints;
    PetscSFNode       *sremotePoints, *newLocalPoints, *newOwners;
    const PetscInt    *localPoints;
    PetscInt          *slocalPoints;
    PetscInt           numRoots, numLeaves, numSubRoots = numSubCells+numSubFaces+numSubVertices, numSubLeaves = 0, l, sl, ll, pStart, pEnd, p, vStart, vEnd;
    PetscMPIInt        rank;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(subdm, &sfPointSub);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    if (numRoots >= 0) {
      /* Only vertices should be shared */
      ierr = PetscMalloc2(pEnd-pStart,&newLocalPoints,numRoots,&newOwners);CHKERRQ(ierr);
      for (p = 0; p < pEnd-pStart; ++p) {
        newLocalPoints[p].rank  = -2;
        newLocalPoints[p].index = -2;
      }
      /* Set subleaves */
      for (l = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subPoint = DMPlexFilterPoint_Internal(point, firstSubVertex, numSubVertices, subVertices);

        if ((point < vStart) && (point >= vEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Should not be mapping anything but vertices, %d", point);
        if (subPoint < 0) continue;
        newLocalPoints[point-pStart].rank  = rank;
        newLocalPoints[point-pStart].index = subPoint;
        ++numSubLeaves;
      }
      /* Must put in owned subpoints */
      for (p = pStart; p < pEnd; ++p) {
        const PetscInt subPoint = DMPlexFilterPoint_Internal(p, firstSubVertex, numSubVertices, subVertices);

        if (subPoint < 0) {
          newOwners[p-pStart].rank  = -3;
          newOwners[p-pStart].index = -3;
        } else {
          newOwners[p-pStart].rank  = rank;
          newOwners[p-pStart].index = subPoint;
        }
      }
      ierr = PetscSFReduceBegin(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sfPoint, MPIU_2INT, newOwners, newLocalPoints);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfPoint, MPIU_2INT, newOwners, newLocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(numSubLeaves,    &slocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(numSubLeaves, &sremotePoints);CHKERRQ(ierr);
      for (l = 0, sl = 0, ll = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subPoint = DMPlexFilterPoint_Internal(point, firstSubVertex, numSubVertices, subVertices);

        if (subPoint < 0) continue;
        if (newLocalPoints[point].rank == rank) {++ll; continue;}
        slocalPoints[sl]        = subPoint;
        sremotePoints[sl].rank  = newLocalPoints[point].rank;
        sremotePoints[sl].index = newLocalPoints[point].index;
        if (sremotePoints[sl].rank  < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote rank for local point %d", point);
        if (sremotePoints[sl].index < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote subpoint for local point %d", point);
        ++sl;
      }
      ierr = PetscFree2(newLocalPoints,newOwners);CHKERRQ(ierr);
      if (sl + ll != numSubLeaves) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatch in number of subleaves %d + %d != %d", sl, ll, numSubLeaves);
      ierr = PetscSFSetGraph(sfPointSub, numSubRoots, sl, slocalPoints, PETSC_OWN_POINTER, sremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
  }
  CHKMEMQ;
  /* Cleanup */
  if (subvertexIS) {ierr = ISRestoreIndices(subvertexIS, &subVertices);CHKERRQ(ierr);}
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  ierr = PetscFree(subCells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCohesiveSubmesh_Interpolated(DM dm, const char labelname[], PetscInt value, DM subdm)
{
  DMLabel        label = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (labelname) {ierr = DMGetLabel(dm, labelname, &label);CHKERRQ(ierr);}
  ierr = DMPlexCreateSubmeshGeneric_Interpolated(dm, label, value, PETSC_FALSE, PETSC_TRUE, 1, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateCohesiveSubmesh - Extract from a mesh with cohesive cells the hypersurface defined by one face of the cells. Optionally, a Label an be given to restrict the cells.

  Input Parameters:
+ dm          - The original mesh
. hasLagrange - The mesh has Lagrange unknowns in the cohesive cells
. label       - A label name, or NULL
- value  - A label value

  Output Parameter:
. subdm - The surface mesh

  Note: This function produces a DMLabel mapping original points in the submesh to their depth. This can be obtained using DMPlexGetSubpointMap().

  Level: developer

.seealso: DMPlexGetSubpointMap(), DMPlexCreateSubmesh()
@*/
PetscErrorCode DMPlexCreateCohesiveSubmesh(DM dm, PetscBool hasLagrange, const char label[], PetscInt value, DM *subdm)
{
  PetscInt       dim, cdim, depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 5);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*subdm, dim-1);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*subdm, cdim);CHKERRQ(ierr);
  if (depth == dim) {
    ierr = DMPlexCreateCohesiveSubmesh_Interpolated(dm, label, value, *subdm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, *subdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexFilter - Extract a subset of mesh cells defined by a label as a separate mesh

  Input Parameters:
+ dm        - The original mesh
. cellLabel - The DMLabel marking cells contained in the new mesh
- value     - The label value to use

  Output Parameter:
. subdm - The new mesh

  Note: This function produces a DMLabel mapping original points in the submesh to their depth. This can be obtained using DMPlexGetSubpointMap().

  Level: developer

.seealso: DMPlexGetSubpointMap(), DMGetLabel(), DMLabelSetValue()
@*/
PetscErrorCode DMPlexFilter(DM dm, DMLabel cellLabel, PetscInt value, DM *subdm)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 3);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*subdm, dim);CHKERRQ(ierr);
  /* Extract submesh in place, could be empty on some procs, could have inconsistency if procs do not both extract a shared cell */
  ierr = DMPlexCreateSubmeshGeneric_Interpolated(dm, cellLabel, value, PETSC_FALSE, PETSC_FALSE, 0, *subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetSubpointMap - Returns a DMLabel with point dimension as values

  Input Parameter:
. dm - The submesh DM

  Output Parameter:
. subpointMap - The DMLabel of all the points from the original mesh in this submesh, or NULL if this is not a submesh

  Level: developer

.seealso: DMPlexCreateSubmesh(), DMPlexGetSubpointIS()
@*/
PetscErrorCode DMPlexGetSubpointMap(DM dm, DMLabel *subpointMap)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointMap, 2);
  *subpointMap = ((DM_Plex*) dm->data)->subpointMap;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetSubpointMap - Sets the DMLabel with point dimension as values

  Input Parameters:
+ dm - The submesh DM
- subpointMap - The DMLabel of all the points from the original mesh in this submesh

  Note: Should normally not be called by the user, since it is set in DMPlexCreateSubmesh()

  Level: developer

.seealso: DMPlexCreateSubmesh(), DMPlexGetSubpointIS()
@*/
PetscErrorCode DMPlexSetSubpointMap(DM dm, DMLabel subpointMap)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DMLabel        tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  tmp  = mesh->subpointMap;
  mesh->subpointMap = subpointMap;
  ierr = PetscObjectReference((PetscObject) mesh->subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubpointIS_Internal(DM dm, IS *subpointIS)
{
  DMLabel        spmap;
  PetscInt       depth, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetSubpointMap(dm, &spmap);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (spmap && depth >= 0) {
    DM_Plex  *mesh = (DM_Plex *) dm->data;
    PetscInt *points, *depths;
    PetscInt  pStart, pEnd, p, off;

    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Submeshes must start the point numbering at 0, not %d", pStart);
    ierr = PetscMalloc1(pEnd, &points);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, depth+1, MPIU_INT, &depths);CHKERRQ(ierr);
    depths[0] = depth;
    depths[1] = 0;
    for(d = 2; d <= depth; ++d) {depths[d] = depth+1 - d;}
    for(d = 0, off = 0; d <= depth; ++d) {
      const PetscInt dep = depths[d];
      PetscInt       depStart, depEnd, n;

      ierr = DMPlexGetDepthStratum(dm, dep, &depStart, &depEnd);CHKERRQ(ierr);
      ierr = DMLabelGetStratumSize(spmap, dep, &n);CHKERRQ(ierr);
      if (((d < 2) && (depth > 1)) || (d == 1)) { /* Only check vertices and cells for now since the map is broken for others */
        if (n != depEnd-depStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d at depth %d should be %d", n, dep, depEnd-depStart);
      } else {
        if (!n) {
          if (d == 0) {
            /* Missing cells */
            for(p = 0; p < depEnd-depStart; ++p, ++off) points[off] = -1;
          } else {
            /* Missing faces */
            for(p = 0; p < depEnd-depStart; ++p, ++off) points[off] = PETSC_MAX_INT;
          }
        }
      }
      if (n) {
        IS              is;
        const PetscInt *opoints;

        ierr = DMLabelGetStratumIS(spmap, dep, &is);CHKERRQ(ierr);
        ierr = ISGetIndices(is, &opoints);CHKERRQ(ierr);
        for(p = 0; p < n; ++p, ++off) points[off] = opoints[p];
        ierr = ISRestoreIndices(is, &opoints);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
      }
    }
    ierr = DMRestoreWorkArray(dm, depth+1, MPIU_INT, &depths);CHKERRQ(ierr);
    if (off != pEnd) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d should be %d", off, pEnd);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, pEnd, points, PETSC_OWN_POINTER, subpointIS);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject) spmap, &mesh->subpointState);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetSubpointIS - Returns an IS covering the entire subdm chart with the original points as data

  Input Parameter:
. dm - The submesh DM

  Output Parameter:
. subpointIS - The IS of all the points from the original mesh in this submesh, or NULL if this is not a submesh

  Note: This IS is guaranteed to be sorted by the construction of the submesh

  Level: developer

.seealso: DMPlexCreateSubmesh(), DMPlexGetSubpointMap()
@*/
PetscErrorCode DMPlexGetSubpointIS(DM dm, IS *subpointIS)
{
  DM_Plex         *mesh = (DM_Plex *) dm->data;
  DMLabel          spmap;
  PetscObjectState state;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointIS, 2);
  ierr = DMPlexGetSubpointMap(dm, &spmap);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject) spmap, &state);CHKERRQ(ierr);
  if (state != mesh->subpointState || !mesh->subpointIS) {ierr = DMPlexCreateSubpointIS_Internal(dm, &mesh->subpointIS);CHKERRQ(ierr);}
  *subpointIS = mesh->subpointIS;
  PetscFunctionReturn(0);
}

/*@
  DMGetEnclosureRelation - Get the relationship between dmA and dmB

  Input Parameters:
+ dmA - The first DM
- dmB - The second DM

  Output Parameter:
. rel - The relation of dmA to dmB

  Level: intermediate

.seealso: DMPlexGetEnclosurePoint()
@*/
PetscErrorCode DMGetEnclosureRelation(DM dmA, DM dmB, DMEnclosureType *rel)
{
  DM             plexA, plexB, sdm;
  DMLabel        spmap;
  PetscInt       pStartA, pEndA, pStartB, pEndB, NpA, NpB;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(rel, 3);
  *rel = DM_ENC_NONE;
  if (!dmA || !dmB) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 1);
  if (dmA == dmB) {*rel = DM_ENC_EQUALITY; PetscFunctionReturn(0);}
  ierr = DMConvert(dmA, DMPLEX, &plexA);CHKERRQ(ierr);
  ierr = DMConvert(dmB, DMPLEX, &plexB);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plexA, &pStartA, &pEndA);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plexB, &pStartB, &pEndB);CHKERRQ(ierr);
  /* Assumption 1: subDMs have smaller charts than the DMs that they originate from
    - The degenerate case of a subdomain which includes all of the domain on some process can be treated as equality */
  if ((pStartA == pStartB) && (pEndA == pEndB)) {
    *rel = DM_ENC_EQUALITY;
    goto end;
  }
  NpA = pEndA - pStartA;
  NpB = pEndB - pStartB;
  if (NpA == NpB) goto end;
  sdm = NpA > NpB ? plexB : plexA; /* The other is the original, enclosing dm */
  ierr = DMPlexGetSubpointMap(sdm, &spmap);CHKERRQ(ierr);
  if (!spmap) goto end;
  /* TODO Check the space mapped to by subpointMap is same size as dm */
  if (NpA > NpB) {
    *rel = DM_ENC_SUPERMESH;
  } else {
    *rel = DM_ENC_SUBMESH;
  }
  end:
  ierr = DMDestroy(&plexA);CHKERRQ(ierr);
  ierr = DMDestroy(&plexB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetEnclosurePoint - Get the point pA in dmA which corresponds to the point pB in dmB

  Input Parameters:
+ dmA   - The first DM
. dmB   - The second DM
. etype - The type of enclosure relation that dmA has to dmB
- pB    - A point of dmB

  Output Parameter:
. pA    - The corresponding point of dmA

  Level: intermediate

.seealso: DMGetEnclosureRelation()
@*/
PetscErrorCode DMGetEnclosurePoint(DM dmA, DM dmB, DMEnclosureType etype, PetscInt pB, PetscInt *pA)
{
  DM              sdm;
  IS              subpointIS;
  const PetscInt *subpoints;
  PetscInt        numSubpoints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* TODO Cache the IS, making it look like an index */
  switch (etype) {
    case DM_ENC_SUPERMESH:
    sdm  = dmB;
    ierr = DMPlexGetSubpointIS(sdm, &subpointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(subpointIS, &subpoints);CHKERRQ(ierr);
    *pA  = subpoints[pB];
    ierr = ISRestoreIndices(subpointIS, &subpoints);CHKERRQ(ierr);
    break;
    case DM_ENC_SUBMESH:
    sdm  = dmA;
    ierr = DMPlexGetSubpointIS(sdm, &subpointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(subpointIS, &numSubpoints);CHKERRQ(ierr);
    ierr = ISGetIndices(subpointIS, &subpoints);CHKERRQ(ierr);
    ierr = PetscFindInt(pB, numSubpoints, subpoints, pA);CHKERRQ(ierr);
    if (*pA < 0) {
      ierr = DMViewFromOptions(dmA, NULL, "-dm_enc_A_view");CHKERRQ(ierr);
      ierr = DMViewFromOptions(dmB, NULL, "-dm_enc_B_view");CHKERRQ(ierr);
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d not found in submesh", pB);
    }
    ierr = ISRestoreIndices(subpointIS, &subpoints);CHKERRQ(ierr);
    break;
    case DM_ENC_EQUALITY:
    case DM_ENC_NONE:
    *pA = pB;break;
    case DM_ENC_UNKNOWN:
    {
      DMEnclosureType enc;

      ierr = DMGetEnclosureRelation(dmA, dmB, &enc);CHKERRQ(ierr);
      ierr = DMGetEnclosurePoint(dmA, dmB, enc, pB, pA);CHKERRQ(ierr);
    }
    break;
    default: SETERRQ1(PetscObjectComm((PetscObject) dmA), PETSC_ERR_ARG_OUTOFRANGE, "Invalid enclosure type %d", (int) etype);
  }
  PetscFunctionReturn(0);
}

#include <petsc/private/dmpleximpl.h>    /*I      "petscdmplex.h"    I*/
#include <petsc/private/dmlabelimpl.h>   /*I      "petscdmlabel.h"   I*/
#include <petscsf.h>

static PetscErrorCode DMPlexCellIsHybrid_Internal(DM dm, PetscInt p, PetscBool *isHybrid)
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetCellType(dm, p, &ct));
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

  PetscFunctionBegin;
  if (cStart) *cStart = -1;
  if (cEnd)   *cEnd   = -1;
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &ctLabel));
  switch (dim) {
    case 1: CHKERRQ(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_POINT_PRISM_TENSOR, cStart, cEnd));break;
    case 2: CHKERRQ(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_SEG_PRISM_TENSOR, cStart, cEnd));break;
    case 3:
      CHKERRQ(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_TRI_PRISM_TENSOR, cStart, cEnd));
      if (*cStart < 0) CHKERRQ(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_QUAD_PRISM_TENSOR, cStart, cEnd));
      break;
    default: PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkBoundaryFaces_Internal(DM dm, PetscInt val, PetscInt cellHeight, DMLabel label)
{
  PetscInt       fStart, fEnd, f;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, &fEnd));
  for (f = fStart; f < fEnd; ++f) {
    PetscInt supportSize;

    CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
    if (supportSize == 1) {
      if (val < 0) {
        PetscInt *closure = NULL;
        PetscInt  clSize, cl, cval;

        CHKERRQ(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &clSize, &closure));
        for (cl = 0; cl < clSize*2; cl += 2) {
          CHKERRQ(DMLabelGetValue(label, closure[cl], &cval));
          if (cval < 0) continue;
          CHKERRQ(DMLabelSetValue(label, f, cval));
          break;
        }
        if (cl == clSize*2) CHKERRQ(DMLabelSetValue(label, f, 1));
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &clSize, &closure));
      } else {
        CHKERRQ(DMLabelSetValue(label, f, val));
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexMarkBoundaryFaces - Mark all faces on the boundary

  Not Collective

  Input Parameters:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexIsInterpolated(dm, &flg));
  PetscCheckFalse(flg != DMPLEX_INTERPOLATED_FULL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not fully interpolated on this rank");
  CHKERRQ(DMPlexMarkBoundaryFaces_Internal(dm, val, 0, label));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLabelComplete_Internal(DM dm, DMLabel label, PetscBool completeCells)
{
  IS              valueIS;
  PetscSF         sfPoint;
  const PetscInt *values;
  PetscInt        numValues, v, cStart, cEnd, nroots;

  PetscFunctionBegin;
  CHKERRQ(DMLabelGetNumValues(label, &numValues));
  CHKERRQ(DMLabelGetValueIS(label, &valueIS));
  CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  CHKERRQ(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    CHKERRQ(DMLabelGetStratumSize(label, values[v], &numPoints));
    CHKERRQ(DMLabelGetStratumIS(label, values[v], &pointIS));
    CHKERRQ(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      PetscInt  q = points[p];
      PetscInt *closure = NULL;
      PetscInt  closureSize, c;

      if (cStart <= q && q < cEnd && !completeCells) { /* skip cells */
        continue;
      }
      CHKERRQ(DMPlexGetTransitiveClosure(dm, q, PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        CHKERRQ(DMLabelSetValue(label, closure[c], values[v]));
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, q, PETSC_TRUE, &closureSize, &closure));
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(valueIS, &values));
  CHKERRQ(ISDestroy(&valueIS));
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  CHKERRQ(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
  if (nroots >= 0) {
    DMLabel         lblRoots, lblLeaves;
    IS              valueIS, pointIS;
    const PetscInt *values;
    PetscInt        numValues, v;

    /* Pull point contributions from remote leaves into local roots */
    CHKERRQ(DMLabelGather(label, sfPoint, &lblLeaves));
    CHKERRQ(DMLabelGetValueIS(lblLeaves, &valueIS));
    CHKERRQ(ISGetLocalSize(valueIS, &numValues));
    CHKERRQ(ISGetIndices(valueIS, &values));
    for (v = 0; v < numValues; ++v) {
      const PetscInt value = values[v];

      CHKERRQ(DMLabelGetStratumIS(lblLeaves, value, &pointIS));
      CHKERRQ(DMLabelInsertIS(label, pointIS, value));
      CHKERRQ(ISDestroy(&pointIS));
    }
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISDestroy(&valueIS));
    CHKERRQ(DMLabelDestroy(&lblLeaves));
    /* Push point contributions from roots into remote leaves */
    CHKERRQ(DMLabelDistribute(label, sfPoint, &lblRoots));
    CHKERRQ(DMLabelGetValueIS(lblRoots, &valueIS));
    CHKERRQ(ISGetLocalSize(valueIS, &numValues));
    CHKERRQ(ISGetIndices(valueIS, &values));
    for (v = 0; v < numValues; ++v) {
      const PetscInt value = values[v];

      CHKERRQ(DMLabelGetStratumIS(lblRoots, value, &pointIS));
      CHKERRQ(DMLabelInsertIS(label, pointIS, value));
      CHKERRQ(ISDestroy(&pointIS));
    }
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISDestroy(&valueIS));
    CHKERRQ(DMLabelDestroy(&lblRoots));
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
  PetscFunctionBegin;
  CHKERRQ(DMPlexLabelComplete_Internal(dm, label, PETSC_TRUE));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMLabelGetNumValues(label, &numValues));
  CHKERRQ(DMLabelGetValueIS(label, &valueIS));
  CHKERRQ(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    CHKERRQ(DMLabelGetStratumSize(label, values[v], &numPoints));
    CHKERRQ(DMLabelGetStratumIS(label, values[v], &pointIS));
    CHKERRQ(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closure));
      for (cl = closureSize-1; cl > 0; --cl) {
        const PetscInt cell = closure[cl*2];
        if ((cell >= cStart) && (cell < cEnd)) {CHKERRQ(DMLabelSetValue(label, cell, values[v])); break;}
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closure));
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(valueIS, &values));
  CHKERRQ(ISDestroy(&valueIS));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  CHKERRQ(DMLabelGetNumValues(label, &numValues));
  CHKERRQ(DMLabelGetValueIS(label, &valueIS));
  CHKERRQ(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    CHKERRQ(DMLabelGetStratumSize(label, values[v], &numPoints));
    CHKERRQ(DMLabelGetStratumIS(label, values[v], &pointIS));
    CHKERRQ(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      const PetscInt face = points[p];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, cl;

      if ((face < fStart) || (face >= fEnd)) continue;
      CHKERRQ(DMPlexGetTransitiveClosure(dm, face, PETSC_FALSE, &closureSize, &closure));
      for (cl = closureSize-1; cl > 0; --cl) {
        const PetscInt cell = closure[cl*2];
        if ((cell >= cStart) && (cell < cEnd)) {CHKERRQ(DMLabelSetValue(label, cell, values[v])); break;}
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, face, PETSC_FALSE, &closureSize, &closure));
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(valueIS, &values));
  CHKERRQ(ISDestroy(&valueIS));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMLabelGetNumValues(label, &numValues));
  CHKERRQ(DMLabelGetValueIS(label, &valueIS));
  CHKERRQ(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    CHKERRQ(DMLabelGetStratumSize(label, values[v], &numPoints));
    CHKERRQ(DMLabelGetStratumIS(label, values[v], &pointIS));
    CHKERRQ(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      PetscInt point = points[p];

      if (point >= cStart && point < cEnd) {
        CHKERRQ(DMLabelClearValue(label,point,values[v]));
      }
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(valueIS, &values));
  CHKERRQ(ISDestroy(&valueIS));
  PetscFunctionReturn(0);
}

/* take (oldEnd, added) pairs, ordered by height and convert them to (oldstart, newstart) pairs, ordered by ascending
 * index (skipping first, which is (0,0)) */
static inline PetscErrorCode DMPlexShiftPointSetUp_Internal(PetscInt depth, PetscInt depthShift[])
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
static inline PetscInt DMPlexShiftPoint_Internal(PetscInt p, PetscInt depth, PetscInt depthShift[])
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
static inline PetscInt DMPlexShiftPointInverse_Internal(PetscInt p, PetscInt depth, PetscInt depthShift[])
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (depth < 0) PetscFunctionReturn(0);
  /* Step 1: Expand chart */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  pEnd = DMPlexShiftPoint_Internal(pEnd,depth,depthShift);
  CHKERRQ(DMPlexSetChart(dmNew, pStart, pEnd));
  CHKERRQ(DMCreateLabel(dmNew,"depth"));
  CHKERRQ(DMPlexGetDepthLabel(dmNew,&depthLabel));
  CHKERRQ(DMCreateLabel(dmNew, "celltype"));
  /* Step 2: Set cone and support sizes */
  for (d = 0; d <= depth; ++d) {
    PetscInt pStartNew, pEndNew;
    IS pIS;

    CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    pStartNew = DMPlexShiftPoint_Internal(pStart, depth, depthShift);
    pEndNew = DMPlexShiftPoint_Internal(pEnd, depth, depthShift);
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF, pEndNew - pStartNew, pStartNew, 1, &pIS));
    CHKERRQ(DMLabelSetStratumIS(depthLabel, d, pIS));
    CHKERRQ(ISDestroy(&pIS));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt       newp = DMPlexShiftPoint_Internal(p, depth, depthShift);
      PetscInt       size;
      DMPolytopeType ct;

      CHKERRQ(DMPlexGetConeSize(dm, p, &size));
      CHKERRQ(DMPlexSetConeSize(dmNew, newp, size));
      CHKERRQ(DMPlexGetSupportSize(dm, p, &size));
      CHKERRQ(DMPlexSetSupportSize(dmNew, newp, size));
      CHKERRQ(DMPlexGetCellType(dm, p, &ct));
      CHKERRQ(DMPlexSetCellType(dmNew, newp, ct));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftPoints_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt      *newpoints;
  PetscInt       depth = 0, maxConeSize, maxSupportSize, maxConeSizeNew, maxSupportSizeNew, pStart, pEnd, p;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (depth < 0) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  CHKERRQ(DMPlexGetMaxSizes(dmNew, &maxConeSizeNew, &maxSupportSizeNew));
  CHKERRQ(PetscMalloc1(PetscMax(PetscMax(maxConeSize, maxSupportSize), PetscMax(maxConeSizeNew, maxSupportSizeNew)),&newpoints));
  /* Step 5: Set cones and supports */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *points = NULL, *orientations = NULL;
    PetscInt        size,sizeNew, i, newp = DMPlexShiftPoint_Internal(p, depth, depthShift);

    CHKERRQ(DMPlexGetConeSize(dm, p, &size));
    CHKERRQ(DMPlexGetCone(dm, p, &points));
    CHKERRQ(DMPlexGetConeOrientation(dm, p, &orientations));
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthShift);
    }
    CHKERRQ(DMPlexSetCone(dmNew, newp, newpoints));
    CHKERRQ(DMPlexSetConeOrientation(dmNew, newp, orientations));
    CHKERRQ(DMPlexGetSupportSize(dm, p, &size));
    CHKERRQ(DMPlexGetSupportSize(dmNew, newp, &sizeNew));
    CHKERRQ(DMPlexGetSupport(dm, p, &points));
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthShift);
    }
    for (i = size; i < sizeNew; ++i) newpoints[i] = 0;
    CHKERRQ(DMPlexSetSupport(dmNew, newp, newpoints));
  }
  CHKERRQ(PetscFree(newpoints));
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

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinateDim(dm, &dim));
  CHKERRQ(DMSetCoordinateDim(dmNew, dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  /* Step 8: Convert coordinates */
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dmNew, 0, &vStartNew, &vEndNew));
  CHKERRQ(DMPlexGetHeightStratum(dmNew, 0, &cStartNew, &cEndNew));
  CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection));
  CHKERRQ(PetscSectionSetNumFields(newCoordSection, 1));
  CHKERRQ(PetscSectionSetFieldComponents(newCoordSection, 0, dim));
  CHKERRQ(PetscSectionGetChart(coordSection, &sStart, &sEnd));
  hasCells = sStart == cStart ? PETSC_TRUE : PETSC_FALSE;
  CHKERRQ(PetscSectionSetChart(newCoordSection, hasCells ? cStartNew : vStartNew, vEndNew));
  if (hasCells) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt cNew = DMPlexShiftPoint_Internal(c, depth, depthShift), dof;

      CHKERRQ(PetscSectionGetDof(coordSection, c, &dof));
      CHKERRQ(PetscSectionSetDof(newCoordSection, cNew, dof));
      CHKERRQ(PetscSectionSetFieldDof(newCoordSection, cNew, 0, dof));
    }
  }
  for (v = vStartNew; v < vEndNew; ++v) {
    CHKERRQ(PetscSectionSetDof(newCoordSection, v, dim));
    CHKERRQ(PetscSectionSetFieldDof(newCoordSection, v, 0, dim));
  }
  CHKERRQ(PetscSectionSetUp(newCoordSection));
  CHKERRQ(DMSetCoordinateSection(dmNew, PETSC_DETERMINE, newCoordSection));
  CHKERRQ(PetscSectionGetStorageSize(newCoordSection, &coordSize));
  CHKERRQ(VecCreate(PETSC_COMM_SELF, &newCoordinates));
  CHKERRQ(PetscObjectSetName((PetscObject) newCoordinates, "coordinates"));
  CHKERRQ(VecSetSizes(newCoordinates, coordSize, PETSC_DETERMINE));
  CHKERRQ(VecSetBlockSize(newCoordinates, dim));
  CHKERRQ(VecSetType(newCoordinates,VECSTANDARD));
  CHKERRQ(DMSetCoordinatesLocal(dmNew, newCoordinates));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(VecGetArray(coordinates, &coords));
  CHKERRQ(VecGetArray(newCoordinates, &newCoords));
  if (hasCells) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt cNew = DMPlexShiftPoint_Internal(c, depth, depthShift), dof, off, noff, d;

      CHKERRQ(PetscSectionGetDof(coordSection, c, &dof));
      CHKERRQ(PetscSectionGetOffset(coordSection, c, &off));
      CHKERRQ(PetscSectionGetOffset(newCoordSection, cNew, &noff));
      for (d = 0; d < dof; ++d) newCoords[noff+d] = coords[off+d];
    }
  }
  for (v = vStart; v < vEnd; ++v) {
    PetscInt dof, off, noff, d;

    CHKERRQ(PetscSectionGetDof(coordSection, v, &dof));
    CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
    CHKERRQ(PetscSectionGetOffset(newCoordSection, DMPlexShiftPoint_Internal(v, depth, depthShift), &noff));
    for (d = 0; d < dof; ++d) newCoords[noff+d] = coords[off+d];
  }
  CHKERRQ(VecRestoreArray(coordinates, &coords));
  CHKERRQ(VecRestoreArray(newCoordinates, &newCoords));
  CHKERRQ(VecDestroy(&newCoordinates));
  CHKERRQ(PetscSectionDestroy(&newCoordSection));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftSF_Single(DM dm, PetscInt depthShift[], PetscSF sf, PetscSF sfNew)
{
  const PetscSFNode *remotePoints;
  PetscSFNode       *gremotePoints;
  const PetscInt    *localPoints;
  PetscInt          *glocalPoints, *newLocation, *newRemoteLocation;
  PetscInt           numRoots, numLeaves, l, pStart, pEnd, depth = 0, totShift = 0;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSFGetGraph(sf, &numRoots, &numLeaves, &localPoints, &remotePoints));
  totShift = DMPlexShiftPoint_Internal(pEnd, depth, depthShift) - pEnd;
  if (numRoots >= 0) {
    CHKERRQ(PetscMalloc2(numRoots, &newLocation, pEnd-pStart, &newRemoteLocation));
    for (l = 0; l < numRoots; ++l) newLocation[l] = DMPlexShiftPoint_Internal(l, depth, depthShift);
    CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, newLocation, newRemoteLocation, MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, newLocation, newRemoteLocation, MPI_REPLACE));
    CHKERRQ(PetscMalloc1(numLeaves, &glocalPoints));
    CHKERRQ(PetscMalloc1(numLeaves, &gremotePoints));
    for (l = 0; l < numLeaves; ++l) {
      glocalPoints[l]        = DMPlexShiftPoint_Internal(localPoints[l], depth, depthShift);
      gremotePoints[l].rank  = remotePoints[l].rank;
      gremotePoints[l].index = newRemoteLocation[localPoints[l]];
    }
    CHKERRQ(PetscFree2(newLocation, newRemoteLocation));
    CHKERRQ(PetscSFSetGraph(sfNew, numRoots + totShift, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftSF_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscSF        sfPoint, sfPointNew;
  PetscBool      useNatural;

  PetscFunctionBegin;
  /* Step 9: Convert pointSF */
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  CHKERRQ(DMGetPointSF(dmNew, &sfPointNew));
  CHKERRQ(DMPlexShiftSF_Single(dm, depthShift, sfPoint, sfPointNew));
  /* Step 9b: Convert naturalSF */
  CHKERRQ(DMGetUseNatural(dm, &useNatural));
  if (useNatural) {
    PetscSF sfNat, sfNatNew;

    CHKERRQ(DMSetUseNatural(dmNew, useNatural));
    CHKERRQ(DMGetNaturalSF(dm, &sfNat));
    CHKERRQ(DMGetNaturalSF(dmNew, &sfNatNew));
    CHKERRQ(DMPlexShiftSF_Single(dm, depthShift, sfNat, sfNatNew));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftLabels_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt           depth = 0, numLabels, l;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  /* Step 10: Convert labels */
  CHKERRQ(DMGetNumLabels(dm, &numLabels));
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, newlabel;
    const char     *lname;
    PetscBool       isDepth, isDim;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, val;

    CHKERRQ(DMGetLabelName(dm, l, &lname));
    CHKERRQ(PetscStrcmp(lname, "depth", &isDepth));
    if (isDepth) continue;
    CHKERRQ(PetscStrcmp(lname, "dim", &isDim));
    if (isDim) continue;
    CHKERRQ(DMCreateLabel(dmNew, lname));
    CHKERRQ(DMGetLabel(dm, lname, &label));
    CHKERRQ(DMGetLabel(dmNew, lname, &newlabel));
    CHKERRQ(DMLabelGetDefaultValue(label,&val));
    CHKERRQ(DMLabelSetDefaultValue(newlabel,val));
    CHKERRQ(DMLabelGetValueIS(label, &valueIS));
    CHKERRQ(ISGetLocalSize(valueIS, &numValues));
    CHKERRQ(ISGetIndices(valueIS, &values));
    for (val = 0; val < numValues; ++val) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        numPoints, p;

      CHKERRQ(DMLabelGetStratumIS(label, values[val], &pointIS));
      CHKERRQ(ISGetLocalSize(pointIS, &numPoints));
      CHKERRQ(ISGetIndices(pointIS, &points));
      for (p = 0; p < numPoints; ++p) {
        const PetscInt newpoint = DMPlexShiftPoint_Internal(points[p], depth, depthShift);

        CHKERRQ(DMLabelSetValue(newlabel, newpoint, values[val]));
      }
      CHKERRQ(ISRestoreIndices(pointIS, &points));
      CHKERRQ(ISDestroy(&pointIS));
    }
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISDestroy(&valueIS));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateVTKLabel_Internal(DM dm, PetscBool createGhostLabel, DM dmNew)
{
  PetscSF            sfPoint;
  DMLabel            vtkLabel, ghostLabel = NULL;
  const PetscSFNode *leafRemote;
  const PetscInt    *leafLocal;
  PetscInt           cellHeight, cStart, cEnd, c, fStart, fEnd, f, numLeaves, l;
  PetscMPIInt        rank;

  PetscFunctionBegin;
   /* Step 11: Make label for output (vtk) and to mark ghost points (ghost) */
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  CHKERRQ(DMPlexGetVTKCellHeight(dmNew, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(PetscSFGetGraph(sfPoint, NULL, &numLeaves, &leafLocal, &leafRemote));
  CHKERRQ(DMCreateLabel(dmNew, "vtk"));
  CHKERRQ(DMGetLabel(dmNew, "vtk", &vtkLabel));
  if (createGhostLabel) {
    CHKERRQ(DMCreateLabel(dmNew, "ghost"));
    CHKERRQ(DMGetLabel(dmNew, "ghost", &ghostLabel));
  }
  for (l = 0, c = cStart; l < numLeaves && c < cEnd; ++l, ++c) {
    for (; c < leafLocal[l] && c < cEnd; ++c) {
      CHKERRQ(DMLabelSetValue(vtkLabel, c, 1));
    }
    if (leafLocal[l] >= cEnd) break;
    if (leafRemote[l].rank == rank) {
      CHKERRQ(DMLabelSetValue(vtkLabel, c, 1));
    } else if (ghostLabel) {
      CHKERRQ(DMLabelSetValue(ghostLabel, c, 2));
    }
  }
  for (; c < cEnd; ++c) {
    CHKERRQ(DMLabelSetValue(vtkLabel, c, 1));
  }
  if (ghostLabel) {
    CHKERRQ(DMPlexGetHeightStratum(dmNew, 1, &fStart, &fEnd));
    for (f = fStart; f < fEnd; ++f) {
      PetscInt numCells;

      CHKERRQ(DMPlexGetSupportSize(dmNew, f, &numCells));
      if (numCells < 2) {
        CHKERRQ(DMLabelSetValue(ghostLabel, f, 1));
      } else {
        const PetscInt *cells = NULL;
        PetscInt        vA, vB;

        CHKERRQ(DMPlexGetSupport(dmNew, f, &cells));
        CHKERRQ(DMLabelGetValue(vtkLabel, cells[0], &vA));
        CHKERRQ(DMLabelGetValue(vtkLabel, cells[1], &vB));
        if (vA != 1 && vB != 1) CHKERRQ(DMLabelSetValue(ghostLabel, f, 1));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftTree_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  DM             refTree;
  PetscSection   pSec;
  PetscInt       *parents, *childIDs;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetReferenceTree(dm,&refTree));
  CHKERRQ(DMPlexSetReferenceTree(dmNew,refTree));
  CHKERRQ(DMPlexGetTree(dm,&pSec,&parents,&childIDs,NULL,NULL));
  if (pSec) {
    PetscInt p, pStart, pEnd, *parentsShifted, pStartShifted, pEndShifted, depth;
    PetscInt *childIDsShifted;
    PetscSection pSecShifted;

    CHKERRQ(PetscSectionGetChart(pSec,&pStart,&pEnd));
    CHKERRQ(DMPlexGetDepth(dm,&depth));
    pStartShifted = DMPlexShiftPoint_Internal(pStart,depth,depthShift);
    pEndShifted   = DMPlexShiftPoint_Internal(pEnd,depth,depthShift);
    CHKERRQ(PetscMalloc2(pEndShifted - pStartShifted,&parentsShifted,pEndShifted-pStartShifted,&childIDsShifted));
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dmNew),&pSecShifted));
    CHKERRQ(PetscSectionSetChart(pSecShifted,pStartShifted,pEndShifted));
    for (p = pStartShifted; p < pEndShifted; p++) {
      /* start off assuming no children */
      CHKERRQ(PetscSectionSetDof(pSecShifted,p,0));
    }
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof;
      PetscInt pNew = DMPlexShiftPoint_Internal(p,depth,depthShift);

      CHKERRQ(PetscSectionGetDof(pSec,p,&dof));
      CHKERRQ(PetscSectionSetDof(pSecShifted,pNew,dof));
    }
    CHKERRQ(PetscSectionSetUp(pSecShifted));
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof;
      PetscInt pNew = DMPlexShiftPoint_Internal(p,depth,depthShift);

      CHKERRQ(PetscSectionGetDof(pSec,p,&dof));
      if (dof) {
        PetscInt off, offNew;

        CHKERRQ(PetscSectionGetOffset(pSec,p,&off));
        CHKERRQ(PetscSectionGetOffset(pSecShifted,pNew,&offNew));
        parentsShifted[offNew] = DMPlexShiftPoint_Internal(parents[off],depth,depthShift);
        childIDsShifted[offNew] = childIDs[off];
      }
    }
    CHKERRQ(DMPlexSetTree(dmNew,pSecShifted,parentsShifted,childIDsShifted));
    CHKERRQ(PetscFree2(parentsShifted,childIDsShifted));
    CHKERRQ(PetscSectionDestroy(&pSecShifted));
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

  PetscFunctionBegin;
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
  nleaves = PetscMax(0, nleaves);
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  /* Count ghost cells */
  CHKERRQ(DMLabelGetValueIS(label, &valueIS));
  CHKERRQ(ISGetLocalSize(valueIS, &numFS));
  CHKERRQ(ISGetIndices(valueIS, &values));
  Ng   = 0;
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f, numBdFaces = 0;

    CHKERRQ(DMLabelGetStratumIS(label, values[fs], &faceIS));
    CHKERRQ(ISGetLocalSize(faceIS, &numFaces));
    CHKERRQ(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      PetscInt numChildren;

      CHKERRQ(PetscFindInt(faces[f], nleaves, leaves, &loc));
      CHKERRQ(DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL));
      /* non-local and ancestors points don't get to register ghosts */
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] >= fStart) && (faces[f] < fEnd)) ++numBdFaces;
    }
    Ng += numBdFaces;
    CHKERRQ(ISRestoreIndices(faceIS, &faces));
    CHKERRQ(ISDestroy(&faceIS));
  }
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(PetscMalloc1(2*(depth+1), &depthShift));
  for (d = 0; d <= depth; d++) {
    PetscInt dEnd;

    CHKERRQ(DMPlexGetDepthStratum(dm,d,NULL,&dEnd));
    depthShift[2*d]   = dEnd;
    depthShift[2*d+1] = 0;
  }
  if (depth >= 0) depthShift[2*depth+1] = Ng;
  CHKERRQ(DMPlexShiftPointSetUp_Internal(depth,depthShift));
  CHKERRQ(DMPlexShiftSizes_Internal(dm, depthShift, gdm));
  /* Step 3: Set cone/support sizes for new points */
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, NULL, &cEnd));
  for (c = cEnd; c < cEnd + Ng; ++c) {
    CHKERRQ(DMPlexSetConeSize(gdm, c, 1));
  }
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    CHKERRQ(DMLabelGetStratumIS(label, values[fs], &faceIS));
    CHKERRQ(ISGetLocalSize(faceIS, &numFaces));
    CHKERRQ(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      PetscInt size, numChildren;

      CHKERRQ(PetscFindInt(faces[f], nleaves, leaves, &loc));
      CHKERRQ(DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL));
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] < fStart) || (faces[f] >= fEnd)) continue;
      CHKERRQ(DMPlexGetSupportSize(dm, faces[f], &size));
      PetscCheckFalse(size != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM has boundary face %d with %d support cells", faces[f], size);
      CHKERRQ(DMPlexSetSupportSize(gdm, faces[f] + Ng, 2));
    }
    CHKERRQ(ISRestoreIndices(faceIS, &faces));
    CHKERRQ(ISDestroy(&faceIS));
  }
  /* Step 4: Setup ghosted DM */
  CHKERRQ(DMSetUp(gdm));
  CHKERRQ(DMPlexShiftPoints_Internal(dm, depthShift, gdm));
  /* Step 6: Set cones and supports for new points */
  ghostCell = cEnd;
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    CHKERRQ(DMLabelGetStratumIS(label, values[fs], &faceIS));
    CHKERRQ(ISGetLocalSize(faceIS, &numFaces));
    CHKERRQ(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      PetscInt newFace = faces[f] + Ng, numChildren;

      CHKERRQ(PetscFindInt(faces[f], nleaves, leaves, &loc));
      CHKERRQ(DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL));
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] < fStart) || (faces[f] >= fEnd)) continue;
      CHKERRQ(DMPlexSetCone(gdm, ghostCell, &newFace));
      CHKERRQ(DMPlexInsertSupport(gdm, newFace, 1, ghostCell));
      ++ghostCell;
    }
    CHKERRQ(ISRestoreIndices(faceIS, &faces));
    CHKERRQ(ISDestroy(&faceIS));
  }
  CHKERRQ(ISRestoreIndices(valueIS, &values));
  CHKERRQ(ISDestroy(&valueIS));
  CHKERRQ(DMPlexShiftCoordinates_Internal(dm, depthShift, gdm));
  CHKERRQ(DMPlexShiftSF_Internal(dm, depthShift, gdm));
  CHKERRQ(DMPlexShiftLabels_Internal(dm, depthShift, gdm));
  CHKERRQ(DMPlexCreateVTKLabel_Internal(dm, PETSC_TRUE, gdm));
  CHKERRQ(DMPlexShiftTree_Internal(dm, depthShift, gdm));
  CHKERRQ(PetscFree(depthShift));
  for (c = cEnd; c < cEnd + Ng; ++c) {
    CHKERRQ(DMPlexSetCellType(gdm, c, DM_POLYTOPE_FV_GHOST));
  }
  /* Step 7: Periodicity */
  CHKERRQ(DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd));
  CHKERRQ(DMSetPeriodicity(gdm, isper, maxCell,  L,  bd));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numGhostCells) PetscValidIntPointer(numGhostCells, 3);
  PetscValidPointer(dmGhosted, 4);
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject)dm), &gdm));
  CHKERRQ(DMSetType(gdm, DMPLEX));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMSetDimension(gdm, dim));
  CHKERRQ(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  CHKERRQ(DMSetBasicAdjacency(gdm, useCone,  useClosure));
  CHKERRQ(DMGetLabel(dm, name, &label));
  if (!label) {
    /* Get label for boundary faces */
    CHKERRQ(DMCreateLabel(dm, name));
    CHKERRQ(DMGetLabel(dm, name, &label));
    CHKERRQ(DMPlexMarkBoundaryFaces(dm, 1, label));
  }
  CHKERRQ(DMPlexConstructGhostCells_Internal(dm, label, &Ng, gdm));
  CHKERRQ(DMCopyDisc(dm, gdm));
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, gdm));
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
  PetscInt         numSP = 0;          /* The number of depths for which we have replicated points */
  const PetscInt  *values;             /* List of depths for which we have replicated points */
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  /* We do not want this label automatically computed, instead we compute it here */
  CHKERRQ(DMCreateLabel(sdm, "celltype"));
  /* Count split points and add cohesive cells */
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  CHKERRQ(PetscMalloc5(depth+1,&depthMax,depth+1,&depthEnd,2*(depth+1),&depthShift,depth+1,&pMaxNew,depth+1,&numHybridPointsOld));
  CHKERRQ(PetscMalloc7(depth+1,&splitIS,depth+1,&unsplitIS,depth+1,&numSplitPoints,depth+1,&numUnsplitPoints,depth+1,&numHybridPoints,depth+1,&splitPoints,depth+1,&unsplitPoints));
  for (d = 0; d <= depth; ++d) {
    CHKERRQ(DMPlexGetDepthStratum(dm, d, NULL, &pMaxNew[d]));
    CHKERRQ(DMPlexGetTensorPrismBounds_Internal(dm, d, &depthMax[d], NULL));
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
    CHKERRQ(DMLabelGetValueIS(label, &valueIS));
    CHKERRQ(ISGetLocalSize(valueIS, &numSP));
    CHKERRQ(ISGetIndices(valueIS, &values));
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    CHKERRQ(DMLabelGetStratumIS(label, dep, &splitIS[dep]));
    if (splitIS[dep]) {
      CHKERRQ(ISGetLocalSize(splitIS[dep], &numSplitPoints[dep]));
      CHKERRQ(ISGetIndices(splitIS[dep], &splitPoints[dep]));
    }
    CHKERRQ(DMLabelGetStratumIS(label, shift2+dep, &unsplitIS[dep]));
    if (unsplitIS[dep]) {
      CHKERRQ(ISGetLocalSize(unsplitIS[dep], &numUnsplitPoints[dep]));
      CHKERRQ(ISGetIndices(unsplitIS[dep], &unsplitPoints[dep]));
    }
  }
  /* Calculate number of hybrid points */
  for (d = 1; d <= depth; ++d) numHybridPoints[d]     = numSplitPoints[d-1] + numUnsplitPoints[d-1]; /* There is a hybrid cell/face/edge for every split face/edge/vertex   */
  for (d = 0; d <= depth; ++d) depthShift[2*d+1]      = numSplitPoints[d] + numHybridPoints[d];
  CHKERRQ(DMPlexShiftPointSetUp_Internal(depth,depthShift));
  /* the end of the points in this stratum that come before the new points:
   * shifting pMaxNew[d] gets the new start of the next stratum, then count back the old hybrid points and the newly
   * added points */
  for (d = 0; d <= depth; ++d) pMaxNew[d]             = DMPlexShiftPoint_Internal(pMaxNew[d],depth,depthShift) - (numHybridPointsOld[d] + numSplitPoints[d] + numHybridPoints[d]);
  CHKERRQ(DMPlexShiftSizes_Internal(dm, depthShift, sdm));
  /* Step 3: Set cone/support sizes for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt  splitp = p    + pMaxNew[dep];
      const PetscInt *support;
      DMPolytopeType  ct;
      PetscInt        coneSize, supportSize, qf, qn, qp, e;

      CHKERRQ(DMPlexGetConeSize(dm, oldp, &coneSize));
      CHKERRQ(DMPlexSetConeSize(sdm, splitp, coneSize));
      CHKERRQ(DMPlexGetSupportSize(dm, oldp, &supportSize));
      CHKERRQ(DMPlexSetSupportSize(sdm, splitp, supportSize));
      CHKERRQ(DMPlexGetCellType(dm, oldp, &ct));
      CHKERRQ(DMPlexSetCellType(sdm, splitp, ct));
      if (dep == depth-1) {
        const PetscInt hybcell = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Add cohesive cells, they are prisms */
        CHKERRQ(DMPlexSetConeSize(sdm, hybcell, 2 + coneSize));
        switch (coneSize) {
          case 2: CHKERRQ(DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_SEG_PRISM_TENSOR));break;
          case 3: CHKERRQ(DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_TRI_PRISM_TENSOR));break;
          case 4: CHKERRQ(DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_QUAD_PRISM_TENSOR));break;
        }
      } else if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        CHKERRQ(DMPlexGetSupport(dm, oldp, &support));
        for (e = 0, qn = 0, qp = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == 1) ++qf;
          if ((val == 1) || (val ==  (shift + 1))) ++qn;
          if ((val == 1) || (val == -(shift + 1))) ++qp;
        }
        /* Split old vertex: Edges into original vertex and new cohesive edge */
        CHKERRQ(DMPlexSetSupportSize(sdm, newp, qn+1));
        /* Split new vertex: Edges into split vertex and new cohesive edge */
        CHKERRQ(DMPlexSetSupportSize(sdm, splitp, qp+1));
        /* Add hybrid edge */
        CHKERRQ(DMPlexSetConeSize(sdm, hybedge, 2));
        CHKERRQ(DMPlexSetSupportSize(sdm, hybedge, qf));
        CHKERRQ(DMPlexSetCellType(sdm, hybedge, DM_POLYTOPE_POINT_PRISM_TENSOR));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        CHKERRQ(DMPlexGetSupport(dm, oldp, &support));
        for (e = 0, qn = 0, qp = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) ++qf;
          if ((val == dim-1) || (val ==  (shift + dim-1))) ++qn;
          if ((val == dim-1) || (val == -(shift + dim-1))) ++qp;
        }
        /* Split old edge: Faces into original edge and cohesive face (positive side?) */
        CHKERRQ(DMPlexSetSupportSize(sdm, newp, qn+1));
        /* Split new edge: Faces into split edge and cohesive face (negative side?) */
        CHKERRQ(DMPlexSetSupportSize(sdm, splitp, qp+1));
        /* Add hybrid face */
        CHKERRQ(DMPlexSetConeSize(sdm, hybface, 4));
        CHKERRQ(DMPlexSetSupportSize(sdm, hybface, qf));
        CHKERRQ(DMPlexSetCellType(sdm, hybface, DM_POLYTOPE_SEG_PRISM_TENSOR));
      }
    }
  }
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numUnsplitPoints[dep]; ++p) {
      const PetscInt  oldp   = unsplitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt *support;
      PetscInt        coneSize, supportSize, qf, e, s;

      CHKERRQ(DMPlexGetConeSize(dm, oldp, &coneSize));
      CHKERRQ(DMPlexGetSupportSize(dm, oldp, &supportSize));
      CHKERRQ(DMPlexGetSupport(dm, oldp, &support));
      if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit vertex: Edges into original vertex, split edges, and new cohesive edge twice */
        for (s = 0, qf = 0; s < supportSize; ++s, ++qf) {
          CHKERRQ(PetscFindInt(support[s], numSplitPoints[dep+1], splitPoints[dep+1], &e));
          if (e >= 0) ++qf;
        }
        CHKERRQ(DMPlexSetSupportSize(sdm, newp, qf+2));
        /* Add hybrid edge */
        CHKERRQ(DMPlexSetConeSize(sdm, hybedge, 2));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          /* Split and unsplit edges produce hybrid faces */
          if (val == 1) ++qf;
          if (val == (shift2 + 1)) ++qf;
        }
        CHKERRQ(DMPlexSetSupportSize(sdm, hybedge, qf));
        CHKERRQ(DMPlexSetCellType(sdm, hybedge, DM_POLYTOPE_POINT_PRISM_TENSOR));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];
        PetscInt       val;

        for (e = 0, qf = 0; e < supportSize; ++e) {
          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) qf += 2;
          else              ++qf;
        }
        /* Unsplit edge: Faces into original edge, split face, and cohesive face twice */
        CHKERRQ(DMPlexSetSupportSize(sdm, newp, qf+2));
        /* Add hybrid face */
        for (e = 0, qf = 0; e < supportSize; ++e) {
          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) ++qf;
        }
        CHKERRQ(DMPlexSetConeSize(sdm, hybface, 4));
        CHKERRQ(DMPlexSetSupportSize(sdm, hybface, qf));
        CHKERRQ(DMPlexSetCellType(sdm, hybface, DM_POLYTOPE_SEG_PRISM_TENSOR));
      }
    }
  }
  /* Step 4: Setup split DM */
  CHKERRQ(DMSetUp(sdm));
  CHKERRQ(DMPlexShiftPoints_Internal(dm, depthShift, sdm));
  CHKERRQ(DMPlexGetMaxSizes(sdm, &maxConeSizeNew, &maxSupportSizeNew));
  CHKERRQ(PetscMalloc3(PetscMax(maxConeSize, maxConeSizeNew)*3,&coneNew,PetscMax(maxConeSize, maxConeSizeNew)*3,&coneONew,PetscMax(maxSupportSize, maxSupportSizeNew),&supportNew));
  /* Step 6: Set cones and supports for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt  splitp = p    + pMaxNew[dep];
      const PetscInt *cone, *support, *ornt;
      DMPolytopeType  ct;
      PetscInt        coneSize, supportSize, q, qf, qn, qp, v, e, s;

      CHKERRQ(DMPlexGetCellType(dm, oldp, &ct));
      CHKERRQ(DMPlexGetConeSize(dm, oldp, &coneSize));
      CHKERRQ(DMPlexGetCone(dm, oldp, &cone));
      CHKERRQ(DMPlexGetConeOrientation(dm, oldp, &ornt));
      CHKERRQ(DMPlexGetSupportSize(dm, oldp, &supportSize));
      CHKERRQ(DMPlexGetSupport(dm, oldp, &support));
      if (dep == depth-1) {
        PetscBool       hasUnsplit = PETSC_FALSE;
        const PetscInt  hybcell    = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        const PetscInt *supportF;

        /* Split face:       copy in old face to new face to start */
        CHKERRQ(DMPlexGetSupport(sdm, newp,  &supportF));
        CHKERRQ(DMPlexSetSupport(sdm, splitp, supportF));
        /* Split old face:   old vertices/edges in cone so no change */
        /* Split new face:   new vertices/edges in cone */
        for (q = 0; q < coneSize; ++q) {
          CHKERRQ(PetscFindInt(cone[q], numSplitPoints[dep-1], splitPoints[dep-1], &v));
          if (v < 0) {
            CHKERRQ(PetscFindInt(cone[q], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
            PetscCheckFalse(v < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[q], dep-1);
            coneNew[2+q] = DMPlexShiftPoint_Internal(cone[q], depth, depthShift) /*cone[q] + depthOffset[dep-1]*/;
            hasUnsplit   = PETSC_TRUE;
          } else {
            coneNew[2+q] = v + pMaxNew[dep-1];
            if (dep > 1) {
              const PetscInt *econe;
              PetscInt        econeSize, r, vs, vu;

              CHKERRQ(DMPlexGetConeSize(dm, cone[q], &econeSize));
              CHKERRQ(DMPlexGetCone(dm, cone[q], &econe));
              for (r = 0; r < econeSize; ++r) {
                CHKERRQ(PetscFindInt(econe[r], numSplitPoints[dep-2],   splitPoints[dep-2],   &vs));
                CHKERRQ(PetscFindInt(econe[r], numUnsplitPoints[dep-2], unsplitPoints[dep-2], &vu));
                if (vs >= 0) continue;
                PetscCheckFalse(vu < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", econe[r], dep-2);
                hasUnsplit   = PETSC_TRUE;
              }
            }
          }
        }
        CHKERRQ(DMPlexSetCone(sdm, splitp, &coneNew[2]));
        CHKERRQ(DMPlexSetConeOrientation(sdm, splitp, ornt));
        /* Face support */
        for (s = 0; s < supportSize; ++s) {
          PetscInt val;

          CHKERRQ(DMLabelGetValue(label, support[s], &val));
          if (val < 0) {
            /* Split old face:   Replace negative side cell with cohesive cell */
             CHKERRQ(DMPlexInsertSupport(sdm, newp, s, hybcell));
          } else {
            /* Split new face:   Replace positive side cell with cohesive cell */
            CHKERRQ(DMPlexInsertSupport(sdm, splitp, s, hybcell));
            /* Get orientation for cohesive face */
            {
              const PetscInt *ncone, *nconeO;
              PetscInt        nconeSize, nc;

              CHKERRQ(DMPlexGetConeSize(dm, support[s], &nconeSize));
              CHKERRQ(DMPlexGetCone(dm, support[s], &ncone));
              CHKERRQ(DMPlexGetConeOrientation(dm, support[s], &nconeO));
              for (nc = 0; nc < nconeSize; ++nc) {
                if (ncone[nc] == oldp) {
                  coneONew[0] = nconeO[nc];
                  break;
                }
              }
              PetscCheckFalse(nc >= nconeSize,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate face %d in neighboring cell %d", oldp, support[s]);
            }
          }
        }
        /* Cohesive cell:    Old and new split face, then new cohesive faces */
        const PetscInt *arr = DMPolytopeTypeGetArrangment(ct, coneONew[0]);

        coneNew[0]  = newp;   /* Extracted negative side orientation above */
        coneNew[1]  = splitp;
        coneONew[1] = coneONew[0];
        for (q = 0; q < coneSize; ++q) {
          /* Hybrid faces must follow order from oriented end face */
          const PetscInt qa = arr[q*2+0];
          const PetscInt qo = arr[q*2+1];
          DMPolytopeType ft = dep == 2 ? DM_POLYTOPE_SEGMENT : DM_POLYTOPE_POINT;

          CHKERRQ(PetscFindInt(cone[qa], numSplitPoints[dep-1], splitPoints[dep-1], &v));
          if (v < 0) {
            CHKERRQ(PetscFindInt(cone[qa], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
            coneNew[2+q]  = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
          } else {
            coneNew[2+q]  = v + pMaxNew[dep] + numSplitPoints[dep];
          }
          coneONew[2+q] = DMPolytopeTypeComposeOrientation(ft, qo, ornt[qa]);
        }
        CHKERRQ(DMPlexSetCone(sdm, hybcell, coneNew));
        CHKERRQ(DMPlexSetConeOrientation(sdm, hybcell, coneONew));
        /* Label the hybrid cells on the boundary of the split */
        if (hasUnsplit) CHKERRQ(DMLabelSetValue(label, -hybcell, dim));
      } else if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Split old vertex: Edges in old split faces and new cohesive edge */
        for (e = 0, qn = 0; e < supportSize; ++e) {
          PetscInt val;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if ((val == 1) || (val == (shift + 1))) {
            supportNew[qn++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qn] = hybedge;
        CHKERRQ(DMPlexSetSupport(sdm, newp, supportNew));
        /* Split new vertex: Edges in new split faces and new cohesive edge */
        for (e = 0, qp = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == 1) {
            CHKERRQ(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qp++] = edge + pMaxNew[dep+1];
          } else if (val == -(shift + 1)) {
            supportNew[qp++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qp] = hybedge;
        CHKERRQ(DMPlexSetSupport(sdm, splitp, supportNew));
        /* Hybrid edge:    Old and new split vertex */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        CHKERRQ(DMPlexSetCone(sdm, hybedge, coneNew));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == 1) {
            CHKERRQ(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        CHKERRQ(DMPlexSetSupport(sdm, hybedge, supportNew));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Split old edge:   old vertices in cone so no change */
        /* Split new edge:   new vertices in cone */
        for (q = 0; q < coneSize; ++q) {
          CHKERRQ(PetscFindInt(cone[q], numSplitPoints[dep-1], splitPoints[dep-1], &v));
          if (v < 0) {
            CHKERRQ(PetscFindInt(cone[q], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
            PetscCheckFalse(v < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[q], dep-1);
            coneNew[q] = DMPlexShiftPoint_Internal(cone[q], depth, depthShift) /*cone[q] + depthOffset[dep-1]*/;
          } else {
            coneNew[q] = v + pMaxNew[dep-1];
          }
        }
        CHKERRQ(DMPlexSetCone(sdm, splitp, coneNew));
        /* Split old edge: Faces in positive side cells and old split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          } else if (val == (shift + dim-1)) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[q++] = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        CHKERRQ(DMPlexSetSupport(sdm, newp, supportNew));
        /* Split new edge: Faces in negative side cells and new split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val, face;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) {
            CHKERRQ(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            PetscCheckFalse(face < 0,comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[q++] = face + pMaxNew[dep+1];
          } else if (val == -(shift + dim-1)) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[q++] = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        CHKERRQ(DMPlexSetSupport(sdm, splitp, supportNew));
        /* Hybrid face */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        for (v = 0; v < coneSize; ++v) {
          PetscInt vertex;
          CHKERRQ(PetscFindInt(cone[v], numSplitPoints[dep-1], splitPoints[dep-1], &vertex));
          if (vertex < 0) {
            CHKERRQ(PetscFindInt(cone[v], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &vertex));
            PetscCheckFalse(vertex < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[v], dep-1);
            coneNew[2+v] = vertex + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
          } else {
            coneNew[2+v] = vertex + pMaxNew[dep] + numSplitPoints[dep];
          }
        }
        CHKERRQ(DMPlexSetCone(sdm, hybface, coneNew));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, face;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) {
            CHKERRQ(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            PetscCheckFalse(face < 0,comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[qf++] = face + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        CHKERRQ(DMPlexSetSupport(sdm, hybface, supportNew));
      }
    }
  }
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numUnsplitPoints[dep]; ++p) {
      const PetscInt  oldp   = unsplitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt *cone, *support;
      PetscInt        coneSize, supportSize, supportSizeNew, q, qf, e, f, s;

      CHKERRQ(DMPlexGetConeSize(dm, oldp, &coneSize));
      CHKERRQ(DMPlexGetCone(dm, oldp, &cone));
      CHKERRQ(DMPlexGetSupportSize(dm, oldp, &supportSize));
      CHKERRQ(DMPlexGetSupport(dm, oldp, &support));
      if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit vertex */
        CHKERRQ(DMPlexGetSupportSize(sdm, newp, &supportSizeNew));
        for (s = 0, q = 0; s < supportSize; ++s) {
          supportNew[q++] = DMPlexShiftPoint_Internal(support[s], depth, depthShift) /*support[s] + depthOffset[dep+1]*/;
          CHKERRQ(PetscFindInt(support[s], numSplitPoints[dep+1], splitPoints[dep+1], &e));
          if (e >= 0) {
            supportNew[q++] = e + pMaxNew[dep+1];
          }
        }
        supportNew[q++] = hybedge;
        supportNew[q++] = hybedge;
        PetscCheckFalse(q != supportSizeNew,comm, PETSC_ERR_ARG_WRONG, "Support size %d != %d for vertex %d", q, supportSizeNew, newp);
        CHKERRQ(DMPlexSetSupport(sdm, newp, supportNew));
        /* Hybrid edge */
        coneNew[0] = newp;
        coneNew[1] = newp;
        CHKERRQ(DMPlexSetCone(sdm, hybedge, coneNew));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          CHKERRQ(DMLabelGetValue(label, support[e], &val));
          if (val == 1) {
            CHKERRQ(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2];
          } else if  (val ==  (shift2 + 1)) {
            CHKERRQ(PetscFindInt(support[e], numUnsplitPoints[dep+1], unsplitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a unsplit edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2] + numSplitPoints[dep+1];
          }
        }
        CHKERRQ(DMPlexSetSupport(sdm, hybedge, supportNew));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit edge: Faces into original edge, split face, and hybrid face twice */
        for (f = 0, qf = 0; f < supportSize; ++f) {
          PetscInt val, face;

          CHKERRQ(DMLabelGetValue(label, support[f], &val));
          if (val == dim-1) {
            CHKERRQ(PetscFindInt(support[f], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            PetscCheckFalse(face < 0,comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[f]);
            supportNew[qf++] = DMPlexShiftPoint_Internal(support[f], depth, depthShift) /*support[f] + depthOffset[dep+1]*/;
            supportNew[qf++] = face + pMaxNew[dep+1];
          } else {
            supportNew[qf++] = DMPlexShiftPoint_Internal(support[f], depth, depthShift) /*support[f] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qf++] = hybface;
        supportNew[qf++] = hybface;
        CHKERRQ(DMPlexGetSupportSize(sdm, newp, &supportSizeNew));
        PetscCheckFalse(qf != supportSizeNew,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support size for unsplit edge %d is %d != %d", newp, qf, supportSizeNew);
        CHKERRQ(DMPlexSetSupport(sdm, newp, supportNew));
        /* Add hybrid face */
        coneNew[0] = newp;
        coneNew[1] = newp;
        CHKERRQ(PetscFindInt(cone[0], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
        PetscCheckFalse(v < 0,comm, PETSC_ERR_ARG_WRONG, "Vertex %d is not an unsplit vertex", cone[0]);
        coneNew[2] = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
        CHKERRQ(PetscFindInt(cone[1], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
        PetscCheckFalse(v < 0,comm, PETSC_ERR_ARG_WRONG, "Vertex %d is not an unsplit vertex", cone[1]);
        coneNew[3] = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
        CHKERRQ(DMPlexSetCone(sdm, hybface, coneNew));
        for (f = 0, qf = 0; f < supportSize; ++f) {
          PetscInt val, face;

          CHKERRQ(DMLabelGetValue(label, support[f], &val));
          if (val == dim-1) {
            CHKERRQ(PetscFindInt(support[f], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            supportNew[qf++] = face + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        CHKERRQ(DMPlexGetSupportSize(sdm, hybface, &supportSizeNew));
        PetscCheckFalse(qf != supportSizeNew,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support size for hybrid face %d is %d != %d", hybface, qf, supportSizeNew);
        CHKERRQ(DMPlexSetSupport(sdm, hybface, supportNew));
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
    CHKERRQ(DMLabelGetStratumIS(label, dep, &pIS));
    if (!pIS) continue;
    dep  = -dep - shift;
    CHKERRQ(ISGetLocalSize(pIS, &numPoints));
    CHKERRQ(ISGetIndices(pIS, &points));
    for (p = 0; p < numPoints; ++p) {
      const PetscInt  oldp = points[p];
      const PetscInt  newp = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*depthOffset[dep] + oldp*/;
      const PetscInt *cone;
      PetscInt        coneSize, c;
      /* PetscBool       replaced = PETSC_FALSE; */

      /* Negative edge: replace split vertex */
      /* Negative cell: replace split face */
      CHKERRQ(DMPlexGetConeSize(sdm, newp, &coneSize));
      CHKERRQ(DMPlexGetCone(sdm, newp, &cone));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt coldp = DMPlexShiftPointInverse_Internal(cone[c],depth,depthShift);
        PetscInt       csplitp, cp, val;

        CHKERRQ(DMLabelGetValue(label, coldp, &val));
        if (val == dep-1) {
          CHKERRQ(PetscFindInt(coldp, numSplitPoints[dep-1], splitPoints[dep-1], &cp));
          PetscCheckFalse(cp < 0,comm, PETSC_ERR_ARG_WRONG, "Point %d is not a split point of dimension %d", oldp, dep-1);
          csplitp  = pMaxNew[dep-1] + cp;
          CHKERRQ(DMPlexInsertCone(sdm, newp, c, csplitp));
          /* replaced = PETSC_TRUE; */
        }
      }
      /* Cells with only a vertex or edge on the submesh have no replacement */
      /* PetscCheck(replaced,comm, PETSC_ERR_ARG_WRONG, "The cone of point %d does not contain split points", oldp); */
    }
    CHKERRQ(ISRestoreIndices(pIS, &points));
    CHKERRQ(ISDestroy(&pIS));
  }
  /* Step 7: Coordinates */
  CHKERRQ(DMPlexShiftCoordinates_Internal(dm, depthShift, sdm));
  CHKERRQ(DMGetCoordinateSection(sdm, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(sdm, &coordinates));
  CHKERRQ(VecGetArray(coordinates, &coords));
  for (v = 0; v < (numSplitPoints ? numSplitPoints[0] : 0); ++v) {
    const PetscInt newp   = DMPlexShiftPoint_Internal(splitPoints[0][v], depth, depthShift) /*depthOffset[0] + splitPoints[0][v]*/;
    const PetscInt splitp = pMaxNew[0] + v;
    PetscInt       dof, off, soff, d;

    CHKERRQ(PetscSectionGetDof(coordSection, newp, &dof));
    CHKERRQ(PetscSectionGetOffset(coordSection, newp, &off));
    CHKERRQ(PetscSectionGetOffset(coordSection, splitp, &soff));
    for (d = 0; d < dof; ++d) coords[soff+d] = coords[off+d];
  }
  CHKERRQ(VecRestoreArray(coordinates, &coords));
  /* Step 8: SF, if I can figure this out we can split the mesh in parallel */
  CHKERRQ(DMPlexShiftSF_Internal(dm, depthShift, sdm));
  /* Step 9: Labels */
  CHKERRQ(DMPlexShiftLabels_Internal(dm, depthShift, sdm));
  CHKERRQ(DMPlexCreateVTKLabel_Internal(dm, PETSC_FALSE, sdm));
  CHKERRQ(DMGetNumLabels(sdm, &numLabels));
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt newp   = DMPlexShiftPoint_Internal(splitPoints[dep][p], depth, depthShift) /*depthOffset[dep] + splitPoints[dep][p]*/;
      const PetscInt splitp = pMaxNew[dep] + p;
      PetscInt       l;

      if (splitLabel) {
        const PetscInt val = 100 + dep;

        CHKERRQ(DMLabelSetValue(splitLabel, newp,    val));
        CHKERRQ(DMLabelSetValue(splitLabel, splitp, -val));
      }
      for (l = 0; l < numLabels; ++l) {
        DMLabel     mlabel;
        const char *lname;
        PetscInt    val;
        PetscBool   isDepth;

        CHKERRQ(DMGetLabelName(sdm, l, &lname));
        CHKERRQ(PetscStrcmp(lname, "depth", &isDepth));
        if (isDepth) continue;
        CHKERRQ(DMGetLabel(sdm, lname, &mlabel));
        CHKERRQ(DMLabelGetValue(mlabel, newp, &val));
        if (val >= 0) {
          CHKERRQ(DMLabelSetValue(mlabel, splitp, val));
        }
      }
    }
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    if (splitIS[dep]) CHKERRQ(ISRestoreIndices(splitIS[dep], &splitPoints[dep]));
    CHKERRQ(ISDestroy(&splitIS[dep]));
    if (unsplitIS[dep]) CHKERRQ(ISRestoreIndices(unsplitIS[dep], &unsplitPoints[dep]));
    CHKERRQ(ISDestroy(&unsplitIS[dep]));
  }
  if (label) {
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISDestroy(&valueIS));
  }
  for (d = 0; d <= depth; ++d) {
    CHKERRQ(DMPlexGetDepthStratum(sdm, d, NULL, &pEnd));
    pMaxNew[d] = pEnd - numHybridPoints[d] - numHybridPointsOld[d];
  }
  CHKERRQ(PetscFree3(coneNew, coneONew, supportNew));
  CHKERRQ(PetscFree5(depthMax, depthEnd, depthShift, pMaxNew, numHybridPointsOld));
  CHKERRQ(PetscFree7(splitIS, unsplitIS, numSplitPoints, numUnsplitPoints, numHybridPoints, splitPoints, unsplitPoints));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmSplit, 4);
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject)dm), &sdm));
  CHKERRQ(DMSetType(sdm, DMPLEX));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMSetDimension(sdm, dim));
  switch (dim) {
  case 2:
  case 3:
    CHKERRQ(DMPlexConstructCohesiveCells_Internal(dm, label, splitLabel, sdm));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct cohesive cells for dimension %d", dim);
  }
  *dmSplit = sdm;
  PetscFunctionReturn(0);
}

/* Returns the side of the surface for a given cell with a face on the surface */
static PetscErrorCode GetSurfaceSide_Static(DM dm, DM subdm, PetscInt numSubpoints, const PetscInt *subpoints, PetscInt cell, PetscInt face, PetscBool *pos)
{
  const PetscInt *cone, *ornt;
  PetscInt        dim, coneSize, c;

  PetscFunctionBegin;
  *pos = PETSC_TRUE;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetConeSize(dm, cell, &coneSize));
  CHKERRQ(DMPlexGetCone(dm, cell, &cone));
  CHKERRQ(DMPlexGetConeOrientation(dm, cell, &ornt));
  for (c = 0; c < coneSize; ++c) {
    if (cone[c] == face) {
      PetscInt o = ornt[c];

      if (subdm) {
        const PetscInt *subcone, *subornt;
        PetscInt        subpoint, subface, subconeSize, sc;

        CHKERRQ(PetscFindInt(cell, numSubpoints, subpoints, &subpoint));
        CHKERRQ(PetscFindInt(face, numSubpoints, subpoints, &subface));
        CHKERRQ(DMPlexGetConeSize(subdm, subpoint, &subconeSize));
        CHKERRQ(DMPlexGetCone(subdm, subpoint, &subcone));
        CHKERRQ(DMPlexGetConeOrientation(subdm, subpoint, &subornt));
        for (sc = 0; sc < subconeSize; ++sc) {
          if (subcone[sc] == subface) {
            o = subornt[0];
            break;
          }
        }
        PetscCheckFalse(sc >= subconeSize,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find subpoint %d (%d) in cone for subpoint %d (%d)", subface, face, subpoint, cell);
      }
      if (o >= 0) *pos = PETSC_TRUE;
      else        *pos = PETSC_FALSE;
      break;
    }
  }
  PetscCheckFalse(c == coneSize,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Cell %d in split face %d support does not have it in the cone", cell, face);
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  if (subdm) {
    CHKERRQ(DMPlexGetSubpointIS(subdm, &subpointIS));
    if (subpointIS) {
      CHKERRQ(ISGetLocalSize(subpointIS, &numSubpoints));
      CHKERRQ(ISGetIndices(subpointIS, &subpoints));
    }
  }
  /* Mark cell on the fault, and its faces which touch the fault: cell orientation for face gives the side of the fault */
  CHKERRQ(DMLabelGetStratumIS(label, dim-1, &dimIS));
  if (!dimIS) PetscFunctionReturn(0);
  CHKERRQ(ISGetLocalSize(dimIS, &numPoints));
  CHKERRQ(ISGetIndices(dimIS, &points));
  for (p = 0; p < numPoints; ++p) { /* Loop over fault faces */
    const PetscInt *support;
    PetscInt        supportSize, s;

    CHKERRQ(DMPlexGetSupportSize(dm, points[p], &supportSize));
#if 0
    if (supportSize != 2) {
      const PetscInt *lp;
      PetscInt        Nlp, pind;

      /* Check that for a cell with a single support face, that face is in the SF */
      /*   THis check only works for the remote side. We would need root side information */
      CHKERRQ(PetscSFGetGraph(dm->sf, NULL, &Nlp, &lp, NULL));
      CHKERRQ(PetscFindInt(points[p], Nlp, lp, &pind));
      PetscCheckFalse(pind < 0,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Split face %d has %d != 2 supports, and the face is not shared with another process", points[p], supportSize);
    }
#endif
    CHKERRQ(DMPlexGetSupport(dm, points[p], &support));
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;
      PetscBool       pos;

      CHKERRQ(GetSurfaceSide_Static(dm, subdm, numSubpoints, subpoints, support[s], points[p], &pos));
      if (pos) CHKERRQ(DMLabelSetValue(label, support[s],  rev*(shift+dim)));
      else     CHKERRQ(DMLabelSetValue(label, support[s], -rev*(shift+dim)));
      if (rev < 0) pos = !pos ? PETSC_TRUE : PETSC_FALSE;
      /* Put faces touching the fault in the label */
      CHKERRQ(DMPlexGetConeSize(dm, support[s], &coneSize));
      CHKERRQ(DMPlexGetCone(dm, support[s], &cone));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];

        CHKERRQ(DMLabelGetValue(label, point, &val));
        if (val == -1) {
          PetscInt *closure = NULL;
          PetscInt  closureSize, cl;

          CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
          for (cl = 0; cl < closureSize*2; cl += 2) {
            const PetscInt clp  = closure[cl];
            PetscInt       bval = -1;

            CHKERRQ(DMLabelGetValue(label, clp, &val));
            if (blabel) CHKERRQ(DMLabelGetValue(blabel, clp, &bval));
            if ((val >= 0) && (val < dim-1) && (bval < 0)) {
              CHKERRQ(DMLabelSetValue(label, point, pos == PETSC_TRUE ? shift+dim-1 : -(shift+dim-1)));
              break;
            }
          }
          CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
        }
      }
    }
  }
  CHKERRQ(ISRestoreIndices(dimIS, &points));
  CHKERRQ(ISDestroy(&dimIS));
  if (subpointIS) CHKERRQ(ISRestoreIndices(subpointIS, &subpoints));
  /* Mark boundary points as unsplit */
  if (blabel) {
    CHKERRQ(DMLabelGetStratumIS(blabel, 1, &dimIS));
    CHKERRQ(ISGetLocalSize(dimIS, &numPoints));
    CHKERRQ(ISGetIndices(dimIS, &points));
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscInt       val, bval;

      CHKERRQ(DMLabelGetValue(blabel, point, &bval));
      if (bval >= 0) {
        CHKERRQ(DMLabelGetValue(label, point, &val));
        if ((val < 0) || (val > dim)) {
          /* This could be a point added from splitting a vertex on an adjacent fault, otherwise its just wrong */
          CHKERRQ(DMLabelClearValue(blabel, point, bval));
        }
      }
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscInt       val, bval;

      CHKERRQ(DMLabelGetValue(blabel, point, &bval));
      if (bval >= 0) {
        const PetscInt *cone,    *support;
        PetscInt        coneSize, supportSize, s, valA, valB, valE;

        /* Mark as unsplit */
        CHKERRQ(DMLabelGetValue(label, point, &val));
        PetscCheckFalse((val < 0) || (val > dim),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %d has label value %d, should be part of the fault", point, val);
        CHKERRQ(DMLabelClearValue(label, point, val));
        CHKERRQ(DMLabelSetValue(label, point, shift2+val));
        /* Check for cross-edge
             A cross-edge has endpoints which are both on the boundary of the surface, but the edge itself is not. */
        if (val != 0) continue;
        CHKERRQ(DMPlexGetSupport(dm, point, &support));
        CHKERRQ(DMPlexGetSupportSize(dm, point, &supportSize));
        for (s = 0; s < supportSize; ++s) {
          CHKERRQ(DMPlexGetCone(dm, support[s], &cone));
          CHKERRQ(DMPlexGetConeSize(dm, support[s], &coneSize));
          PetscCheckFalse(coneSize != 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %D has %D vertices != 2", support[s], coneSize);
          CHKERRQ(DMLabelGetValue(blabel, cone[0], &valA));
          CHKERRQ(DMLabelGetValue(blabel, cone[1], &valB));
          CHKERRQ(DMLabelGetValue(blabel, support[s], &valE));
          if ((valE < 0) && (valA >= 0) && (valB >= 0) && (cone[0] != cone[1])) CHKERRQ(DMLabelSetValue(blabel, support[s], 2));
        }
      }
    }
    CHKERRQ(ISRestoreIndices(dimIS, &points));
    CHKERRQ(ISDestroy(&dimIS));
  }
  /* Search for other cells/faces/edges connected to the fault by a vertex */
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMLabelGetStratumIS(label, 0, &dimIS));
  /* TODO Why are we including cross edges here? Shouldn't they be in the star of boundary vertices? */
  if (blabel) CHKERRQ(DMLabelGetStratumIS(blabel, 2, &crossEdgeIS));
  if (dimIS && crossEdgeIS) {
    IS vertIS = dimIS;

    CHKERRQ(ISExpand(vertIS, crossEdgeIS, &dimIS));
    CHKERRQ(ISDestroy(&crossEdgeIS));
    CHKERRQ(ISDestroy(&vertIS));
  }
  if (!dimIS) {
    PetscFunctionReturn(0);
  }
  CHKERRQ(ISGetLocalSize(dimIS, &numPoints));
  CHKERRQ(ISGetIndices(dimIS, &points));
  for (p = 0; p < numPoints; ++p) { /* Loop over fault vertices */
    PetscInt *star = NULL;
    PetscInt  starSize, s;
    PetscInt  again = 1;  /* 0: Finished 1: Keep iterating after a change 2: No change */

    /* All points connected to the fault are inside a cell, so at the top level we will only check cells */
    CHKERRQ(DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star));
    while (again) {
      PetscCheckFalse(again > 1,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Could not classify all cells connected to the fault");
      again = 0;
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt  point = star[s];
        const PetscInt *cone;
        PetscInt        coneSize, c;

        if ((point < cStart) || (point >= cEnd)) continue;
        CHKERRQ(DMLabelGetValue(label, point, &val));
        if (val != -1) continue;
        again = again == 1 ? 1 : 2;
        CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
        CHKERRQ(DMPlexGetCone(dm, point, &cone));
        for (c = 0; c < coneSize; ++c) {
          CHKERRQ(DMLabelGetValue(label, cone[c], &val));
          if (val != -1) {
            const PetscInt *ccone;
            PetscInt        cconeSize, cc, side;

            PetscCheckFalse(PetscAbs(val) < shift,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d on cell %d has an invalid label %d", cone[c], point, val);
            if (val > 0) side =  1;
            else         side = -1;
            CHKERRQ(DMLabelSetValue(label, point, side*(shift+dim)));
            /* Mark cell faces which touch the fault */
            CHKERRQ(DMPlexGetConeSize(dm, point, &cconeSize));
            CHKERRQ(DMPlexGetCone(dm, point, &ccone));
            for (cc = 0; cc < cconeSize; ++cc) {
              PetscInt *closure = NULL;
              PetscInt  closureSize, cl;

              CHKERRQ(DMLabelGetValue(label, ccone[cc], &val));
              if (val != -1) continue;
              CHKERRQ(DMPlexGetTransitiveClosure(dm, ccone[cc], PETSC_TRUE, &closureSize, &closure));
              for (cl = 0; cl < closureSize*2; cl += 2) {
                const PetscInt clp = closure[cl];

                CHKERRQ(DMLabelGetValue(label, clp, &val));
                if (val == -1) continue;
                CHKERRQ(DMLabelSetValue(label, ccone[cc], side*(shift+dim-1)));
                break;
              }
              CHKERRQ(DMPlexRestoreTransitiveClosure(dm, ccone[cc], PETSC_TRUE, &closureSize, &closure));
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

      CHKERRQ(DMLabelGetValue(label, point, &val));
      if (val == -1) {
        PetscInt      *sstar = NULL;
        PetscInt       sstarSize, ss;
        PetscBool      marked = PETSC_FALSE, isHybrid;

        CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &sstarSize, &sstar));
        for (ss = 0; ss < sstarSize*2; ss += 2) {
          const PetscInt spoint = sstar[ss];

          if ((spoint < cStart) || (spoint >= cEnd)) continue;
          CHKERRQ(DMLabelGetValue(label, spoint, &val));
          PetscCheckFalse(val == -1,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Cell %d in star of %d does not have a valid label", spoint, point);
          CHKERRQ(DMLabelGetValue(depthLabel, point, &dep));
          if (val > 0) {
            CHKERRQ(DMLabelSetValue(label, point,   shift+dep));
          } else {
            CHKERRQ(DMLabelSetValue(label, point, -(shift+dep)));
          }
          marked = PETSC_TRUE;
          break;
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &sstarSize, &sstar));
        CHKERRQ(DMPlexCellIsHybrid_Internal(dm, point, &isHybrid));
        PetscCheckFalse(!isHybrid && !marked,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d could not be classified", point);
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star));
  }
  CHKERRQ(ISRestoreIndices(dimIS, &points));
  CHKERRQ(ISDestroy(&dimIS));
  /* If any faces touching the fault divide cells on either side, split them
       This only occurs without a surface boundary */
  CHKERRQ(DMLabelGetStratumIS(label,   shift+dim-1,  &facePosIS));
  CHKERRQ(DMLabelGetStratumIS(label, -(shift+dim-1), &faceNegIS));
  CHKERRQ(ISExpand(facePosIS, faceNegIS, &dimIS));
  CHKERRQ(ISDestroy(&facePosIS));
  CHKERRQ(ISDestroy(&faceNegIS));
  CHKERRQ(ISGetLocalSize(dimIS, &numPoints));
  CHKERRQ(ISGetIndices(dimIS, &points));
  for (p = 0; p < numPoints; ++p) {
    const PetscInt  point = points[p];
    const PetscInt *support;
    PetscInt        supportSize, valA, valB;

    CHKERRQ(DMPlexGetSupportSize(dm, point, &supportSize));
    if (supportSize != 2) continue;
    CHKERRQ(DMPlexGetSupport(dm, point, &support));
    CHKERRQ(DMLabelGetValue(label, support[0], &valA));
    CHKERRQ(DMLabelGetValue(label, support[1], &valB));
    if ((valA == -1) || (valB == -1)) continue;
    if (valA*valB > 0) continue;
    /* Split the face */
    CHKERRQ(DMLabelGetValue(label, point, &valA));
    CHKERRQ(DMLabelClearValue(label, point, valA));
    CHKERRQ(DMLabelSetValue(label, point, dim-1));
    /* Label its closure:
      unmarked: label as unsplit
      incident: relabel as split
      split:    do nothing
    */
    {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        CHKERRQ(DMLabelGetValue(label, closure[cl], &valA));
        if (valA == -1) { /* Mark as unsplit */
          CHKERRQ(DMLabelGetValue(depthLabel, closure[cl], &dep));
          CHKERRQ(DMLabelSetValue(label, closure[cl], shift2+dep));
        } else if (((valA >= shift) && (valA < shift2)) || ((valA <= -shift) && (valA > -shift2))) {
          CHKERRQ(DMLabelGetValue(depthLabel, closure[cl], &dep));
          CHKERRQ(DMLabelClearValue(label, closure[cl], valA));
          CHKERRQ(DMLabelSetValue(label, closure[cl], dep));
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
    }
  }
  CHKERRQ(ISRestoreIndices(dimIS, &points));
  CHKERRQ(ISDestroy(&dimIS));
  PetscFunctionReturn(0);
}

/* Check that no cell have all vertices on the fault */
PetscErrorCode DMPlexCheckValidSubmesh_Private(DM dm, DMLabel label, DM subdm)
{
  IS              subpointIS;
  const PetscInt *dmpoints;
  PetscInt        defaultValue, cStart, cEnd, c, vStart, vEnd;

  PetscFunctionBegin;
  if (!label) PetscFunctionReturn(0);
  CHKERRQ(DMLabelGetDefaultValue(label, &defaultValue));
  CHKERRQ(DMPlexGetSubpointIS(subdm, &subpointIS));
  if (!subpointIS) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetHeightStratum(subdm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(ISGetIndices(subpointIS, &dmpoints));
  for (c = cStart; c < cEnd; ++c) {
    PetscBool invalidCell = PETSC_TRUE;
    PetscInt *closure     = NULL;
    PetscInt  closureSize, cl;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, dmpoints[c], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      PetscInt value = 0;

      if ((closure[cl] < vStart) || (closure[cl] >= vEnd)) continue;
      CHKERRQ(DMLabelGetValue(label, closure[cl], &value));
      if (value == defaultValue) {invalidCell = PETSC_FALSE; break;}
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, dmpoints[c], PETSC_TRUE, &closureSize, &closure));
    if (invalidCell) {
      CHKERRQ(ISRestoreIndices(subpointIS, &dmpoints));
      CHKERRQ(ISDestroy(&subpointIS));
      CHKERRQ(DMDestroy(&subdm));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ambiguous submesh. Cell %D has all of its vertices on the submesh.", dmpoints[c]);
    }
  }
  CHKERRQ(ISRestoreIndices(subpointIS, &dmpoints));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (bdlabel) PetscValidPointer(bdlabel, 3);
  if (hybridLabel) PetscValidPointer(hybridLabel, 4);
  if (splitLabel)  PetscValidPointer(splitLabel, 5);
  if (dmInterface) PetscValidPointer(dmInterface, 6);
  PetscValidPointer(dmHybrid, 7);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexCreateSubmesh(dm, label, 1, PETSC_FALSE, &idm));
  CHKERRQ(DMPlexCheckValidSubmesh_Private(dm, label, idm));
  CHKERRQ(DMPlexOrient(idm));
  CHKERRQ(DMPlexGetSubpointMap(idm, &subpointMap));
  CHKERRQ(DMLabelDuplicate(subpointMap, &hlabel));
  CHKERRQ(DMLabelClearStratum(hlabel, dim));
  if (splitLabel) {
    const char *name;
    char        sname[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscObjectGetName((PetscObject) hlabel, &name));
    CHKERRQ(PetscStrncpy(sname, name, PETSC_MAX_PATH_LEN));
    CHKERRQ(PetscStrcat(sname, " split"));
    CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, sname, &slabel));
  }
  CHKERRQ(DMPlexLabelCohesiveComplete(dm, hlabel, bdlabel, PETSC_FALSE, idm));
  if (dmInterface) {*dmInterface = idm;}
  else             CHKERRQ(DMDestroy(&idm));
  CHKERRQ(DMPlexConstructCohesiveCells(dm, hlabel, slabel, dmHybrid));
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *dmHybrid));
  if (hybridLabel) *hybridLabel = hlabel;
  else             CHKERRQ(DMLabelDestroy(&hlabel));
  if (splitLabel)  *splitLabel  = slabel;
  {
    DM      cdm;
    DMLabel ctLabel;

    /* We need to somehow share the celltype label with the coordinate dm */
    CHKERRQ(DMGetCoordinateDM(*dmHybrid, &cdm));
    CHKERRQ(DMPlexGetCellTypeLabel(*dmHybrid, &ctLabel));
    CHKERRQ(DMSetLabel(cdm, ctLabel));
  }
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

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV      = 0;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetDimension(dm, &dim));
  pSize = PetscMax(depth, dim) + 1;
  CHKERRQ(PetscMalloc2(pSize, &pStart, pSize, &pEnd));
  for (d = 0; d <= depth; ++d) {
    CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, depth-d, &pStart[d], &pEnd[d]));
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  if (vertexLabel) CHKERRQ(DMLabelGetStratumIS(vertexLabel, value, &subvertexIS));
  if (subvertexIS) {
    CHKERRQ(ISGetSize(subvertexIS, &numSubVerticesInitial));
    CHKERRQ(ISGetIndices(subvertexIS, &subvertices));
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numCells = 0, c;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      if ((point >= pStart[depth]) && (point < pEnd[depth])) star[numCells++] = point;
    }
    for (c = 0; c < numCells; ++c) {
      const PetscInt cell    = star[c];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, cl;
      PetscInt       cellLoc, numCorners = 0, faceSize = 0;

      CHKERRQ(DMLabelGetValue(subpointMap, cell, &cellLoc));
      if (cellLoc == 2) continue;
      PetscCheckFalse(cellLoc >= 0,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Cell %d has dimension %d in the surface label", cell, cellLoc);
      CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt point = closure[cl];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          ++numCorners;
          CHKERRQ(DMLabelGetValue(vertexLabel, point, &vertexLoc));
          if (vertexLoc == value) closure[faceSize++] = point;
        }
      }
      if (!(*nFV)) CHKERRQ(DMPlexGetNumFaceVertices(dm, dim, numCorners, nFV));
      PetscCheckFalse(faceSize > *nFV,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
      if (faceSize == *nFV) {
        const PetscInt *cells = NULL;
        PetscInt        numCells, nc;

        ++(*numFaces);
        for (cl = 0; cl < faceSize; ++cl) {
          CHKERRQ(DMLabelSetValue(subpointMap, closure[cl], 0));
        }
        CHKERRQ(DMPlexGetJoin(dm, faceSize, closure, &numCells, &cells));
        for (nc = 0; nc < numCells; ++nc) {
          CHKERRQ(DMLabelSetValue(subpointMap, cells[nc], 2));
        }
        CHKERRQ(DMPlexRestoreJoin(dm, faceSize, closure, &numCells, &cells));
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
  }
  if (subvertexIS) {
    CHKERRQ(ISRestoreIndices(subvertexIS, &subvertices));
  }
  CHKERRQ(ISDestroy(&subvertexIS));
  CHKERRQ(PetscFree2(pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DMLabel subpointMap, DM subdm)
{
  IS               subvertexIS = NULL;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd;
  PetscInt         dim, d, numSubVerticesInitial = 0, v;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(PetscMalloc2(dim+1, &pStart, dim+1, &pEnd));
  for (d = 0; d <= dim; ++d) {
    CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, dim-d, &pStart[d], &pEnd[d]));
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  if (vertexLabel) {
    CHKERRQ(DMLabelGetStratumIS(vertexLabel, value, &subvertexIS));
    if (subvertexIS) {
      CHKERRQ(ISGetSize(subvertexIS, &numSubVerticesInitial));
      CHKERRQ(ISGetIndices(subvertexIS, &subvertices));
    }
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numFaces = 0, f;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      PetscInt       faceLoc;

      if ((point >= pStart[dim-1]) && (point < pEnd[dim-1])) {
        if (markedFaces) {
          CHKERRQ(DMLabelGetValue(vertexLabel, point, &faceLoc));
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

      CHKERRQ(DMLabelGetValue(subpointMap, face, &faceLoc));
      if (faceLoc == dim-1) continue;
      PetscCheckFalse(faceLoc >= 0,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d has dimension %d in the surface label", face, faceLoc);
      CHKERRQ(DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        const PetscInt point = closure[c];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          CHKERRQ(DMLabelGetValue(vertexLabel, point, &vertexLoc));
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
              CHKERRQ(DMLabelSetValue(subpointMap, point, d));
              break;
            }
          }
        }
        CHKERRQ(DMPlexGetSupportSize(dm, face, &supportSize));
        CHKERRQ(DMPlexGetSupport(dm, face, &support));
        for (s = 0; s < supportSize; ++s) {
          CHKERRQ(DMLabelSetValue(subpointMap, support[s], dim));
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
  }
  if (subvertexIS) CHKERRQ(ISRestoreIndices(subvertexIS, &subvertices));
  CHKERRQ(ISDestroy(&subvertexIS));
  CHKERRQ(PetscFree2(pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkCohesiveSubmesh_Uninterpolated(DM dm, PetscBool hasLagrange, const char labelname[], PetscInt value, DMLabel subpointMap, PetscInt *numFaces, PetscInt *nFV, PetscInt *subCells[], DM subdm)
{
  DMLabel         label = NULL;
  const PetscInt *cone;
  PetscInt        dim, cMax, cEnd, c, subc = 0, p, coneSize = -1;

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV = 0;
  if (labelname) CHKERRQ(DMGetLabel(dm, labelname, &label));
  *subCells = NULL;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetTensorPrismBounds_Internal(dm, dim, &cMax, &cEnd));
  if (cMax < 0) PetscFunctionReturn(0);
  if (label) {
    for (c = cMax; c < cEnd; ++c) {
      PetscInt val;

      CHKERRQ(DMLabelGetValue(label, c, &val));
      if (val == value) {
        ++(*numFaces);
        CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
      }
    }
  } else {
    *numFaces = cEnd - cMax;
    CHKERRQ(DMPlexGetConeSize(dm, cMax, &coneSize));
  }
  CHKERRQ(PetscMalloc1(*numFaces *2, subCells));
  if (!(*numFaces)) PetscFunctionReturn(0);
  *nFV = hasLagrange ? coneSize/3 : coneSize/2;
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cells;
    PetscInt        numCells;

    if (label) {
      PetscInt val;

      CHKERRQ(DMLabelGetValue(label, c, &val));
      if (val != value) continue;
    }
    CHKERRQ(DMPlexGetCone(dm, c, &cone));
    for (p = 0; p < *nFV; ++p) {
      CHKERRQ(DMLabelSetValue(subpointMap, cone[p], 0));
    }
    /* Negative face */
    CHKERRQ(DMPlexGetJoin(dm, *nFV, cone, &numCells, &cells));
    /* Not true in parallel
    PetscCheckFalse(numCells != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells"); */
    for (p = 0; p < numCells; ++p) {
      CHKERRQ(DMLabelSetValue(subpointMap, cells[p], 2));
      (*subCells)[subc++] = cells[p];
    }
    CHKERRQ(DMPlexRestoreJoin(dm, *nFV, cone, &numCells, &cells));
    /* Positive face is not included */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkCohesiveSubmesh_Interpolated(DM dm, DMLabel label, PetscInt value, DMLabel subpointMap, DM subdm)
{
  PetscInt      *pStart, *pEnd;
  PetscInt       dim, cMax, cEnd, c, d;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetTensorPrismBounds_Internal(dm, dim, &cMax, &cEnd));
  if (cMax < 0) PetscFunctionReturn(0);
  CHKERRQ(PetscMalloc2(dim+1,&pStart,dim+1,&pEnd));
  for (d = 0; d <= dim; ++d) CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]));
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cone;
    PetscInt       *closure = NULL;
    PetscInt        fconeSize, coneSize, closureSize, cl, val;

    if (label) {
      CHKERRQ(DMLabelGetValue(label, c, &val));
      if (val != value) continue;
    }
    CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, c, &cone));
    CHKERRQ(DMPlexGetConeSize(dm, cone[0], &fconeSize));
    PetscCheckFalse(coneSize != (fconeSize ? fconeSize : 1) + 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells");
    /* Negative face */
    CHKERRQ(DMPlexGetTransitiveClosure(dm, cone[0], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];

      for (d = 0; d <= dim; ++d) {
        if ((point >= pStart[d]) && (point < pEnd[d])) {
          CHKERRQ(DMLabelSetValue(subpointMap, point, d));
          break;
        }
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cone[0], PETSC_TRUE, &closureSize, &closure));
    /* Cells -- positive face is not included */
    for (cl = 0; cl < 1; ++cl) {
      const PetscInt *support;
      PetscInt        supportSize, s;

      CHKERRQ(DMPlexGetSupportSize(dm, cone[cl], &supportSize));
      /* PetscCheckFalse(supportSize != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive faces should separate two cells"); */
      CHKERRQ(DMPlexGetSupport(dm, cone[cl], &support));
      for (s = 0; s < supportSize; ++s) {
        CHKERRQ(DMLabelSetValue(subpointMap, support[s], dim));
      }
    }
  }
  CHKERRQ(PetscFree2(pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFaceOrientation(DM dm, PetscInt cell, PetscInt numCorners, PetscInt indices[], PetscInt oppositeVertex, PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  MPI_Comm       comm;
  PetscBool      posOrient = PETSC_FALSE;
  const PetscInt debug     = 0;
  PetscInt       cellDim, faceSize, f;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(DMGetDimension(dm, &cellDim));
  if (debug) CHKERRQ(PetscPrintf(comm, "cellDim: %d numCorners: %d\n", cellDim, numCorners));

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
    CHKERRQ(PetscSortInt(faceSizeTri, sortedIndices));
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
    PetscCheck(found,comm, PETSC_ERR_ARG_WRONG, "Invalid tri crossface");
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
    CHKERRQ(PetscSortInt(faceSizeQuad, sortedIndices));
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
    PetscCheck(found,comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossface");
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
    CHKERRQ(PetscSortInt(faceSizeHex, sortedIndices));
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
    PetscCheck(found,comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
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
    CHKERRQ(PetscSortInt(faceSizeTet, sortedIndices));
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
    PetscCheck(found,comm, PETSC_ERR_ARG_WRONG, "Invalid tet crossface");
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
    CHKERRQ(PetscSortInt(faceSizeQuadHex, sortedIndices));
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
    PetscCheck(found,comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown cell type for faceOrientation().");
  if (!posOrient) {
    if (debug) CHKERRQ(PetscPrintf(comm, "  Reversing initial face orientation\n"));
    for (f = 0; f < faceSize; ++f) faceVertices[f] = origVertices[faceSize-1 - f];
  } else {
    if (debug) CHKERRQ(PetscPrintf(comm, "  Keeping initial face orientation\n"));
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

  Output Parameters:
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetConeSize(dm, cell, &coneSize));
  CHKERRQ(DMPlexGetCone(dm, cell, &cone));
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
  CHKERRQ(DMPlexGetFaceOrientation(dm, cell, numCorners, indices, oppositeVertex, origVertices, faceVertices, posOriented));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(DMPlexGetConeSize(subdm, subcell, &coneSize));
  PetscCheckFalse(coneSize != 1,comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size of cell %d is %d != 1", cell, coneSize);
#if 0
  /* Cannot use this because support() has not been constructed yet */
  CHKERRQ(DMPlexGetJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces));
#else
  {
    PetscInt f;

    numFaces = 0;
    CHKERRQ(DMGetWorkArray(subdm, 1, MPIU_INT, (void **) &faces));
    for (f = firstFace; f < *newFacePoint; ++f) {
      PetscInt dof, off, d;

      CHKERRQ(PetscSectionGetDof(submesh->coneSection, f, &dof));
      CHKERRQ(PetscSectionGetOffset(submesh->coneSection, f, &off));
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
  PetscCheckFalse(numFaces > 1,comm, PETSC_ERR_ARG_WRONG, "Vertex set had %d faces, not one", numFaces);
  else if (numFaces == 1) {
    /* Add the other cell neighbor for this face */
    CHKERRQ(DMPlexSetCone(subdm, subcell, faces));
  } else {
    PetscInt *indices, *origVertices, *orientedVertices, *orientedSubVertices, v, ov;
    PetscBool posOriented;

    CHKERRQ(DMGetWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), MPIU_INT, &orientedVertices));
    origVertices        = &orientedVertices[numFaceVertices];
    indices             = &orientedVertices[numFaceVertices*2];
    orientedSubVertices = &orientedVertices[numFaceVertices*3];
    CHKERRQ(DMPlexGetOrientedFace(dm, cell, numFaceVertices, faceVertices, numCorners, indices, origVertices, orientedVertices, &posOriented));
    /* TODO: I know that routine should return a permutation, not the indices */
    for (v = 0; v < numFaceVertices; ++v) {
      const PetscInt vertex = faceVertices[v], subvertex = subfaceVertices[v];
      for (ov = 0; ov < numFaceVertices; ++ov) {
        if (orientedVertices[ov] == vertex) {
          orientedSubVertices[ov] = subvertex;
          break;
        }
      }
      PetscCheckFalse(ov == numFaceVertices,comm, PETSC_ERR_PLIB, "Could not find face vertex %d in orientated set", vertex);
    }
    CHKERRQ(DMPlexSetCone(subdm, *newFacePoint, orientedSubVertices));
    CHKERRQ(DMPlexSetCone(subdm, subcell, newFacePoint));
    CHKERRQ(DMRestoreWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), MPIU_INT, &orientedVertices));
    ++(*newFacePoint);
  }
#if 0
  CHKERRQ(DMPlexRestoreJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces));
#else
  CHKERRQ(DMRestoreWorkArray(subdm, 1, MPIU_INT, (void **) &faces));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  /* Create subpointMap which marks the submesh */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap));
  CHKERRQ(DMPlexSetSubpointMap(subdm, subpointMap));
  CHKERRQ(DMLabelDestroy(&subpointMap));
  if (vertexLabel) CHKERRQ(DMPlexMarkSubmesh_Uninterpolated(dm, vertexLabel, value, subpointMap, &numSubFaces, &nFV, subdm));
  /* Setup chart */
  CHKERRQ(DMLabelGetStratumSize(subpointMap, 0, &numSubVertices));
  CHKERRQ(DMLabelGetStratumSize(subpointMap, 2, &numSubCells));
  CHKERRQ(DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices));
  CHKERRQ(DMPlexSetVTKCellHeight(subdm, 1));
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  CHKERRQ(DMLabelGetStratumIS(subpointMap, 0, &subvertexIS));
  if (subvertexIS) CHKERRQ(ISGetIndices(subvertexIS, &subVertices));
  CHKERRQ(DMLabelGetStratumIS(subpointMap, 2, &subcellIS));
  if (subcellIS) CHKERRQ(ISGetIndices(subcellIS, &subCells));
  for (c = 0; c < numSubCells; ++c) {
    CHKERRQ(DMPlexSetConeSize(subdm, c, 1));
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    CHKERRQ(DMPlexSetConeSize(subdm, f, nFV));
  }
  CHKERRQ(DMSetUp(subdm));
  /* Create face cones */
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  CHKERRQ(DMGetWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt cell    = subCells[c];
    const PetscInt subcell = c;
    PetscInt      *closure = NULL;
    PetscInt       closureSize, cl, numCorners = 0, faceSize = 0;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];
      PetscInt       subVertex;

      if ((point >= vStart) && (point < vEnd)) {
        ++numCorners;
        CHKERRQ(PetscFindInt(point, numSubVertices, subVertices, &subVertex));
        if (subVertex >= 0) {
          closure[faceSize] = point;
          subface[faceSize] = firstSubVertex+subVertex;
          ++faceSize;
        }
      }
    }
    PetscCheckFalse(faceSize > nFV,comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
    if (faceSize == nFV) {
      CHKERRQ(DMPlexInsertFace_Internal(dm, subdm, faceSize, closure, subface, numCorners, cell, subcell, firstSubFace, &newFacePoint));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  }
  CHKERRQ(DMRestoreWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  CHKERRQ(DMPlexSymmetrize(subdm));
  CHKERRQ(DMPlexStratify(subdm));
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize, v;
    const char  *name;

    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMGetCoordinateSection(subdm, &subCoordSection));
    CHKERRQ(PetscSectionSetNumFields(subCoordSection, 1));
    CHKERRQ(PetscSectionGetFieldComponents(coordSection, 0, &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(subCoordSection, 0, numComp));
    CHKERRQ(PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices));
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      CHKERRQ(PetscSectionGetDof(coordSection, vertex, &dof));
      CHKERRQ(PetscSectionSetDof(subCoordSection, subvertex, dof));
      CHKERRQ(PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof));
    }
    CHKERRQ(PetscSectionSetUp(subCoordSection));
    CHKERRQ(PetscSectionGetStorageSize(subCoordSection, &coordSize));
    CHKERRQ(VecCreate(PETSC_COMM_SELF, &subCoordinates));
    CHKERRQ(PetscObjectGetName((PetscObject)coordinates,&name));
    CHKERRQ(PetscObjectSetName((PetscObject)subCoordinates,name));
    CHKERRQ(VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE));
    CHKERRQ(VecSetType(subCoordinates,VECSTANDARD));
    if (coordSize) {
      CHKERRQ(VecGetArray(coordinates,    &coords));
      CHKERRQ(VecGetArray(subCoordinates, &subCoords));
      for (v = 0; v < numSubVertices; ++v) {
        const PetscInt vertex    = subVertices[v];
        const PetscInt subvertex = firstSubVertex+v;
        PetscInt       dof, off, sdof, soff, d;

        CHKERRQ(PetscSectionGetDof(coordSection, vertex, &dof));
        CHKERRQ(PetscSectionGetOffset(coordSection, vertex, &off));
        CHKERRQ(PetscSectionGetDof(subCoordSection, subvertex, &sdof));
        CHKERRQ(PetscSectionGetOffset(subCoordSection, subvertex, &soff));
        PetscCheckFalse(dof != sdof,comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
        for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
      }
      CHKERRQ(VecRestoreArray(coordinates,    &coords));
      CHKERRQ(VecRestoreArray(subCoordinates, &subCoords));
    }
    CHKERRQ(DMSetCoordinatesLocal(subdm, subCoordinates));
    CHKERRQ(VecDestroy(&subCoordinates));
  }
  /* Cleanup */
  if (subvertexIS) CHKERRQ(ISRestoreIndices(subvertexIS, &subVertices));
  CHKERRQ(ISDestroy(&subvertexIS));
  if (subcellIS) CHKERRQ(ISRestoreIndices(subcellIS, &subCells));
  CHKERRQ(ISDestroy(&subcellIS));
  PetscFunctionReturn(0);
}

static inline PetscInt DMPlexFilterPoint_Internal(PetscInt point, PetscInt firstSubPoint, PetscInt numSubPoints, const PetscInt subPoints[])
{
  PetscInt       subPoint;
  PetscErrorCode ierr;

  ierr = PetscFindInt(point, numSubPoints, subPoints, &subPoint); if (ierr < 0) return ierr;
  return subPoint < 0 ? subPoint : firstSubPoint+subPoint;
}

static PetscErrorCode DMPlexFilterLabels_Internal(DM dm, const PetscInt numSubPoints[], const PetscInt *subpoints[], const PetscInt firstSubPoint[], DM subdm)
{
  PetscInt       Nl, l, d;

  PetscFunctionBegin;
  CHKERRQ(DMGetNumLabels(dm, &Nl));
  for (l = 0; l < Nl; ++l) {
    DMLabel         label, newlabel;
    const char     *lname;
    PetscBool       isDepth, isDim, isCelltype, isVTK;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        Nv, v;

    CHKERRQ(DMGetLabelName(dm, l, &lname));
    CHKERRQ(PetscStrcmp(lname, "depth", &isDepth));
    CHKERRQ(PetscStrcmp(lname, "dim", &isDim));
    CHKERRQ(PetscStrcmp(lname, "celltype", &isCelltype));
    CHKERRQ(PetscStrcmp(lname, "vtk", &isVTK));
    if (isDepth || isDim || isCelltype || isVTK) continue;
    CHKERRQ(DMCreateLabel(subdm, lname));
    CHKERRQ(DMGetLabel(dm, lname, &label));
    CHKERRQ(DMGetLabel(subdm, lname, &newlabel));
    CHKERRQ(DMLabelGetDefaultValue(label, &v));
    CHKERRQ(DMLabelSetDefaultValue(newlabel, v));
    CHKERRQ(DMLabelGetValueIS(label, &valueIS));
    CHKERRQ(ISGetLocalSize(valueIS, &Nv));
    CHKERRQ(ISGetIndices(valueIS, &values));
    for (v = 0; v < Nv; ++v) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        Np, p;

      CHKERRQ(DMLabelGetStratumIS(label, values[v], &pointIS));
      CHKERRQ(ISGetLocalSize(pointIS, &Np));
      CHKERRQ(ISGetIndices(pointIS, &points));
      for (p = 0; p < Np; ++p) {
        const PetscInt point = points[p];
        PetscInt       subp;

        CHKERRQ(DMPlexGetPointDepth(dm, point, &d));
        subp = DMPlexFilterPoint_Internal(point, firstSubPoint[d], numSubPoints[d], subpoints[d]);
        if (subp >= 0) CHKERRQ(DMLabelSetValue(newlabel, subp, values[v]));
      }
      CHKERRQ(ISRestoreIndices(pointIS, &points));
      CHKERRQ(ISDestroy(&pointIS));
    }
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISDestroy(&valueIS));
  }
  PetscFunctionReturn(0);
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  /* Create subpointMap which marks the submesh */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap));
  CHKERRQ(DMPlexSetSubpointMap(subdm, subpointMap));
  if (cellHeight) {
    if (isCohesive) CHKERRQ(DMPlexMarkCohesiveSubmesh_Interpolated(dm, label, value, subpointMap, subdm));
    else            CHKERRQ(DMPlexMarkSubmesh_Interpolated(dm, label, value, markedFaces, subpointMap, subdm));
  } else {
    DMLabel         depth;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints=0;

    CHKERRQ(DMPlexGetDepthLabel(dm, &depth));
    CHKERRQ(DMLabelGetStratumIS(label, value, &pointIS));
    if (pointIS) {
      CHKERRQ(ISGetIndices(pointIS, &points));
      CHKERRQ(ISGetLocalSize(pointIS, &numPoints));
    }
    for (p = 0; p < numPoints; ++p) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, c, pdim;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        CHKERRQ(DMLabelGetValue(depth, closure[c], &pdim));
        CHKERRQ(DMLabelSetValue(subpointMap, closure[c], pdim));
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure));
    }
    if (pointIS) CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }
  /* Setup chart */
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(PetscMalloc4(dim+1,&numSubPoints,dim+1,&firstSubPoint,dim+1,&subpointIS,dim+1,&subpoints));
  for (d = 0; d <= dim; ++d) {
    CHKERRQ(DMLabelGetStratumSize(subpointMap, d, &numSubPoints[d]));
    totSubPoints += numSubPoints[d];
  }
  CHKERRQ(DMPlexSetChart(subdm, 0, totSubPoints));
  CHKERRQ(DMPlexSetVTKCellHeight(subdm, cellHeight));
  /* Set cone sizes */
  firstSubPoint[dim] = 0;
  firstSubPoint[0]   = firstSubPoint[dim] + numSubPoints[dim];
  if (dim > 1) {firstSubPoint[dim-1] = firstSubPoint[0]     + numSubPoints[0];}
  if (dim > 2) {firstSubPoint[dim-2] = firstSubPoint[dim-1] + numSubPoints[dim-1];}
  for (d = 0; d <= dim; ++d) {
    CHKERRQ(DMLabelGetStratumIS(subpointMap, d, &subpointIS[d]));
    if (subpointIS[d]) CHKERRQ(ISGetIndices(subpointIS[d], &subpoints[d]));
  }
  /* We do not want this label automatically computed, instead we compute it here */
  CHKERRQ(DMCreateLabel(subdm, "celltype"));
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, coneSizeNew, c, val;
      DMPolytopeType  ct;

      CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
      CHKERRQ(DMPlexSetConeSize(subdm, subpoint, coneSize));
      CHKERRQ(DMPlexGetCellType(dm, point, &ct));
      CHKERRQ(DMPlexSetCellType(subdm, subpoint, ct));
      if (cellHeight && (d == dim)) {
        CHKERRQ(DMPlexGetCone(dm, point, &cone));
        for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
          CHKERRQ(DMLabelGetValue(subpointMap, cone[c], &val));
          if (val >= 0) coneSizeNew++;
        }
        CHKERRQ(DMPlexSetConeSize(subdm, subpoint, coneSizeNew));
        CHKERRQ(DMPlexSetCellType(subdm, subpoint, DM_POLYTOPE_FV_GHOST));
      }
    }
  }
  CHKERRQ(DMLabelDestroy(&subpointMap));
  CHKERRQ(DMSetUp(subdm));
  /* Set cones */
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  CHKERRQ(PetscMalloc2(maxConeSize,&coneNew,maxConeSize,&orntNew));
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, subconeSize, coneSizeNew, c, subc, fornt = 0;

      if (d == dim-1) {
        const PetscInt *support, *cone, *ornt;
        PetscInt        supportSize, coneSize, s, subc;

        CHKERRQ(DMPlexGetSupport(dm, point, &support));
        CHKERRQ(DMPlexGetSupportSize(dm, point, &supportSize));
        for (s = 0; s < supportSize; ++s) {
          PetscBool isHybrid;

          CHKERRQ(DMPlexCellIsHybrid_Internal(dm, support[s], &isHybrid));
          if (!isHybrid) continue;
          CHKERRQ(PetscFindInt(support[s], numSubPoints[d+1], subpoints[d+1], &subc));
          if (subc >= 0) {
            const PetscInt ccell = subpoints[d+1][subc];

            CHKERRQ(DMPlexGetCone(dm, ccell, &cone));
            CHKERRQ(DMPlexGetConeSize(dm, ccell, &coneSize));
            CHKERRQ(DMPlexGetConeOrientation(dm, ccell, &ornt));
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
      CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
      CHKERRQ(DMPlexGetConeSize(subdm, subpoint, &subconeSize));
      CHKERRQ(DMPlexGetCone(dm, point, &cone));
      CHKERRQ(DMPlexGetConeOrientation(dm, point, &ornt));
      for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
        CHKERRQ(PetscFindInt(cone[c], numSubPoints[d-1], subpoints[d-1], &subc));
        if (subc >= 0) {
          coneNew[coneSizeNew] = firstSubPoint[d-1] + subc;
          orntNew[coneSizeNew] = ornt[c];
          ++coneSizeNew;
        }
      }
      PetscCheckFalse(coneSizeNew != subconeSize,comm, PETSC_ERR_PLIB, "Number of cone points located %d does not match subcone size %d", coneSizeNew, subconeSize);
      CHKERRQ(DMPlexSetCone(subdm, subpoint, coneNew));
      CHKERRQ(DMPlexSetConeOrientation(subdm, subpoint, orntNew));
      if (fornt < 0) CHKERRQ(DMPlexOrientPoint(subdm, subpoint, fornt));
    }
  }
  CHKERRQ(PetscFree2(coneNew,orntNew));
  CHKERRQ(DMPlexSymmetrize(subdm));
  CHKERRQ(DMPlexStratify(subdm));
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     cdim, numComp, coordSize;
    const char  *name;

    CHKERRQ(DMGetCoordinateDim(dm, &cdim));
    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMGetCoordinateSection(subdm, &subCoordSection));
    CHKERRQ(PetscSectionSetNumFields(subCoordSection, 1));
    CHKERRQ(PetscSectionGetFieldComponents(coordSection, 0, &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(subCoordSection, 0, numComp));
    CHKERRQ(PetscSectionSetChart(subCoordSection, firstSubPoint[0], firstSubPoint[0]+numSubPoints[0]));
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt       dof;

      CHKERRQ(PetscSectionGetDof(coordSection, vertex, &dof));
      CHKERRQ(PetscSectionSetDof(subCoordSection, subvertex, dof));
      CHKERRQ(PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof));
    }
    CHKERRQ(PetscSectionSetUp(subCoordSection));
    CHKERRQ(PetscSectionGetStorageSize(subCoordSection, &coordSize));
    CHKERRQ(VecCreate(PETSC_COMM_SELF, &subCoordinates));
    CHKERRQ(PetscObjectGetName((PetscObject)coordinates,&name));
    CHKERRQ(PetscObjectSetName((PetscObject)subCoordinates,name));
    CHKERRQ(VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE));
    CHKERRQ(VecSetBlockSize(subCoordinates, cdim));
    CHKERRQ(VecSetType(subCoordinates,VECSTANDARD));
    CHKERRQ(VecGetArray(coordinates,    &coords));
    CHKERRQ(VecGetArray(subCoordinates, &subCoords));
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt dof, off, sdof, soff, d;

      CHKERRQ(PetscSectionGetDof(coordSection, vertex, &dof));
      CHKERRQ(PetscSectionGetOffset(coordSection, vertex, &off));
      CHKERRQ(PetscSectionGetDof(subCoordSection, subvertex, &sdof));
      CHKERRQ(PetscSectionGetOffset(subCoordSection, subvertex, &soff));
      PetscCheckFalse(dof != sdof,comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    CHKERRQ(VecRestoreArray(coordinates,    &coords));
    CHKERRQ(VecRestoreArray(subCoordinates, &subCoords));
    CHKERRQ(DMSetCoordinatesLocal(subdm, subCoordinates));
    CHKERRQ(VecDestroy(&subCoordinates));
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

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
    CHKERRQ(DMGetPointSF(dm, &sfPoint));
    CHKERRQ(DMGetPointSF(subdm, &sfPointSub));
    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    CHKERRQ(DMPlexGetChart(subdm, NULL, &numSubroots));
    CHKERRQ(DMPlexGetSubpointIS(subdm, &subpIS));
    if (subpIS) {
      CHKERRQ(ISGetIndices(subpIS, &subpoints));
      CHKERRQ(ISGetLocalSize(subpIS, &numSubpoints));
    }
    CHKERRQ(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
    if (numRoots >= 0) {
      CHKERRQ(PetscMalloc2(pEnd-pStart,&newLocalPoints,numRoots,&newOwners));
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
      CHKERRQ(PetscSFReduceBegin(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      CHKERRQ(PetscSFReduceEnd(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      CHKERRQ(PetscSFBcastBegin(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      CHKERRQ(PetscMalloc1(numSubleaves, &slocalPoints));
      CHKERRQ(PetscMalloc1(numSubleaves, &sremotePoints));
      for (l = 0, sl = 0, ll = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subpoint = DMPlexFilterPoint_Internal(point, 0, numSubpoints, subpoints);

        if (subpoint < 0) continue;
        if (newLocalPoints[point].rank == rank) {++ll; continue;}
        slocalPoints[sl]        = subpoint;
        sremotePoints[sl].rank  = newLocalPoints[point].rank;
        sremotePoints[sl].index = newLocalPoints[point].index;
        PetscCheckFalse(sremotePoints[sl].rank  < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote rank for local point %d", point);
        PetscCheckFalse(sremotePoints[sl].index < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote subpoint for local point %d", point);
        ++sl;
      }
      PetscCheckFalse(sl + ll != numSubleaves,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatch in number of subleaves %d + %d != %d", sl, ll, numSubleaves);
      CHKERRQ(PetscFree2(newLocalPoints,newOwners));
      CHKERRQ(PetscSFSetGraph(sfPointSub, numSubroots, sl, slocalPoints, PETSC_OWN_POINTER, sremotePoints, PETSC_OWN_POINTER));
    }
    if (subpIS) {
      CHKERRQ(ISRestoreIndices(subpIS, &subpoints));
    }
  }
  /* Filter labels */
  CHKERRQ(DMPlexFilterLabels_Internal(dm, numSubPoints, subpoints, firstSubPoint, subdm));
  /* Cleanup */
  for (d = 0; d <= dim; ++d) {
    if (subpointIS[d]) CHKERRQ(ISRestoreIndices(subpointIS[d], &subpoints[d]));
    CHKERRQ(ISDestroy(&subpointIS[d]));
  }
  CHKERRQ(PetscFree4(numSubPoints,firstSubPoint,subpointIS,subpoints));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DM subdm)
{
  PetscFunctionBegin;
  CHKERRQ(DMPlexCreateSubmeshGeneric_Interpolated(dm, vertexLabel, value, markedFaces, PETSC_FALSE, 1, subdm));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 5);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject)dm), subdm));
  CHKERRQ(DMSetType(*subdm, DMPLEX));
  CHKERRQ(DMSetDimension(*subdm, dim-1));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMSetCoordinateDim(*subdm, cdim));
  CHKERRQ(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheckFalse(interpolated == DMPLEX_INTERPOLATED_PARTIAL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated) {
    CHKERRQ(DMPlexCreateSubmesh_Interpolated(dm, vertexLabel, value, markedFaces, *subdm));
  } else {
    CHKERRQ(DMPlexCreateSubmesh_Uninterpolated(dm, vertexLabel, value, *subdm));
  }
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *subdm));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  /* Create subpointMap which marks the submesh */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap));
  CHKERRQ(DMPlexSetSubpointMap(subdm, subpointMap));
  CHKERRQ(DMLabelDestroy(&subpointMap));
  CHKERRQ(DMPlexMarkCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, subpointMap, &numSubFaces, &nFV, &subCells, subdm));
  /* Setup chart */
  CHKERRQ(DMLabelGetStratumSize(subpointMap, 0, &numSubVertices));
  CHKERRQ(DMLabelGetStratumSize(subpointMap, 2, &numSubCells));
  CHKERRQ(DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices));
  CHKERRQ(DMPlexSetVTKCellHeight(subdm, 1));
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  CHKERRQ(DMLabelGetStratumIS(subpointMap, 0, &subvertexIS));
  if (subvertexIS) CHKERRQ(ISGetIndices(subvertexIS, &subVertices));
  for (c = 0; c < numSubCells; ++c) {
    CHKERRQ(DMPlexSetConeSize(subdm, c, 1));
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    CHKERRQ(DMPlexSetConeSize(subdm, f, nFV));
  }
  CHKERRQ(DMSetUp(subdm));
  /* Create face cones */
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  CHKERRQ(DMGetWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt  cell    = subCells[c];
    const PetscInt  subcell = c;
    const PetscInt *cone, *cells;
    PetscBool       isHybrid;
    PetscInt        numCells, subVertex, p, v;

    CHKERRQ(DMPlexCellIsHybrid_Internal(dm, cell, &isHybrid));
    if (!isHybrid) continue;
    CHKERRQ(DMPlexGetCone(dm, cell, &cone));
    for (v = 0; v < nFV; ++v) {
      CHKERRQ(PetscFindInt(cone[v], numSubVertices, subVertices, &subVertex));
      subface[v] = firstSubVertex+subVertex;
    }
    CHKERRQ(DMPlexSetCone(subdm, newFacePoint, subface));
    CHKERRQ(DMPlexSetCone(subdm, subcell, &newFacePoint));
    CHKERRQ(DMPlexGetJoin(dm, nFV, cone, &numCells, &cells));
    /* Not true in parallel
    PetscCheckFalse(numCells != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells"); */
    for (p = 0; p < numCells; ++p) {
      PetscInt  negsubcell;
      PetscBool isHybrid;

      CHKERRQ(DMPlexCellIsHybrid_Internal(dm, cells[p], &isHybrid));
      if (isHybrid) continue;
      /* I know this is a crap search */
      for (negsubcell = 0; negsubcell < numSubCells; ++negsubcell) {
        if (subCells[negsubcell] == cells[p]) break;
      }
      PetscCheckFalse(negsubcell == numSubCells,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find negative face neighbor for cohesive cell %d", cell);
      CHKERRQ(DMPlexSetCone(subdm, negsubcell, &newFacePoint));
    }
    CHKERRQ(DMPlexRestoreJoin(dm, nFV, cone, &numCells, &cells));
    ++newFacePoint;
  }
  CHKERRQ(DMRestoreWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  CHKERRQ(DMPlexSymmetrize(subdm));
  CHKERRQ(DMPlexStratify(subdm));
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     cdim, numComp, coordSize, v;
    const char  *name;

    CHKERRQ(DMGetCoordinateDim(dm, &cdim));
    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMGetCoordinateSection(subdm, &subCoordSection));
    CHKERRQ(PetscSectionSetNumFields(subCoordSection, 1));
    CHKERRQ(PetscSectionGetFieldComponents(coordSection, 0, &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(subCoordSection, 0, numComp));
    CHKERRQ(PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices));
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      CHKERRQ(PetscSectionGetDof(coordSection, vertex, &dof));
      CHKERRQ(PetscSectionSetDof(subCoordSection, subvertex, dof));
      CHKERRQ(PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof));
    }
    CHKERRQ(PetscSectionSetUp(subCoordSection));
    CHKERRQ(PetscSectionGetStorageSize(subCoordSection, &coordSize));
    CHKERRQ(VecCreate(PETSC_COMM_SELF, &subCoordinates));
    CHKERRQ(PetscObjectGetName((PetscObject)coordinates,&name));
    CHKERRQ(PetscObjectSetName((PetscObject)subCoordinates,name));
    CHKERRQ(VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE));
    CHKERRQ(VecSetBlockSize(subCoordinates, cdim));
    CHKERRQ(VecSetType(subCoordinates,VECSTANDARD));
    CHKERRQ(VecGetArray(coordinates,    &coords));
    CHKERRQ(VecGetArray(subCoordinates, &subCoords));
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof, off, sdof, soff, d;

      CHKERRQ(PetscSectionGetDof(coordSection, vertex, &dof));
      CHKERRQ(PetscSectionGetOffset(coordSection, vertex, &off));
      CHKERRQ(PetscSectionGetDof(subCoordSection, subvertex, &sdof));
      CHKERRQ(PetscSectionGetOffset(subCoordSection, subvertex, &soff));
      PetscCheckFalse(dof != sdof,comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    CHKERRQ(VecRestoreArray(coordinates,    &coords));
    CHKERRQ(VecRestoreArray(subCoordinates, &subCoords));
    CHKERRQ(DMSetCoordinatesLocal(subdm, subCoordinates));
    CHKERRQ(VecDestroy(&subCoordinates));
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

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
    CHKERRQ(DMGetPointSF(dm, &sfPoint));
    CHKERRQ(DMGetPointSF(subdm, &sfPointSub));
    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    CHKERRQ(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
    if (numRoots >= 0) {
      /* Only vertices should be shared */
      CHKERRQ(PetscMalloc2(pEnd-pStart,&newLocalPoints,numRoots,&newOwners));
      for (p = 0; p < pEnd-pStart; ++p) {
        newLocalPoints[p].rank  = -2;
        newLocalPoints[p].index = -2;
      }
      /* Set subleaves */
      for (l = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subPoint = DMPlexFilterPoint_Internal(point, firstSubVertex, numSubVertices, subVertices);

        PetscCheckFalse((point < vStart) && (point >= vEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Should not be mapping anything but vertices, %d", point);
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
      CHKERRQ(PetscSFReduceBegin(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      CHKERRQ(PetscSFReduceEnd(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      CHKERRQ(PetscSFBcastBegin(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      CHKERRQ(PetscMalloc1(numSubLeaves,    &slocalPoints));
      CHKERRQ(PetscMalloc1(numSubLeaves, &sremotePoints));
      for (l = 0, sl = 0, ll = 0; l < numLeaves; ++l) {
        const PetscInt point    = localPoints[l];
        const PetscInt subPoint = DMPlexFilterPoint_Internal(point, firstSubVertex, numSubVertices, subVertices);

        if (subPoint < 0) continue;
        if (newLocalPoints[point].rank == rank) {++ll; continue;}
        slocalPoints[sl]        = subPoint;
        sremotePoints[sl].rank  = newLocalPoints[point].rank;
        sremotePoints[sl].index = newLocalPoints[point].index;
        PetscCheckFalse(sremotePoints[sl].rank  < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote rank for local point %d", point);
        PetscCheckFalse(sremotePoints[sl].index < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid remote subpoint for local point %d", point);
        ++sl;
      }
      CHKERRQ(PetscFree2(newLocalPoints,newOwners));
      PetscCheckFalse(sl + ll != numSubLeaves,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatch in number of subleaves %d + %d != %d", sl, ll, numSubLeaves);
      CHKERRQ(PetscSFSetGraph(sfPointSub, numSubRoots, sl, slocalPoints, PETSC_OWN_POINTER, sremotePoints, PETSC_OWN_POINTER));
    }
  }
  CHKMEMQ;
  /* Cleanup */
  if (subvertexIS) CHKERRQ(ISRestoreIndices(subvertexIS, &subVertices));
  CHKERRQ(ISDestroy(&subvertexIS));
  CHKERRQ(PetscFree(subCells));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCohesiveSubmesh_Interpolated(DM dm, const char labelname[], PetscInt value, DM subdm)
{
  DMLabel        label = NULL;

  PetscFunctionBegin;
  if (labelname) CHKERRQ(DMGetLabel(dm, labelname, &label));
  CHKERRQ(DMPlexCreateSubmeshGeneric_Interpolated(dm, label, value, PETSC_FALSE, PETSC_TRUE, 1, subdm));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateCohesiveSubmesh - Extract from a mesh with cohesive cells the hypersurface defined by one face of the cells. Optionally, a Label can be given to restrict the cells.

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 5);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject)dm), subdm));
  CHKERRQ(DMSetType(*subdm, DMPLEX));
  CHKERRQ(DMSetDimension(*subdm, dim-1));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMSetCoordinateDim(*subdm, cdim));
  if (depth == dim) {
    CHKERRQ(DMPlexCreateCohesiveSubmesh_Interpolated(dm, label, value, *subdm));
  } else {
    CHKERRQ(DMPlexCreateCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, *subdm));
  }
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *subdm));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 4);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), subdm));
  CHKERRQ(DMSetType(*subdm, DMPLEX));
  CHKERRQ(DMSetDimension(*subdm, dim));
  /* Extract submesh in place, could be empty on some procs, could have inconsistency if procs do not both extract a shared cell */
  CHKERRQ(DMPlexCreateSubmeshGeneric_Interpolated(dm, cellLabel, value, PETSC_FALSE, PETSC_FALSE, 0, *subdm));
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *subdm));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  tmp  = mesh->subpointMap;
  mesh->subpointMap = subpointMap;
  CHKERRQ(PetscObjectReference((PetscObject) mesh->subpointMap));
  CHKERRQ(DMLabelDestroy(&tmp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubpointIS_Internal(DM dm, IS *subpointIS)
{
  DMLabel        spmap;
  PetscInt       depth, d;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetSubpointMap(dm, &spmap));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (spmap && depth >= 0) {
    DM_Plex  *mesh = (DM_Plex *) dm->data;
    PetscInt *points, *depths;
    PetscInt  pStart, pEnd, p, off;

    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCheck(!pStart,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Submeshes must start the point numbering at 0, not %d", pStart);
    CHKERRQ(PetscMalloc1(pEnd, &points));
    CHKERRQ(DMGetWorkArray(dm, depth+1, MPIU_INT, &depths));
    depths[0] = depth;
    depths[1] = 0;
    for (d = 2; d <= depth; ++d) {depths[d] = depth+1 - d;}
    for (d = 0, off = 0; d <= depth; ++d) {
      const PetscInt dep = depths[d];
      PetscInt       depStart, depEnd, n;

      CHKERRQ(DMPlexGetDepthStratum(dm, dep, &depStart, &depEnd));
      CHKERRQ(DMLabelGetStratumSize(spmap, dep, &n));
      if (((d < 2) && (depth > 1)) || (d == 1)) { /* Only check vertices and cells for now since the map is broken for others */
        PetscCheckFalse(n != depEnd-depStart,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d at depth %d should be %d", n, dep, depEnd-depStart);
      } else {
        if (!n) {
          if (d == 0) {
            /* Missing cells */
            for (p = 0; p < depEnd-depStart; ++p, ++off) points[off] = -1;
          } else {
            /* Missing faces */
            for (p = 0; p < depEnd-depStart; ++p, ++off) points[off] = PETSC_MAX_INT;
          }
        }
      }
      if (n) {
        IS              is;
        const PetscInt *opoints;

        CHKERRQ(DMLabelGetStratumIS(spmap, dep, &is));
        CHKERRQ(ISGetIndices(is, &opoints));
        for (p = 0; p < n; ++p, ++off) points[off] = opoints[p];
        CHKERRQ(ISRestoreIndices(is, &opoints));
        CHKERRQ(ISDestroy(&is));
      }
    }
    CHKERRQ(DMRestoreWorkArray(dm, depth+1, MPIU_INT, &depths));
    PetscCheckFalse(off != pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d should be %d", off, pEnd);
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, pEnd, points, PETSC_OWN_POINTER, subpointIS));
    CHKERRQ(PetscObjectStateGet((PetscObject) spmap, &mesh->subpointState));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointIS, 2);
  CHKERRQ(DMPlexGetSubpointMap(dm, &spmap));
  CHKERRQ(PetscObjectStateGet((PetscObject) spmap, &state));
  if (state != mesh->subpointState || !mesh->subpointIS) CHKERRQ(DMPlexCreateSubpointIS_Internal(dm, &mesh->subpointIS));
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

.seealso: DMGetEnclosurePoint()
@*/
PetscErrorCode DMGetEnclosureRelation(DM dmA, DM dmB, DMEnclosureType *rel)
{
  DM             plexA, plexB, sdm;
  DMLabel        spmap;
  PetscInt       pStartA, pEndA, pStartB, pEndB, NpA, NpB;

  PetscFunctionBegin;
  PetscValidPointer(rel, 3);
  *rel = DM_ENC_NONE;
  if (!dmA || !dmB) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  if (dmA == dmB) {*rel = DM_ENC_EQUALITY; PetscFunctionReturn(0);}
  CHKERRQ(DMConvert(dmA, DMPLEX, &plexA));
  CHKERRQ(DMConvert(dmB, DMPLEX, &plexB));
  CHKERRQ(DMPlexGetChart(plexA, &pStartA, &pEndA));
  CHKERRQ(DMPlexGetChart(plexB, &pStartB, &pEndB));
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
  CHKERRQ(DMPlexGetSubpointMap(sdm, &spmap));
  if (!spmap) goto end;
  /* TODO Check the space mapped to by subpointMap is same size as dm */
  if (NpA > NpB) {
    *rel = DM_ENC_SUPERMESH;
  } else {
    *rel = DM_ENC_SUBMESH;
  }
  end:
  CHKERRQ(DMDestroy(&plexA));
  CHKERRQ(DMDestroy(&plexB));
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

  PetscFunctionBegin;
  /* TODO Cache the IS, making it look like an index */
  switch (etype) {
    case DM_ENC_SUPERMESH:
    sdm  = dmB;
    CHKERRQ(DMPlexGetSubpointIS(sdm, &subpointIS));
    CHKERRQ(ISGetIndices(subpointIS, &subpoints));
    *pA  = subpoints[pB];
    CHKERRQ(ISRestoreIndices(subpointIS, &subpoints));
    break;
    case DM_ENC_SUBMESH:
    sdm  = dmA;
    CHKERRQ(DMPlexGetSubpointIS(sdm, &subpointIS));
    CHKERRQ(ISGetLocalSize(subpointIS, &numSubpoints));
    CHKERRQ(ISGetIndices(subpointIS, &subpoints));
    CHKERRQ(PetscFindInt(pB, numSubpoints, subpoints, pA));
    if (*pA < 0) {
      CHKERRQ(DMViewFromOptions(dmA, NULL, "-dm_enc_A_view"));
      CHKERRQ(DMViewFromOptions(dmB, NULL, "-dm_enc_B_view"));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d not found in submesh", pB);
    }
    CHKERRQ(ISRestoreIndices(subpointIS, &subpoints));
    break;
    case DM_ENC_EQUALITY:
    case DM_ENC_NONE:
    *pA = pB;break;
    case DM_ENC_UNKNOWN:
    {
      DMEnclosureType enc;

      CHKERRQ(DMGetEnclosureRelation(dmA, dmB, &enc));
      CHKERRQ(DMGetEnclosurePoint(dmA, dmB, enc, pB, pA));
    }
    break;
    default: SETERRQ(PetscObjectComm((PetscObject) dmA), PETSC_ERR_ARG_OUTOFRANGE, "Invalid enclosure type %d", (int) etype);
  }
  PetscFunctionReturn(0);
}

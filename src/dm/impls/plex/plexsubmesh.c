#include <petsc/private/dmpleximpl.h>    /*I      "petscdmplex.h"    I*/
#include <petsc/private/dmlabelimpl.h>   /*I      "petscdmlabel.h"   I*/
#include <petscsf.h>

static PetscErrorCode DMPlexCellIsHybrid_Internal(DM dm, PetscInt p, PetscBool *isHybrid)
{
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCellType(dm, p, &ct));
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
  PetscCall(DMPlexGetCellTypeLabel(dm, &ctLabel));
  switch (dim) {
    case 1: PetscCall(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_POINT_PRISM_TENSOR, cStart, cEnd));break;
    case 2: PetscCall(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_SEG_PRISM_TENSOR, cStart, cEnd));break;
    case 3:
      PetscCall(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_TRI_PRISM_TENSOR, cStart, cEnd));
      if (*cStart < 0) PetscCall(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_QUAD_PRISM_TENSOR, cStart, cEnd));
      break;
    default: PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkBoundaryFaces_Internal(DM dm, PetscInt val, PetscInt cellHeight, DMLabel label)
{
  PetscInt       fStart, fEnd, f;

  PetscFunctionBegin;
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, &fEnd));
  for (f = fStart; f < fEnd; ++f) {
    PetscInt supportSize;

    PetscCall(DMPlexGetSupportSize(dm, f, &supportSize));
    if (supportSize == 1) {
      if (val < 0) {
        PetscInt *closure = NULL;
        PetscInt  clSize, cl, cval;

        PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &clSize, &closure));
        for (cl = 0; cl < clSize*2; cl += 2) {
          PetscCall(DMLabelGetValue(label, closure[cl], &cval));
          if (cval < 0) continue;
          PetscCall(DMLabelSetValue(label, f, cval));
          break;
        }
        if (cl == clSize*2) PetscCall(DMLabelSetValue(label, f, 1));
        PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &clSize, &closure));
      } else {
        PetscCall(DMLabelSetValue(label, f, val));
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
  PetscCall(DMPlexIsInterpolated(dm, &flg));
  PetscCheckFalse(flg != DMPLEX_INTERPOLATED_FULL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not fully interpolated on this rank");
  PetscCall(DMPlexMarkBoundaryFaces_Internal(dm, val, 0, label));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLabelComplete_Internal(DM dm, DMLabel label, PetscBool completeCells)
{
  IS              valueIS;
  PetscSF         sfPoint;
  const PetscInt *values;
  PetscInt        numValues, v, cStart, cEnd, nroots;

  PetscFunctionBegin;
  PetscCall(DMLabelGetNumValues(label, &numValues));
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  PetscCall(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    PetscCall(DMLabelGetStratumSize(label, values[v], &numPoints));
    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    PetscCall(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      PetscInt  q = points[p];
      PetscInt *closure = NULL;
      PetscInt  closureSize, c;

      if (cStart <= q && q < cEnd && !completeCells) { /* skip cells */
        continue;
      }
      PetscCall(DMPlexGetTransitiveClosure(dm, q, PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        PetscCall(DMLabelSetValue(label, closure[c], values[v]));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, q, PETSC_TRUE, &closureSize, &closure));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
  if (nroots >= 0) {
    DMLabel         lblRoots, lblLeaves;
    IS              valueIS, pointIS;
    const PetscInt *values;
    PetscInt        numValues, v;

    /* Pull point contributions from remote leaves into local roots */
    PetscCall(DMLabelGather(label, sfPoint, &lblLeaves));
    PetscCall(DMLabelGetValueIS(lblLeaves, &valueIS));
    PetscCall(ISGetLocalSize(valueIS, &numValues));
    PetscCall(ISGetIndices(valueIS, &values));
    for (v = 0; v < numValues; ++v) {
      const PetscInt value = values[v];

      PetscCall(DMLabelGetStratumIS(lblLeaves, value, &pointIS));
      PetscCall(DMLabelInsertIS(label, pointIS, value));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
    PetscCall(DMLabelDestroy(&lblLeaves));
    /* Push point contributions from roots into remote leaves */
    PetscCall(DMLabelDistribute(label, sfPoint, &lblRoots));
    PetscCall(DMLabelGetValueIS(lblRoots, &valueIS));
    PetscCall(ISGetLocalSize(valueIS, &numValues));
    PetscCall(ISGetIndices(valueIS, &values));
    for (v = 0; v < numValues; ++v) {
      const PetscInt value = values[v];

      PetscCall(DMLabelGetStratumIS(lblRoots, value, &pointIS));
      PetscCall(DMLabelInsertIS(label, pointIS, value));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
    PetscCall(DMLabelDestroy(&lblRoots));
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
  PetscCall(DMPlexLabelComplete_Internal(dm, label, PETSC_TRUE));
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
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelGetNumValues(label, &numValues));
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    PetscCall(DMLabelGetStratumSize(label, values[v], &numPoints));
    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    PetscCall(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      PetscCall(DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closure));
      for (cl = closureSize-1; cl > 0; --cl) {
        const PetscInt cell = closure[cl*2];
        if ((cell >= cStart) && (cell < cEnd)) {PetscCall(DMLabelSetValue(label, cell, values[v])); break;}
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closure));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
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
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMLabelGetNumValues(label, &numValues));
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    PetscCall(DMLabelGetStratumSize(label, values[v], &numPoints));
    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    PetscCall(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      const PetscInt face = points[p];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, cl;

      if ((face < fStart) || (face >= fEnd)) continue;
      PetscCall(DMPlexGetTransitiveClosure(dm, face, PETSC_FALSE, &closureSize, &closure));
      for (cl = closureSize-1; cl > 0; --cl) {
        const PetscInt cell = closure[cl*2];
        if ((cell >= cStart) && (cell < cEnd)) {PetscCall(DMLabelSetValue(label, cell, values[v])); break;}
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, face, PETSC_FALSE, &closureSize, &closure));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
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
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelGetNumValues(label, &numValues));
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetIndices(valueIS, &values));
  for (v = 0; v < numValues; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    PetscCall(DMLabelGetStratumSize(label, values[v], &numPoints));
    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    PetscCall(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      PetscInt point = points[p];

      if (point >= cStart && point < cEnd) {
        PetscCall(DMLabelClearValue(label,point,values[v]));
      }
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
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
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (depth < 0) PetscFunctionReturn(0);
  /* Step 1: Expand chart */
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  pEnd = DMPlexShiftPoint_Internal(pEnd,depth,depthShift);
  PetscCall(DMPlexSetChart(dmNew, pStart, pEnd));
  PetscCall(DMCreateLabel(dmNew,"depth"));
  PetscCall(DMPlexGetDepthLabel(dmNew,&depthLabel));
  PetscCall(DMCreateLabel(dmNew, "celltype"));
  /* Step 2: Set cone and support sizes */
  for (d = 0; d <= depth; ++d) {
    PetscInt pStartNew, pEndNew;
    IS pIS;

    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    pStartNew = DMPlexShiftPoint_Internal(pStart, depth, depthShift);
    pEndNew = DMPlexShiftPoint_Internal(pEnd, depth, depthShift);
    PetscCall(ISCreateStride(PETSC_COMM_SELF, pEndNew - pStartNew, pStartNew, 1, &pIS));
    PetscCall(DMLabelSetStratumIS(depthLabel, d, pIS));
    PetscCall(ISDestroy(&pIS));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt       newp = DMPlexShiftPoint_Internal(p, depth, depthShift);
      PetscInt       size;
      DMPolytopeType ct;

      PetscCall(DMPlexGetConeSize(dm, p, &size));
      PetscCall(DMPlexSetConeSize(dmNew, newp, size));
      PetscCall(DMPlexGetSupportSize(dm, p, &size));
      PetscCall(DMPlexSetSupportSize(dmNew, newp, size));
      PetscCall(DMPlexGetCellType(dm, p, &ct));
      PetscCall(DMPlexSetCellType(dmNew, newp, ct));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftPoints_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt      *newpoints;
  PetscInt       depth = 0, maxConeSize, maxSupportSize, maxConeSizeNew, maxSupportSizeNew, pStart, pEnd, p;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (depth < 0) PetscFunctionReturn(0);
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  PetscCall(DMPlexGetMaxSizes(dmNew, &maxConeSizeNew, &maxSupportSizeNew));
  PetscCall(PetscMalloc1(PetscMax(PetscMax(maxConeSize, maxSupportSize), PetscMax(maxConeSizeNew, maxSupportSizeNew)),&newpoints));
  /* Step 5: Set cones and supports */
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *points = NULL, *orientations = NULL;
    PetscInt        size,sizeNew, i, newp = DMPlexShiftPoint_Internal(p, depth, depthShift);

    PetscCall(DMPlexGetConeSize(dm, p, &size));
    PetscCall(DMPlexGetCone(dm, p, &points));
    PetscCall(DMPlexGetConeOrientation(dm, p, &orientations));
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthShift);
    }
    PetscCall(DMPlexSetCone(dmNew, newp, newpoints));
    PetscCall(DMPlexSetConeOrientation(dmNew, newp, orientations));
    PetscCall(DMPlexGetSupportSize(dm, p, &size));
    PetscCall(DMPlexGetSupportSize(dmNew, newp, &sizeNew));
    PetscCall(DMPlexGetSupport(dm, p, &points));
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthShift);
    }
    for (i = size; i < sizeNew; ++i) newpoints[i] = 0;
    PetscCall(DMPlexSetSupport(dmNew, newp, newpoints));
  }
  PetscCall(PetscFree(newpoints));
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
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(DMSetCoordinateDim(dmNew, dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  /* Step 8: Convert coordinates */
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dmNew, 0, &vStartNew, &vEndNew));
  PetscCall(DMPlexGetHeightStratum(dmNew, 0, &cStartNew, &cEndNew));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection));
  PetscCall(PetscSectionSetNumFields(newCoordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(newCoordSection, 0, dim));
  PetscCall(PetscSectionGetChart(coordSection, &sStart, &sEnd));
  hasCells = sStart == cStart ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscSectionSetChart(newCoordSection, hasCells ? cStartNew : vStartNew, vEndNew));
  if (hasCells) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt cNew = DMPlexShiftPoint_Internal(c, depth, depthShift), dof;

      PetscCall(PetscSectionGetDof(coordSection, c, &dof));
      PetscCall(PetscSectionSetDof(newCoordSection, cNew, dof));
      PetscCall(PetscSectionSetFieldDof(newCoordSection, cNew, 0, dof));
    }
  }
  for (v = vStartNew; v < vEndNew; ++v) {
    PetscCall(PetscSectionSetDof(newCoordSection, v, dim));
    PetscCall(PetscSectionSetFieldDof(newCoordSection, v, 0, dim));
  }
  PetscCall(PetscSectionSetUp(newCoordSection));
  PetscCall(DMSetCoordinateSection(dmNew, PETSC_DETERMINE, newCoordSection));
  PetscCall(PetscSectionGetStorageSize(newCoordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &newCoordinates));
  PetscCall(PetscObjectSetName((PetscObject) newCoordinates, "coordinates"));
  PetscCall(VecSetSizes(newCoordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(newCoordinates, dim));
  PetscCall(VecSetType(newCoordinates,VECSTANDARD));
  PetscCall(DMSetCoordinatesLocal(dmNew, newCoordinates));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetArray(coordinates, &coords));
  PetscCall(VecGetArray(newCoordinates, &newCoords));
  if (hasCells) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt cNew = DMPlexShiftPoint_Internal(c, depth, depthShift), dof, off, noff, d;

      PetscCall(PetscSectionGetDof(coordSection, c, &dof));
      PetscCall(PetscSectionGetOffset(coordSection, c, &off));
      PetscCall(PetscSectionGetOffset(newCoordSection, cNew, &noff));
      for (d = 0; d < dof; ++d) newCoords[noff+d] = coords[off+d];
    }
  }
  for (v = vStart; v < vEnd; ++v) {
    PetscInt dof, off, noff, d;

    PetscCall(PetscSectionGetDof(coordSection, v, &dof));
    PetscCall(PetscSectionGetOffset(coordSection, v, &off));
    PetscCall(PetscSectionGetOffset(newCoordSection, DMPlexShiftPoint_Internal(v, depth, depthShift), &noff));
    for (d = 0; d < dof; ++d) newCoords[noff+d] = coords[off+d];
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(VecRestoreArray(newCoordinates, &newCoords));
  PetscCall(VecDestroy(&newCoordinates));
  PetscCall(PetscSectionDestroy(&newCoordSection));
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
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSFGetGraph(sf, &numRoots, &numLeaves, &localPoints, &remotePoints));
  totShift = DMPlexShiftPoint_Internal(pEnd, depth, depthShift) - pEnd;
  if (numRoots >= 0) {
    PetscCall(PetscMalloc2(numRoots, &newLocation, pEnd-pStart, &newRemoteLocation));
    for (l = 0; l < numRoots; ++l) newLocation[l] = DMPlexShiftPoint_Internal(l, depth, depthShift);
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, newLocation, newRemoteLocation, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, newLocation, newRemoteLocation, MPI_REPLACE));
    PetscCall(PetscMalloc1(numLeaves, &glocalPoints));
    PetscCall(PetscMalloc1(numLeaves, &gremotePoints));
    for (l = 0; l < numLeaves; ++l) {
      glocalPoints[l]        = DMPlexShiftPoint_Internal(localPoints[l], depth, depthShift);
      gremotePoints[l].rank  = remotePoints[l].rank;
      gremotePoints[l].index = newRemoteLocation[localPoints[l]];
    }
    PetscCall(PetscFree2(newLocation, newRemoteLocation));
    PetscCall(PetscSFSetGraph(sfNew, numRoots + totShift, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftSF_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscSF        sfPoint, sfPointNew;
  PetscBool      useNatural;

  PetscFunctionBegin;
  /* Step 9: Convert pointSF */
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(DMGetPointSF(dmNew, &sfPointNew));
  PetscCall(DMPlexShiftSF_Single(dm, depthShift, sfPoint, sfPointNew));
  /* Step 9b: Convert naturalSF */
  PetscCall(DMGetUseNatural(dm, &useNatural));
  if (useNatural) {
    PetscSF sfNat, sfNatNew;

    PetscCall(DMSetUseNatural(dmNew, useNatural));
    PetscCall(DMGetNaturalSF(dm, &sfNat));
    PetscCall(DMGetNaturalSF(dmNew, &sfNatNew));
    PetscCall(DMPlexShiftSF_Single(dm, depthShift, sfNat, sfNatNew));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexShiftLabels_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt           depth = 0, numLabels, l;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  /* Step 10: Convert labels */
  PetscCall(DMGetNumLabels(dm, &numLabels));
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, newlabel;
    const char     *lname;
    PetscBool       isDepth, isDim;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, val;

    PetscCall(DMGetLabelName(dm, l, &lname));
    PetscCall(PetscStrcmp(lname, "depth", &isDepth));
    if (isDepth) continue;
    PetscCall(PetscStrcmp(lname, "dim", &isDim));
    if (isDim) continue;
    PetscCall(DMCreateLabel(dmNew, lname));
    PetscCall(DMGetLabel(dm, lname, &label));
    PetscCall(DMGetLabel(dmNew, lname, &newlabel));
    PetscCall(DMLabelGetDefaultValue(label,&val));
    PetscCall(DMLabelSetDefaultValue(newlabel,val));
    PetscCall(DMLabelGetValueIS(label, &valueIS));
    PetscCall(ISGetLocalSize(valueIS, &numValues));
    PetscCall(ISGetIndices(valueIS, &values));
    for (val = 0; val < numValues; ++val) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        numPoints, p;

      PetscCall(DMLabelGetStratumIS(label, values[val], &pointIS));
      PetscCall(ISGetLocalSize(pointIS, &numPoints));
      PetscCall(ISGetIndices(pointIS, &points));
      for (p = 0; p < numPoints; ++p) {
        const PetscInt newpoint = DMPlexShiftPoint_Internal(points[p], depth, depthShift);

        PetscCall(DMLabelSetValue(newlabel, newpoint, values[val]));
      }
      PetscCall(ISRestoreIndices(pointIS, &points));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
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
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(DMPlexGetVTKCellHeight(dmNew, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(PetscSFGetGraph(sfPoint, NULL, &numLeaves, &leafLocal, &leafRemote));
  PetscCall(DMCreateLabel(dmNew, "vtk"));
  PetscCall(DMGetLabel(dmNew, "vtk", &vtkLabel));
  if (createGhostLabel) {
    PetscCall(DMCreateLabel(dmNew, "ghost"));
    PetscCall(DMGetLabel(dmNew, "ghost", &ghostLabel));
  }
  for (l = 0, c = cStart; l < numLeaves && c < cEnd; ++l, ++c) {
    for (; c < leafLocal[l] && c < cEnd; ++c) {
      PetscCall(DMLabelSetValue(vtkLabel, c, 1));
    }
    if (leafLocal[l] >= cEnd) break;
    if (leafRemote[l].rank == rank) {
      PetscCall(DMLabelSetValue(vtkLabel, c, 1));
    } else if (ghostLabel) {
      PetscCall(DMLabelSetValue(ghostLabel, c, 2));
    }
  }
  for (; c < cEnd; ++c) {
    PetscCall(DMLabelSetValue(vtkLabel, c, 1));
  }
  if (ghostLabel) {
    PetscCall(DMPlexGetHeightStratum(dmNew, 1, &fStart, &fEnd));
    for (f = fStart; f < fEnd; ++f) {
      PetscInt numCells;

      PetscCall(DMPlexGetSupportSize(dmNew, f, &numCells));
      if (numCells < 2) {
        PetscCall(DMLabelSetValue(ghostLabel, f, 1));
      } else {
        const PetscInt *cells = NULL;
        PetscInt        vA, vB;

        PetscCall(DMPlexGetSupport(dmNew, f, &cells));
        PetscCall(DMLabelGetValue(vtkLabel, cells[0], &vA));
        PetscCall(DMLabelGetValue(vtkLabel, cells[1], &vB));
        if (vA != 1 && vB != 1) PetscCall(DMLabelSetValue(ghostLabel, f, 1));
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
  PetscCall(DMPlexGetReferenceTree(dm,&refTree));
  PetscCall(DMPlexSetReferenceTree(dmNew,refTree));
  PetscCall(DMPlexGetTree(dm,&pSec,&parents,&childIDs,NULL,NULL));
  if (pSec) {
    PetscInt p, pStart, pEnd, *parentsShifted, pStartShifted, pEndShifted, depth;
    PetscInt *childIDsShifted;
    PetscSection pSecShifted;

    PetscCall(PetscSectionGetChart(pSec,&pStart,&pEnd));
    PetscCall(DMPlexGetDepth(dm,&depth));
    pStartShifted = DMPlexShiftPoint_Internal(pStart,depth,depthShift);
    pEndShifted   = DMPlexShiftPoint_Internal(pEnd,depth,depthShift);
    PetscCall(PetscMalloc2(pEndShifted - pStartShifted,&parentsShifted,pEndShifted-pStartShifted,&childIDsShifted));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dmNew),&pSecShifted));
    PetscCall(PetscSectionSetChart(pSecShifted,pStartShifted,pEndShifted));
    for (p = pStartShifted; p < pEndShifted; p++) {
      /* start off assuming no children */
      PetscCall(PetscSectionSetDof(pSecShifted,p,0));
    }
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof;
      PetscInt pNew = DMPlexShiftPoint_Internal(p,depth,depthShift);

      PetscCall(PetscSectionGetDof(pSec,p,&dof));
      PetscCall(PetscSectionSetDof(pSecShifted,pNew,dof));
    }
    PetscCall(PetscSectionSetUp(pSecShifted));
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof;
      PetscInt pNew = DMPlexShiftPoint_Internal(p,depth,depthShift);

      PetscCall(PetscSectionGetDof(pSec,p,&dof));
      if (dof) {
        PetscInt off, offNew;

        PetscCall(PetscSectionGetOffset(pSec,p,&off));
        PetscCall(PetscSectionGetOffset(pSecShifted,pNew,&offNew));
        parentsShifted[offNew] = DMPlexShiftPoint_Internal(parents[off],depth,depthShift);
        childIDsShifted[offNew] = childIDs[off];
      }
    }
    PetscCall(DMPlexSetTree(dmNew,pSecShifted,parentsShifted,childIDsShifted));
    PetscCall(PetscFree2(parentsShifted,childIDsShifted));
    PetscCall(PetscSectionDestroy(&pSecShifted));
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
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
  nleaves = PetscMax(0, nleaves);
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  /* Count ghost cells */
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &numFS));
  PetscCall(ISGetIndices(valueIS, &values));
  Ng   = 0;
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f, numBdFaces = 0;

    PetscCall(DMLabelGetStratumIS(label, values[fs], &faceIS));
    PetscCall(ISGetLocalSize(faceIS, &numFaces));
    PetscCall(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      PetscInt numChildren;

      PetscCall(PetscFindInt(faces[f], nleaves, leaves, &loc));
      PetscCall(DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL));
      /* non-local and ancestors points don't get to register ghosts */
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] >= fStart) && (faces[f] < fEnd)) ++numBdFaces;
    }
    Ng += numBdFaces;
    PetscCall(ISRestoreIndices(faceIS, &faces));
    PetscCall(ISDestroy(&faceIS));
  }
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(PetscMalloc1(2*(depth+1), &depthShift));
  for (d = 0; d <= depth; d++) {
    PetscInt dEnd;

    PetscCall(DMPlexGetDepthStratum(dm,d,NULL,&dEnd));
    depthShift[2*d]   = dEnd;
    depthShift[2*d+1] = 0;
  }
  if (depth >= 0) depthShift[2*depth+1] = Ng;
  PetscCall(DMPlexShiftPointSetUp_Internal(depth,depthShift));
  PetscCall(DMPlexShiftSizes_Internal(dm, depthShift, gdm));
  /* Step 3: Set cone/support sizes for new points */
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &cEnd));
  for (c = cEnd; c < cEnd + Ng; ++c) {
    PetscCall(DMPlexSetConeSize(gdm, c, 1));
  }
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    PetscCall(DMLabelGetStratumIS(label, values[fs], &faceIS));
    PetscCall(ISGetLocalSize(faceIS, &numFaces));
    PetscCall(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      PetscInt size, numChildren;

      PetscCall(PetscFindInt(faces[f], nleaves, leaves, &loc));
      PetscCall(DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL));
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] < fStart) || (faces[f] >= fEnd)) continue;
      PetscCall(DMPlexGetSupportSize(dm, faces[f], &size));
      PetscCheckFalse(size != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM has boundary face %d with %d support cells", faces[f], size);
      PetscCall(DMPlexSetSupportSize(gdm, faces[f] + Ng, 2));
    }
    PetscCall(ISRestoreIndices(faceIS, &faces));
    PetscCall(ISDestroy(&faceIS));
  }
  /* Step 4: Setup ghosted DM */
  PetscCall(DMSetUp(gdm));
  PetscCall(DMPlexShiftPoints_Internal(dm, depthShift, gdm));
  /* Step 6: Set cones and supports for new points */
  ghostCell = cEnd;
  for (fs = 0; fs < numFS; ++fs) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    PetscCall(DMLabelGetStratumIS(label, values[fs], &faceIS));
    PetscCall(ISGetLocalSize(faceIS, &numFaces));
    PetscCall(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      PetscInt newFace = faces[f] + Ng, numChildren;

      PetscCall(PetscFindInt(faces[f], nleaves, leaves, &loc));
      PetscCall(DMPlexGetTreeChildren(dm,faces[f],&numChildren,NULL));
      if (loc >= 0 || numChildren) continue;
      if ((faces[f] < fStart) || (faces[f] >= fEnd)) continue;
      PetscCall(DMPlexSetCone(gdm, ghostCell, &newFace));
      PetscCall(DMPlexInsertSupport(gdm, newFace, 1, ghostCell));
      ++ghostCell;
    }
    PetscCall(ISRestoreIndices(faceIS, &faces));
    PetscCall(ISDestroy(&faceIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscCall(DMPlexShiftCoordinates_Internal(dm, depthShift, gdm));
  PetscCall(DMPlexShiftSF_Internal(dm, depthShift, gdm));
  PetscCall(DMPlexShiftLabels_Internal(dm, depthShift, gdm));
  PetscCall(DMPlexCreateVTKLabel_Internal(dm, PETSC_TRUE, gdm));
  PetscCall(DMPlexShiftTree_Internal(dm, depthShift, gdm));
  PetscCall(PetscFree(depthShift));
  for (c = cEnd; c < cEnd + Ng; ++c) {
    PetscCall(DMPlexSetCellType(gdm, c, DM_POLYTOPE_FV_GHOST));
  }
  /* Step 7: Periodicity */
  PetscCall(DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd));
  PetscCall(DMSetPeriodicity(gdm, isper, maxCell,  L,  bd));
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
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), &gdm));
  PetscCall(DMSetType(gdm, DMPLEX));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(gdm, dim));
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(gdm, useCone,  useClosure));
  PetscCall(DMGetLabel(dm, name, &label));
  if (!label) {
    /* Get label for boundary faces */
    PetscCall(DMCreateLabel(dm, name));
    PetscCall(DMGetLabel(dm, name, &label));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
  }
  PetscCall(DMPlexConstructGhostCells_Internal(dm, label, &Ng, gdm));
  PetscCall(DMCopyDisc(dm, gdm));
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, gdm));
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
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  /* We do not want this label automatically computed, instead we compute it here */
  PetscCall(DMCreateLabel(sdm, "celltype"));
  /* Count split points and add cohesive cells */
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  PetscCall(PetscMalloc5(depth+1,&depthMax,depth+1,&depthEnd,2*(depth+1),&depthShift,depth+1,&pMaxNew,depth+1,&numHybridPointsOld));
  PetscCall(PetscMalloc7(depth+1,&splitIS,depth+1,&unsplitIS,depth+1,&numSplitPoints,depth+1,&numUnsplitPoints,depth+1,&numHybridPoints,depth+1,&splitPoints,depth+1,&unsplitPoints));
  for (d = 0; d <= depth; ++d) {
    PetscCall(DMPlexGetDepthStratum(dm, d, NULL, &pMaxNew[d]));
    PetscCall(DMPlexGetTensorPrismBounds_Internal(dm, d, &depthMax[d], NULL));
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
    PetscCall(DMLabelGetValueIS(label, &valueIS));
    PetscCall(ISGetLocalSize(valueIS, &numSP));
    PetscCall(ISGetIndices(valueIS, &values));
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    PetscCall(DMLabelGetStratumIS(label, dep, &splitIS[dep]));
    if (splitIS[dep]) {
      PetscCall(ISGetLocalSize(splitIS[dep], &numSplitPoints[dep]));
      PetscCall(ISGetIndices(splitIS[dep], &splitPoints[dep]));
    }
    PetscCall(DMLabelGetStratumIS(label, shift2+dep, &unsplitIS[dep]));
    if (unsplitIS[dep]) {
      PetscCall(ISGetLocalSize(unsplitIS[dep], &numUnsplitPoints[dep]));
      PetscCall(ISGetIndices(unsplitIS[dep], &unsplitPoints[dep]));
    }
  }
  /* Calculate number of hybrid points */
  for (d = 1; d <= depth; ++d) numHybridPoints[d]     = numSplitPoints[d-1] + numUnsplitPoints[d-1]; /* There is a hybrid cell/face/edge for every split face/edge/vertex   */
  for (d = 0; d <= depth; ++d) depthShift[2*d+1]      = numSplitPoints[d] + numHybridPoints[d];
  PetscCall(DMPlexShiftPointSetUp_Internal(depth,depthShift));
  /* the end of the points in this stratum that come before the new points:
   * shifting pMaxNew[d] gets the new start of the next stratum, then count back the old hybrid points and the newly
   * added points */
  for (d = 0; d <= depth; ++d) pMaxNew[d]             = DMPlexShiftPoint_Internal(pMaxNew[d],depth,depthShift) - (numHybridPointsOld[d] + numSplitPoints[d] + numHybridPoints[d]);
  PetscCall(DMPlexShiftSizes_Internal(dm, depthShift, sdm));
  /* Step 3: Set cone/support sizes for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt  splitp = p    + pMaxNew[dep];
      const PetscInt *support;
      DMPolytopeType  ct;
      PetscInt        coneSize, supportSize, qf, qn, qp, e;

      PetscCall(DMPlexGetConeSize(dm, oldp, &coneSize));
      PetscCall(DMPlexSetConeSize(sdm, splitp, coneSize));
      PetscCall(DMPlexGetSupportSize(dm, oldp, &supportSize));
      PetscCall(DMPlexSetSupportSize(sdm, splitp, supportSize));
      PetscCall(DMPlexGetCellType(dm, oldp, &ct));
      PetscCall(DMPlexSetCellType(sdm, splitp, ct));
      if (dep == depth-1) {
        const PetscInt hybcell = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Add cohesive cells, they are prisms */
        PetscCall(DMPlexSetConeSize(sdm, hybcell, 2 + coneSize));
        switch (coneSize) {
          case 2: PetscCall(DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_SEG_PRISM_TENSOR));break;
          case 3: PetscCall(DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_TRI_PRISM_TENSOR));break;
          case 4: PetscCall(DMPlexSetCellType(sdm, hybcell, DM_POLYTOPE_QUAD_PRISM_TENSOR));break;
        }
      } else if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        PetscCall(DMPlexGetSupport(dm, oldp, &support));
        for (e = 0, qn = 0, qp = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == 1) ++qf;
          if ((val == 1) || (val ==  (shift + 1))) ++qn;
          if ((val == 1) || (val == -(shift + 1))) ++qp;
        }
        /* Split old vertex: Edges into original vertex and new cohesive edge */
        PetscCall(DMPlexSetSupportSize(sdm, newp, qn+1));
        /* Split new vertex: Edges into split vertex and new cohesive edge */
        PetscCall(DMPlexSetSupportSize(sdm, splitp, qp+1));
        /* Add hybrid edge */
        PetscCall(DMPlexSetConeSize(sdm, hybedge, 2));
        PetscCall(DMPlexSetSupportSize(sdm, hybedge, qf));
        PetscCall(DMPlexSetCellType(sdm, hybedge, DM_POLYTOPE_POINT_PRISM_TENSOR));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        PetscCall(DMPlexGetSupport(dm, oldp, &support));
        for (e = 0, qn = 0, qp = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) ++qf;
          if ((val == dim-1) || (val ==  (shift + dim-1))) ++qn;
          if ((val == dim-1) || (val == -(shift + dim-1))) ++qp;
        }
        /* Split old edge: Faces into original edge and cohesive face (positive side?) */
        PetscCall(DMPlexSetSupportSize(sdm, newp, qn+1));
        /* Split new edge: Faces into split edge and cohesive face (negative side?) */
        PetscCall(DMPlexSetSupportSize(sdm, splitp, qp+1));
        /* Add hybrid face */
        PetscCall(DMPlexSetConeSize(sdm, hybface, 4));
        PetscCall(DMPlexSetSupportSize(sdm, hybface, qf));
        PetscCall(DMPlexSetCellType(sdm, hybface, DM_POLYTOPE_SEG_PRISM_TENSOR));
      }
    }
  }
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numUnsplitPoints[dep]; ++p) {
      const PetscInt  oldp   = unsplitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt *support;
      PetscInt        coneSize, supportSize, qf, e, s;

      PetscCall(DMPlexGetConeSize(dm, oldp, &coneSize));
      PetscCall(DMPlexGetSupportSize(dm, oldp, &supportSize));
      PetscCall(DMPlexGetSupport(dm, oldp, &support));
      if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit vertex: Edges into original vertex, split edges, and new cohesive edge twice */
        for (s = 0, qf = 0; s < supportSize; ++s, ++qf) {
          PetscCall(PetscFindInt(support[s], numSplitPoints[dep+1], splitPoints[dep+1], &e));
          if (e >= 0) ++qf;
        }
        PetscCall(DMPlexSetSupportSize(sdm, newp, qf+2));
        /* Add hybrid edge */
        PetscCall(DMPlexSetConeSize(sdm, hybedge, 2));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          /* Split and unsplit edges produce hybrid faces */
          if (val == 1) ++qf;
          if (val == (shift2 + 1)) ++qf;
        }
        PetscCall(DMPlexSetSupportSize(sdm, hybedge, qf));
        PetscCall(DMPlexSetCellType(sdm, hybedge, DM_POLYTOPE_POINT_PRISM_TENSOR));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];
        PetscInt       val;

        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) qf += 2;
          else              ++qf;
        }
        /* Unsplit edge: Faces into original edge, split face, and cohesive face twice */
        PetscCall(DMPlexSetSupportSize(sdm, newp, qf+2));
        /* Add hybrid face */
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) ++qf;
        }
        PetscCall(DMPlexSetConeSize(sdm, hybface, 4));
        PetscCall(DMPlexSetSupportSize(sdm, hybface, qf));
        PetscCall(DMPlexSetCellType(sdm, hybface, DM_POLYTOPE_SEG_PRISM_TENSOR));
      }
    }
  }
  /* Step 4: Setup split DM */
  PetscCall(DMSetUp(sdm));
  PetscCall(DMPlexShiftPoints_Internal(dm, depthShift, sdm));
  PetscCall(DMPlexGetMaxSizes(sdm, &maxConeSizeNew, &maxSupportSizeNew));
  PetscCall(PetscMalloc3(PetscMax(maxConeSize, maxConeSizeNew)*3,&coneNew,PetscMax(maxConeSize, maxConeSizeNew)*3,&coneONew,PetscMax(maxSupportSize, maxSupportSizeNew),&supportNew));
  /* Step 6: Set cones and supports for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt  splitp = p    + pMaxNew[dep];
      const PetscInt *cone, *support, *ornt;
      DMPolytopeType  ct;
      PetscInt        coneSize, supportSize, q, qf, qn, qp, v, e, s;

      PetscCall(DMPlexGetCellType(dm, oldp, &ct));
      PetscCall(DMPlexGetConeSize(dm, oldp, &coneSize));
      PetscCall(DMPlexGetCone(dm, oldp, &cone));
      PetscCall(DMPlexGetConeOrientation(dm, oldp, &ornt));
      PetscCall(DMPlexGetSupportSize(dm, oldp, &supportSize));
      PetscCall(DMPlexGetSupport(dm, oldp, &support));
      if (dep == depth-1) {
        PetscBool       hasUnsplit = PETSC_FALSE;
        const PetscInt  hybcell    = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        const PetscInt *supportF;

        /* Split face:       copy in old face to new face to start */
        PetscCall(DMPlexGetSupport(sdm, newp,  &supportF));
        PetscCall(DMPlexSetSupport(sdm, splitp, supportF));
        /* Split old face:   old vertices/edges in cone so no change */
        /* Split new face:   new vertices/edges in cone */
        for (q = 0; q < coneSize; ++q) {
          PetscCall(PetscFindInt(cone[q], numSplitPoints[dep-1], splitPoints[dep-1], &v));
          if (v < 0) {
            PetscCall(PetscFindInt(cone[q], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
            PetscCheckFalse(v < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[q], dep-1);
            coneNew[2+q] = DMPlexShiftPoint_Internal(cone[q], depth, depthShift) /*cone[q] + depthOffset[dep-1]*/;
            hasUnsplit   = PETSC_TRUE;
          } else {
            coneNew[2+q] = v + pMaxNew[dep-1];
            if (dep > 1) {
              const PetscInt *econe;
              PetscInt        econeSize, r, vs, vu;

              PetscCall(DMPlexGetConeSize(dm, cone[q], &econeSize));
              PetscCall(DMPlexGetCone(dm, cone[q], &econe));
              for (r = 0; r < econeSize; ++r) {
                PetscCall(PetscFindInt(econe[r], numSplitPoints[dep-2],   splitPoints[dep-2],   &vs));
                PetscCall(PetscFindInt(econe[r], numUnsplitPoints[dep-2], unsplitPoints[dep-2], &vu));
                if (vs >= 0) continue;
                PetscCheckFalse(vu < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", econe[r], dep-2);
                hasUnsplit   = PETSC_TRUE;
              }
            }
          }
        }
        PetscCall(DMPlexSetCone(sdm, splitp, &coneNew[2]));
        PetscCall(DMPlexSetConeOrientation(sdm, splitp, ornt));
        /* Face support */
        for (s = 0; s < supportSize; ++s) {
          PetscInt val;

          PetscCall(DMLabelGetValue(label, support[s], &val));
          if (val < 0) {
            /* Split old face:   Replace negative side cell with cohesive cell */
             PetscCall(DMPlexInsertSupport(sdm, newp, s, hybcell));
          } else {
            /* Split new face:   Replace positive side cell with cohesive cell */
            PetscCall(DMPlexInsertSupport(sdm, splitp, s, hybcell));
            /* Get orientation for cohesive face */
            {
              const PetscInt *ncone, *nconeO;
              PetscInt        nconeSize, nc;

              PetscCall(DMPlexGetConeSize(dm, support[s], &nconeSize));
              PetscCall(DMPlexGetCone(dm, support[s], &ncone));
              PetscCall(DMPlexGetConeOrientation(dm, support[s], &nconeO));
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

          PetscCall(PetscFindInt(cone[qa], numSplitPoints[dep-1], splitPoints[dep-1], &v));
          if (v < 0) {
            PetscCall(PetscFindInt(cone[qa], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
            coneNew[2+q]  = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
          } else {
            coneNew[2+q]  = v + pMaxNew[dep] + numSplitPoints[dep];
          }
          coneONew[2+q] = DMPolytopeTypeComposeOrientation(ft, qo, ornt[qa]);
        }
        PetscCall(DMPlexSetCone(sdm, hybcell, coneNew));
        PetscCall(DMPlexSetConeOrientation(sdm, hybcell, coneONew));
        /* Label the hybrid cells on the boundary of the split */
        if (hasUnsplit) PetscCall(DMLabelSetValue(label, -hybcell, dim));
      } else if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Split old vertex: Edges in old split faces and new cohesive edge */
        for (e = 0, qn = 0; e < supportSize; ++e) {
          PetscInt val;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if ((val == 1) || (val == (shift + 1))) {
            supportNew[qn++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qn] = hybedge;
        PetscCall(DMPlexSetSupport(sdm, newp, supportNew));
        /* Split new vertex: Edges in new split faces and new cohesive edge */
        for (e = 0, qp = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == 1) {
            PetscCall(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qp++] = edge + pMaxNew[dep+1];
          } else if (val == -(shift + 1)) {
            supportNew[qp++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qp] = hybedge;
        PetscCall(DMPlexSetSupport(sdm, splitp, supportNew));
        /* Hybrid edge:    Old and new split vertex */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        PetscCall(DMPlexSetCone(sdm, hybedge, coneNew));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == 1) {
            PetscCall(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        PetscCall(DMPlexSetSupport(sdm, hybedge, supportNew));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1];

        /* Split old edge:   old vertices in cone so no change */
        /* Split new edge:   new vertices in cone */
        for (q = 0; q < coneSize; ++q) {
          PetscCall(PetscFindInt(cone[q], numSplitPoints[dep-1], splitPoints[dep-1], &v));
          if (v < 0) {
            PetscCall(PetscFindInt(cone[q], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
            PetscCheckFalse(v < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[q], dep-1);
            coneNew[q] = DMPlexShiftPoint_Internal(cone[q], depth, depthShift) /*cone[q] + depthOffset[dep-1]*/;
          } else {
            coneNew[q] = v + pMaxNew[dep-1];
          }
        }
        PetscCall(DMPlexSetCone(sdm, splitp, coneNew));
        /* Split old edge: Faces in positive side cells and old split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          } else if (val == (shift + dim-1)) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[q++] = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        PetscCall(DMPlexSetSupport(sdm, newp, supportNew));
        /* Split new edge: Faces in negative side cells and new split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val, face;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) {
            PetscCall(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            PetscCheckFalse(face < 0,comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[q++] = face + pMaxNew[dep+1];
          } else if (val == -(shift + dim-1)) {
            supportNew[q++] = DMPlexShiftPoint_Internal(support[e], depth, depthShift) /*support[e] + depthOffset[dep+1]*/;
          }
        }
        supportNew[q++] = p + pMaxNew[dep+1] + numSplitPoints[dep+1];
        PetscCall(DMPlexSetSupport(sdm, splitp, supportNew));
        /* Hybrid face */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        for (v = 0; v < coneSize; ++v) {
          PetscInt vertex;
          PetscCall(PetscFindInt(cone[v], numSplitPoints[dep-1], splitPoints[dep-1], &vertex));
          if (vertex < 0) {
            PetscCall(PetscFindInt(cone[v], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &vertex));
            PetscCheckFalse(vertex < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate point %d in split or unsplit points of depth %d", cone[v], dep-1);
            coneNew[2+v] = vertex + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
          } else {
            coneNew[2+v] = vertex + pMaxNew[dep] + numSplitPoints[dep];
          }
        }
        PetscCall(DMPlexSetCone(sdm, hybface, coneNew));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, face;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == dim-1) {
            PetscCall(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            PetscCheckFalse(face < 0,comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[qf++] = face + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        PetscCall(DMPlexSetSupport(sdm, hybface, supportNew));
      }
    }
  }
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numUnsplitPoints[dep]; ++p) {
      const PetscInt  oldp   = unsplitPoints[dep][p];
      const PetscInt  newp   = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*oldp + depthOffset[dep]*/;
      const PetscInt *cone, *support;
      PetscInt        coneSize, supportSize, supportSizeNew, q, qf, e, f, s;

      PetscCall(DMPlexGetConeSize(dm, oldp, &coneSize));
      PetscCall(DMPlexGetCone(dm, oldp, &cone));
      PetscCall(DMPlexGetSupportSize(dm, oldp, &supportSize));
      PetscCall(DMPlexGetSupport(dm, oldp, &support));
      if (dep == 0) {
        const PetscInt hybedge = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit vertex */
        PetscCall(DMPlexGetSupportSize(sdm, newp, &supportSizeNew));
        for (s = 0, q = 0; s < supportSize; ++s) {
          supportNew[q++] = DMPlexShiftPoint_Internal(support[s], depth, depthShift) /*support[s] + depthOffset[dep+1]*/;
          PetscCall(PetscFindInt(support[s], numSplitPoints[dep+1], splitPoints[dep+1], &e));
          if (e >= 0) {
            supportNew[q++] = e + pMaxNew[dep+1];
          }
        }
        supportNew[q++] = hybedge;
        supportNew[q++] = hybedge;
        PetscCheckFalse(q != supportSizeNew,comm, PETSC_ERR_ARG_WRONG, "Support size %d != %d for vertex %d", q, supportSizeNew, newp);
        PetscCall(DMPlexSetSupport(sdm, newp, supportNew));
        /* Hybrid edge */
        coneNew[0] = newp;
        coneNew[1] = newp;
        PetscCall(DMPlexSetCone(sdm, hybedge, coneNew));
        for (e = 0, qf = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          PetscCall(DMLabelGetValue(label, support[e], &val));
          if (val == 1) {
            PetscCall(PetscFindInt(support[e], numSplitPoints[dep+1], splitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2];
          } else if  (val ==  (shift2 + 1)) {
            PetscCall(PetscFindInt(support[e], numUnsplitPoints[dep+1], unsplitPoints[dep+1], &edge));
            PetscCheckFalse(edge < 0,comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a unsplit edge", support[e]);
            supportNew[qf++] = edge + pMaxNew[dep+2] + numSplitPoints[dep+2] + numSplitPoints[dep+1];
          }
        }
        PetscCall(DMPlexSetSupport(sdm, hybedge, supportNew));
      } else if (dep == dim-2) {
        const PetscInt hybface = p + pMaxNew[dep+1] + numSplitPoints[dep+1] + numSplitPoints[dep];

        /* Unsplit edge: Faces into original edge, split face, and hybrid face twice */
        for (f = 0, qf = 0; f < supportSize; ++f) {
          PetscInt val, face;

          PetscCall(DMLabelGetValue(label, support[f], &val));
          if (val == dim-1) {
            PetscCall(PetscFindInt(support[f], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            PetscCheckFalse(face < 0,comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[f]);
            supportNew[qf++] = DMPlexShiftPoint_Internal(support[f], depth, depthShift) /*support[f] + depthOffset[dep+1]*/;
            supportNew[qf++] = face + pMaxNew[dep+1];
          } else {
            supportNew[qf++] = DMPlexShiftPoint_Internal(support[f], depth, depthShift) /*support[f] + depthOffset[dep+1]*/;
          }
        }
        supportNew[qf++] = hybface;
        supportNew[qf++] = hybface;
        PetscCall(DMPlexGetSupportSize(sdm, newp, &supportSizeNew));
        PetscCheckFalse(qf != supportSizeNew,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support size for unsplit edge %d is %d != %d", newp, qf, supportSizeNew);
        PetscCall(DMPlexSetSupport(sdm, newp, supportNew));
        /* Add hybrid face */
        coneNew[0] = newp;
        coneNew[1] = newp;
        PetscCall(PetscFindInt(cone[0], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
        PetscCheckFalse(v < 0,comm, PETSC_ERR_ARG_WRONG, "Vertex %d is not an unsplit vertex", cone[0]);
        coneNew[2] = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
        PetscCall(PetscFindInt(cone[1], numUnsplitPoints[dep-1], unsplitPoints[dep-1], &v));
        PetscCheckFalse(v < 0,comm, PETSC_ERR_ARG_WRONG, "Vertex %d is not an unsplit vertex", cone[1]);
        coneNew[3] = v + pMaxNew[dep] + numSplitPoints[dep] + numSplitPoints[dep-1];
        PetscCall(DMPlexSetCone(sdm, hybface, coneNew));
        for (f = 0, qf = 0; f < supportSize; ++f) {
          PetscInt val, face;

          PetscCall(DMLabelGetValue(label, support[f], &val));
          if (val == dim-1) {
            PetscCall(PetscFindInt(support[f], numSplitPoints[dep+1], splitPoints[dep+1], &face));
            supportNew[qf++] = face + pMaxNew[dep+2] + numSplitPoints[dep+2];
          }
        }
        PetscCall(DMPlexGetSupportSize(sdm, hybface, &supportSizeNew));
        PetscCheckFalse(qf != supportSizeNew,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support size for hybrid face %d is %d != %d", hybface, qf, supportSizeNew);
        PetscCall(DMPlexSetSupport(sdm, hybface, supportNew));
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
    PetscCall(DMLabelGetStratumIS(label, dep, &pIS));
    if (!pIS) continue;
    dep  = -dep - shift;
    PetscCall(ISGetLocalSize(pIS, &numPoints));
    PetscCall(ISGetIndices(pIS, &points));
    for (p = 0; p < numPoints; ++p) {
      const PetscInt  oldp = points[p];
      const PetscInt  newp = DMPlexShiftPoint_Internal(oldp, depth, depthShift) /*depthOffset[dep] + oldp*/;
      const PetscInt *cone;
      PetscInt        coneSize, c;
      /* PetscBool       replaced = PETSC_FALSE; */

      /* Negative edge: replace split vertex */
      /* Negative cell: replace split face */
      PetscCall(DMPlexGetConeSize(sdm, newp, &coneSize));
      PetscCall(DMPlexGetCone(sdm, newp, &cone));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt coldp = DMPlexShiftPointInverse_Internal(cone[c],depth,depthShift);
        PetscInt       csplitp, cp, val;

        PetscCall(DMLabelGetValue(label, coldp, &val));
        if (val == dep-1) {
          PetscCall(PetscFindInt(coldp, numSplitPoints[dep-1], splitPoints[dep-1], &cp));
          PetscCheckFalse(cp < 0,comm, PETSC_ERR_ARG_WRONG, "Point %d is not a split point of dimension %d", oldp, dep-1);
          csplitp  = pMaxNew[dep-1] + cp;
          PetscCall(DMPlexInsertCone(sdm, newp, c, csplitp));
          /* replaced = PETSC_TRUE; */
        }
      }
      /* Cells with only a vertex or edge on the submesh have no replacement */
      /* PetscCheck(replaced,comm, PETSC_ERR_ARG_WRONG, "The cone of point %d does not contain split points", oldp); */
    }
    PetscCall(ISRestoreIndices(pIS, &points));
    PetscCall(ISDestroy(&pIS));
  }
  /* Step 7: Coordinates */
  PetscCall(DMPlexShiftCoordinates_Internal(dm, depthShift, sdm));
  PetscCall(DMGetCoordinateSection(sdm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(sdm, &coordinates));
  PetscCall(VecGetArray(coordinates, &coords));
  for (v = 0; v < (numSplitPoints ? numSplitPoints[0] : 0); ++v) {
    const PetscInt newp   = DMPlexShiftPoint_Internal(splitPoints[0][v], depth, depthShift) /*depthOffset[0] + splitPoints[0][v]*/;
    const PetscInt splitp = pMaxNew[0] + v;
    PetscInt       dof, off, soff, d;

    PetscCall(PetscSectionGetDof(coordSection, newp, &dof));
    PetscCall(PetscSectionGetOffset(coordSection, newp, &off));
    PetscCall(PetscSectionGetOffset(coordSection, splitp, &soff));
    for (d = 0; d < dof; ++d) coords[soff+d] = coords[off+d];
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  /* Step 8: SF, if I can figure this out we can split the mesh in parallel */
  PetscCall(DMPlexShiftSF_Internal(dm, depthShift, sdm));
  /* Step 9: Labels */
  PetscCall(DMPlexShiftLabels_Internal(dm, depthShift, sdm));
  PetscCall(DMPlexCreateVTKLabel_Internal(dm, PETSC_FALSE, sdm));
  PetscCall(DMGetNumLabels(sdm, &numLabels));
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt newp   = DMPlexShiftPoint_Internal(splitPoints[dep][p], depth, depthShift) /*depthOffset[dep] + splitPoints[dep][p]*/;
      const PetscInt splitp = pMaxNew[dep] + p;
      PetscInt       l;

      if (splitLabel) {
        const PetscInt val = 100 + dep;

        PetscCall(DMLabelSetValue(splitLabel, newp,    val));
        PetscCall(DMLabelSetValue(splitLabel, splitp, -val));
      }
      for (l = 0; l < numLabels; ++l) {
        DMLabel     mlabel;
        const char *lname;
        PetscInt    val;
        PetscBool   isDepth;

        PetscCall(DMGetLabelName(sdm, l, &lname));
        PetscCall(PetscStrcmp(lname, "depth", &isDepth));
        if (isDepth) continue;
        PetscCall(DMGetLabel(sdm, lname, &mlabel));
        PetscCall(DMLabelGetValue(mlabel, newp, &val));
        if (val >= 0) {
          PetscCall(DMLabelSetValue(mlabel, splitp, val));
        }
      }
    }
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    if (splitIS[dep]) PetscCall(ISRestoreIndices(splitIS[dep], &splitPoints[dep]));
    PetscCall(ISDestroy(&splitIS[dep]));
    if (unsplitIS[dep]) PetscCall(ISRestoreIndices(unsplitIS[dep], &unsplitPoints[dep]));
    PetscCall(ISDestroy(&unsplitIS[dep]));
  }
  if (label) {
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
  }
  for (d = 0; d <= depth; ++d) {
    PetscCall(DMPlexGetDepthStratum(sdm, d, NULL, &pEnd));
    pMaxNew[d] = pEnd - numHybridPoints[d] - numHybridPointsOld[d];
  }
  PetscCall(PetscFree3(coneNew, coneONew, supportNew));
  PetscCall(PetscFree5(depthMax, depthEnd, depthShift, pMaxNew, numHybridPointsOld));
  PetscCall(PetscFree7(splitIS, unsplitIS, numSplitPoints, numUnsplitPoints, numHybridPoints, splitPoints, unsplitPoints));
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
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), &sdm));
  PetscCall(DMSetType(sdm, DMPLEX));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(sdm, dim));
  switch (dim) {
  case 2:
  case 3:
    PetscCall(DMPlexConstructCohesiveCells_Internal(dm, label, splitLabel, sdm));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
  PetscCall(DMPlexGetCone(dm, cell, &cone));
  PetscCall(DMPlexGetConeOrientation(dm, cell, &ornt));
  for (c = 0; c < coneSize; ++c) {
    if (cone[c] == face) {
      PetscInt o = ornt[c];

      if (subdm) {
        const PetscInt *subcone, *subornt;
        PetscInt        subpoint, subface, subconeSize, sc;

        PetscCall(PetscFindInt(cell, numSubpoints, subpoints, &subpoint));
        PetscCall(PetscFindInt(face, numSubpoints, subpoints, &subface));
        PetscCall(DMPlexGetConeSize(subdm, subpoint, &subconeSize));
        PetscCall(DMPlexGetCone(subdm, subpoint, &subcone));
        PetscCall(DMPlexGetConeOrientation(subdm, subpoint, &subornt));
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
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  if (subdm) {
    PetscCall(DMPlexGetSubpointIS(subdm, &subpointIS));
    if (subpointIS) {
      PetscCall(ISGetLocalSize(subpointIS, &numSubpoints));
      PetscCall(ISGetIndices(subpointIS, &subpoints));
    }
  }
  /* Mark cell on the fault, and its faces which touch the fault: cell orientation for face gives the side of the fault */
  PetscCall(DMLabelGetStratumIS(label, dim-1, &dimIS));
  if (!dimIS) PetscFunctionReturn(0);
  PetscCall(ISGetLocalSize(dimIS, &numPoints));
  PetscCall(ISGetIndices(dimIS, &points));
  for (p = 0; p < numPoints; ++p) { /* Loop over fault faces */
    const PetscInt *support;
    PetscInt        supportSize, s;

    PetscCall(DMPlexGetSupportSize(dm, points[p], &supportSize));
#if 0
    if (supportSize != 2) {
      const PetscInt *lp;
      PetscInt        Nlp, pind;

      /* Check that for a cell with a single support face, that face is in the SF */
      /*   THis check only works for the remote side. We would need root side information */
      PetscCall(PetscSFGetGraph(dm->sf, NULL, &Nlp, &lp, NULL));
      PetscCall(PetscFindInt(points[p], Nlp, lp, &pind));
      PetscCheckFalse(pind < 0,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Split face %d has %d != 2 supports, and the face is not shared with another process", points[p], supportSize);
    }
#endif
    PetscCall(DMPlexGetSupport(dm, points[p], &support));
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone;
      PetscInt        coneSize, c;
      PetscBool       pos;

      PetscCall(GetSurfaceSide_Static(dm, subdm, numSubpoints, subpoints, support[s], points[p], &pos));
      if (pos) PetscCall(DMLabelSetValue(label, support[s],  rev*(shift+dim)));
      else     PetscCall(DMLabelSetValue(label, support[s], -rev*(shift+dim)));
      if (rev < 0) pos = !pos ? PETSC_TRUE : PETSC_FALSE;
      /* Put faces touching the fault in the label */
      PetscCall(DMPlexGetConeSize(dm, support[s], &coneSize));
      PetscCall(DMPlexGetCone(dm, support[s], &cone));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];

        PetscCall(DMLabelGetValue(label, point, &val));
        if (val == -1) {
          PetscInt *closure = NULL;
          PetscInt  closureSize, cl;

          PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
          for (cl = 0; cl < closureSize*2; cl += 2) {
            const PetscInt clp  = closure[cl];
            PetscInt       bval = -1;

            PetscCall(DMLabelGetValue(label, clp, &val));
            if (blabel) PetscCall(DMLabelGetValue(blabel, clp, &bval));
            if ((val >= 0) && (val < dim-1) && (bval < 0)) {
              PetscCall(DMLabelSetValue(label, point, pos == PETSC_TRUE ? shift+dim-1 : -(shift+dim-1)));
              break;
            }
          }
          PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
        }
      }
    }
  }
  PetscCall(ISRestoreIndices(dimIS, &points));
  PetscCall(ISDestroy(&dimIS));
  if (subpointIS) PetscCall(ISRestoreIndices(subpointIS, &subpoints));
  /* Mark boundary points as unsplit */
  if (blabel) {
    PetscCall(DMLabelGetStratumIS(blabel, 1, &dimIS));
    PetscCall(ISGetLocalSize(dimIS, &numPoints));
    PetscCall(ISGetIndices(dimIS, &points));
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscInt       val, bval;

      PetscCall(DMLabelGetValue(blabel, point, &bval));
      if (bval >= 0) {
        PetscCall(DMLabelGetValue(label, point, &val));
        if ((val < 0) || (val > dim)) {
          /* This could be a point added from splitting a vertex on an adjacent fault, otherwise its just wrong */
          PetscCall(DMLabelClearValue(blabel, point, bval));
        }
      }
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscInt       val, bval;

      PetscCall(DMLabelGetValue(blabel, point, &bval));
      if (bval >= 0) {
        const PetscInt *cone,    *support;
        PetscInt        coneSize, supportSize, s, valA, valB, valE;

        /* Mark as unsplit */
        PetscCall(DMLabelGetValue(label, point, &val));
        PetscCheckFalse((val < 0) || (val > dim),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %d has label value %d, should be part of the fault", point, val);
        PetscCall(DMLabelClearValue(label, point, val));
        PetscCall(DMLabelSetValue(label, point, shift2+val));
        /* Check for cross-edge
             A cross-edge has endpoints which are both on the boundary of the surface, but the edge itself is not. */
        if (val != 0) continue;
        PetscCall(DMPlexGetSupport(dm, point, &support));
        PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
        for (s = 0; s < supportSize; ++s) {
          PetscCall(DMPlexGetCone(dm, support[s], &cone));
          PetscCall(DMPlexGetConeSize(dm, support[s], &coneSize));
          PetscCheckFalse(coneSize != 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %D has %D vertices != 2", support[s], coneSize);
          PetscCall(DMLabelGetValue(blabel, cone[0], &valA));
          PetscCall(DMLabelGetValue(blabel, cone[1], &valB));
          PetscCall(DMLabelGetValue(blabel, support[s], &valE));
          if ((valE < 0) && (valA >= 0) && (valB >= 0) && (cone[0] != cone[1])) PetscCall(DMLabelSetValue(blabel, support[s], 2));
        }
      }
    }
    PetscCall(ISRestoreIndices(dimIS, &points));
    PetscCall(ISDestroy(&dimIS));
  }
  /* Search for other cells/faces/edges connected to the fault by a vertex */
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelGetStratumIS(label, 0, &dimIS));
  /* TODO Why are we including cross edges here? Shouldn't they be in the star of boundary vertices? */
  if (blabel) PetscCall(DMLabelGetStratumIS(blabel, 2, &crossEdgeIS));
  if (dimIS && crossEdgeIS) {
    IS vertIS = dimIS;

    PetscCall(ISExpand(vertIS, crossEdgeIS, &dimIS));
    PetscCall(ISDestroy(&crossEdgeIS));
    PetscCall(ISDestroy(&vertIS));
  }
  if (!dimIS) {
    PetscFunctionReturn(0);
  }
  PetscCall(ISGetLocalSize(dimIS, &numPoints));
  PetscCall(ISGetIndices(dimIS, &points));
  for (p = 0; p < numPoints; ++p) { /* Loop over fault vertices */
    PetscInt *star = NULL;
    PetscInt  starSize, s;
    PetscInt  again = 1;  /* 0: Finished 1: Keep iterating after a change 2: No change */

    /* All points connected to the fault are inside a cell, so at the top level we will only check cells */
    PetscCall(DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star));
    while (again) {
      PetscCheckFalse(again > 1,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Could not classify all cells connected to the fault");
      again = 0;
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt  point = star[s];
        const PetscInt *cone;
        PetscInt        coneSize, c;

        if ((point < cStart) || (point >= cEnd)) continue;
        PetscCall(DMLabelGetValue(label, point, &val));
        if (val != -1) continue;
        again = again == 1 ? 1 : 2;
        PetscCall(DMPlexGetConeSize(dm, point, &coneSize));
        PetscCall(DMPlexGetCone(dm, point, &cone));
        for (c = 0; c < coneSize; ++c) {
          PetscCall(DMLabelGetValue(label, cone[c], &val));
          if (val != -1) {
            const PetscInt *ccone;
            PetscInt        cconeSize, cc, side;

            PetscCheckFalse(PetscAbs(val) < shift,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d on cell %d has an invalid label %d", cone[c], point, val);
            if (val > 0) side =  1;
            else         side = -1;
            PetscCall(DMLabelSetValue(label, point, side*(shift+dim)));
            /* Mark cell faces which touch the fault */
            PetscCall(DMPlexGetConeSize(dm, point, &cconeSize));
            PetscCall(DMPlexGetCone(dm, point, &ccone));
            for (cc = 0; cc < cconeSize; ++cc) {
              PetscInt *closure = NULL;
              PetscInt  closureSize, cl;

              PetscCall(DMLabelGetValue(label, ccone[cc], &val));
              if (val != -1) continue;
              PetscCall(DMPlexGetTransitiveClosure(dm, ccone[cc], PETSC_TRUE, &closureSize, &closure));
              for (cl = 0; cl < closureSize*2; cl += 2) {
                const PetscInt clp = closure[cl];

                PetscCall(DMLabelGetValue(label, clp, &val));
                if (val == -1) continue;
                PetscCall(DMLabelSetValue(label, ccone[cc], side*(shift+dim-1)));
                break;
              }
              PetscCall(DMPlexRestoreTransitiveClosure(dm, ccone[cc], PETSC_TRUE, &closureSize, &closure));
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

      PetscCall(DMLabelGetValue(label, point, &val));
      if (val == -1) {
        PetscInt      *sstar = NULL;
        PetscInt       sstarSize, ss;
        PetscBool      marked = PETSC_FALSE, isHybrid;

        PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &sstarSize, &sstar));
        for (ss = 0; ss < sstarSize*2; ss += 2) {
          const PetscInt spoint = sstar[ss];

          if ((spoint < cStart) || (spoint >= cEnd)) continue;
          PetscCall(DMLabelGetValue(label, spoint, &val));
          PetscCheckFalse(val == -1,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Cell %d in star of %d does not have a valid label", spoint, point);
          PetscCall(DMLabelGetValue(depthLabel, point, &dep));
          if (val > 0) {
            PetscCall(DMLabelSetValue(label, point,   shift+dep));
          } else {
            PetscCall(DMLabelSetValue(label, point, -(shift+dep)));
          }
          marked = PETSC_TRUE;
          break;
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &sstarSize, &sstar));
        PetscCall(DMPlexCellIsHybrid_Internal(dm, point, &isHybrid));
        PetscCheckFalse(!isHybrid && !marked,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d could not be classified", point);
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star));
  }
  PetscCall(ISRestoreIndices(dimIS, &points));
  PetscCall(ISDestroy(&dimIS));
  /* If any faces touching the fault divide cells on either side, split them
       This only occurs without a surface boundary */
  PetscCall(DMLabelGetStratumIS(label,   shift+dim-1,  &facePosIS));
  PetscCall(DMLabelGetStratumIS(label, -(shift+dim-1), &faceNegIS));
  PetscCall(ISExpand(facePosIS, faceNegIS, &dimIS));
  PetscCall(ISDestroy(&facePosIS));
  PetscCall(ISDestroy(&faceNegIS));
  PetscCall(ISGetLocalSize(dimIS, &numPoints));
  PetscCall(ISGetIndices(dimIS, &points));
  for (p = 0; p < numPoints; ++p) {
    const PetscInt  point = points[p];
    const PetscInt *support;
    PetscInt        supportSize, valA, valB;

    PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
    if (supportSize != 2) continue;
    PetscCall(DMPlexGetSupport(dm, point, &support));
    PetscCall(DMLabelGetValue(label, support[0], &valA));
    PetscCall(DMLabelGetValue(label, support[1], &valB));
    if ((valA == -1) || (valB == -1)) continue;
    if (valA*valB > 0) continue;
    /* Split the face */
    PetscCall(DMLabelGetValue(label, point, &valA));
    PetscCall(DMLabelClearValue(label, point, valA));
    PetscCall(DMLabelSetValue(label, point, dim-1));
    /* Label its closure:
      unmarked: label as unsplit
      incident: relabel as split
      split:    do nothing
    */
    {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        PetscCall(DMLabelGetValue(label, closure[cl], &valA));
        if (valA == -1) { /* Mark as unsplit */
          PetscCall(DMLabelGetValue(depthLabel, closure[cl], &dep));
          PetscCall(DMLabelSetValue(label, closure[cl], shift2+dep));
        } else if (((valA >= shift) && (valA < shift2)) || ((valA <= -shift) && (valA > -shift2))) {
          PetscCall(DMLabelGetValue(depthLabel, closure[cl], &dep));
          PetscCall(DMLabelClearValue(label, closure[cl], valA));
          PetscCall(DMLabelSetValue(label, closure[cl], dep));
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure));
    }
  }
  PetscCall(ISRestoreIndices(dimIS, &points));
  PetscCall(ISDestroy(&dimIS));
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
  PetscCall(DMLabelGetDefaultValue(label, &defaultValue));
  PetscCall(DMPlexGetSubpointIS(subdm, &subpointIS));
  if (!subpointIS) PetscFunctionReturn(0);
  PetscCall(DMPlexGetHeightStratum(subdm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(ISGetIndices(subpointIS, &dmpoints));
  for (c = cStart; c < cEnd; ++c) {
    PetscBool invalidCell = PETSC_TRUE;
    PetscInt *closure     = NULL;
    PetscInt  closureSize, cl;

    PetscCall(DMPlexGetTransitiveClosure(dm, dmpoints[c], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      PetscInt value = 0;

      if ((closure[cl] < vStart) || (closure[cl] >= vEnd)) continue;
      PetscCall(DMLabelGetValue(label, closure[cl], &value));
      if (value == defaultValue) {invalidCell = PETSC_FALSE; break;}
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, dmpoints[c], PETSC_TRUE, &closureSize, &closure));
    if (invalidCell) {
      PetscCall(ISRestoreIndices(subpointIS, &dmpoints));
      PetscCall(ISDestroy(&subpointIS));
      PetscCall(DMDestroy(&subdm));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ambiguous submesh. Cell %D has all of its vertices on the submesh.", dmpoints[c]);
    }
  }
  PetscCall(ISRestoreIndices(subpointIS, &dmpoints));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexCreateSubmesh(dm, label, 1, PETSC_FALSE, &idm));
  PetscCall(DMPlexCheckValidSubmesh_Private(dm, label, idm));
  PetscCall(DMPlexOrient(idm));
  PetscCall(DMPlexGetSubpointMap(idm, &subpointMap));
  PetscCall(DMLabelDuplicate(subpointMap, &hlabel));
  PetscCall(DMLabelClearStratum(hlabel, dim));
  if (splitLabel) {
    const char *name;
    char        sname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscObjectGetName((PetscObject) hlabel, &name));
    PetscCall(PetscStrncpy(sname, name, PETSC_MAX_PATH_LEN));
    PetscCall(PetscStrcat(sname, " split"));
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, sname, &slabel));
  }
  PetscCall(DMPlexLabelCohesiveComplete(dm, hlabel, bdlabel, PETSC_FALSE, idm));
  if (dmInterface) {*dmInterface = idm;}
  else             PetscCall(DMDestroy(&idm));
  PetscCall(DMPlexConstructCohesiveCells(dm, hlabel, slabel, dmHybrid));
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, *dmHybrid));
  if (hybridLabel) *hybridLabel = hlabel;
  else             PetscCall(DMLabelDestroy(&hlabel));
  if (splitLabel)  *splitLabel  = slabel;
  {
    DM      cdm;
    DMLabel ctLabel;

    /* We need to somehow share the celltype label with the coordinate dm */
    PetscCall(DMGetCoordinateDM(*dmHybrid, &cdm));
    PetscCall(DMPlexGetCellTypeLabel(*dmHybrid, &ctLabel));
    PetscCall(DMSetLabel(cdm, ctLabel));
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
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));
  pSize = PetscMax(depth, dim) + 1;
  PetscCall(PetscMalloc2(pSize, &pStart, pSize, &pEnd));
  for (d = 0; d <= depth; ++d) {
    PetscCall(DMPlexGetSimplexOrBoxCells(dm, depth-d, &pStart[d], &pEnd[d]));
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  if (vertexLabel) PetscCall(DMLabelGetStratumIS(vertexLabel, value, &subvertexIS));
  if (subvertexIS) {
    PetscCall(ISGetSize(subvertexIS, &numSubVerticesInitial));
    PetscCall(ISGetIndices(subvertexIS, &subvertices));
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numCells = 0, c;

    PetscCall(DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      if ((point >= pStart[depth]) && (point < pEnd[depth])) star[numCells++] = point;
    }
    for (c = 0; c < numCells; ++c) {
      const PetscInt cell    = star[c];
      PetscInt      *closure = NULL;
      PetscInt       closureSize, cl;
      PetscInt       cellLoc, numCorners = 0, faceSize = 0;

      PetscCall(DMLabelGetValue(subpointMap, cell, &cellLoc));
      if (cellLoc == 2) continue;
      PetscCheckFalse(cellLoc >= 0,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Cell %d has dimension %d in the surface label", cell, cellLoc);
      PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt point = closure[cl];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          ++numCorners;
          PetscCall(DMLabelGetValue(vertexLabel, point, &vertexLoc));
          if (vertexLoc == value) closure[faceSize++] = point;
        }
      }
      if (!(*nFV)) PetscCall(DMPlexGetNumFaceVertices(dm, dim, numCorners, nFV));
      PetscCheckFalse(faceSize > *nFV,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
      if (faceSize == *nFV) {
        const PetscInt *cells = NULL;
        PetscInt        numCells, nc;

        ++(*numFaces);
        for (cl = 0; cl < faceSize; ++cl) {
          PetscCall(DMLabelSetValue(subpointMap, closure[cl], 0));
        }
        PetscCall(DMPlexGetJoin(dm, faceSize, closure, &numCells, &cells));
        for (nc = 0; nc < numCells; ++nc) {
          PetscCall(DMLabelSetValue(subpointMap, cells[nc], 2));
        }
        PetscCall(DMPlexRestoreJoin(dm, faceSize, closure, &numCells, &cells));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
  }
  if (subvertexIS) {
    PetscCall(ISRestoreIndices(subvertexIS, &subvertices));
  }
  PetscCall(ISDestroy(&subvertexIS));
  PetscCall(PetscFree2(pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DMLabel subpointMap, DM subdm)
{
  IS               subvertexIS = NULL;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd;
  PetscInt         dim, d, numSubVerticesInitial = 0, v;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscMalloc2(dim+1, &pStart, dim+1, &pEnd));
  for (d = 0; d <= dim; ++d) {
    PetscCall(DMPlexGetSimplexOrBoxCells(dm, dim-d, &pStart[d], &pEnd[d]));
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  if (vertexLabel) {
    PetscCall(DMLabelGetStratumIS(vertexLabel, value, &subvertexIS));
    if (subvertexIS) {
      PetscCall(ISGetSize(subvertexIS, &numSubVerticesInitial));
      PetscCall(ISGetIndices(subvertexIS, &subvertices));
    }
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numFaces = 0, f;

    PetscCall(DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      PetscInt       faceLoc;

      if ((point >= pStart[dim-1]) && (point < pEnd[dim-1])) {
        if (markedFaces) {
          PetscCall(DMLabelGetValue(vertexLabel, point, &faceLoc));
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

      PetscCall(DMLabelGetValue(subpointMap, face, &faceLoc));
      if (faceLoc == dim-1) continue;
      PetscCheckFalse(faceLoc >= 0,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d has dimension %d in the surface label", face, faceLoc);
      PetscCall(DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        const PetscInt point = closure[c];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          PetscCall(DMLabelGetValue(vertexLabel, point, &vertexLoc));
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
              PetscCall(DMLabelSetValue(subpointMap, point, d));
              break;
            }
          }
        }
        PetscCall(DMPlexGetSupportSize(dm, face, &supportSize));
        PetscCall(DMPlexGetSupport(dm, face, &support));
        for (s = 0; s < supportSize; ++s) {
          PetscCall(DMLabelSetValue(subpointMap, support[s], dim));
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star));
  }
  if (subvertexIS) PetscCall(ISRestoreIndices(subvertexIS, &subvertices));
  PetscCall(ISDestroy(&subvertexIS));
  PetscCall(PetscFree2(pStart, pEnd));
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
  if (labelname) PetscCall(DMGetLabel(dm, labelname, &label));
  *subCells = NULL;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetTensorPrismBounds_Internal(dm, dim, &cMax, &cEnd));
  if (cMax < 0) PetscFunctionReturn(0);
  if (label) {
    for (c = cMax; c < cEnd; ++c) {
      PetscInt val;

      PetscCall(DMLabelGetValue(label, c, &val));
      if (val == value) {
        ++(*numFaces);
        PetscCall(DMPlexGetConeSize(dm, c, &coneSize));
      }
    }
  } else {
    *numFaces = cEnd - cMax;
    PetscCall(DMPlexGetConeSize(dm, cMax, &coneSize));
  }
  PetscCall(PetscMalloc1(*numFaces *2, subCells));
  if (!(*numFaces)) PetscFunctionReturn(0);
  *nFV = hasLagrange ? coneSize/3 : coneSize/2;
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cells;
    PetscInt        numCells;

    if (label) {
      PetscInt val;

      PetscCall(DMLabelGetValue(label, c, &val));
      if (val != value) continue;
    }
    PetscCall(DMPlexGetCone(dm, c, &cone));
    for (p = 0; p < *nFV; ++p) {
      PetscCall(DMLabelSetValue(subpointMap, cone[p], 0));
    }
    /* Negative face */
    PetscCall(DMPlexGetJoin(dm, *nFV, cone, &numCells, &cells));
    /* Not true in parallel
    PetscCheckFalse(numCells != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells"); */
    for (p = 0; p < numCells; ++p) {
      PetscCall(DMLabelSetValue(subpointMap, cells[p], 2));
      (*subCells)[subc++] = cells[p];
    }
    PetscCall(DMPlexRestoreJoin(dm, *nFV, cone, &numCells, &cells));
    /* Positive face is not included */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMarkCohesiveSubmesh_Interpolated(DM dm, DMLabel label, PetscInt value, DMLabel subpointMap, DM subdm)
{
  PetscInt      *pStart, *pEnd;
  PetscInt       dim, cMax, cEnd, c, d;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetTensorPrismBounds_Internal(dm, dim, &cMax, &cEnd));
  if (cMax < 0) PetscFunctionReturn(0);
  PetscCall(PetscMalloc2(dim+1,&pStart,dim+1,&pEnd));
  for (d = 0; d <= dim; ++d) PetscCall(DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]));
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cone;
    PetscInt       *closure = NULL;
    PetscInt        fconeSize, coneSize, closureSize, cl, val;

    if (label) {
      PetscCall(DMLabelGetValue(label, c, &val));
      if (val != value) continue;
    }
    PetscCall(DMPlexGetConeSize(dm, c, &coneSize));
    PetscCall(DMPlexGetCone(dm, c, &cone));
    PetscCall(DMPlexGetConeSize(dm, cone[0], &fconeSize));
    PetscCheckFalse(coneSize != (fconeSize ? fconeSize : 1) + 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells");
    /* Negative face */
    PetscCall(DMPlexGetTransitiveClosure(dm, cone[0], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];

      for (d = 0; d <= dim; ++d) {
        if ((point >= pStart[d]) && (point < pEnd[d])) {
          PetscCall(DMLabelSetValue(subpointMap, point, d));
          break;
        }
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cone[0], PETSC_TRUE, &closureSize, &closure));
    /* Cells -- positive face is not included */
    for (cl = 0; cl < 1; ++cl) {
      const PetscInt *support;
      PetscInt        supportSize, s;

      PetscCall(DMPlexGetSupportSize(dm, cone[cl], &supportSize));
      /* PetscCheckFalse(supportSize != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive faces should separate two cells"); */
      PetscCall(DMPlexGetSupport(dm, cone[cl], &support));
      for (s = 0; s < supportSize; ++s) {
        PetscCall(DMLabelSetValue(subpointMap, support[s], dim));
      }
    }
  }
  PetscCall(PetscFree2(pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFaceOrientation(DM dm, PetscInt cell, PetscInt numCorners, PetscInt indices[], PetscInt oppositeVertex, PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  MPI_Comm       comm;
  PetscBool      posOrient = PETSC_FALSE;
  const PetscInt debug     = 0;
  PetscInt       cellDim, faceSize, f;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(DMGetDimension(dm, &cellDim));
  if (debug) PetscCall(PetscPrintf(comm, "cellDim: %d numCorners: %d\n", cellDim, numCorners));

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
    PetscCall(PetscSortInt(faceSizeTri, sortedIndices));
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
    PetscCall(PetscSortInt(faceSizeQuad, sortedIndices));
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
    PetscCall(PetscSortInt(faceSizeHex, sortedIndices));
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
    PetscCall(PetscSortInt(faceSizeTet, sortedIndices));
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
    PetscCall(PetscSortInt(faceSizeQuadHex, sortedIndices));
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
    if (debug) PetscCall(PetscPrintf(comm, "  Reversing initial face orientation\n"));
    for (f = 0; f < faceSize; ++f) faceVertices[f] = origVertices[faceSize-1 - f];
  } else {
    if (debug) PetscCall(PetscPrintf(comm, "  Keeping initial face orientation\n"));
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
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
  PetscCall(DMPlexGetCone(dm, cell, &cone));
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
  PetscCall(DMPlexGetFaceOrientation(dm, cell, numCorners, indices, oppositeVertex, origVertices, faceVertices, posOriented));
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
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(DMPlexGetConeSize(subdm, subcell, &coneSize));
  PetscCheckFalse(coneSize != 1,comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size of cell %d is %d != 1", cell, coneSize);
#if 0
  /* Cannot use this because support() has not been constructed yet */
  PetscCall(DMPlexGetJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces));
#else
  {
    PetscInt f;

    numFaces = 0;
    PetscCall(DMGetWorkArray(subdm, 1, MPIU_INT, (void **) &faces));
    for (f = firstFace; f < *newFacePoint; ++f) {
      PetscInt dof, off, d;

      PetscCall(PetscSectionGetDof(submesh->coneSection, f, &dof));
      PetscCall(PetscSectionGetOffset(submesh->coneSection, f, &off));
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
    PetscCall(DMPlexSetCone(subdm, subcell, faces));
  } else {
    PetscInt *indices, *origVertices, *orientedVertices, *orientedSubVertices, v, ov;
    PetscBool posOriented;

    PetscCall(DMGetWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), MPIU_INT, &orientedVertices));
    origVertices        = &orientedVertices[numFaceVertices];
    indices             = &orientedVertices[numFaceVertices*2];
    orientedSubVertices = &orientedVertices[numFaceVertices*3];
    PetscCall(DMPlexGetOrientedFace(dm, cell, numFaceVertices, faceVertices, numCorners, indices, origVertices, orientedVertices, &posOriented));
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
    PetscCall(DMPlexSetCone(subdm, *newFacePoint, orientedSubVertices));
    PetscCall(DMPlexSetCone(subdm, subcell, newFacePoint));
    PetscCall(DMRestoreWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), MPIU_INT, &orientedVertices));
    ++(*newFacePoint);
  }
#if 0
  PetscCall(DMPlexRestoreJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces));
#else
  PetscCall(DMRestoreWorkArray(subdm, 1, MPIU_INT, (void **) &faces));
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
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  /* Create subpointMap which marks the submesh */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap));
  PetscCall(DMPlexSetSubpointMap(subdm, subpointMap));
  PetscCall(DMLabelDestroy(&subpointMap));
  if (vertexLabel) PetscCall(DMPlexMarkSubmesh_Uninterpolated(dm, vertexLabel, value, subpointMap, &numSubFaces, &nFV, subdm));
  /* Setup chart */
  PetscCall(DMLabelGetStratumSize(subpointMap, 0, &numSubVertices));
  PetscCall(DMLabelGetStratumSize(subpointMap, 2, &numSubCells));
  PetscCall(DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices));
  PetscCall(DMPlexSetVTKCellHeight(subdm, 1));
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  PetscCall(DMLabelGetStratumIS(subpointMap, 0, &subvertexIS));
  if (subvertexIS) PetscCall(ISGetIndices(subvertexIS, &subVertices));
  PetscCall(DMLabelGetStratumIS(subpointMap, 2, &subcellIS));
  if (subcellIS) PetscCall(ISGetIndices(subcellIS, &subCells));
  for (c = 0; c < numSubCells; ++c) {
    PetscCall(DMPlexSetConeSize(subdm, c, 1));
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    PetscCall(DMPlexSetConeSize(subdm, f, nFV));
  }
  PetscCall(DMSetUp(subdm));
  /* Create face cones */
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  PetscCall(DMGetWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt cell    = subCells[c];
    const PetscInt subcell = c;
    PetscInt      *closure = NULL;
    PetscInt       closureSize, cl, numCorners = 0, faceSize = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];
      PetscInt       subVertex;

      if ((point >= vStart) && (point < vEnd)) {
        ++numCorners;
        PetscCall(PetscFindInt(point, numSubVertices, subVertices, &subVertex));
        if (subVertex >= 0) {
          closure[faceSize] = point;
          subface[faceSize] = firstSubVertex+subVertex;
          ++faceSize;
        }
      }
    }
    PetscCheckFalse(faceSize > nFV,comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
    if (faceSize == nFV) {
      PetscCall(DMPlexInsertFace_Internal(dm, subdm, faceSize, closure, subface, numCorners, cell, subcell, firstSubFace, &newFacePoint));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  }
  PetscCall(DMRestoreWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  PetscCall(DMPlexSymmetrize(subdm));
  PetscCall(DMPlexStratify(subdm));
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize, v;
    const char  *name;

    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(subdm, &subCoordSection));
    PetscCall(PetscSectionSetNumFields(subCoordSection, 1));
    PetscCall(PetscSectionGetFieldComponents(coordSection, 0, &numComp));
    PetscCall(PetscSectionSetFieldComponents(subCoordSection, 0, numComp));
    PetscCall(PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices));
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      PetscCall(PetscSectionGetDof(coordSection, vertex, &dof));
      PetscCall(PetscSectionSetDof(subCoordSection, subvertex, dof));
      PetscCall(PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof));
    }
    PetscCall(PetscSectionSetUp(subCoordSection));
    PetscCall(PetscSectionGetStorageSize(subCoordSection, &coordSize));
    PetscCall(VecCreate(PETSC_COMM_SELF, &subCoordinates));
    PetscCall(PetscObjectGetName((PetscObject)coordinates,&name));
    PetscCall(PetscObjectSetName((PetscObject)subCoordinates,name));
    PetscCall(VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE));
    PetscCall(VecSetType(subCoordinates,VECSTANDARD));
    if (coordSize) {
      PetscCall(VecGetArray(coordinates,    &coords));
      PetscCall(VecGetArray(subCoordinates, &subCoords));
      for (v = 0; v < numSubVertices; ++v) {
        const PetscInt vertex    = subVertices[v];
        const PetscInt subvertex = firstSubVertex+v;
        PetscInt       dof, off, sdof, soff, d;

        PetscCall(PetscSectionGetDof(coordSection, vertex, &dof));
        PetscCall(PetscSectionGetOffset(coordSection, vertex, &off));
        PetscCall(PetscSectionGetDof(subCoordSection, subvertex, &sdof));
        PetscCall(PetscSectionGetOffset(subCoordSection, subvertex, &soff));
        PetscCheckFalse(dof != sdof,comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
        for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
      }
      PetscCall(VecRestoreArray(coordinates,    &coords));
      PetscCall(VecRestoreArray(subCoordinates, &subCoords));
    }
    PetscCall(DMSetCoordinatesLocal(subdm, subCoordinates));
    PetscCall(VecDestroy(&subCoordinates));
  }
  /* Cleanup */
  if (subvertexIS) PetscCall(ISRestoreIndices(subvertexIS, &subVertices));
  PetscCall(ISDestroy(&subvertexIS));
  if (subcellIS) PetscCall(ISRestoreIndices(subcellIS, &subCells));
  PetscCall(ISDestroy(&subcellIS));
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
  PetscCall(DMGetNumLabels(dm, &Nl));
  for (l = 0; l < Nl; ++l) {
    DMLabel         label, newlabel;
    const char     *lname;
    PetscBool       isDepth, isDim, isCelltype, isVTK;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        Nv, v;

    PetscCall(DMGetLabelName(dm, l, &lname));
    PetscCall(PetscStrcmp(lname, "depth", &isDepth));
    PetscCall(PetscStrcmp(lname, "dim", &isDim));
    PetscCall(PetscStrcmp(lname, "celltype", &isCelltype));
    PetscCall(PetscStrcmp(lname, "vtk", &isVTK));
    if (isDepth || isDim || isCelltype || isVTK) continue;
    PetscCall(DMCreateLabel(subdm, lname));
    PetscCall(DMGetLabel(dm, lname, &label));
    PetscCall(DMGetLabel(subdm, lname, &newlabel));
    PetscCall(DMLabelGetDefaultValue(label, &v));
    PetscCall(DMLabelSetDefaultValue(newlabel, v));
    PetscCall(DMLabelGetValueIS(label, &valueIS));
    PetscCall(ISGetLocalSize(valueIS, &Nv));
    PetscCall(ISGetIndices(valueIS, &values));
    for (v = 0; v < Nv; ++v) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        Np, p;

      PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
      PetscCall(ISGetLocalSize(pointIS, &Np));
      PetscCall(ISGetIndices(pointIS, &points));
      for (p = 0; p < Np; ++p) {
        const PetscInt point = points[p];
        PetscInt       subp;

        PetscCall(DMPlexGetPointDepth(dm, point, &d));
        subp = DMPlexFilterPoint_Internal(point, firstSubPoint[d], numSubPoints[d], subpoints[d]);
        if (subp >= 0) PetscCall(DMLabelSetValue(newlabel, subp, values[v]));
      }
      PetscCall(ISRestoreIndices(pointIS, &points));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
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
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Create subpointMap which marks the submesh */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap));
  PetscCall(DMPlexSetSubpointMap(subdm, subpointMap));
  if (cellHeight) {
    if (isCohesive) PetscCall(DMPlexMarkCohesiveSubmesh_Interpolated(dm, label, value, subpointMap, subdm));
    else            PetscCall(DMPlexMarkSubmesh_Interpolated(dm, label, value, markedFaces, subpointMap, subdm));
  } else {
    DMLabel         depth;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints=0;

    PetscCall(DMPlexGetDepthLabel(dm, &depth));
    PetscCall(DMLabelGetStratumIS(label, value, &pointIS));
    if (pointIS) {
      PetscCall(ISGetIndices(pointIS, &points));
      PetscCall(ISGetLocalSize(pointIS, &numPoints));
    }
    for (p = 0; p < numPoints; ++p) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, c, pdim;

      PetscCall(DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        PetscCall(DMLabelGetValue(depth, closure[c], &pdim));
        PetscCall(DMLabelSetValue(subpointMap, closure[c], pdim));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure));
    }
    if (pointIS) PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  /* Setup chart */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscMalloc4(dim+1,&numSubPoints,dim+1,&firstSubPoint,dim+1,&subpointIS,dim+1,&subpoints));
  for (d = 0; d <= dim; ++d) {
    PetscCall(DMLabelGetStratumSize(subpointMap, d, &numSubPoints[d]));
    totSubPoints += numSubPoints[d];
  }
  PetscCall(DMPlexSetChart(subdm, 0, totSubPoints));
  PetscCall(DMPlexSetVTKCellHeight(subdm, cellHeight));
  /* Set cone sizes */
  firstSubPoint[dim] = 0;
  firstSubPoint[0]   = firstSubPoint[dim] + numSubPoints[dim];
  if (dim > 1) {firstSubPoint[dim-1] = firstSubPoint[0]     + numSubPoints[0];}
  if (dim > 2) {firstSubPoint[dim-2] = firstSubPoint[dim-1] + numSubPoints[dim-1];}
  for (d = 0; d <= dim; ++d) {
    PetscCall(DMLabelGetStratumIS(subpointMap, d, &subpointIS[d]));
    if (subpointIS[d]) PetscCall(ISGetIndices(subpointIS[d], &subpoints[d]));
  }
  /* We do not want this label automatically computed, instead we compute it here */
  PetscCall(DMCreateLabel(subdm, "celltype"));
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, coneSizeNew, c, val;
      DMPolytopeType  ct;

      PetscCall(DMPlexGetConeSize(dm, point, &coneSize));
      PetscCall(DMPlexSetConeSize(subdm, subpoint, coneSize));
      PetscCall(DMPlexGetCellType(dm, point, &ct));
      PetscCall(DMPlexSetCellType(subdm, subpoint, ct));
      if (cellHeight && (d == dim)) {
        PetscCall(DMPlexGetCone(dm, point, &cone));
        for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
          PetscCall(DMLabelGetValue(subpointMap, cone[c], &val));
          if (val >= 0) coneSizeNew++;
        }
        PetscCall(DMPlexSetConeSize(subdm, subpoint, coneSizeNew));
        PetscCall(DMPlexSetCellType(subdm, subpoint, DM_POLYTOPE_FV_GHOST));
      }
    }
  }
  PetscCall(DMLabelDestroy(&subpointMap));
  PetscCall(DMSetUp(subdm));
  /* Set cones */
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  PetscCall(PetscMalloc2(maxConeSize,&coneNew,maxConeSize,&orntNew));
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, subconeSize, coneSizeNew, c, subc, fornt = 0;

      if (d == dim-1) {
        const PetscInt *support, *cone, *ornt;
        PetscInt        supportSize, coneSize, s, subc;

        PetscCall(DMPlexGetSupport(dm, point, &support));
        PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
        for (s = 0; s < supportSize; ++s) {
          PetscBool isHybrid;

          PetscCall(DMPlexCellIsHybrid_Internal(dm, support[s], &isHybrid));
          if (!isHybrid) continue;
          PetscCall(PetscFindInt(support[s], numSubPoints[d+1], subpoints[d+1], &subc));
          if (subc >= 0) {
            const PetscInt ccell = subpoints[d+1][subc];

            PetscCall(DMPlexGetCone(dm, ccell, &cone));
            PetscCall(DMPlexGetConeSize(dm, ccell, &coneSize));
            PetscCall(DMPlexGetConeOrientation(dm, ccell, &ornt));
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
      PetscCall(DMPlexGetConeSize(dm, point, &coneSize));
      PetscCall(DMPlexGetConeSize(subdm, subpoint, &subconeSize));
      PetscCall(DMPlexGetCone(dm, point, &cone));
      PetscCall(DMPlexGetConeOrientation(dm, point, &ornt));
      for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
        PetscCall(PetscFindInt(cone[c], numSubPoints[d-1], subpoints[d-1], &subc));
        if (subc >= 0) {
          coneNew[coneSizeNew] = firstSubPoint[d-1] + subc;
          orntNew[coneSizeNew] = ornt[c];
          ++coneSizeNew;
        }
      }
      PetscCheckFalse(coneSizeNew != subconeSize,comm, PETSC_ERR_PLIB, "Number of cone points located %d does not match subcone size %d", coneSizeNew, subconeSize);
      PetscCall(DMPlexSetCone(subdm, subpoint, coneNew));
      PetscCall(DMPlexSetConeOrientation(subdm, subpoint, orntNew));
      if (fornt < 0) PetscCall(DMPlexOrientPoint(subdm, subpoint, fornt));
    }
  }
  PetscCall(PetscFree2(coneNew,orntNew));
  PetscCall(DMPlexSymmetrize(subdm));
  PetscCall(DMPlexStratify(subdm));
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     cdim, numComp, coordSize;
    const char  *name;

    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(subdm, &subCoordSection));
    PetscCall(PetscSectionSetNumFields(subCoordSection, 1));
    PetscCall(PetscSectionGetFieldComponents(coordSection, 0, &numComp));
    PetscCall(PetscSectionSetFieldComponents(subCoordSection, 0, numComp));
    PetscCall(PetscSectionSetChart(subCoordSection, firstSubPoint[0], firstSubPoint[0]+numSubPoints[0]));
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt       dof;

      PetscCall(PetscSectionGetDof(coordSection, vertex, &dof));
      PetscCall(PetscSectionSetDof(subCoordSection, subvertex, dof));
      PetscCall(PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof));
    }
    PetscCall(PetscSectionSetUp(subCoordSection));
    PetscCall(PetscSectionGetStorageSize(subCoordSection, &coordSize));
    PetscCall(VecCreate(PETSC_COMM_SELF, &subCoordinates));
    PetscCall(PetscObjectGetName((PetscObject)coordinates,&name));
    PetscCall(PetscObjectSetName((PetscObject)subCoordinates,name));
    PetscCall(VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(subCoordinates, cdim));
    PetscCall(VecSetType(subCoordinates,VECSTANDARD));
    PetscCall(VecGetArray(coordinates,    &coords));
    PetscCall(VecGetArray(subCoordinates, &subCoords));
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt dof, off, sdof, soff, d;

      PetscCall(PetscSectionGetDof(coordSection, vertex, &dof));
      PetscCall(PetscSectionGetOffset(coordSection, vertex, &off));
      PetscCall(PetscSectionGetDof(subCoordSection, subvertex, &sdof));
      PetscCall(PetscSectionGetOffset(subCoordSection, subvertex, &soff));
      PetscCheckFalse(dof != sdof,comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    PetscCall(VecRestoreArray(coordinates,    &coords));
    PetscCall(VecRestoreArray(subCoordinates, &subCoords));
    PetscCall(DMSetCoordinatesLocal(subdm, subCoordinates));
    PetscCall(VecDestroy(&subCoordinates));
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

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
    PetscCall(DMGetPointSF(dm, &sfPoint));
    PetscCall(DMGetPointSF(subdm, &sfPointSub));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(DMPlexGetChart(subdm, NULL, &numSubroots));
    PetscCall(DMPlexGetSubpointIS(subdm, &subpIS));
    if (subpIS) {
      PetscCall(ISGetIndices(subpIS, &subpoints));
      PetscCall(ISGetLocalSize(subpIS, &numSubpoints));
    }
    PetscCall(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
    if (numRoots >= 0) {
      PetscCall(PetscMalloc2(pEnd-pStart,&newLocalPoints,numRoots,&newOwners));
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
      PetscCall(PetscSFReduceBegin(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      PetscCall(PetscSFReduceEnd(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      PetscCall(PetscSFBcastBegin(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      PetscCall(PetscMalloc1(numSubleaves, &slocalPoints));
      PetscCall(PetscMalloc1(numSubleaves, &sremotePoints));
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
      PetscCall(PetscFree2(newLocalPoints,newOwners));
      PetscCall(PetscSFSetGraph(sfPointSub, numSubroots, sl, slocalPoints, PETSC_OWN_POINTER, sremotePoints, PETSC_OWN_POINTER));
    }
    if (subpIS) {
      PetscCall(ISRestoreIndices(subpIS, &subpoints));
    }
  }
  /* Filter labels */
  PetscCall(DMPlexFilterLabels_Internal(dm, numSubPoints, subpoints, firstSubPoint, subdm));
  /* Cleanup */
  for (d = 0; d <= dim; ++d) {
    if (subpointIS[d]) PetscCall(ISRestoreIndices(subpointIS[d], &subpoints[d]));
    PetscCall(ISDestroy(&subpointIS[d]));
  }
  PetscCall(PetscFree4(numSubPoints,firstSubPoint,subpointIS,subpoints));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, PetscBool markedFaces, DM subdm)
{
  PetscFunctionBegin;
  PetscCall(DMPlexCreateSubmeshGeneric_Interpolated(dm, vertexLabel, value, markedFaces, PETSC_FALSE, 1, subdm));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), subdm));
  PetscCall(DMSetType(*subdm, DMPLEX));
  PetscCall(DMSetDimension(*subdm, dim-1));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMSetCoordinateDim(*subdm, cdim));
  PetscCall(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheckFalse(interpolated == DMPLEX_INTERPOLATED_PARTIAL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated) {
    PetscCall(DMPlexCreateSubmesh_Interpolated(dm, vertexLabel, value, markedFaces, *subdm));
  } else {
    PetscCall(DMPlexCreateSubmesh_Uninterpolated(dm, vertexLabel, value, *subdm));
  }
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, *subdm));
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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  /* Create subpointMap which marks the submesh */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "subpoint_map", &subpointMap));
  PetscCall(DMPlexSetSubpointMap(subdm, subpointMap));
  PetscCall(DMLabelDestroy(&subpointMap));
  PetscCall(DMPlexMarkCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, subpointMap, &numSubFaces, &nFV, &subCells, subdm));
  /* Setup chart */
  PetscCall(DMLabelGetStratumSize(subpointMap, 0, &numSubVertices));
  PetscCall(DMLabelGetStratumSize(subpointMap, 2, &numSubCells));
  PetscCall(DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices));
  PetscCall(DMPlexSetVTKCellHeight(subdm, 1));
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  PetscCall(DMLabelGetStratumIS(subpointMap, 0, &subvertexIS));
  if (subvertexIS) PetscCall(ISGetIndices(subvertexIS, &subVertices));
  for (c = 0; c < numSubCells; ++c) {
    PetscCall(DMPlexSetConeSize(subdm, c, 1));
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    PetscCall(DMPlexSetConeSize(subdm, f, nFV));
  }
  PetscCall(DMSetUp(subdm));
  /* Create face cones */
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  PetscCall(DMGetWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt  cell    = subCells[c];
    const PetscInt  subcell = c;
    const PetscInt *cone, *cells;
    PetscBool       isHybrid;
    PetscInt        numCells, subVertex, p, v;

    PetscCall(DMPlexCellIsHybrid_Internal(dm, cell, &isHybrid));
    if (!isHybrid) continue;
    PetscCall(DMPlexGetCone(dm, cell, &cone));
    for (v = 0; v < nFV; ++v) {
      PetscCall(PetscFindInt(cone[v], numSubVertices, subVertices, &subVertex));
      subface[v] = firstSubVertex+subVertex;
    }
    PetscCall(DMPlexSetCone(subdm, newFacePoint, subface));
    PetscCall(DMPlexSetCone(subdm, subcell, &newFacePoint));
    PetscCall(DMPlexGetJoin(dm, nFV, cone, &numCells, &cells));
    /* Not true in parallel
    PetscCheckFalse(numCells != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells"); */
    for (p = 0; p < numCells; ++p) {
      PetscInt  negsubcell;
      PetscBool isHybrid;

      PetscCall(DMPlexCellIsHybrid_Internal(dm, cells[p], &isHybrid));
      if (isHybrid) continue;
      /* I know this is a crap search */
      for (negsubcell = 0; negsubcell < numSubCells; ++negsubcell) {
        if (subCells[negsubcell] == cells[p]) break;
      }
      PetscCheckFalse(negsubcell == numSubCells,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find negative face neighbor for cohesive cell %d", cell);
      PetscCall(DMPlexSetCone(subdm, negsubcell, &newFacePoint));
    }
    PetscCall(DMPlexRestoreJoin(dm, nFV, cone, &numCells, &cells));
    ++newFacePoint;
  }
  PetscCall(DMRestoreWorkArray(subdm, maxConeSize, MPIU_INT, (void**) &subface));
  PetscCall(DMPlexSymmetrize(subdm));
  PetscCall(DMPlexStratify(subdm));
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     cdim, numComp, coordSize, v;
    const char  *name;

    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(subdm, &subCoordSection));
    PetscCall(PetscSectionSetNumFields(subCoordSection, 1));
    PetscCall(PetscSectionGetFieldComponents(coordSection, 0, &numComp));
    PetscCall(PetscSectionSetFieldComponents(subCoordSection, 0, numComp));
    PetscCall(PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices));
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      PetscCall(PetscSectionGetDof(coordSection, vertex, &dof));
      PetscCall(PetscSectionSetDof(subCoordSection, subvertex, dof));
      PetscCall(PetscSectionSetFieldDof(subCoordSection, subvertex, 0, dof));
    }
    PetscCall(PetscSectionSetUp(subCoordSection));
    PetscCall(PetscSectionGetStorageSize(subCoordSection, &coordSize));
    PetscCall(VecCreate(PETSC_COMM_SELF, &subCoordinates));
    PetscCall(PetscObjectGetName((PetscObject)coordinates,&name));
    PetscCall(PetscObjectSetName((PetscObject)subCoordinates,name));
    PetscCall(VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(subCoordinates, cdim));
    PetscCall(VecSetType(subCoordinates,VECSTANDARD));
    PetscCall(VecGetArray(coordinates,    &coords));
    PetscCall(VecGetArray(subCoordinates, &subCoords));
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof, off, sdof, soff, d;

      PetscCall(PetscSectionGetDof(coordSection, vertex, &dof));
      PetscCall(PetscSectionGetOffset(coordSection, vertex, &off));
      PetscCall(PetscSectionGetDof(subCoordSection, subvertex, &sdof));
      PetscCall(PetscSectionGetOffset(subCoordSection, subvertex, &soff));
      PetscCheckFalse(dof != sdof,comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    PetscCall(VecRestoreArray(coordinates,    &coords));
    PetscCall(VecRestoreArray(subCoordinates, &subCoords));
    PetscCall(DMSetCoordinatesLocal(subdm, subCoordinates));
    PetscCall(VecDestroy(&subCoordinates));
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

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
    PetscCall(DMGetPointSF(dm, &sfPoint));
    PetscCall(DMGetPointSF(subdm, &sfPointSub));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
    if (numRoots >= 0) {
      /* Only vertices should be shared */
      PetscCall(PetscMalloc2(pEnd-pStart,&newLocalPoints,numRoots,&newOwners));
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
      PetscCall(PetscSFReduceBegin(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      PetscCall(PetscSFReduceEnd(sfPoint, MPIU_2INT, newLocalPoints, newOwners, MPI_MAXLOC));
      PetscCall(PetscSFBcastBegin(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sfPoint, MPIU_2INT, newOwners, newLocalPoints,MPI_REPLACE));
      PetscCall(PetscMalloc1(numSubLeaves,    &slocalPoints));
      PetscCall(PetscMalloc1(numSubLeaves, &sremotePoints));
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
      PetscCall(PetscFree2(newLocalPoints,newOwners));
      PetscCheckFalse(sl + ll != numSubLeaves,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatch in number of subleaves %d + %d != %d", sl, ll, numSubLeaves);
      PetscCall(PetscSFSetGraph(sfPointSub, numSubRoots, sl, slocalPoints, PETSC_OWN_POINTER, sremotePoints, PETSC_OWN_POINTER));
    }
  }
  CHKMEMQ;
  /* Cleanup */
  if (subvertexIS) PetscCall(ISRestoreIndices(subvertexIS, &subVertices));
  PetscCall(ISDestroy(&subvertexIS));
  PetscCall(PetscFree(subCells));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCohesiveSubmesh_Interpolated(DM dm, const char labelname[], PetscInt value, DM subdm)
{
  DMLabel        label = NULL;

  PetscFunctionBegin;
  if (labelname) PetscCall(DMGetLabel(dm, labelname, &label));
  PetscCall(DMPlexCreateSubmeshGeneric_Interpolated(dm, label, value, PETSC_FALSE, PETSC_TRUE, 1, subdm));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), subdm));
  PetscCall(DMSetType(*subdm, DMPLEX));
  PetscCall(DMSetDimension(*subdm, dim-1));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMSetCoordinateDim(*subdm, cdim));
  if (depth == dim) {
    PetscCall(DMPlexCreateCohesiveSubmesh_Interpolated(dm, label, value, *subdm));
  } else {
    PetscCall(DMPlexCreateCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, *subdm));
  }
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, *subdm));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), subdm));
  PetscCall(DMSetType(*subdm, DMPLEX));
  PetscCall(DMSetDimension(*subdm, dim));
  /* Extract submesh in place, could be empty on some procs, could have inconsistency if procs do not both extract a shared cell */
  PetscCall(DMPlexCreateSubmeshGeneric_Interpolated(dm, cellLabel, value, PETSC_FALSE, PETSC_FALSE, 0, *subdm));
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, *subdm));
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
  PetscCall(PetscObjectReference((PetscObject) mesh->subpointMap));
  PetscCall(DMLabelDestroy(&tmp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateSubpointIS_Internal(DM dm, IS *subpointIS)
{
  DMLabel        spmap;
  PetscInt       depth, d;

  PetscFunctionBegin;
  PetscCall(DMPlexGetSubpointMap(dm, &spmap));
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (spmap && depth >= 0) {
    DM_Plex  *mesh = (DM_Plex *) dm->data;
    PetscInt *points, *depths;
    PetscInt  pStart, pEnd, p, off;

    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCheck(!pStart,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Submeshes must start the point numbering at 0, not %d", pStart);
    PetscCall(PetscMalloc1(pEnd, &points));
    PetscCall(DMGetWorkArray(dm, depth+1, MPIU_INT, &depths));
    depths[0] = depth;
    depths[1] = 0;
    for (d = 2; d <= depth; ++d) {depths[d] = depth+1 - d;}
    for (d = 0, off = 0; d <= depth; ++d) {
      const PetscInt dep = depths[d];
      PetscInt       depStart, depEnd, n;

      PetscCall(DMPlexGetDepthStratum(dm, dep, &depStart, &depEnd));
      PetscCall(DMLabelGetStratumSize(spmap, dep, &n));
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

        PetscCall(DMLabelGetStratumIS(spmap, dep, &is));
        PetscCall(ISGetIndices(is, &opoints));
        for (p = 0; p < n; ++p, ++off) points[off] = opoints[p];
        PetscCall(ISRestoreIndices(is, &opoints));
        PetscCall(ISDestroy(&is));
      }
    }
    PetscCall(DMRestoreWorkArray(dm, depth+1, MPIU_INT, &depths));
    PetscCheckFalse(off != pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d should be %d", off, pEnd);
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, pEnd, points, PETSC_OWN_POINTER, subpointIS));
    PetscCall(PetscObjectStateGet((PetscObject) spmap, &mesh->subpointState));
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
  PetscCall(DMPlexGetSubpointMap(dm, &spmap));
  PetscCall(PetscObjectStateGet((PetscObject) spmap, &state));
  if (state != mesh->subpointState || !mesh->subpointIS) PetscCall(DMPlexCreateSubpointIS_Internal(dm, &mesh->subpointIS));
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
  PetscCall(DMConvert(dmA, DMPLEX, &plexA));
  PetscCall(DMConvert(dmB, DMPLEX, &plexB));
  PetscCall(DMPlexGetChart(plexA, &pStartA, &pEndA));
  PetscCall(DMPlexGetChart(plexB, &pStartB, &pEndB));
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
  PetscCall(DMPlexGetSubpointMap(sdm, &spmap));
  if (!spmap) goto end;
  /* TODO Check the space mapped to by subpointMap is same size as dm */
  if (NpA > NpB) {
    *rel = DM_ENC_SUPERMESH;
  } else {
    *rel = DM_ENC_SUBMESH;
  }
  end:
  PetscCall(DMDestroy(&plexA));
  PetscCall(DMDestroy(&plexB));
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
    PetscCall(DMPlexGetSubpointIS(sdm, &subpointIS));
    PetscCall(ISGetIndices(subpointIS, &subpoints));
    *pA  = subpoints[pB];
    PetscCall(ISRestoreIndices(subpointIS, &subpoints));
    break;
    case DM_ENC_SUBMESH:
    sdm  = dmA;
    PetscCall(DMPlexGetSubpointIS(sdm, &subpointIS));
    PetscCall(ISGetLocalSize(subpointIS, &numSubpoints));
    PetscCall(ISGetIndices(subpointIS, &subpoints));
    PetscCall(PetscFindInt(pB, numSubpoints, subpoints, pA));
    if (*pA < 0) {
      PetscCall(DMViewFromOptions(dmA, NULL, "-dm_enc_A_view"));
      PetscCall(DMViewFromOptions(dmB, NULL, "-dm_enc_B_view"));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d not found in submesh", pB);
    }
    PetscCall(ISRestoreIndices(subpointIS, &subpoints));
    break;
    case DM_ENC_EQUALITY:
    case DM_ENC_NONE:
    *pA = pB;break;
    case DM_ENC_UNKNOWN:
    {
      DMEnclosureType enc;

      PetscCall(DMGetEnclosureRelation(dmA, dmB, &enc));
      PetscCall(DMGetEnclosurePoint(dmA, dmB, enc, pB, pA));
    }
    break;
    default: SETERRQ(PetscObjectComm((PetscObject) dmA), PETSC_ERR_ARG_OUTOFRANGE, "Invalid enclosure type %d", (int) etype);
  }
  PetscFunctionReturn(0);
}

#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkBoundaryFaces"
/*@
  DMPlexMarkBoundaryFaces - Mark all faces on the boundary

  Not Collective

  Input Parameter:
. dm - The original DM

  Output Parameter:
. label - The DMLabel marking boundary faces with value 1

  Level: developer

.seealso: DMLabelCreate(), DMPlexCreateLabel()
@*/
PetscErrorCode DMPlexMarkBoundaryFaces(DM dm, DMLabel label)
{
  PetscInt       fStart, fEnd, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {
    PetscInt supportSize;

    ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
    if (supportSize == 1) {
      ierr = DMLabelSetValue(label, f, 1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexLabelComplete"
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
  IS              valueIS;
  const PetscInt *values;
  PetscInt        numValues, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
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
      PetscInt  closureSize, c;

      ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        ierr = DMLabelSetValue(label, closure[c], values[v]);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexShiftPoint_Internal"
PETSC_STATIC_INLINE PetscInt DMPlexShiftPoint_Internal(PetscInt p, PetscInt depth, PetscInt depthEnd[], PetscInt depthShift[])
{
  if (depth < 0) return p;
  /* Cells    */ if (p < depthEnd[depth])   return p;
  /* Vertices */ if (p < depthEnd[0])       return p + depthShift[depth];
  /* Faces    */ if (p < depthEnd[depth-1]) return p + depthShift[depth] + depthShift[0];
  /* Edges    */                            return p + depthShift[depth] + depthShift[0] + depthShift[depth-1];
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexShiftSizes_Internal"
static PetscErrorCode DMPlexShiftSizes_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt      *depthEnd;
  PetscInt       depth = 0, d, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth < 0) PetscFunctionReturn(0);
  ierr = PetscMalloc((depth+1) * sizeof(PetscInt), &depthEnd);CHKERRQ(ierr);
  /* Step 1: Expand chart */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    pEnd += depthShift[d];
    ierr  = DMPlexGetDepthStratum(dm, d, NULL, &depthEnd[d]);CHKERRQ(ierr);
  }
  ierr = DMPlexSetChart(dmNew, pStart, pEnd);CHKERRQ(ierr);
  /* Step 2: Set cone and support sizes */
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt newp = DMPlexShiftPoint_Internal(p, depth, depthEnd, depthShift);
      PetscInt size;

      ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(dmNew, newp, size);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(dmNew, newp, size);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(depthEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexShiftPoints_Internal"
static PetscErrorCode DMPlexShiftPoints_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt      *depthEnd, *newpoints;
  PetscInt       depth = 0, d, maxConeSize, maxSupportSize, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth < 0) PetscFunctionReturn(0);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1,PetscInt,&depthEnd,PetscMax(maxConeSize, maxSupportSize),PetscInt,&newpoints);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, NULL, &depthEnd[d]);CHKERRQ(ierr);
  }
  /* Step 5: Set cones and supports */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *points = NULL, *orientations = NULL;
    PetscInt        size, i, newp = DMPlexShiftPoint_Internal(p, depth, depthEnd, depthShift);

    ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &points);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &orientations);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthEnd, depthShift);
    }
    ierr = DMPlexSetCone(dmNew, newp, newpoints);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dmNew, newp, orientations);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &points);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) {
      newpoints[i] = DMPlexShiftPoint_Internal(points[i], depth, depthEnd, depthShift);
    }
    ierr = DMPlexSetSupport(dmNew, newp, newpoints);CHKERRQ(ierr);
  }
  ierr = PetscFree2(depthEnd,newpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexShiftCoordinates_Internal"
static PetscErrorCode DMPlexShiftCoordinates_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscSection   coordSection, newCoordSection;
  Vec            coordinates, newCoordinates;
  PetscScalar   *coords, *newCoords;
  PetscInt      *depthEnd, coordSize;
  PetscInt       dim, depth = 0, d, vStart, vEnd, vStartNew, vEndNew, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((depth+1) * sizeof(PetscInt), &depthEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, NULL, &depthEnd[d]);CHKERRQ(ierr);
  }
  /* Step 8: Convert coordinates */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmNew, 0, &vStartNew, &vEndNew);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(newCoordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(newCoordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newCoordSection, vStartNew, vEndNew);CHKERRQ(ierr);
  for (v = vStartNew; v < vEndNew; ++v) {
    ierr = PetscSectionSetDof(newCoordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(newCoordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(newCoordSection);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(dmNew, newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newCoordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), &newCoordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newCoordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(newCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newCoordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmNew, newCoordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt dof, off, noff, d;

    ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(newCoordSection, DMPlexShiftPoint_Internal(v, depth, depthEnd, depthShift), &noff);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) {
      newCoords[noff+d] = coords[off+d];
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(newCoordinates, &newCoords);CHKERRQ(ierr);
  ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&newCoordSection);CHKERRQ(ierr);
  ierr = PetscFree(depthEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexShiftSF_Internal"
static PetscErrorCode DMPlexShiftSF_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscInt          *depthEnd;
  PetscInt           depth = 0, d;
  PetscSF            sfPoint, sfPointNew;
  const PetscSFNode *remotePoints;
  PetscSFNode       *gremotePoints;
  const PetscInt    *localPoints;
  PetscInt          *glocalPoints, *newLocation, *newRemoteLocation;
  PetscInt           numRoots, numLeaves, l, pStart, pEnd, totShift = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((depth+1) * sizeof(PetscInt), &depthEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    totShift += depthShift[d];
    ierr      = DMPlexGetDepthStratum(dm, d, NULL, &depthEnd[d]);CHKERRQ(ierr);
  }
  /* Step 9: Convert pointSF */
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMGetPointSF(dmNew, &sfPointNew);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots >= 0) {
    ierr = PetscMalloc2(numRoots,PetscInt,&newLocation,pEnd-pStart,PetscInt,&newRemoteLocation);CHKERRQ(ierr);
    for (l=0; l<numRoots; l++) newLocation[l] = DMPlexShiftPoint_Internal(l, depth, depthEnd, depthShift);
    ierr = PetscSFBcastBegin(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscMalloc(numLeaves * sizeof(PetscInt),    &glocalPoints);CHKERRQ(ierr);
    ierr = PetscMalloc(numLeaves * sizeof(PetscSFNode), &gremotePoints);CHKERRQ(ierr);
    for (l = 0; l < numLeaves; ++l) {
      glocalPoints[l]        = DMPlexShiftPoint_Internal(localPoints[l], depth, depthEnd, depthShift);
      gremotePoints[l].rank  = remotePoints[l].rank;
      gremotePoints[l].index = newRemoteLocation[localPoints[l]];
    }
    ierr = PetscFree2(newLocation,newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfPointNew, numRoots + totShift, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  ierr = PetscFree(depthEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexShiftLabels_Internal"
static PetscErrorCode DMPlexShiftLabels_Internal(DM dm, PetscInt depthShift[], DM dmNew)
{
  PetscSF            sfPoint;
  DMLabel            vtkLabel, ghostLabel;
  PetscInt          *depthEnd;
  const PetscSFNode *leafRemote;
  const PetscInt    *leafLocal;
  PetscInt           depth = 0, d, numLeaves, numLabels, l, cStart, cEnd, c, fStart, fEnd, f;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((depth+1) * sizeof(PetscInt), &depthEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, NULL, &depthEnd[d]);CHKERRQ(ierr);
  }
  /* Step 10: Convert labels */
  ierr = DMPlexGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, newlabel;
    const char     *lname;
    PetscBool       isDepth;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, val;

    ierr = DMPlexGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = DMPlexCreateLabel(dmNew, lname);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, lname, &label);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dmNew, lname, &newlabel);CHKERRQ(ierr);
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
        const PetscInt newpoint = DMPlexShiftPoint_Internal(points[p], depth, depthEnd, depthShift);

        ierr = DMLabelSetValue(newlabel, newpoint, values[val]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  ierr = PetscFree(depthEnd);CHKERRQ(ierr);
  /* Step 11: Make label for output (vtk) and to mark ghost points (ghost) */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &numLeaves, &leafLocal, &leafRemote);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(dmNew, "vtk");CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(dmNew, "ghost");CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dmNew, "vtk", &vtkLabel);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dmNew, "ghost", &ghostLabel);CHKERRQ(ierr);
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
  if (0) {
    ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMLabelView(vtkLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
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
      if (!vA && !vB) {ierr = DMLabelSetValue(ghostLabel, f, 1);CHKERRQ(ierr);}
    }
  }
  if (0) {
    ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMLabelView(ghostLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexConstructGhostCells_Internal"
static PetscErrorCode DMPlexConstructGhostCells_Internal(DM dm, DMLabel label, PetscInt *numGhostCells, DM gdm)
{
  IS              valueIS;
  const PetscInt *values;
  PetscInt       *depthShift;
  PetscInt        depth = 0, numFS, fs, ghostCell, cEnd, c;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Count ghost cells */
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(valueIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);

  *numGhostCells = 0;
  for (fs = 0; fs < numFS; ++fs) {
    PetscInt numBdFaces;

    ierr = DMLabelGetStratumSize(label, values[fs], &numBdFaces);CHKERRQ(ierr);

    *numGhostCells += numBdFaces;
  }
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((depth+1) * sizeof(PetscInt), &depthShift);CHKERRQ(ierr);
  ierr = PetscMemzero(depthShift, (depth+1) * sizeof(PetscInt));CHKERRQ(ierr);
  if (depth >= 0) depthShift[depth] = *numGhostCells;
  ierr = DMPlexShiftSizes_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  /* Step 3: Set cone/support sizes for new points */
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  for (c = cEnd; c < cEnd + *numGhostCells; ++c) {
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
      PetscInt size;

      ierr = DMPlexGetSupportSize(dm, faces[f], &size);CHKERRQ(ierr);
      if (size != 1) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "DM has boundary face %d with %d support cells", faces[f], size);
      ierr = DMPlexSetSupportSize(gdm, faces[f] + *numGhostCells, 2);CHKERRQ(ierr);
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
    for (f = 0; f < numFaces; ++f, ++ghostCell) {
      PetscInt newFace = faces[f] + *numGhostCells;

      ierr = DMPlexSetCone(gdm, ghostCell, &newFace);CHKERRQ(ierr);
      ierr = DMPlexInsertSupport(gdm, newFace, 1, ghostCell);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  /* Step 7: Stratify */
  ierr = DMPlexStratify(gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftCoordinates_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftSF_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = DMPlexShiftLabels_Internal(dm, depthShift, gdm);CHKERRQ(ierr);
  ierr = PetscFree(depthShift);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexConstructGhostCells"
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
*/
PetscErrorCode DMPlexConstructGhostCells(DM dm, const char labelName[], PetscInt *numGhostCells, DM *dmGhosted)
{
  DM             gdm;
  DMLabel        label;
  const char    *name = labelName ? labelName : "Face Sets";
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(numGhostCells, 3);
  PetscValidPointer(dmGhosted, 4);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &gdm);CHKERRQ(ierr);
  ierr = DMSetType(gdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(gdm, dim);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) {
    /* Get label for boundary faces */
    ierr = DMPlexCreateLabel(dm, name);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, label);CHKERRQ(ierr);
  }
  ierr = DMPlexConstructGhostCells_Internal(dm, label, numGhostCells, gdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(gdm);CHKERRQ(ierr);
  *dmGhosted = gdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexConstructCohesiveCells_Internal"
static PetscErrorCode DMPlexConstructCohesiveCells_Internal(DM dm, DMLabel label, DM sdm)
{
  MPI_Comm        comm;
  IS              valueIS, *pointIS;
  const PetscInt *values, **splitPoints;
  PetscSection    coordSection;
  Vec             coordinates;
  PetscScalar    *coords;
  PetscInt       *depthShift, *depthOffset, *pMaxNew, *numSplitPoints, *coneNew, *coneONew, *supportNew;
  PetscInt        shift = 100, depth = 0, dep, dim, d, numSP = 0, sp, maxConeSize, maxSupportSize, numLabels, vStart, vEnd, pEnd, p, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  /* Count split points and add cohesive cells */
  if (label) {
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numSP);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = PetscMalloc6(depth+1,PetscInt,&depthShift,depth+1,PetscInt,&depthOffset,depth+1,PetscInt,&pMaxNew,maxConeSize*3,PetscInt,&coneNew,maxConeSize*3,PetscInt,&coneONew,maxSupportSize,PetscInt,&supportNew);CHKERRQ(ierr);
  ierr = PetscMalloc3(depth+1,IS,&pointIS,depth+1,PetscInt,&numSplitPoints,depth+1,const PetscInt*,&splitPoints);CHKERRQ(ierr);
  ierr = PetscMemzero(depthShift, (depth+1) * sizeof(PetscInt));CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr              = DMPlexGetDepthStratum(dm, d, NULL, &pMaxNew[d]);CHKERRQ(ierr);
    numSplitPoints[d] = 0;
    splitPoints[d]    = NULL;
    pointIS[d]        = NULL;
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    ierr = DMLabelGetStratumSize(label, dep, &depthShift[dep]);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, dep, &pointIS[dep]);CHKERRQ(ierr);
    if (pointIS[dep]) {
      ierr = ISGetLocalSize(pointIS[dep], &numSplitPoints[dep]);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS[dep], &splitPoints[dep]);CHKERRQ(ierr);
    }
  }
  if (depth >= 0) {
    /* Calculate number of additional points */
    depthShift[depth] = depthShift[depth-1]; /* There is a cohesive cell for every split face   */
    depthShift[1]    += depthShift[0];       /* There is a cohesive edge for every split vertex */
    /* Calculate hybrid bound for each dimension */
    pMaxNew[0] += depthShift[depth];
    if (depth > 1) pMaxNew[dim-1] += depthShift[depth] + depthShift[0];
    if (depth > 2) pMaxNew[1]     += depthShift[depth] + depthShift[0] + depthShift[dim-1];

    /* Calculate point offset for each dimension */
    depthOffset[depth] = 0;
    depthOffset[0]     = depthOffset[depth] + depthShift[depth];
    if (depth > 1) depthOffset[dim-1] = depthOffset[0]     + depthShift[0];
    if (depth > 2) depthOffset[1]     = depthOffset[dim-1] + depthShift[dim-1];
  }
  ierr = DMPlexShiftSizes_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  /* Step 3: Set cone/support sizes for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = depthOffset[dep] + oldp;
      const PetscInt  splitp = pMaxNew[dep] + p;
      const PetscInt *support;
      PetscInt        coneSize, supportSize, q, e;

      ierr = DMPlexGetConeSize(dm, oldp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, splitp, coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, oldp, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(sdm, splitp, supportSize);CHKERRQ(ierr);
      if (dep == depth-1) {
        const PetscInt ccell = pMaxNew[depth] + p;
        /* Add cohesive cells, they are prisms */
        ierr = DMPlexSetConeSize(sdm, ccell, 2 + coneSize);CHKERRQ(ierr);
      } else if (dep == 0) {
        const PetscInt cedge = pMaxNew[1] + (depthShift[1] - depthShift[0]) + p;

        ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
        /* Split old vertex: Edges in old split faces and new cohesive edge */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == 1) || (val == (shift + 1))) ++q;
        }
        ierr = DMPlexSetSupportSize(sdm, newp, q+1);CHKERRQ(ierr);
        /* Split new vertex: Edges in new split faces and new cohesive edge */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == 1) || (val == -(shift + 1))) ++q;
        }
        ierr = DMPlexSetSupportSize(sdm, splitp, q+1);CHKERRQ(ierr);
        /* Add cohesive edges */
        ierr = DMPlexSetConeSize(sdm, cedge, 2);CHKERRQ(ierr);
        /* Punt for now on support, you loop over closure, extract faces, check which ones are in the label */
      } else if (dep == dim-2) {
        ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
        /* Split old edge: Faces in positive side cells and old split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == dim-1) || (val == (shift + dim-1))) ++q;
        }
        ierr = DMPlexSetSupportSize(sdm, newp, q);CHKERRQ(ierr);
        /* Split new edge: Faces in negative side cells and new split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == dim-1) || (val == -(shift + dim-1))) ++q;
        }
        ierr = DMPlexSetSupportSize(sdm, splitp, q);CHKERRQ(ierr);
      }
    }
  }
  /* Step 4: Setup split DM */
  ierr = DMSetUp(sdm);CHKERRQ(ierr);
  ierr = DMPlexShiftPoints_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  /* Step 6: Set cones and supports for new points */
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt  oldp   = splitPoints[dep][p];
      const PetscInt  newp   = depthOffset[dep] + oldp;
      const PetscInt  splitp = pMaxNew[dep] + p;
      const PetscInt *cone, *support, *ornt;
      PetscInt        coneSize, supportSize, q, v, e, s;

      ierr = DMPlexGetConeSize(dm, oldp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, oldp, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, oldp, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, oldp, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, oldp, &support);CHKERRQ(ierr);
      if (dep == depth-1) {
        const PetscInt  ccell = pMaxNew[depth] + p;
        const PetscInt *supportF;

        /* Split face:       copy in old face to new face to start */
        ierr = DMPlexGetSupport(sdm, newp,  &supportF);CHKERRQ(ierr);
        ierr = DMPlexSetSupport(sdm, splitp, supportF);CHKERRQ(ierr);
        /* Split old face:   old vertices/edges in cone so no change */
        /* Split new face:   new vertices/edges in cone */
        for (q = 0; q < coneSize; ++q) {
          ierr = PetscFindInt(cone[q], numSplitPoints[dim-2], splitPoints[dim-2], &v);CHKERRQ(ierr);

          coneNew[2+q] = pMaxNew[dim-2] + v;
        }
        ierr = DMPlexSetCone(sdm, splitp, &coneNew[2]);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(sdm, splitp, ornt);CHKERRQ(ierr);
        /* Face support */
        for (s = 0; s < supportSize; ++s) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[s], &val);CHKERRQ(ierr);
          if (val < 0) {
            /* Split old face:   Replace negative side cell with cohesive cell */
             ierr = DMPlexInsertSupport(sdm, newp, s, ccell);CHKERRQ(ierr);
          } else {
            /* Split new face:   Replace positive side cell with cohesive cell */
            ierr = DMPlexInsertSupport(sdm, splitp, s, ccell);CHKERRQ(ierr);
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
        /* Cohesive cell:    Old and new split face, then new cohesive edges */
        coneNew[0] = newp;   /* Extracted negative side orientation above */
        coneNew[1] = splitp; coneONew[1] = coneONew[0];
        if (dim > 2) {
          PetscInt *closure = NULL, closureSize, cl;

          ierr = DMPlexGetTransitiveClosure(dm, oldp, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          for (cl = 0, q = 0; cl < closureSize*2; cl += 2) {
            const PetscInt clp = closure[cl];

            if ((clp >= vStart) && (clp < vEnd)) {
              ierr = PetscFindInt(clp, numSplitPoints[0], splitPoints[0], &v);CHKERRQ(ierr);
              coneNew[2+q]  = pMaxNew[1] + (depthShift[1] - depthShift[0]) + v;
              coneONew[2+q] = 0;
              ++q;
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm, oldp, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          if (q != coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of split face vertices %d should be %d", q, coneSize);
        } else {
          for (q = 0; q < coneSize; ++q) {
            coneNew[2+q]  = (pMaxNew[1] - pMaxNew[dim-2]) + (depthShift[1] - depthShift[0]) + coneNew[2+q];
            coneONew[2+q] = 0;
          }
        }
        ierr = DMPlexSetCone(sdm, ccell, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(sdm, ccell, coneONew);CHKERRQ(ierr);
      } else if (dep == 0) {
        const PetscInt cedge = pMaxNew[1] + (depthShift[1] - depthShift[0]) + p;

        /* Split old vertex: Edges in old split faces and new cohesive edge */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == 1) || (val == (shift + 1))) {
            supportNew[q++] = depthOffset[1] + support[e];
          }
        }
        supportNew[q] = cedge;

        ierr = DMPlexSetSupport(sdm, newp, supportNew);CHKERRQ(ierr);
        /* Split new vertex: Edges in new split faces and new cohesive edge */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val, edge;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == 1) {
            ierr = PetscFindInt(support[e], numSplitPoints[1], splitPoints[1], &edge);CHKERRQ(ierr);
            if (edge < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Edge %d is not a split edge", support[e]);
            supportNew[q++] = pMaxNew[1] + edge;
          } else if (val == -(shift + 1)) {
            supportNew[q++] = depthOffset[1] + support[e];
          }
        }
        supportNew[q] = cedge;
        ierr          = DMPlexSetSupport(sdm, splitp, supportNew);CHKERRQ(ierr);
        /* Cohesive edge:    Old and new split vertex, punting on support */
        coneNew[0] = newp;
        coneNew[1] = splitp;
        ierr       = DMPlexSetCone(sdm, cedge, coneNew);CHKERRQ(ierr);
      } else if (dep == dim-2) {
        /* Split old edge:   old vertices in cone so no change */
        /* Split new edge:   new vertices in cone */
        for (q = 0; q < coneSize; ++q) {
          ierr = PetscFindInt(cone[q], numSplitPoints[dim-3], splitPoints[dim-3], &v);CHKERRQ(ierr);

          coneNew[q] = pMaxNew[dim-3] + v;
        }
        ierr = DMPlexSetCone(sdm, splitp, coneNew);CHKERRQ(ierr);
        /* Split old edge: Faces in positive side cells and old split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if ((val == dim-1) || (val == (shift + dim-1))) {
            supportNew[q++] = depthOffset[dim-1] + support[e];
          }
        }
        ierr = DMPlexSetSupport(sdm, newp, supportNew);CHKERRQ(ierr);
        /* Split new edge: Faces in negative side cells and new split faces */
        for (e = 0, q = 0; e < supportSize; ++e) {
          PetscInt val, face;

          ierr = DMLabelGetValue(label, support[e], &val);CHKERRQ(ierr);
          if (val == dim-1) {
            ierr = PetscFindInt(support[e], numSplitPoints[dim-1], splitPoints[dim-1], &face);CHKERRQ(ierr);
            if (face < 0) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Face %d is not a split face", support[e]);
            supportNew[q++] = pMaxNew[dim-1] + face;
          } else if (val == -(shift + dim-1)) {
            supportNew[q++] = depthOffset[dim-1] + support[e];
          }
        }
        ierr = DMPlexSetSupport(sdm, splitp, supportNew);CHKERRQ(ierr);
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
      const PetscInt  newp = depthOffset[dep] + oldp;
      const PetscInt *cone;
      PetscInt        coneSize, c;
      /* PetscBool       replaced = PETSC_FALSE; */

      /* Negative edge: replace split vertex */
      /* Negative cell: replace split face */
      ierr = DMPlexGetConeSize(sdm, newp, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(sdm, newp, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        const PetscInt coldp = cone[c] - depthOffset[dep-1];
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
  /* Step 7: Stratify */
  ierr = DMPlexStratify(sdm);CHKERRQ(ierr);
  /* Step 8: Coordinates */
  ierr = DMPlexShiftCoordinates_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(sdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(sdm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < (numSplitPoints ? numSplitPoints[0] : 0); ++v) {
    const PetscInt newp   = depthOffset[0] + splitPoints[0][v];
    const PetscInt splitp = pMaxNew[0] + v;
    PetscInt       dof, off, soff, d;

    ierr = PetscSectionGetDof(coordSection, newp, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, newp, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, splitp, &soff);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) coords[soff+d] = coords[off+d];
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  /* Step 9: SF, if I can figure this out we can split the mesh in parallel */
  ierr = DMPlexShiftSF_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  /* Step 10: Labels */
  ierr = DMPlexShiftLabels_Internal(dm, depthShift, sdm);CHKERRQ(ierr);
  ierr = DMPlexGetNumLabels(sdm, &numLabels);CHKERRQ(ierr);
  for (dep = 0; dep <= depth; ++dep) {
    for (p = 0; p < numSplitPoints[dep]; ++p) {
      const PetscInt newp   = depthOffset[dep] + splitPoints[dep][p];
      const PetscInt splitp = pMaxNew[dep] + p;
      PetscInt       l;

      for (l = 0; l < numLabels; ++l) {
        DMLabel     mlabel;
        const char *lname;
        PetscInt    val;
        PetscBool   isDepth;

        ierr = DMPlexGetLabelName(sdm, l, &lname);CHKERRQ(ierr);
        ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = DMPlexGetLabel(sdm, lname, &mlabel);CHKERRQ(ierr);
        ierr = DMLabelGetValue(mlabel, newp, &val);CHKERRQ(ierr);
        if (val >= 0) {
          ierr = DMLabelSetValue(mlabel, splitp, val);CHKERRQ(ierr);
#if 0
          /* Do not put cohesive edges into the label */
          if (dep == 0) {
            const PetscInt cedge = pMaxNew[1] + (depthShift[1] - depthShift[0]) + p;
            ierr = DMLabelSetValue(mlabel, cedge, val);CHKERRQ(ierr);
          }
#endif
        }
      }
    }
  }
  for (sp = 0; sp < numSP; ++sp) {
    const PetscInt dep = values[sp];

    if ((dep < 0) || (dep > depth)) continue;
    if (pointIS[dep]) {ierr = ISRestoreIndices(pointIS[dep], &splitPoints[dep]);CHKERRQ(ierr);}
    ierr = ISDestroy(&pointIS[dep]);CHKERRQ(ierr);
  }
  if (label) {
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  ierr = DMPlexGetChart(sdm, NULL, &pEnd);CHKERRQ(ierr);
  if (depth > 0) pMaxNew[0] += depthShift[0];        /* Account for shadow vertices */
  if (depth > 1) pMaxNew[1]  = pEnd - depthShift[0]; /* There is a hybrid edge for every shadow vertex */
  if (depth > 2) pMaxNew[2]  = -1;                   /* There are no hybrid faces */
  ierr = DMPlexSetHybridBounds(sdm, depth >= 0 ? pMaxNew[depth] : PETSC_DETERMINE, depth>1 ? pMaxNew[depth-1] : PETSC_DETERMINE, depth>2 ? pMaxNew[1] : PETSC_DETERMINE, depth >= 0 ? pMaxNew[0] : PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscFree6(depthShift, depthOffset, pMaxNew, coneNew, coneONew, supportNew);CHKERRQ(ierr);
  ierr = PetscFree3(pointIS, numSplitPoints, splitPoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexConstructCohesiveCells"
/*@C
  DMPlexConstructCohesiveCells - Construct cohesive cells which split the face along an internal interface

  Collective on dm

  Input Parameters:
+ dm - The original DM
- labelName - The label specifying the boundary faces (this could be auto-generated)

  Output Parameters:
- dmSplit - The new DM

  Level: developer

.seealso: DMCreate(), DMPlexLabelCohesiveComplete()
@*/
PetscErrorCode DMPlexConstructCohesiveCells(DM dm, DMLabel label, DM *dmSplit)
{
  DM             sdm;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmSplit, 4);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(sdm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
  case 3:
    ierr = DMPlexConstructCohesiveCells_Internal(dm, label, sdm);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct cohesive cells for dimension %d", dim);
  }
  *dmSplit = sdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexLabelCohesiveComplete"
/*@
  DMPlexLabelCohesiveComplete - Starting with a label marking vertices on an internal surface, we add all other mesh pieces
  to complete the surface

  Input Parameters:
+ dm - The DM
. label - A DMLabel marking the surface vertices
. flip  - Flag to flip the submesh normal and replace points on the other side
- subdm - The subDM associated with the label, or NULL

  Output Parameter:
. label - A DMLabel marking all surface points

  Level: developer

.seealso: DMPlexConstructCohesiveCells(), DMPlexLabelComplete()
@*/
PetscErrorCode DMPlexLabelCohesiveComplete(DM dm, DMLabel label, PetscBool flip, DM subdm)
{
  DMLabel         depthLabel;
  IS              dimIS, subpointIS;
  const PetscInt *points, *subpoints;
  const PetscInt  rev   = flip ? -1 : 1;
  PetscInt        shift = 100, dim, dep, cStart, cEnd, numPoints, numSubpoints, p, val;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (subdm) {
    ierr = DMPlexCreateSubpointIS(subdm, &subpointIS);CHKERRQ(ierr);
    if (subpointIS) {
      ierr = ISGetLocalSize(subpointIS, &numSubpoints);CHKERRQ(ierr);
      ierr = ISGetIndices(subpointIS, &subpoints);CHKERRQ(ierr);
    }
  }
  /* Cell orientation for face gives the side of the fault */
  ierr = DMLabelGetStratumIS(label, dim-1, &dimIS);CHKERRQ(ierr);
  if (!dimIS) PetscFunctionReturn(0);
  ierr = ISGetLocalSize(dimIS, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(dimIS, &points);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) {
    const PetscInt *support;
    PetscInt        supportSize, s;

    ierr = DMPlexGetSupportSize(dm, points[p], &supportSize);CHKERRQ(ierr);
    if (supportSize != 2) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Split face %d has %d != 2 supports", points[p], supportSize);
    ierr = DMPlexGetSupport(dm, points[p], &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, c;
      PetscBool       pos = PETSC_TRUE;

      ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        if (cone[c] == points[p]) {
          PetscInt o = ornt[c];

          if (subdm) {
            const PetscInt *subcone, *subornt;
            PetscInt        subpoint, subface, subconeSize, sc;

            ierr = PetscFindInt(support[s], numSubpoints, subpoints, &subpoint);CHKERRQ(ierr);
            ierr = PetscFindInt(points[p],  numSubpoints, subpoints, &subface);CHKERRQ(ierr);
            ierr = DMPlexGetConeSize(subdm, subpoint, &subconeSize);CHKERRQ(ierr);
            ierr = DMPlexGetCone(subdm, subpoint, &subcone);CHKERRQ(ierr);
            ierr = DMPlexGetConeOrientation(subdm, subpoint, &subornt);CHKERRQ(ierr);
            for (sc = 0; sc < subconeSize; ++sc) {
              if (subcone[sc] == subface) {
                o = subornt[0];
                break;
              }
            }
            if (sc >= subconeSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find point %d in cone for subpoint %d", points[p], subpoint);
          }
          if (o >= 0) {
            ierr = DMLabelSetValue(label, support[s],  rev*(shift+dim));CHKERRQ(ierr);
            pos  = rev > 0 ? PETSC_TRUE : PETSC_FALSE;
          } else {
            ierr = DMLabelSetValue(label, support[s], -rev*(shift+dim));CHKERRQ(ierr);
            pos  = rev > 0 ? PETSC_FALSE : PETSC_TRUE;
          }
          break;
        }
      }
      if (c == coneSize) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Cell split face %d support does not have it in the cone", points[p]);
      /* Put faces touching the fault in the label */
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];

        ierr = DMLabelGetValue(label, point, &val);CHKERRQ(ierr);
        if (val == -1) {
          PetscInt *closure = NULL;
          PetscInt  closureSize, cl;

          ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          for (cl = 0; cl < closureSize*2; cl += 2) {
            const PetscInt clp = closure[cl];

            ierr = DMLabelGetValue(label, clp, &val);CHKERRQ(ierr);
            if ((val >= 0) && (val < dim-1)) {
              ierr = DMLabelSetValue(label, point, pos == PETSC_TRUE ? shift+dim-1 : -(shift+dim-1));CHKERRQ(ierr);
              break;
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        }
      }
    }
  }
  if (subdm) {
    if (subpointIS) {ierr = ISRestoreIndices(subpointIS, &subpoints);CHKERRQ(ierr);}
    ierr = ISDestroy(&subpointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(dimIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&dimIS);CHKERRQ(ierr);
  /* Search for other cells/faces/edges connected to the fault by a vertex */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, 0, &dimIS);CHKERRQ(ierr);
  if (!dimIS) PetscFunctionReturn(0);
  ierr = ISGetLocalSize(dimIS, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(dimIS, &points);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) {
    PetscInt *star = NULL;
    PetscInt  starSize, s;
    PetscInt  again = 1;  /* 0: Finished 1: Keep iterating after a change 2: No change */

    /* First mark cells connected to the fault */
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
        again = 2;
        ierr  = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
        ierr  = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
        for (c = 0; c < coneSize; ++c) {
          ierr = DMLabelGetValue(label, cone[c], &val);CHKERRQ(ierr);
          if (val != -1) {
            if (abs(val) < shift) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Face %d on cell %d has an invalid label %d", cone[c], point, val);
            if (val > 0) {
              ierr = DMLabelSetValue(label, point,   shift+dim);CHKERRQ(ierr);
            } else {
              ierr = DMLabelSetValue(label, point, -(shift+dim));CHKERRQ(ierr);
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
        PetscInt  *sstar = NULL;
        PetscInt   sstarSize, ss;
        PetscBool  marked = PETSC_FALSE;

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
        if (!marked) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d could not be classified", point);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(dimIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&dimIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkSubmesh_Uninterpolated"
/* Here we need the explicit assumption that:

     For any marked cell, the marked vertices constitute a single face
*/
static PetscErrorCode DMPlexMarkSubmesh_Uninterpolated(DM dm, DMLabel vertexLabel, PetscInt value, DMLabel subpointMap, PetscInt *numFaces, PetscInt *nFV, DM subdm)
{
  IS               subvertexIS = NULL;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd, *pMax, pSize;
  PetscInt         depth, dim, d, numSubVerticesInitial = 0, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV      = 0;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  pSize = PetscMax(depth, dim) + 1;
  ierr = PetscMalloc3(pSize,PetscInt,&pStart,pSize,PetscInt,&pEnd,pSize,PetscInt,&pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, depth >= 0 ? &pMax[depth] : NULL, depth>1 ? &pMax[depth-1] : NULL, depth>2 ? &pMax[1] : NULL, &pMax[0]);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
    if (pMax[d] >= 0) pEnd[d] = PetscMin(pEnd[d], pMax[d]);
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
  ierr = PetscFree3(pStart,pEnd,pMax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkSubmesh_Interpolated"
static PetscErrorCode DMPlexMarkSubmesh_Interpolated(DM dm, DMLabel vertexLabel, PetscInt value, DMLabel subpointMap, DM subdm)
{
  IS               subvertexIS;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd, *pMax;
  PetscInt         dim, d, numSubVerticesInitial = 0, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim+1,PetscInt,&pStart,dim+1,PetscInt,&pEnd,dim+1,PetscInt,&pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pMax[dim], dim>1 ? &pMax[dim-1] : NULL, dim > 2 ? &pMax[1] : NULL, &pMax[0]);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
    if (pMax[d] >= 0) pEnd[d] = PetscMin(pEnd[d], pMax[d]);
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  ierr = DMLabelGetStratumIS(vertexLabel, value, &subvertexIS);CHKERRQ(ierr);
  if (subvertexIS) {
    ierr = ISGetSize(subvertexIS, &numSubVerticesInitial);CHKERRQ(ierr);
    ierr = ISGetIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = NULL;
    PetscInt       starSize, s, numFaces = 0, f;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      if ((point >= pStart[dim-1]) && (point < pEnd[dim-1])) star[numFaces++] = point;
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
  if (subvertexIS) {
    ierr = ISRestoreIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  ierr = PetscFree3(pStart,pEnd,pMax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkCohesiveSubmesh_Uninterpolated"
static PetscErrorCode DMPlexMarkCohesiveSubmesh_Uninterpolated(DM dm, PetscBool hasLagrange, const char labelname[], PetscInt value, DMLabel subpointMap, PetscInt *numFaces, PetscInt *nFV, PetscInt *subCells[], DM subdm)
{
  DMLabel         label = NULL;
  const PetscInt *cone;
  PetscInt        dim, cMax, cEnd, c, subc = 0, p, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV = 0;
  if (labelname) {ierr = DMPlexGetLabel(dm, labelname, &label);CHKERRQ(ierr);}
  *subCells = NULL;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
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
  *nFV = hasLagrange ? coneSize/3 : coneSize/2;
  ierr = PetscMalloc(*numFaces *2 * sizeof(PetscInt), subCells);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkCohesiveSubmesh_Interpolated"
static PetscErrorCode DMPlexMarkCohesiveSubmesh_Interpolated(DM dm, PetscBool hasLagrange, const char labelname[], PetscInt value, DMLabel subpointMap, DM subdm)
{
  DMLabel        label = NULL;
  PetscInt      *pStart, *pEnd;
  PetscInt       dim, cMax, cEnd, c, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (labelname) {ierr = DMPlexGetLabel(dm, labelname, &label);CHKERRQ(ierr);}
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  if (cMax < 0) PetscFunctionReturn(0);
  ierr = PetscMalloc2(dim+1,PetscInt,&pStart,dim+1,PetscInt,&pEnd);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  for (c = cMax; c < cEnd; ++c) {
    const PetscInt *cone;
    PetscInt       *closure = NULL;
    PetscInt        coneSize, closureSize, cl;

    if (label) {
      PetscInt val;

      ierr = DMLabelGetValue(label, c, &val);CHKERRQ(ierr);
      if (val != value) continue;
    }
    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    if (hasLagrange) {
      if (coneSize != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells");
    } else {
      if (coneSize != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive cells should separate two cells");
    }
    /* Negative face */
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
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
      if (supportSize != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cohesive faces should separate two cells");
      ierr = DMPlexGetSupport(dm, cone[cl], &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        ierr = DMLabelSetValue(subpointMap, support[s], dim);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree2(pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetFaceOrientation"
PetscErrorCode DMPlexGetFaceOrientation(DM dm, PetscInt cell, PetscInt numCorners, PetscInt indices[], PetscInt oppositeVertex, PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  MPI_Comm       comm;
  PetscBool      posOrient = PETSC_FALSE;
  const PetscInt debug     = 0;
  PetscInt       cellDim, faceSize, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &cellDim);CHKERRQ(ierr);
  if (debug) {PetscPrintf(comm, "cellDim: %d numCorners: %d\n", cellDim, numCorners);CHKERRQ(ierr);}

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

    faceSize = faceSizeTri;
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

    faceSize = faceSizeQuad;
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

    faceSize = faceSizeHex;
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

    faceSize = faceSizeTet;
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

    faceSize = faceSizeQuadHex;
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetOrientedFace"
/*
    Given a cell and a face, as a set of vertices,
      return the oriented face, as a set of vertices, in faceVertices
    The orientation is such that the face normal points out of the cell
*/
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertFace_Internal"
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
    ierr     = DMGetWorkArray(subdm, 1, PETSC_INT, (void **) &faces);CHKERRQ(ierr);
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

    ierr                = DMGetWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), PETSC_INT, &orientedVertices);CHKERRQ(ierr);
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
    ierr = DMRestoreWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), PETSC_INT, &orientedVertices);CHKERRQ(ierr);
    ++(*newFacePoint);
  }
#if 0
  ierr = DMPlexRestoreJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
#else
  ierr = DMRestoreWorkArray(subdm, 1, PETSC_INT, (void **) &faces);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubmesh_Uninterpolated"
static PetscErrorCode DMPlexCreateSubmesh_Uninterpolated(DM dm, const char vertexLabelName[], PetscInt value, DM subdm)
{
  MPI_Comm        comm;
  DMLabel         vertexLabel, subpointMap;
  IS              subvertexIS,  subcellIS;
  const PetscInt *subVertices, *subCells;
  PetscInt        numSubVertices, firstSubVertex, numSubCells;
  PetscInt       *subface, maxConeSize, numSubFaces = 0, firstSubFace, newFacePoint, nFV = 0;
  PetscInt        vStart, vEnd, c, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate("subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  if (vertexLabelName) {
    ierr = DMPlexGetLabel(dm, vertexLabelName, &vertexLabel);CHKERRQ(ierr);
    ierr = DMPlexMarkSubmesh_Uninterpolated(dm, vertexLabel, value, subpointMap, &numSubFaces, &nFV, subdm);CHKERRQ(ierr);
  }
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
  ierr = DMGetWorkArray(subdm, maxConeSize, PETSC_INT, (void**) &subface);CHKERRQ(ierr);
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
  ierr = DMRestoreWorkArray(subdm, maxConeSize, PETSC_INT, (void**) &subface);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize, v;

    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
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
    ierr = VecCreate(comm, &subCoordinates);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubmesh_Interpolated"
static PetscErrorCode DMPlexCreateSubmesh_Interpolated(DM dm, const char vertexLabelName[], PetscInt value, DM subdm)
{
  MPI_Comm         comm;
  DMLabel          subpointMap, vertexLabel;
  IS              *subpointIS;
  const PetscInt **subpoints;
  PetscInt        *numSubPoints, *firstSubPoint, *coneNew, *coneONew;
  PetscInt         totSubPoints = 0, maxConeSize, dim, p, d, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate("subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  if (vertexLabelName) {
    ierr = DMPlexGetLabel(dm, vertexLabelName, &vertexLabel);CHKERRQ(ierr);
    ierr = DMPlexMarkSubmesh_Interpolated(dm, vertexLabel, value, subpointMap, subdm);CHKERRQ(ierr);
  }
  /* Setup chart */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim+1,PetscInt,&numSubPoints,dim+1,PetscInt,&firstSubPoint,dim+1,IS,&subpointIS,dim+1,const PetscInt *,&subpoints);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumSize(subpointMap, d, &numSubPoints[d]);CHKERRQ(ierr);
    totSubPoints += numSubPoints[d];
  }
  ierr = DMPlexSetChart(subdm, 0, totSubPoints);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, 1);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubPoint[dim] = 0;
  firstSubPoint[0]   = firstSubPoint[dim] + numSubPoints[dim];
  if (dim > 1) {firstSubPoint[dim-1] = firstSubPoint[0]     + numSubPoints[0];}
  if (dim > 2) {firstSubPoint[dim-2] = firstSubPoint[dim-1] + numSubPoints[dim-1];}
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumIS(subpointMap, d, &subpointIS[d]);CHKERRQ(ierr);
    ierr = ISGetIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);
  }
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, coneSizeNew, c, val;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(subdm, subpoint, coneSize);CHKERRQ(ierr);
      if (d == dim) {
        ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
        for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
          ierr = DMLabelGetValue(subpointMap, cone[c], &val);CHKERRQ(ierr);
          if (val >= 0) coneSizeNew++;
        }
        ierr = DMPlexSetConeSize(subdm, subpoint, coneSizeNew);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Set cones */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxConeSize,PetscInt,&coneNew,maxConeSize,PetscInt,&coneONew);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, subconeSize, coneSizeNew, c, subc;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(subdm, subpoint, &subconeSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, point, &ornt);CHKERRQ(ierr);
      for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
        ierr = PetscFindInt(cone[c], numSubPoints[d-1], subpoints[d-1], &subc);CHKERRQ(ierr);
        if (subc >= 0) {
          coneNew[coneSizeNew]  = firstSubPoint[d-1] + subc;
          coneONew[coneSizeNew] = ornt[c];
          ++coneSizeNew;
        }
      }
      if (coneSizeNew != subconeSize) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of cone points located %d does not match subcone size %d", coneSizeNew, subconeSize);
      ierr = DMPlexSetCone(subdm, subpoint, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(subdm, subpoint, coneONew);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(coneNew,coneONew);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize;

    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
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
    ierr = VecCreate(comm, &subCoordinates);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
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
  /* Cleanup */
  for (d = 0; d <= dim; ++d) {
    ierr = ISRestoreIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);
    ierr = ISDestroy(&subpointIS[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(numSubPoints,firstSubPoint,subpointIS,subpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubmesh"
/*@C
  DMPlexCreateSubmesh - Extract a hypersurface from the mesh using vertices defined by a label

  Input Parameters:
+ dm           - The original mesh
. vertexLabel  - The DMLabel marking vertices contained in the surface
- value        - The label value to use

  Output Parameter:
. subdm - The surface mesh

  Note: This function produces a DMLabel mapping original points in the submesh to their depth. This can be obtained using DMPlexGetSubpointMap().

  Level: developer

.seealso: DMPlexGetSubpointMap(), DMPlexGetLabel(), DMLabelSetValue()
@*/
PetscErrorCode DMPlexCreateSubmesh(DM dm, const char vertexLabel[], PetscInt value, DM *subdm)
{
  PetscInt       dim, depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 3);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*subdm, dim-1);CHKERRQ(ierr);
  if (depth == dim) {
    ierr = DMPlexCreateSubmesh_Interpolated(dm, vertexLabel, value, *subdm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateSubmesh_Uninterpolated(dm, vertexLabel, value, *subdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexFilterPoint_Internal"
PETSC_STATIC_INLINE PetscInt DMPlexFilterPoint_Internal(PetscInt point, PetscInt firstSubPoint, PetscInt numSubPoints, const PetscInt subPoints[])
{
  PetscInt       subPoint;
  PetscErrorCode ierr;

  ierr = PetscFindInt(point, numSubPoints, subPoints, &subPoint); if (ierr < 0) return ierr;
  return subPoint < 0 ? subPoint : firstSubPoint+subPoint;
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCohesiveSubmesh_Uninterpolated"
static PetscErrorCode DMPlexCreateCohesiveSubmesh_Uninterpolated(DM dm, PetscBool hasLagrange, const char label[], PetscInt value, DM subdm)
{
  MPI_Comm        comm;
  DMLabel         subpointMap;
  IS              subvertexIS;
  const PetscInt *subVertices;
  PetscInt        numSubVertices, firstSubVertex, numSubCells, *subCells = NULL;
  PetscInt       *subface, maxConeSize, numSubFaces, firstSubFace, newFacePoint, nFV;
  PetscInt        cMax, c, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate("subpoint_map", &subpointMap);CHKERRQ(ierr);
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
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(subdm, maxConeSize, PETSC_INT, (void**) &subface);CHKERRQ(ierr);
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt  cell    = subCells[c];
    const PetscInt  subcell = c;
    const PetscInt *cone, *cells;
    PetscInt        numCells, subVertex, p, v;

    if (cell < cMax) continue;
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
      PetscInt negsubcell;

      if (cells[p] >= cMax) continue;
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
  ierr = DMRestoreWorkArray(subdm, maxConeSize, PETSC_INT, (void**) &subface);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize, v;

    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
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
    ierr = VecCreate(comm, &subCoordinates);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
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
      ierr = PetscMalloc2(pEnd-pStart,PetscSFNode,&newLocalPoints,numRoots,PetscSFNode,&newOwners);CHKERRQ(ierr);
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
      ierr = PetscMalloc(numSubLeaves * sizeof(PetscInt),    &slocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc(numSubLeaves * sizeof(PetscSFNode), &sremotePoints);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCohesiveSubmesh_Interpolated"
static PetscErrorCode DMPlexCreateCohesiveSubmesh_Interpolated(DM dm, PetscBool hasLagrange, const char label[], PetscInt value, DM subdm)
{
  MPI_Comm         comm;
  DMLabel          subpointMap;
  IS              *subpointIS;
  const PetscInt **subpoints;
  PetscInt        *numSubPoints, *firstSubPoint, *coneNew;
  PetscInt         totSubPoints = 0, maxConeSize, dim, p, d, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate("subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  ierr = DMPlexMarkCohesiveSubmesh_Interpolated(dm, hasLagrange, label, value, subpointMap, subdm);CHKERRQ(ierr);
  /* Setup chart */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim+1,PetscInt,&numSubPoints,dim+1,PetscInt,&firstSubPoint,dim+1,IS,&subpointIS,dim+1,const PetscInt *,&subpoints);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumSize(subpointMap, d, &numSubPoints[d]);CHKERRQ(ierr);
    totSubPoints += numSubPoints[d];
  }
  ierr = DMPlexSetChart(subdm, 0, totSubPoints);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, 1);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubPoint[dim] = 0;
  firstSubPoint[0]   = firstSubPoint[dim] + numSubPoints[dim];
  if (dim > 1) {firstSubPoint[dim-1] = firstSubPoint[0]     + numSubPoints[0];}
  if (dim > 2) {firstSubPoint[dim-2] = firstSubPoint[dim-1] + numSubPoints[dim-1];}
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumIS(subpointMap, d, &subpointIS[d]);CHKERRQ(ierr);
    ierr = ISGetIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);
  }
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, coneSizeNew, c, val;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(subdm, subpoint, coneSize);CHKERRQ(ierr);
      if (d == dim) {
        ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
        for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
          ierr = DMLabelGetValue(subpointMap, cone[c], &val);CHKERRQ(ierr);
          if (val >= 0) coneSizeNew++;
        }
        ierr = DMPlexSetConeSize(subdm, subpoint, coneSizeNew);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Set cones */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(maxConeSize * sizeof(PetscInt), &coneNew);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, subconeSize, coneSizeNew, c, subc;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(subdm, subpoint, &subconeSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
        ierr = PetscFindInt(cone[c], numSubPoints[d-1], subpoints[d-1], &subc);CHKERRQ(ierr);
        if (subc >= 0) coneNew[coneSizeNew++] = firstSubPoint[d-1] + subc;
      }
      if (coneSizeNew != subconeSize) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of cone points located %d does not match subcone size %d", coneSizeNew, subconeSize);
      ierr = DMPlexSetCone(subdm, subpoint, coneNew);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(coneNew);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     numComp, coordSize;

    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
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
    ierr = VecCreate(comm, &subCoordinates);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
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
  /* Cleanup */
  for (d = 0; d <= dim; ++d) {
    ierr = ISRestoreIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);
    ierr = ISDestroy(&subpointIS[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(numSubPoints,firstSubPoint,subpointIS,subpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCohesiveSubmesh"
/*
  DMPlexCreateCohesiveSubmesh - Extract from a mesh with cohesive cells the hypersurface defined by one face of the cells. Optionally, a Label an be given to restrict the cells.

  Input Parameters:
+ dm          - The original mesh
. hasLagrange - The mesh has Lagrange unknowns in the cohesive cells
. label       - A label name, or PETSC_NULL
- value  - A label value

  Output Parameter:
. subdm - The surface mesh

  Note: This function produces a DMLabel mapping original points in the submesh to their depth. This can be obtained using DMPlexGetSubpointMap().

  Level: developer

.seealso: DMPlexGetSubpointMap(), DMPlexCreateSubmesh()
*/
PetscErrorCode DMPlexCreateCohesiveSubmesh(DM dm, PetscBool hasLagrange, const char label[], PetscInt value, DM *subdm)
{
  PetscInt       dim, depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdm, 5);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*subdm, dim-1);CHKERRQ(ierr);
  if (depth == dim) {
    ierr = DMPlexCreateCohesiveSubmesh_Interpolated(dm, hasLagrange, label, value, *subdm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateCohesiveSubmesh_Uninterpolated(dm, hasLagrange, label, value, *subdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetSubpointMap"
/*@
  DMPlexGetSubpointMap - Returns a DMLabel with point dimension as values

  Input Parameter:
. dm - The submesh DM

  Output Parameter:
. subpointMap - The DMLabel of all the points from the original mesh in this submesh, or NULL if this is not a submesh

  Level: developer

.seealso: DMPlexCreateSubmesh(), DMPlexCreateSubpointIS()
@*/
PetscErrorCode DMPlexGetSubpointMap(DM dm, DMLabel *subpointMap)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointMap, 2);
  *subpointMap = mesh->subpointMap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetSubpointMap"
/* Note: Should normally not be called by the user, since it is set in DMPlexCreateSubmesh() */
PetscErrorCode DMPlexSetSubpointMap(DM dm, DMLabel subpointMap)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DMLabel        tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  tmp  = mesh->subpointMap;
  mesh->subpointMap = subpointMap;
  ++mesh->subpointMap->refct;
  ierr = DMLabelDestroy(&tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubpointIS"
/*@
  DMPlexCreateSubpointIS - Creates an IS covering the entire subdm chart with the original points as data

  Input Parameter:
. dm - The submesh DM

  Output Parameter:
. subpointIS - The IS of all the points from the original mesh in this submesh, or NULL if this is not a submesh

  Note: This is IS is guaranteed to be sorted by the construction of the submesh

  Level: developer

.seealso: DMPlexCreateSubmesh(), DMPlexGetSubpointMap()
@*/
PetscErrorCode DMPlexCreateSubpointIS(DM dm, IS *subpointIS)
{
  MPI_Comm        comm;
  DMLabel         subpointMap;
  IS              is;
  const PetscInt *opoints;
  PetscInt       *points, *depths;
  PetscInt        depth, depStart, depEnd, d, pStart, pEnd, p, n, off;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointIS, 2);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  *subpointIS = NULL;
  ierr = DMPlexGetSubpointMap(dm, &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (subpointMap && depth >= 0) {
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Submeshes must start the point numbering at 0, not %d", pStart);
    ierr = DMGetWorkArray(dm, depth+1, PETSC_INT, &depths);CHKERRQ(ierr);
    depths[0] = depth;
    depths[1] = 0;
    for(d = 2; d <= depth; ++d) {depths[d] = depth+1 - d;}
    ierr = PetscMalloc(pEnd * sizeof(PetscInt), &points);CHKERRQ(ierr);
    for(d = 0, off = 0; d <= depth; ++d) {
      const PetscInt dep = depths[d];

      ierr = DMPlexGetDepthStratum(dm, dep, &depStart, &depEnd);CHKERRQ(ierr);
      ierr = DMLabelGetStratumSize(subpointMap, dep, &n);CHKERRQ(ierr);
      if (((d < 2) && (depth > 1)) || (d == 1)) { /* Only check vertices and cells for now since the map is broken for others */
        if (n != depEnd-depStart) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d at depth %d should be %d", n, dep, depEnd-depStart);
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
        ierr = DMLabelGetStratumIS(subpointMap, dep, &is);CHKERRQ(ierr);
        ierr = ISGetIndices(is, &opoints);CHKERRQ(ierr);
        for(p = 0; p < n; ++p, ++off) points[off] = opoints[p];
        ierr = ISRestoreIndices(is, &opoints);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
      }
    }
    ierr = DMRestoreWorkArray(dm, depth+1, PETSC_INT, &depths);CHKERRQ(ierr);
    if (off != pEnd) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d should be %d", off, pEnd);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, pEnd, points, PETSC_OWN_POINTER, subpointIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

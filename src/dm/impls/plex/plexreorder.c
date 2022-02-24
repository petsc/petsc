#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/matorderimpl.h> /*I      "petscmat.h"      I*/

static PetscErrorCode DMPlexCreateOrderingClosure_Static(DM dm, PetscInt numPoints, const PetscInt pperm[], PetscInt **clperm, PetscInt **invclperm)
{
  PetscInt      *perm, *iperm;
  PetscInt       depth, d, pStart, pEnd, fStart, fMax, fEnd, p;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscMalloc1(pEnd-pStart,&perm));
  CHKERRQ(PetscMalloc1(pEnd-pStart,&iperm));
  for (p = pStart; p < pEnd; ++p) iperm[p] = -1;
  for (d = depth; d > 0; --d) {
    CHKERRQ(DMPlexGetDepthStratum(dm, d,   &pStart, &pEnd));
    CHKERRQ(DMPlexGetDepthStratum(dm, d-1, &fStart, &fEnd));
    fMax = fStart;
    for (p = pStart; p < pEnd; ++p) {
      const PetscInt *cone;
      PetscInt        point, coneSize, c;

      if (d == depth) {
        perm[p]         = pperm[p];
        iperm[pperm[p]] = p;
      }
      point = perm[p];
      CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
      CHKERRQ(DMPlexGetCone(dm, point, &cone));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt oldc = cone[c];
        const PetscInt newc = iperm[oldc];

        if (newc < 0) {
          perm[fMax]  = oldc;
          iperm[oldc] = fMax++;
        }
      }
    }
    PetscCheckFalse(fMax != fEnd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of depth %d faces %d does not match permuted number %d", d, fEnd-fStart, fMax-fStart);
  }
  *clperm    = perm;
  *invclperm = iperm;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetOrdering - Calculate a reordering of the mesh

  Collective on dm

  Input Parameters:
+ dm - The DMPlex object
. otype - type of reordering, one of the following:
$     MATORDERINGNATURAL - Natural
$     MATORDERINGND - Nested Dissection
$     MATORDERING1WD - One-way Dissection
$     MATORDERINGRCM - Reverse Cuthill-McKee
$     MATORDERINGQMD - Quotient Minimum Degree
- label - [Optional] Label used to segregate ordering into sets, or NULL

  Output Parameter:
. perm - The point permutation as an IS, perm[old point number] = new point number

  Note: The label is used to group sets of points together by label value. This makes it easy to reorder a mesh which
  has different types of cells, and then loop over each set of reordered cells for assembly.

  Level: intermediate

.seealso: DMPlexPermute(), MatGetOrdering()
@*/
PetscErrorCode DMPlexGetOrdering(DM dm, MatOrderingType otype, DMLabel label, IS *perm)
{
  PetscInt       numCells = 0;
  PetscInt      *start = NULL, *adjacency = NULL, *cperm, *clperm = NULL, *invclperm = NULL, *mask, *xls, pStart, pEnd, c, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(perm, 4);
  CHKERRQ(DMPlexCreateNeighborCSR(dm, 0, &numCells, &start, &adjacency));
  CHKERRQ(PetscMalloc3(numCells,&cperm,numCells,&mask,numCells*2,&xls));
  if (numCells) {
    /* Shift for Fortran numbering */
    for (i = 0; i < start[numCells]; ++i) ++adjacency[i];
    for (i = 0; i <= numCells; ++i)       ++start[i];
    CHKERRQ(SPARSEPACKgenrcm(&numCells, start, adjacency, cperm, mask, xls));
  }
  CHKERRQ(PetscFree(start));
  CHKERRQ(PetscFree(adjacency));
  /* Shift for Fortran numbering */
  for (c = 0; c < numCells; ++c) --cperm[c];
  /* Segregate */
  if (label) {
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, numPoints = 0;
    PetscInt       *sperm, *vsize, *voff, v;

    CHKERRQ(DMLabelGetValueIS(label, &valueIS));
    CHKERRQ(ISSort(valueIS));
    CHKERRQ(ISGetLocalSize(valueIS, &numValues));
    CHKERRQ(ISGetIndices(valueIS, &values));
    CHKERRQ(PetscCalloc3(numCells,&sperm,numValues,&vsize,numValues+1,&voff));
    for (v = 0; v < numValues; ++v) {
      CHKERRQ(DMLabelGetStratumSize(label, values[v], &vsize[v]));
      if (v < numValues-1) voff[v+2] += vsize[v] + voff[v+1];
      numPoints += vsize[v];
    }
    PetscCheckFalse(numPoints != numCells,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label only covers %D cells < %D total", numPoints, numCells);
    for (c = 0; c < numCells; ++c) {
      const PetscInt oldc = cperm[c];
      PetscInt       val, vloc;

      CHKERRQ(DMLabelGetValue(label, oldc, &val));
      PetscCheckFalse(val == -1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cell %D not present in label", oldc);
      CHKERRQ(PetscFindInt(val, numValues, values, &vloc));
      PetscCheckFalse(vloc < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Value %D not present label", val);
      sperm[voff[vloc+1]++] = oldc;
    }
    for (v = 0; v < numValues; ++v) {
      PetscCheckFalse(voff[v+1] - voff[v] != vsize[v],PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of %D values found is %D != %D", values[v], voff[v+1] - voff[v], vsize[v]);
    }
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISDestroy(&valueIS));
    CHKERRQ(PetscArraycpy(cperm, sperm, numCells));
    CHKERRQ(PetscFree3(sperm, vsize, voff));
  }
  /* Construct closure */
  CHKERRQ(DMPlexCreateOrderingClosure_Static(dm, numCells, cperm, &clperm, &invclperm));
  CHKERRQ(PetscFree3(cperm,mask,xls));
  CHKERRQ(PetscFree(clperm));
  /* Invert permutation */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd-pStart, invclperm, PETSC_OWN_POINTER, perm));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetOrdering1D - Reorder the vertices so that the mesh is in a line

  Collective on dm

  Input Parameter:
. dm - The DMPlex object

  Output Parameter:
. perm - The point permutation as an IS, perm[old point number] = new point number

  Level: intermediate

.seealso: DMPlexGetOrdering(), DMPlexPermute(), MatGetOrdering()
@*/
PetscErrorCode DMPlexGetOrdering1D(DM dm, IS *perm)
{
  PetscInt       *points;
  const PetscInt *support, *cone;
  PetscInt        dim, pStart, pEnd, cStart, cEnd, c, vStart, vEnd, v, suppSize, lastCell = 0;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  PetscCheck(dim == 1, PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Input mesh must be one dimensional, not %" PetscInt_FMT, dim);
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &points));
  for (c = cStart; c < cEnd; ++c) points[c] = c;
  for (v = vStart; v < vEnd; ++v) points[v] = v;
  for (v = vStart; v < vEnd; ++v) {
    CHKERRQ(DMPlexGetSupportSize(dm, v, &suppSize));
    CHKERRQ(DMPlexGetSupport(dm, v, &support));
    if (suppSize == 1) {lastCell = support[0]; break;}
  }
  if (v < vEnd) {
    PetscInt pos = cEnd;

    points[v] = pos++;
    while (lastCell >= cStart) {
      CHKERRQ(DMPlexGetCone(dm, lastCell, &cone));
      if (cone[0] == v) v = cone[1];
      else              v = cone[0];
      CHKERRQ(DMPlexGetSupport(dm, v, &support));
      CHKERRQ(DMPlexGetSupportSize(dm, v, &suppSize));
      if (suppSize == 1) {lastCell = -1;}
      else {
        if (support[0] == lastCell) lastCell = support[1];
        else                        lastCell = support[0];
      }
      points[v] = pos++;
    }
    PetscCheck(pos == pEnd, PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Last vertex was %" PetscInt_FMT ", not %" PetscInt_FMT, pos, pEnd);
  }
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd-pStart, points, PETSC_OWN_POINTER, perm));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPermute - Reorder the mesh according to the input permutation

  Collective on dm

  Input Parameters:
+ dm - The DMPlex object
- perm - The point permutation, perm[old point number] = new point number

  Output Parameter:
. pdm - The permuted DM

  Level: intermediate

.seealso: MatPermute()
@*/
PetscErrorCode DMPlexPermute(DM dm, IS perm, DM *pdm)
{
  DM_Plex       *plex = (DM_Plex *) dm->data, *plexNew;
  PetscInt       dim, cdim;
  const char    *name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(perm, IS_CLASSID, 2);
  PetscValidPointer(pdm, 3);
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), pdm));
  CHKERRQ(DMSetType(*pdm, DMPLEX));
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(PetscObjectSetName((PetscObject) *pdm, name));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMSetDimension(*pdm, dim));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMSetCoordinateDim(*pdm, cdim));
  CHKERRQ(DMCopyDisc(dm, *pdm));
  if (dm->localSection) {
    PetscSection section, sectionNew;

    CHKERRQ(DMGetLocalSection(dm, &section));
    CHKERRQ(PetscSectionPermute(section, perm, &sectionNew));
    CHKERRQ(DMSetLocalSection(*pdm, sectionNew));
    CHKERRQ(PetscSectionDestroy(&sectionNew));
  }
  plexNew = (DM_Plex *) (*pdm)->data;
  /* Ignore ltogmap, ltogmapb */
  /* Ignore sf, sectionSF */
  /* Ignore globalVertexNumbers, globalCellNumbers */
  /* Reorder labels */
  {
    PetscInt numLabels, l;
    DMLabel  label, labelNew;

    CHKERRQ(DMGetNumLabels(dm, &numLabels));
    for (l = 0; l < numLabels; ++l) {
      CHKERRQ(DMGetLabelByNum(dm, l, &label));
      CHKERRQ(DMLabelPermute(label, perm, &labelNew));
      CHKERRQ(DMAddLabel(*pdm, labelNew));
      CHKERRQ(DMLabelDestroy(&labelNew));
    }
    CHKERRQ(DMGetLabel(*pdm, "depth", &(*pdm)->depthLabel));
    if (plex->subpointMap) CHKERRQ(DMLabelPermute(plex->subpointMap, perm, &plexNew->subpointMap));
  }
  /* Reorder topology */
  {
    const PetscInt *pperm;
    PetscInt        maxConeSize, maxSupportSize, n, pStart, pEnd, p;

    CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
    plexNew->maxConeSize    = maxConeSize;
    plexNew->maxSupportSize = maxSupportSize;
    CHKERRQ(PetscSectionDestroy(&plexNew->coneSection));
    CHKERRQ(PetscSectionPermute(plex->coneSection, perm, &plexNew->coneSection));
    CHKERRQ(PetscSectionGetStorageSize(plexNew->coneSection, &n));
    CHKERRQ(PetscMalloc1(n, &plexNew->cones));
    CHKERRQ(PetscMalloc1(n, &plexNew->coneOrientations));
    CHKERRQ(ISGetIndices(perm, &pperm));
    CHKERRQ(PetscSectionGetChart(plex->coneSection, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      CHKERRQ(PetscSectionGetDof(plexNew->coneSection, pperm[p], &dof));
      CHKERRQ(PetscSectionGetOffset(plex->coneSection, p, &off));
      CHKERRQ(PetscSectionGetOffset(plexNew->coneSection, pperm[p], &offNew));
      for (d = 0; d < dof; ++d) {
        plexNew->cones[offNew+d]            = pperm[plex->cones[off+d]];
        plexNew->coneOrientations[offNew+d] = plex->coneOrientations[off+d];
      }
    }
    CHKERRQ(PetscSectionDestroy(&plexNew->supportSection));
    CHKERRQ(PetscSectionPermute(plex->supportSection, perm, &plexNew->supportSection));
    CHKERRQ(PetscSectionGetStorageSize(plexNew->supportSection, &n));
    CHKERRQ(PetscMalloc1(n, &plexNew->supports));
    CHKERRQ(PetscSectionGetChart(plex->supportSection, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      CHKERRQ(PetscSectionGetDof(plexNew->supportSection, pperm[p], &dof));
      CHKERRQ(PetscSectionGetOffset(plex->supportSection, p, &off));
      CHKERRQ(PetscSectionGetOffset(plexNew->supportSection, pperm[p], &offNew));
      for (d = 0; d < dof; ++d) {
        plexNew->supports[offNew+d] = pperm[plex->supports[off+d]];
      }
    }
    CHKERRQ(ISRestoreIndices(perm, &pperm));
  }
  /* Remap coordinates */
  {
    DM              cdm, cdmNew;
    PetscSection    csection, csectionNew;
    Vec             coordinates, coordinatesNew;
    PetscScalar    *coords, *coordsNew;
    const PetscInt *pperm;
    PetscInt        pStart, pEnd, p;
    const char     *name;

    CHKERRQ(DMGetCoordinateDM(dm, &cdm));
    CHKERRQ(DMGetLocalSection(cdm, &csection));
    CHKERRQ(PetscSectionPermute(csection, perm, &csectionNew));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(VecDuplicate(coordinates, &coordinatesNew));
    CHKERRQ(PetscObjectGetName((PetscObject)coordinates,&name));
    CHKERRQ(PetscObjectSetName((PetscObject)coordinatesNew,name));
    CHKERRQ(VecGetArray(coordinates, &coords));
    CHKERRQ(VecGetArray(coordinatesNew, &coordsNew));
    CHKERRQ(PetscSectionGetChart(csectionNew, &pStart, &pEnd));
    CHKERRQ(ISGetIndices(perm, &pperm));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      CHKERRQ(PetscSectionGetDof(csectionNew, p, &dof));
      CHKERRQ(PetscSectionGetOffset(csection, p, &off));
      CHKERRQ(PetscSectionGetOffset(csectionNew, pperm[p], &offNew));
      for (d = 0; d < dof; ++d) coordsNew[offNew+d] = coords[off+d];
    }
    CHKERRQ(ISRestoreIndices(perm, &pperm));
    CHKERRQ(VecRestoreArray(coordinates, &coords));
    CHKERRQ(VecRestoreArray(coordinatesNew, &coordsNew));
    CHKERRQ(DMGetCoordinateDM(*pdm, &cdmNew));
    CHKERRQ(DMSetLocalSection(cdmNew, csectionNew));
    CHKERRQ(DMSetCoordinatesLocal(*pdm, coordinatesNew));
    CHKERRQ(PetscSectionDestroy(&csectionNew));
    CHKERRQ(VecDestroy(&coordinatesNew));
  }
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *pdm));
  (*pdm)->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/matorderimpl.h> /*I      "petscmat.h"      I*/

static PetscErrorCode DMPlexCreateOrderingClosure_Static(DM dm, PetscInt numPoints, const PetscInt pperm[], PetscInt **clperm, PetscInt **invclperm)
{
  PetscInt      *perm, *iperm;
  PetscInt       depth, d, pStart, pEnd, fStart, fMax, fEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart,&perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart,&iperm);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) iperm[p] = -1;
  for (d = depth; d > 0; --d) {
    ierr = DMPlexGetDepthStratum(dm, d,   &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, d-1, &fStart, &fEnd);CHKERRQ(ierr);
    fMax = fStart;
    for (p = pStart; p < pEnd; ++p) {
      const PetscInt *cone;
      PetscInt        point, coneSize, c;

      if (d == depth) {
        perm[p]         = pperm[p];
        iperm[pperm[p]] = p;
      }
      point = perm[p];
      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        const PetscInt oldc = cone[c];
        const PetscInt newc = iperm[oldc];

        if (newc < 0) {
          perm[fMax]  = oldc;
          iperm[oldc] = fMax++;
        }
      }
    }
    if (fMax != fEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of depth %d faces %d does not match permuted number %d", d, fEnd-fStart, fMax-fStart);
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

.seealso: MatGetOrdering()
@*/
PetscErrorCode DMPlexGetOrdering(DM dm, MatOrderingType otype, DMLabel label, IS *perm)
{
  PetscInt       numCells = 0;
  PetscInt      *start = NULL, *adjacency = NULL, *cperm, *clperm = NULL, *invclperm = NULL, *mask, *xls, pStart, pEnd, c, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(perm, 4);
  ierr = DMPlexCreateNeighborCSR(dm, 0, &numCells, &start, &adjacency);CHKERRQ(ierr);
  ierr = PetscMalloc3(numCells,&cperm,numCells,&mask,numCells*2,&xls);CHKERRQ(ierr);
  if (numCells) {
    /* Shift for Fortran numbering */
    for (i = 0; i < start[numCells]; ++i) ++adjacency[i];
    for (i = 0; i <= numCells; ++i)       ++start[i];
    ierr = SPARSEPACKgenrcm(&numCells, start, adjacency, cperm, mask, xls);CHKERRQ(ierr);
  }
  ierr = PetscFree(start);CHKERRQ(ierr);
  ierr = PetscFree(adjacency);CHKERRQ(ierr);
  /* Shift for Fortran numbering */
  for (c = 0; c < numCells; ++c) --cperm[c];
  /* Segregate */
  if (label) {
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, numPoints = 0;
    PetscInt       *sperm, *vsize, *voff, v;

    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISSort(valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = PetscCalloc3(numCells,&sperm,numValues,&vsize,numValues+1,&voff);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      ierr = DMLabelGetStratumSize(label, values[v], &vsize[v]);CHKERRQ(ierr);
      if (v < numValues-1) voff[v+2] += vsize[v] + voff[v+1];
      numPoints += vsize[v];
    }
    if (numPoints != numCells) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label only covers %D cells < %D total", numPoints, numCells);
    for (c = 0; c < numCells; ++c) {
      const PetscInt oldc = cperm[c];
      PetscInt       val, vloc;

      ierr = DMLabelGetValue(label, oldc, &val);CHKERRQ(ierr);
      if (val == -1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cell %D not present in label", oldc);
      ierr = PetscFindInt(val, numValues, values, &vloc);CHKERRQ(ierr);
      if (vloc < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Value %D not present label", val);
      sperm[voff[vloc+1]++] = oldc;
    }
    for (v = 0; v < numValues; ++v) {
      if (voff[v+1] - voff[v] != vsize[v]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of %D values found is %D != %D", values[v], voff[v+1] - voff[v], vsize[v]);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    ierr = PetscArraycpy(cperm, sperm, numCells);CHKERRQ(ierr);
    ierr = PetscFree3(sperm, vsize, voff);CHKERRQ(ierr);
  }
  /* Construct closure */
  ierr = DMPlexCreateOrderingClosure_Static(dm, numCells, cperm, &clperm, &invclperm);CHKERRQ(ierr);
  ierr = PetscFree3(cperm,mask,xls);CHKERRQ(ierr);
  ierr = PetscFree(clperm);CHKERRQ(ierr);
  /* Invert permutation */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd-pStart, invclperm, PETSC_OWN_POINTER, perm);CHKERRQ(ierr);
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
  PetscSection   section, sectionNew;
  PetscInt       dim, cdim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(perm, IS_CLASSID, 2);
  PetscValidPointer(pdm, 3);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), pdm);CHKERRQ(ierr);
  ierr = DMSetType(*pdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(*pdm, dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*pdm, cdim);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, *pdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  if (section) {
    ierr = PetscSectionPermute(section, perm, &sectionNew);CHKERRQ(ierr);
    ierr = DMSetLocalSection(*pdm, sectionNew);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&sectionNew);CHKERRQ(ierr);
  }
  plexNew = (DM_Plex *) (*pdm)->data;
  /* Ignore ltogmap, ltogmapb */
  /* Ignore sf, sectionSF */
  /* Ignore globalVertexNumbers, globalCellNumbers */
  /* Reorder labels */
  {
    PetscInt numLabels, l;
    DMLabel  label, labelNew;

    ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
    for (l = 0; l < numLabels; ++l) {
      ierr = DMGetLabelByNum(dm, l, &label);CHKERRQ(ierr);
      ierr = DMLabelPermute(label, perm, &labelNew);CHKERRQ(ierr);
      ierr = DMAddLabel(*pdm, labelNew);CHKERRQ(ierr);
      ierr = DMLabelDestroy(&labelNew);CHKERRQ(ierr);
    }
    ierr = DMGetLabel(*pdm, "depth", &(*pdm)->depthLabel);CHKERRQ(ierr);
    if (plex->subpointMap) {ierr = DMLabelPermute(plex->subpointMap, perm, &plexNew->subpointMap);CHKERRQ(ierr);}
  }
  /* Reorder topology */
  {
    const PetscInt *pperm;
    PetscInt        maxConeSize, maxSupportSize, n, pStart, pEnd, p;

    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    plexNew->maxConeSize    = maxConeSize;
    plexNew->maxSupportSize = maxSupportSize;
    ierr = PetscSectionDestroy(&plexNew->coneSection);CHKERRQ(ierr);
    ierr = PetscSectionPermute(plex->coneSection, perm, &plexNew->coneSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(plexNew->coneSection, &n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &plexNew->cones);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &plexNew->coneOrientations);CHKERRQ(ierr);
    ierr = ISGetIndices(perm, &pperm);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(plex->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      ierr = PetscSectionGetDof(plexNew->coneSection, pperm[p], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plex->coneSection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plexNew->coneSection, pperm[p], &offNew);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        plexNew->cones[offNew+d]            = pperm[plex->cones[off+d]];
        plexNew->coneOrientations[offNew+d] = plex->coneOrientations[off+d];
      }
    }
    ierr = PetscSectionDestroy(&plexNew->supportSection);CHKERRQ(ierr);
    ierr = PetscSectionPermute(plex->supportSection, perm, &plexNew->supportSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(plexNew->supportSection, &n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &plexNew->supports);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(plex->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      ierr = PetscSectionGetDof(plexNew->supportSection, pperm[p], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plex->supportSection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(plexNew->supportSection, pperm[p], &offNew);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        plexNew->supports[offNew+d] = pperm[plex->supports[off+d]];
      }
    }
    ierr = ISRestoreIndices(perm, &pperm);CHKERRQ(ierr);
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

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &csection);CHKERRQ(ierr);
    ierr = PetscSectionPermute(csection, perm, &csectionNew);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecDuplicate(coordinates, &coordinatesNew);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)coordinates,&name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)coordinatesNew,name);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecGetArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(csectionNew, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = ISGetIndices(perm, &pperm);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, offNew, d;

      ierr = PetscSectionGetDof(csectionNew, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(csection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(csectionNew, pperm[p], &offNew);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) coordsNew[offNew+d] = coords[off+d];
    }
    ierr = ISRestoreIndices(perm, &pperm);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(*pdm, &cdmNew);CHKERRQ(ierr);
    ierr = DMSetLocalSection(cdmNew, csectionNew);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*pdm, coordinatesNew);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&csectionNew);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinatesNew);CHKERRQ(ierr);
  }
  (*pdm)->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

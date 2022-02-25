#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/petscfeimpl.h>  /* For PetscFEInterpolate_Static() */

#include <petscdmplextransform.h>
#include <petscsf.h>

/*@
  DMPlexCreateProcessSF - Create an SF which just has process connectivity

  Collective on dm

  Input Parameters:
+ dm      - The DM
- sfPoint - The PetscSF which encodes point connectivity

  Output Parameters:
+ processRanks - A list of process neighbors, or NULL
- sfProcess    - An SF encoding the process connectivity, or NULL

  Level: developer

.seealso: PetscSFCreate(), DMPlexCreateTwoSidedProcessSF()
@*/
PetscErrorCode DMPlexCreateProcessSF(DM dm, PetscSF sfPoint, IS *processRanks, PetscSF *sfProcess)
{
  PetscInt           numRoots, numLeaves, l;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscInt          *ranks, *ranksNew;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) {PetscValidPointer(processRanks, 3);}
  if (sfProcess)    {PetscValidPointer(sfProcess, 4);}
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &ranks);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    ranks[l] = remotePoints[l].rank;
  }
  ierr = PetscSortRemoveDupsInt(&numLeaves, ranks);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &ranksNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeaves, &remotePointsNew);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    ranksNew[l]              = ranks[l];
    localPointsNew[l]        = l;
    remotePointsNew[l].index = 0;
    remotePointsNew[l].rank  = ranksNew[l];
  }
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  if (processRanks) {ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), numLeaves, ranksNew, PETSC_OWN_POINTER, processRanks);CHKERRQ(ierr);}
  else              {ierr = PetscFree(ranksNew);CHKERRQ(ierr);}
  if (sfProcess) {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *sfProcess, "Process SF");CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(*sfProcess);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*sfProcess, size, numLeaves, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateCoarsePointIS - Creates an IS covering the coarse DM chart with the fine points as data

  Collective on dm

  Input Parameter:
. dm - The coarse DM

  Output Parameter:
. fpointIS - The IS of all the fine points which exist in the original coarse mesh

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementUniform(), DMPlexGetSubpointIS()
@*/
PetscErrorCode DMPlexCreateCoarsePointIS(DM dm, IS *fpointIS)
{
  DMPlexTransform tr;
  PetscInt       *fpoints;
  PetscInt        pStart, pEnd, p, vStart, vEnd, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetUp(tr);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &fpoints);CHKERRQ(ierr);
  for (p = 0; p < pEnd-pStart; ++p) fpoints[p] = -1;
  for (v = vStart; v < vEnd; ++v) {
    PetscInt vNew = -1; /* quiet overzealous may be used uninitialized check */

    ierr = DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_POINT, DM_POLYTOPE_POINT, p, 0, &vNew);CHKERRQ(ierr);
    fpoints[v-pStart] = vNew;
  }
  ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, pEnd-pStart, fpoints, PETSC_OWN_POINTER, fpointIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexSetTransformType - Set the transform type for uniform refinement

  Input Parameters:
+ dm - The DM
- type - The transform type for uniform refinement

  Level: developer

.seealso: DMPlexTransformType, DMRefine(), DMPlexGetTransformType(), DMPlexSetRefinementUniform()
@*/
PetscErrorCode DMPlexSetTransformType(DM dm, DMPlexTransformType type)
{
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  if (type) PetscValidCharPointer(type, 2);
  ierr = PetscFree(mesh->transformType);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type, &mesh->transformType);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetTransformType - Retrieve the transform type for uniform refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. type - The transform type for uniform refinement

  Level: developer

.seealso: DMPlexTransformType, DMRefine(), DMPlexSetTransformType(), DMPlexGetRefinementUniform()
@*/
PetscErrorCode DMPlexGetTransformType(DM dm, DMPlexTransformType *type)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscValidPointer(type, 2);
  *type = mesh->transformType;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRefinementUniform - Set the flag for uniform refinement

  Input Parameters:
+ dm - The DM
- refinementUniform - The flag for uniform refinement

  Level: developer

.seealso: DMRefine(), DMPlexGetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexSetRefinementUniform(DM dm, PetscBool refinementUniform)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  mesh->refinementUniform = refinementUniform;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRefinementUniform - Retrieve the flag for uniform refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. refinementUniform - The flag for uniform refinement

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexGetRefinementUniform(DM dm, PetscBool *refinementUniform)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscValidPointer(refinementUniform,  2);
  *refinementUniform = mesh->refinementUniform;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRefinementLimit - Set the maximum cell volume for refinement

  Input Parameters:
+ dm - The DM
- refinementLimit - The maximum cell volume in the refined mesh

  Level: developer

.seealso: DMRefine(), DMPlexGetRefinementLimit(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform()
@*/
PetscErrorCode DMPlexSetRefinementLimit(DM dm, PetscReal refinementLimit)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementLimit = refinementLimit;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRefinementLimit - Retrieve the maximum cell volume for refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. refinementLimit - The maximum cell volume in the refined mesh

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementLimit(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform()
@*/
PetscErrorCode DMPlexGetRefinementLimit(DM dm, PetscReal *refinementLimit)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementLimit,  2);
  /* if (mesh->refinementLimit < 0) = getMaxVolume()/2.0; */
  *refinementLimit = mesh->refinementLimit;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRefinementFunction - Set the function giving the maximum cell volume for refinement

  Input Parameters:
+ dm - The DM
- refinementFunc - Function giving the maximum cell volume in the refined mesh

  Note: The calling sequence is refinementFunc(coords, limit)
$ coords - Coordinates of the current point, usually a cell centroid
$ limit  - The maximum cell volume for a cell containing this point

  Level: developer

.seealso: DMRefine(), DMPlexGetRefinementFunction(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexSetRefinementFunction(DM dm, PetscErrorCode (*refinementFunc)(const PetscReal [], PetscReal *))
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementFunc = refinementFunc;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRefinementFunction - Get the function giving the maximum cell volume for refinement

  Input Parameter:
. dm - The DM

  Output Parameter:
. refinementFunc - Function giving the maximum cell volume in the refined mesh

  Note: The calling sequence is refinementFunc(coords, limit)
$ coords - Coordinates of the current point, usually a cell centroid
$ limit  - The maximum cell volume for a cell containing this point

  Level: developer

.seealso: DMRefine(), DMPlexSetRefinementFunction(), DMPlexGetRefinementUniform(), DMPlexSetRefinementUniform(), DMPlexGetRefinementLimit(), DMPlexSetRefinementLimit()
@*/
PetscErrorCode DMPlexGetRefinementFunction(DM dm, PetscErrorCode (**refinementFunc)(const PetscReal [], PetscReal *))
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementFunc,  2);
  *refinementFunc = mesh->refinementFunc;
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefine_Plex(DM dm, MPI_Comm comm, DM *rdm)
{
  PetscBool      isUniform;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetRefinementUniform(dm, &isUniform);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-initref_dm_view");CHKERRQ(ierr);
  if (isUniform) {
    DMPlexTransform     tr;
    DM                  cdm, rcdm;
    DMPlexTransformType trType;
    const char         *prefix;
    PetscOptions       options;

    ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr);CHKERRQ(ierr);
    ierr = DMPlexTransformSetDM(tr, dm);CHKERRQ(ierr);
    ierr = DMPlexGetTransformType(dm, &trType);CHKERRQ(ierr);
    if (trType) {ierr = DMPlexTransformSetType(tr, trType);CHKERRQ(ierr);}
    ierr = PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) tr,  prefix);CHKERRQ(ierr);
    ierr = PetscObjectGetOptions((PetscObject) dm, &options);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject) tr, options);CHKERRQ(ierr);
    ierr = DMPlexTransformSetFromOptions(tr);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject) tr, NULL);CHKERRQ(ierr);
    ierr = DMPlexTransformSetUp(tr);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) tr, NULL, "-dm_plex_transform_view");CHKERRQ(ierr);
    ierr = DMPlexTransformApply(tr, dm, rdm);CHKERRQ(ierr);
    ierr = DMPlexSetRegularRefinement(*rdm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, *rdm);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(*rdm, &rcdm);CHKERRQ(ierr);
    ierr = DMCopyDisc(cdm, rcdm);CHKERRQ(ierr);
    ierr = DMPlexTransformCreateDiscLabels(tr, *rdm);CHKERRQ(ierr);
    ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
  } else {
    ierr = DMPlexRefine_Internal(dm, NULL, NULL, NULL, rdm);CHKERRQ(ierr);
  }
  if (*rdm) {
    ((DM_Plex *) (*rdm)->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
    ((DM_Plex *) (*rdm)->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
  }
  ierr = DMViewFromOptions(dm, NULL, "-postref_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefineHierarchy_Plex(DM dm, PetscInt nlevels, DM rdm[])
{
  DM             cdm = dm;
  PetscInt       r;
  PetscBool      isUniform, localized;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetRefinementUniform(dm, &isUniform);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  if (isUniform) {
    for (r = 0; r < nlevels; ++r) {
      DMPlexTransform tr;
      DM              codm, rcodm;
      const char     *prefix;

      ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) cdm), &tr);CHKERRQ(ierr);
      ierr = PetscObjectGetOptionsPrefix((PetscObject) cdm, &prefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject) tr,   prefix);CHKERRQ(ierr);
      ierr = DMPlexTransformSetDM(tr, cdm);CHKERRQ(ierr);
      ierr = DMPlexTransformSetFromOptions(tr);CHKERRQ(ierr);
      ierr = DMPlexTransformSetUp(tr);CHKERRQ(ierr);
      ierr = DMPlexTransformApply(tr, cdm, &rdm[r]);CHKERRQ(ierr);
      ierr = DMSetCoarsenLevel(rdm[r], cdm->leveldown);CHKERRQ(ierr);
      ierr = DMSetRefineLevel(rdm[r], cdm->levelup+1);CHKERRQ(ierr);
      ierr = DMCopyDisc(cdm, rdm[r]);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(dm, &codm);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(rdm[r], &rcodm);CHKERRQ(ierr);
      ierr = DMCopyDisc(codm, rcodm);CHKERRQ(ierr);
      ierr = DMPlexTransformCreateDiscLabels(tr, rdm[r]);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(rdm[r], cdm);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(rdm[r], PETSC_TRUE);CHKERRQ(ierr);
      if (rdm[r]) {
        ((DM_Plex *) (rdm[r])->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
        ((DM_Plex *) (rdm[r])->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
      }
      cdm  = rdm[r];
      ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
    }
  } else {
    for (r = 0; r < nlevels; ++r) {
      ierr = DMRefine(cdm, PetscObjectComm((PetscObject) dm), &rdm[r]);CHKERRQ(ierr);
      ierr = DMCopyDisc(cdm, rdm[r]);CHKERRQ(ierr);
      if (localized) {ierr = DMLocalizeCoordinates(rdm[r]);CHKERRQ(ierr);}
      ierr = DMSetCoarseDM(rdm[r], cdm);CHKERRQ(ierr);
      if (rdm[r]) {
        ((DM_Plex *) (rdm[r])->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
        ((DM_Plex *) (rdm[r])->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
      }
      cdm  = rdm[r];
    }
  }
  PetscFunctionReturn(0);
}

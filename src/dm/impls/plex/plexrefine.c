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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) {PetscValidPointer(processRanks, 3);}
  if (sfProcess)    {PetscValidPointer(sfProcess, 4);}
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  CHKERRQ(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
  CHKERRQ(PetscMalloc1(numLeaves, &ranks));
  for (l = 0; l < numLeaves; ++l) {
    ranks[l] = remotePoints[l].rank;
  }
  CHKERRQ(PetscSortRemoveDupsInt(&numLeaves, ranks));
  CHKERRQ(PetscMalloc1(numLeaves, &ranksNew));
  CHKERRQ(PetscMalloc1(numLeaves, &localPointsNew));
  CHKERRQ(PetscMalloc1(numLeaves, &remotePointsNew));
  for (l = 0; l < numLeaves; ++l) {
    ranksNew[l]              = ranks[l];
    localPointsNew[l]        = l;
    remotePointsNew[l].index = 0;
    remotePointsNew[l].rank  = ranksNew[l];
  }
  CHKERRQ(PetscFree(ranks));
  if (processRanks) CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm), numLeaves, ranksNew, PETSC_OWN_POINTER, processRanks));
  else              CHKERRQ(PetscFree(ranksNew));
  if (sfProcess) {
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess));
    CHKERRQ(PetscObjectSetName((PetscObject) *sfProcess, "Process SF"));
    CHKERRQ(PetscSFSetFromOptions(*sfProcess));
    CHKERRQ(PetscSFSetGraph(*sfProcess, size, numLeaves, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr));
  CHKERRQ(DMPlexTransformSetUp(tr));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &fpoints));
  for (p = 0; p < pEnd-pStart; ++p) fpoints[p] = -1;
  for (v = vStart; v < vEnd; ++v) {
    PetscInt vNew = -1; /* quiet overzealous may be used uninitialized check */

    CHKERRQ(DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_POINT, DM_POLYTOPE_POINT, p, 0, &vNew));
    fpoints[v-pStart] = vNew;
  }
  CHKERRQ(DMPlexTransformDestroy(&tr));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, pEnd-pStart, fpoints, PETSC_OWN_POINTER, fpointIS));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  if (type) PetscValidCharPointer(type, 2);
  CHKERRQ(PetscFree(mesh->transformType));
  CHKERRQ(PetscStrallocpy(type, &mesh->transformType));
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
  PetscValidBoolPointer(refinementUniform,  2);
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
  PetscValidRealPointer(refinementLimit,  2);
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetRefinementUniform(dm, &isUniform));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-initref_dm_view"));
  if (isUniform) {
    DMPlexTransform     tr;
    DM                  cdm, rcdm;
    DMPlexTransformType trType;
    const char         *prefix;
    PetscOptions       options;

    CHKERRQ(DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr));
    CHKERRQ(DMPlexTransformSetDM(tr, dm));
    CHKERRQ(DMPlexGetTransformType(dm, &trType));
    if (trType) CHKERRQ(DMPlexTransformSetType(tr, trType));
    CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) tr,  prefix));
    CHKERRQ(PetscObjectGetOptions((PetscObject) dm, &options));
    CHKERRQ(PetscObjectSetOptions((PetscObject) tr, options));
    CHKERRQ(DMPlexTransformSetFromOptions(tr));
    CHKERRQ(PetscObjectSetOptions((PetscObject) tr, NULL));
    CHKERRQ(DMPlexTransformSetUp(tr));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) tr, NULL, "-dm_plex_transform_view"));
    CHKERRQ(DMPlexTransformApply(tr, dm, rdm));
    CHKERRQ(DMPlexSetRegularRefinement(*rdm, PETSC_TRUE));
    CHKERRQ(DMCopyDisc(dm, *rdm));
    CHKERRQ(DMGetCoordinateDM(dm, &cdm));
    CHKERRQ(DMGetCoordinateDM(*rdm, &rcdm));
    CHKERRQ(DMCopyDisc(cdm, rcdm));
    CHKERRQ(DMPlexTransformCreateDiscLabels(tr, *rdm));
    CHKERRQ(DMPlexTransformDestroy(&tr));
  } else {
    CHKERRQ(DMPlexRefine_Internal(dm, NULL, NULL, NULL, rdm));
  }
  if (*rdm) {
    ((DM_Plex *) (*rdm)->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
    ((DM_Plex *) (*rdm)->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
  }
  CHKERRQ(DMViewFromOptions(dm, NULL, "-postref_dm_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefineHierarchy_Plex(DM dm, PetscInt nlevels, DM rdm[])
{
  DM             cdm = dm;
  PetscInt       r;
  PetscBool      isUniform, localized;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetRefinementUniform(dm, &isUniform));
  CHKERRQ(DMGetCoordinatesLocalized(dm, &localized));
  if (isUniform) {
    for (r = 0; r < nlevels; ++r) {
      DMPlexTransform tr;
      DM              codm, rcodm;
      const char     *prefix;

      CHKERRQ(DMPlexTransformCreate(PetscObjectComm((PetscObject) cdm), &tr));
      CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) cdm, &prefix));
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) tr,   prefix));
      CHKERRQ(DMPlexTransformSetDM(tr, cdm));
      CHKERRQ(DMPlexTransformSetFromOptions(tr));
      CHKERRQ(DMPlexTransformSetUp(tr));
      CHKERRQ(DMPlexTransformApply(tr, cdm, &rdm[r]));
      CHKERRQ(DMSetCoarsenLevel(rdm[r], cdm->leveldown));
      CHKERRQ(DMSetRefineLevel(rdm[r], cdm->levelup+1));
      CHKERRQ(DMCopyDisc(cdm, rdm[r]));
      CHKERRQ(DMGetCoordinateDM(dm, &codm));
      CHKERRQ(DMGetCoordinateDM(rdm[r], &rcodm));
      CHKERRQ(DMCopyDisc(codm, rcodm));
      CHKERRQ(DMPlexTransformCreateDiscLabels(tr, rdm[r]));
      CHKERRQ(DMSetCoarseDM(rdm[r], cdm));
      CHKERRQ(DMPlexSetRegularRefinement(rdm[r], PETSC_TRUE));
      if (rdm[r]) {
        ((DM_Plex *) (rdm[r])->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
        ((DM_Plex *) (rdm[r])->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
      }
      cdm  = rdm[r];
      CHKERRQ(DMPlexTransformDestroy(&tr));
    }
  } else {
    for (r = 0; r < nlevels; ++r) {
      CHKERRQ(DMRefine(cdm, PetscObjectComm((PetscObject) dm), &rdm[r]));
      CHKERRQ(DMCopyDisc(cdm, rdm[r]));
      if (localized) CHKERRQ(DMLocalizeCoordinates(rdm[r]));
      CHKERRQ(DMSetCoarseDM(rdm[r], cdm));
      if (rdm[r]) {
        ((DM_Plex *) (rdm[r])->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
        ((DM_Plex *) (rdm[r])->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
      }
      cdm  = rdm[r];
    }
  }
  PetscFunctionReturn(0);
}

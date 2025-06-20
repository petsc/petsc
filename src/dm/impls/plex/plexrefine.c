#include <petsc/private/dmpleximpl.h>  /*I      "petscdmplex.h"   I*/
#include <petsc/private/petscfeimpl.h> /* For PetscFEInterpolate_Static() */

#include <petscdmplextransform.h> /*I      "petscdmplextransform.h"   I*/
#include <petscsf.h>

/*@
  DMPlexCreateProcessSF - Create an `PetscSF` which just has process connectivity

  Collective

  Input Parameters:
+ dm      - The `DM`
- sfPoint - The `PetscSF` which encodes point connectivity

  Output Parameters:
+ processRanks - A list of process neighbors, or `NULL`
- sfProcess    - An `PetscSF` encoding the process connectivity, or `NULL`

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `PetscSF`, `PetscSFCreate()`, `DMPlexCreateTwoSidedProcessSF()`
@*/
PetscErrorCode DMPlexCreateProcessSF(DM dm, PetscSF sfPoint, IS *processRanks, PetscSF *sfProcess)
{
  PetscInt           numRoots, numLeaves, l;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscMPIInt       *ranks;
  PetscInt          *ranksNew;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) PetscAssertPointer(processRanks, 3);
  if (sfProcess) PetscAssertPointer(sfProcess, 4);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
  PetscCall(PetscMalloc1(numLeaves, &ranks));
  for (l = 0; l < numLeaves; ++l) ranks[l] = (PetscMPIInt)remotePoints[l].rank;
  PetscCall(PetscSortRemoveDupsMPIInt(&numLeaves, ranks));
  PetscCall(PetscMalloc1(numLeaves, &ranksNew));
  PetscCall(PetscMalloc1(numLeaves, &localPointsNew));
  PetscCall(PetscMalloc1(numLeaves, &remotePointsNew));
  for (l = 0; l < numLeaves; ++l) {
    ranksNew[l]              = ranks[l];
    localPointsNew[l]        = l;
    remotePointsNew[l].index = 0;
    remotePointsNew[l].rank  = ranksNew[l];
  }
  PetscCall(PetscFree(ranks));
  if (processRanks) PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), numLeaves, ranksNew, PETSC_OWN_POINTER, processRanks));
  else PetscCall(PetscFree(ranksNew));
  if (sfProcess) {
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess));
    PetscCall(PetscObjectSetName((PetscObject)*sfProcess, "Process SF"));
    PetscCall(PetscSFSetFromOptions(*sfProcess));
    PetscCall(PetscSFSetGraph(*sfProcess, size, numLeaves, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCreateCoarsePointIS - Creates an `IS` covering the coarse `DM` chart with the fine points as data

  Collective

  Input Parameter:
. dm - The coarse `DM`

  Output Parameter:
. fpointIS - The `IS` of all the fine points which exist in the original coarse mesh

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `IS`, `DMRefine()`, `DMPlexSetRefinementUniform()`, `DMPlexGetSubpointIS()`
@*/
PetscErrorCode DMPlexCreateCoarsePointIS(DM dm, IS *fpointIS)
{
  DMPlexTransform tr;
  PetscInt       *fpoints;
  PetscInt        pStart, pEnd, p, vStart, vEnd, v;

  PetscFunctionBegin;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscMalloc1(pEnd - pStart, &fpoints));
  for (p = 0; p < pEnd - pStart; ++p) fpoints[p] = -1;
  for (v = vStart; v < vEnd; ++v) {
    PetscInt vNew = -1; /* quiet overzealous may be used uninitialized check */

    PetscCall(DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_POINT, DM_POLYTOPE_POINT, p, 0, &vNew));
    fpoints[v - pStart] = vNew;
  }
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, pEnd - pStart, fpoints, PETSC_OWN_POINTER, fpointIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetTransformType - Set the transform type for uniform refinement

  Input Parameters:
+ dm   - The `DM`
- type - The transform type for uniform refinement

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexTransformType`, `DMRefine()`, `DMPlexGetTransformType()`, `DMPlexSetRefinementUniform()`
@*/
PetscErrorCode DMPlexSetTransformType(DM dm, DMPlexTransformType type)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  if (type) PetscAssertPointer(type, 2);
  PetscCall(PetscFree(mesh->transformType));
  PetscCall(PetscStrallocpy(type, &mesh->transformType));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetTransformType - Retrieve the transform type for uniform refinement

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. type - The transform type for uniform refinement

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexTransformType`, `DMRefine()`, `DMPlexSetTransformType()`, `DMPlexGetRefinementUniform()`
@*/
PetscErrorCode DMPlexGetTransformType(DM dm, DMPlexTransformType *type)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscAssertPointer(type, 2);
  *type = mesh->transformType;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexSetTransform(DM dm, DMPlexTransform tr)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  if (tr) PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)tr));
  PetscCall(DMPlexTransformDestroy(&mesh->transform));
  mesh->transform = tr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexGetTransform(DM dm, DMPlexTransform *tr)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscAssertPointer(tr, 2);
  *tr = mesh->transform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexSetSaveTransform(DM dm, PetscBool save)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  mesh->saveTransform = save;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexGetSaveTransform(DM dm, PetscBool *save)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscAssertPointer(save, 2);
  *save = mesh->saveTransform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetRefinementUniform - Set the flag for uniform refinement

  Input Parameters:
+ dm                - The `DM`
- refinementUniform - The flag for uniform refinement

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexGetRefinementUniform()`, `DMPlexGetRefinementLimit()`, `DMPlexSetRefinementLimit()`
@*/
PetscErrorCode DMPlexSetRefinementUniform(DM dm, PetscBool refinementUniform)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  mesh->refinementUniform = refinementUniform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetRefinementUniform - Retrieve the flag for uniform refinement

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. refinementUniform - The flag for uniform refinement

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexSetRefinementUniform()`, `DMPlexGetRefinementLimit()`, `DMPlexSetRefinementLimit()`
@*/
PetscErrorCode DMPlexGetRefinementUniform(DM dm, PetscBool *refinementUniform)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscAssertPointer(refinementUniform, 2);
  *refinementUniform = mesh->refinementUniform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetRefinementLimit - Set the maximum cell volume for refinement

  Input Parameters:
+ dm              - The `DM`
- refinementLimit - The maximum cell volume in the refined mesh

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexGetRefinementLimit()`, `DMPlexGetRefinementUniform()`, `DMPlexSetRefinementUniform()`
@*/
PetscErrorCode DMPlexSetRefinementLimit(DM dm, PetscReal refinementLimit)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementLimit = refinementLimit;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetRefinementLimit - Retrieve the maximum cell volume for refinement

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. refinementLimit - The maximum cell volume in the refined mesh

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexSetRefinementLimit()`, `DMPlexGetRefinementUniform()`, `DMPlexSetRefinementUniform()`
@*/
PetscErrorCode DMPlexGetRefinementLimit(DM dm, PetscReal *refinementLimit)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(refinementLimit, 2);
  /* if (mesh->refinementLimit < 0) = getMaxVolume()/2.0; */
  *refinementLimit = mesh->refinementLimit;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexSetRefinementFunction - Set the function giving the maximum cell volume for refinement

  Input Parameters:
+ dm             - The `DM`
- refinementFunc - Function giving the maximum cell volume in the refined mesh

  Calling Sequence of `refinementFunc`:
+ coords - Coordinates of the current point, usually a cell centroid
- limit  - The maximum cell volume for a cell containing this point

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexGetRefinementFunction()`, `DMPlexGetRefinementUniform()`, `DMPlexSetRefinementUniform()`, `DMPlexGetRefinementLimit()`, `DMPlexSetRefinementLimit()`
@*/
PetscErrorCode DMPlexSetRefinementFunction(DM dm, PetscErrorCode (*refinementFunc)(const PetscReal coords[], PetscReal *limit))
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementFunc = refinementFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetRefinementFunction - Get the function giving the maximum cell volume for refinement

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. refinementFunc - Function giving the maximum cell volume in the refined mesh

  Calling Sequence of `refinementFunc`:
+ coords - Coordinates of the current point, usually a cell centroid
- limit  - The maximum cell volume for a cell containing this point

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexSetRefinementFunction()`, `DMPlexGetRefinementUniform()`, `DMPlexSetRefinementUniform()`, `DMPlexGetRefinementLimit()`, `DMPlexSetRefinementLimit()`
@*/
PetscErrorCode DMPlexGetRefinementFunction(DM dm, PetscErrorCode (**refinementFunc)(const PetscReal coords[], PetscReal *limit))
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(refinementFunc, 2);
  *refinementFunc = mesh->refinementFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMRefine_Plex(DM dm, MPI_Comm comm, DM *rdm)
{
  PetscBool isUniform;

  PetscFunctionBegin;
  PetscCall(DMPlexGetRefinementUniform(dm, &isUniform));
  PetscCall(DMViewFromOptions(dm, NULL, "-initref_dm_view"));
  if (isUniform) {
    DMPlexTransform     tr;
    DM                  cdm, rcdm;
    DMPlexTransformType trType;
    const char         *prefix;
    PetscOptions        options;
    PetscInt            cDegree;
    PetscBool           useCeed, save;

    PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
    PetscCall(DMPlexTransformSetDM(tr, dm));
    PetscCall(DMPlexGetTransformType(dm, &trType));
    if (trType) PetscCall(DMPlexTransformSetType(tr, trType));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, prefix));
    PetscCall(PetscObjectGetOptions((PetscObject)dm, &options));
    PetscCall(PetscObjectSetOptions((PetscObject)tr, options));
    PetscCall(DMPlexTransformSetFromOptions(tr));
    PetscCall(PetscObjectSetOptions((PetscObject)tr, NULL));
    PetscCall(DMPlexTransformSetUp(tr));
    PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
    PetscCall(DMPlexTransformApply(tr, dm, rdm));
    PetscCall(DMPlexSetRegularRefinement(*rdm, PETSC_TRUE));
    PetscCall(DMPlexGetUseCeed(dm, &useCeed));
    PetscCall(DMPlexSetUseCeed(*rdm, useCeed));
    PetscCall(DMSetMatType(*rdm, dm->mattype));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateDM(*rdm, &rcdm));
    PetscCall(DMGetCoordinateDegree_Internal(dm, &cDegree));
    {
      PetscDS cds, rcds;

      PetscCall(DMPlexCreateCoordinateSpace(*rdm, cDegree, PETSC_FALSE, PETSC_TRUE));
      PetscCall(DMGetCoordinateDM(*rdm, &rcdm));
      PetscCall(DMGetDS(cdm, &cds));
      PetscCall(DMGetDS(rcdm, &rcds));
      PetscCall(PetscDSCopyConstants(cds, rcds));
    }
    PetscCall(DMPlexGetUseCeed(cdm, &useCeed));
    PetscCall(DMPlexSetUseCeed(rcdm, useCeed));
    if (useCeed) {
      PetscCall(DMPlexSetUseMatClosurePermutation(rcdm, PETSC_FALSE));
      PetscCall(DMUseTensorOrder(rcdm, PETSC_TRUE));
    }
    PetscCall(DMPlexTransformCreateDiscLabels(tr, *rdm));
    PetscCall(DMPlexGetSaveTransform(dm, &save));
    if (save) PetscCall(DMPlexSetTransform(*rdm, tr));
    PetscCall(DMPlexTransformDestroy(&tr));
  } else {
    PetscCall(DMPlexRefine_Internal(dm, NULL, NULL, NULL, rdm));
  }
  if (*rdm) {
    ((DM_Plex *)(*rdm)->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
    ((DM_Plex *)(*rdm)->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
  }
  PetscCall(DMViewFromOptions(*rdm, NULL, "-postref_dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMRefineHierarchy_Plex(DM dm, PetscInt nlevels, DM rdm[])
{
  DM        cdm = dm;
  PetscInt  r;
  PetscBool isUniform, localized, useCeed;

  PetscFunctionBegin;
  PetscCall(DMPlexGetRefinementUniform(dm, &isUniform));
  PetscCall(DMGetCoordinatesLocalized(dm, &localized));
  if (isUniform) {
    for (r = 0; r < nlevels; ++r) {
      DMPlexTransform tr;
      DM              codm, rcodm;
      const char     *prefix;

      PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)cdm), &tr));
      PetscCall(PetscObjectGetOptionsPrefix((PetscObject)cdm, &prefix));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, prefix));
      PetscCall(DMPlexTransformSetDM(tr, cdm));
      PetscCall(DMPlexTransformSetFromOptions(tr));
      PetscCall(DMPlexTransformSetUp(tr));
      PetscCall(DMPlexTransformApply(tr, cdm, &rdm[r]));
      PetscCall(DMSetCoarsenLevel(rdm[r], cdm->leveldown));
      PetscCall(DMSetRefineLevel(rdm[r], cdm->levelup + 1));
      PetscCall(DMSetMatType(rdm[r], dm->mattype));
      PetscCall(DMPlexGetUseCeed(dm, &useCeed));
      PetscCall(DMPlexSetUseCeed(rdm[r], useCeed));
      PetscCall(DMCopyDisc(cdm, rdm[r]));
      PetscCall(DMGetCoordinateDM(dm, &codm));
      PetscCall(DMGetCoordinateDM(rdm[r], &rcodm));
      PetscCall(DMCopyDisc(codm, rcodm));
      PetscCall(DMPlexGetUseCeed(codm, &useCeed));
      PetscCall(DMPlexSetUseCeed(rcodm, useCeed));
      if (useCeed) {
        PetscCall(DMPlexSetUseMatClosurePermutation(rcodm, PETSC_FALSE));
        PetscCall(DMUseTensorOrder(rcodm, PETSC_TRUE));
      }
      PetscCall(DMPlexTransformCreateDiscLabels(tr, rdm[r]));
      PetscCall(DMSetCoarseDM(rdm[r], cdm));
      PetscCall(DMPlexSetRegularRefinement(rdm[r], PETSC_TRUE));
      if (rdm[r]) {
        ((DM_Plex *)rdm[r]->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
        ((DM_Plex *)rdm[r]->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
      }
      cdm = rdm[r];
      PetscCall(DMPlexTransformDestroy(&tr));
    }
  } else {
    for (r = 0; r < nlevels; ++r) {
      PetscCall(DMRefine(cdm, PetscObjectComm((PetscObject)dm), &rdm[r]));
      PetscCall(DMPlexGetUseCeed(dm, &useCeed));
      PetscCall(DMPlexSetUseCeed(rdm[r], useCeed));
      PetscCall(DMCopyDisc(cdm, rdm[r]));
      if (localized) PetscCall(DMLocalizeCoordinates(rdm[r]));
      PetscCall(DMSetCoarseDM(rdm[r], cdm));
      if (rdm[r]) {
        ((DM_Plex *)rdm[r]->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
        ((DM_Plex *)rdm[r]->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
      }
      cdm = rdm[r];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

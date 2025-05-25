#define PETSCDM_DLL
#include <petsc/private/dmswarmimpl.h> /*I   "petscdmswarm.h"   I*/
#include <petscsf.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdt.h>
#include "../src/dm/impls/swarm/data_bucket.h"

#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */

PetscClassId DMSWARMCELLDM_CLASSID;

/*@
  DMSwarmCellDMDestroy - destroy a `DMSwarmCellDM`

  Collective

  Input Parameter:
. celldm - address of `DMSwarmCellDM`

  Level: advanced

.seealso: `DMSwarmCellDM`, `DMSwarmCellDMCreate()`
@*/
PetscErrorCode DMSwarmCellDMDestroy(DMSwarmCellDM *celldm)
{
  PetscFunctionBegin;
  if (!*celldm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*celldm, DMSWARMCELLDM_CLASSID, 1);
  if (--((PetscObject)*celldm)->refct > 0) {
    *celldm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscTryTypeMethod(*celldm, destroy);
  for (PetscInt f = 0; f < (*celldm)->Nf; ++f) PetscCall(PetscFree((*celldm)->dmFields[f]));
  PetscCall(PetscFree((*celldm)->dmFields));
  for (PetscInt f = 0; f < (*celldm)->Nfc; ++f) PetscCall(PetscFree((*celldm)->coordFields[f]));
  PetscCall(PetscFree((*celldm)->coordFields));
  PetscCall(PetscFree((*celldm)->cellid));
  PetscCall(DMSwarmSortDestroy(&(*celldm)->sort));
  PetscCall(DMDestroy(&(*celldm)->dm));
  PetscCall(PetscHeaderDestroy(celldm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmCellDMView - view a `DMSwarmCellDM`

  Collective

  Input Parameters:
+ celldm - `DMSwarmCellDM`
- viewer - viewer to display field, for example `PETSC_VIEWER_STDOUT_WORLD`

  Level: advanced

.seealso: `DMSwarmCellDM`, `DMSwarmCellDMCreate()`
@*/
PetscErrorCode DMSwarmCellDMView(DMSwarmCellDM celldm, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)celldm), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(celldm, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)celldm, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "solution field%s:", celldm->Nf > 1 ? "s" : ""));
    for (PetscInt f = 0; f < celldm->Nf; ++f) PetscCall(PetscViewerASCIIPrintf(viewer, " %s", celldm->dmFields[f]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "coordinate field%s:", celldm->Nfc > 1 ? "s" : ""));
    for (PetscInt f = 0; f < celldm->Nfc; ++f) PetscCall(PetscViewerASCIIPrintf(viewer, " %s", celldm->coordFields[f]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT));
    PetscCall(DMView(celldm->dm, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscTryTypeMethod(celldm, view, viewer);
  if (isascii) PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmCellDMGetDM - Returns the background `DM` for the `DMSwarm`

  Not Collective

  Input Parameter:
. celldm - The `DMSwarmCellDM` object

  Output Parameter:
. dm - The `DM` object

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMGetDM(DMSwarmCellDM celldm, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  *dm = celldm->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmCellDMGetFields - Returns the `DM` fields for the `DMSwarm`

  Not Collective

  Input Parameter:
. celldm - The `DMSwarmCellDM` object

  Output Parameters:
+ Nf    - The number of fields
- names - The array of field names in the `DMSWARM`

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMGetFields(DMSwarmCellDM celldm, PetscInt *Nf, const char **names[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  if (Nf) {
    PetscAssertPointer(Nf, 2);
    *Nf = celldm->Nf;
  }
  if (names) {
    PetscAssertPointer(names, 3);
    *names = (const char **)celldm->dmFields;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmCellDMGetCoordinateFields - Returns the `DM` coordinate fields for the `DMSwarm`

  Not Collective

  Input Parameter:
. celldm - The `DMSwarmCellDM` object

  Output Parameters:
+ Nfc   - The number of coordinate fields
- names - The array of coordinate field names in the `DMSWARM`

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMGetCoordinateFields(DMSwarmCellDM celldm, PetscInt *Nfc, const char **names[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  if (Nfc) {
    PetscAssertPointer(Nfc, 2);
    *Nfc = celldm->Nfc;
  }
  if (names) {
    PetscAssertPointer(names, 3);
    *names = (const char **)celldm->coordFields;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmCellDMGetCellID - Returns the cell id field name for the `DMSwarm`

  Not Collective

  Input Parameter:
. celldm - The `DMSwarmCellDM` object

  Output Parameters:
. cellid - The cell id field name in the `DMSWARM`

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMGetCellID(DMSwarmCellDM celldm, const char *cellid[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  PetscAssertPointer(cellid, 2);
  *cellid = celldm->cellid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmCellDMGetSort - Returns the sort context over the active `DMSwarmCellDM` for the `DMSwarm`

  Not Collective

  Input Parameter:
. celldm - The `DMSwarmCellDM` object

  Output Parameter:
. sort - The `DMSwarmSort` object

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMGetSort(DMSwarmCellDM celldm, DMSwarmSort *sort)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  PetscAssertPointer(sort, 2);
  *sort = celldm->sort;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmCellDMSetSort - Sets the sort context over the active `DMSwarmCellDM` for the `DMSwarm`

  Not Collective

  Input Parameters:
+ celldm - The `DMSwarmCellDM` object
- sort   - The `DMSwarmSort` object

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMSetSort(DMSwarmCellDM celldm, DMSwarmSort sort)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  PetscAssertPointer(sort, 2);
  celldm->sort = sort;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmCellDMGetBlockSize - Returns the total blocksize for the `DM` fields

  Not Collective

  Input Parameters:
+ celldm - The `DMSwarmCellDM` object
- sw     - The `DMSwarm` object

  Output Parameter:
. bs - The total block size

  Level: intermediate

.seealso: `DMSwarmCellDM`, `DM`, `DMSwarmSetCellDM()`
@*/
PetscErrorCode DMSwarmCellDMGetBlockSize(DMSwarmCellDM celldm, DM sw, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(celldm, DMSWARMCELLDM_CLASSID, 1);
  PetscValidHeaderSpecific(sw, DM_CLASSID, 2);
  PetscAssertPointer(bs, 3);
  *bs = 0;
  for (PetscInt f = 0; f < celldm->Nf; ++f) {
    PetscInt fbs;

    PetscCall(DMSwarmGetFieldInfo(sw, celldm->dmFields[f], &fbs, NULL));
    *bs += fbs;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmCellDMCreate - create a `DMSwarmCellDM`

  Collective

  Input Parameters:
+ dm          - The background `DM` for the `DMSwarm`
. Nf          - The number of swarm fields defined over `dm`
. dmFields    - The swarm field names for the `dm` fields
. Nfc         - The number of swarm fields to use for coordinates over `dm`
- coordFields - The swarm field names for the `dm` coordinate fields

  Output Parameter:
. celldm - The new `DMSwarmCellDM`

  Level: advanced

.seealso: `DMSwarmCellDM`, `DMSWARM`, `DMSetType()`
@*/
PetscErrorCode DMSwarmCellDMCreate(DM dm, PetscInt Nf, const char *dmFields[], PetscInt Nfc, const char *coordFields[], DMSwarmCellDM *celldm)
{
  DMSwarmCellDM b;
  const char   *name;
  char          cellid[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (Nf) PetscAssertPointer(dmFields, 3);
  if (Nfc) PetscAssertPointer(coordFields, 5);
  PetscCall(DMInitializePackage());

  PetscCall(PetscHeaderCreate(b, DMSWARMCELLDM_CLASSID, "DMSwarmCellDM", "Background DM for a Swarm", "DM", PetscObjectComm((PetscObject)dm), DMSwarmCellDMDestroy, DMSwarmCellDMView));
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(PetscObjectSetName((PetscObject)b, name));
  PetscCall(PetscObjectReference((PetscObject)dm));
  b->dm  = dm;
  b->Nf  = Nf;
  b->Nfc = Nfc;
  PetscCall(PetscMalloc1(b->Nf, &b->dmFields));
  for (PetscInt f = 0; f < b->Nf; ++f) PetscCall(PetscStrallocpy(dmFields[f], &b->dmFields[f]));
  PetscCall(PetscMalloc1(b->Nfc, &b->coordFields));
  for (PetscInt f = 0; f < b->Nfc; ++f) PetscCall(PetscStrallocpy(coordFields[f], &b->coordFields[f]));
  PetscCall(PetscSNPrintf(cellid, PETSC_MAX_PATH_LEN, "%s_cellid", name));
  PetscCall(PetscStrallocpy(cellid, &b->cellid));
  *celldm = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Coordinate insertition/addition API */
/*@
  DMSwarmSetPointsUniformCoordinates - Set point coordinates in a `DMSWARM` on a regular (ijk) grid

  Collective

  Input Parameters:
+ sw      - the `DMSWARM`
. min     - minimum coordinate values in the x, y, z directions (array of length dim)
. max     - maximum coordinate values in the x, y, z directions (array of length dim)
. npoints - number of points in each spatial direction (array of length dim)
- mode    - indicates whether to append points to the swarm (`ADD_VALUES`), or over-ride existing points (`INSERT_VALUES`)

  Level: beginner

  Notes:
  When using mode = `INSERT_VALUES`, this method will reset the number of particles in the `DMSWARM`
  to be `npoints[0]` x `npoints[1]` (2D) or `npoints[0]` x `npoints[1]` x `npoints[2]` (3D). When using mode = `ADD_VALUES`,
  new points will be appended to any already existing in the `DMSWARM`

.seealso: `DM`, `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmSetPointsUniformCoordinates(DM sw, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)
{
  PetscReal          lmin[] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal          lmax[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt           i, j, k, bs, b, n_estimate, n_curr, n_new_est, p, n_found, Nfc;
  const PetscScalar *_coor;
  DMSwarmCellDM      celldm;
  DM                 dm;
  PetscReal          dx[3];
  PetscInt           _npoints[] = {0, 0, 1};
  Vec                pos;
  PetscScalar       *_pos;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscSF            sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  const char       **coordFields, *cellid;

  PetscFunctionBegin;
  DMSWARMPICVALID(sw);
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);

  PetscCall(DMSwarmCellDMGetDM(celldm, &dm));
  PetscCall(DMGetLocalBoundingBox(dm, lmin, lmax));
  PetscCall(DMGetCoordinateDim(dm, &bs));

  for (b = 0; b < bs; b++) {
    if (npoints[b] > 1) {
      dx[b] = (max[b] - min[b]) / ((PetscReal)(npoints[b] - 1));
    } else {
      dx[b] = 0.0;
    }
    _npoints[b] = npoints[b];
  }

  /* determine number of points living in the bounding box */
  n_estimate = 0;
  for (k = 0; k < _npoints[2]; k++) {
    for (j = 0; j < _npoints[1]; j++) {
      for (i = 0; i < _npoints[0]; i++) {
        PetscReal xp[] = {0.0, 0.0, 0.0};
        PetscInt  ijk[3];
        PetscBool point_inside = PETSC_TRUE;

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        for (b = 0; b < bs; b++) xp[b] = min[b] + ijk[b] * dx[b];
        for (b = 0; b < bs; b++) {
          if (xp[b] < lmin[b]) point_inside = PETSC_FALSE;
          if (xp[b] > lmax[b]) point_inside = PETSC_FALSE;
        }
        if (point_inside) n_estimate++;
      }
    }
  }

  /* create candidate list */
  PetscCall(VecCreate(PETSC_COMM_SELF, &pos));
  PetscCall(VecSetSizes(pos, bs * n_estimate, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(pos, bs));
  PetscCall(VecSetFromOptions(pos));
  PetscCall(VecGetArray(pos, &_pos));

  n_estimate = 0;
  for (k = 0; k < _npoints[2]; k++) {
    for (j = 0; j < _npoints[1]; j++) {
      for (i = 0; i < _npoints[0]; i++) {
        PetscReal xp[] = {0.0, 0.0, 0.0};
        PetscInt  ijk[3];
        PetscBool point_inside = PETSC_TRUE;

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        for (b = 0; b < bs; b++) xp[b] = min[b] + ijk[b] * dx[b];
        for (b = 0; b < bs; b++) {
          if (xp[b] < lmin[b]) point_inside = PETSC_FALSE;
          if (xp[b] > lmax[b]) point_inside = PETSC_FALSE;
        }
        if (point_inside) {
          for (b = 0; b < bs; b++) _pos[bs * n_estimate + b] = xp[b];
          n_estimate++;
        }
      }
    }
  }
  PetscCall(VecRestoreArray(pos, &_pos));

  /* locate points */
  PetscCall(DMLocatePoints(dm, pos, DM_POINTLOCATION_NONE, &sfcell));
  PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) n_found++;
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    PetscCall(DMSwarmGetLocalSize(sw, &n_curr));
    n_new_est = n_curr + n_found;
    PetscCall(DMSwarmSetLocalSizes(sw, n_new_est, -1));
  }
  if (mode == INSERT_VALUES) {
    n_curr    = 0;
    n_new_est = n_found;
    PetscCall(DMSwarmSetLocalSizes(sw, n_new_est, -1));
  }

  /* initialize new coords, cell owners, pid */
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(VecGetArrayRead(pos, &_coor));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (b = 0; b < bs; b++) swarm_coor[bs * (n_curr + n_found) + b] = PetscRealPart(_coor[bs * p + b]);
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(VecRestoreArrayRead(pos, &_coor));

  PetscCall(PetscSFDestroy(&sfcell));
  PetscCall(VecDestroy(&pos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetPointCoordinates - Set point coordinates in a `DMSWARM` from a user defined list

  Collective

  Input Parameters:
+ sw        - the `DMSWARM`
. npoints   - the number of points to insert
. coor      - the coordinate values
. redundant - if set to `PETSC_TRUE`, it is assumed that `npoints` and `coor` are only valid on rank 0 and should be broadcast to other ranks
- mode      - indicates whether to append points to the swarm (`ADD_VALUES`), or over-ride existing points (`INSERT_VALUES`)

  Level: beginner

  Notes:
  If the user has specified `redundant` as `PETSC_FALSE`, the cell `DM` will attempt to locate the coordinates provided by `coor` within
  its sub-domain. If they any values within `coor` are not located in the sub-domain, they will be ignored and will not get
  added to the `DMSWARM`.

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`, `DMSwarmSetPointsUniformCoordinates()`
@*/
PetscErrorCode DMSwarmSetPointCoordinates(DM sw, PetscInt npoints, PetscReal coor[], PetscBool redundant, InsertMode mode)
{
  PetscReal          gmin[] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal          gmax[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt           i, N, bs, b, n_estimate, n_curr, n_new_est, p, n_found;
  Vec                coorlocal;
  const PetscScalar *_coor;
  DMSwarmCellDM      celldm;
  DM                 dm;
  Vec                pos;
  PetscScalar       *_pos;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscSF            sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  PetscReal         *my_coor;
  PetscInt           my_npoints, Nfc;
  PetscMPIInt        rank;
  MPI_Comm           comm;
  const char       **coordFields, *cellid;

  PetscFunctionBegin;
  DMSWARMPICVALID(sw);
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);

  PetscCall(DMSwarmCellDMGetDM(celldm, &dm));
  PetscCall(DMGetCoordinatesLocal(dm, &coorlocal));
  PetscCall(VecGetSize(coorlocal, &N));
  PetscCall(VecGetBlockSize(coorlocal, &bs));
  N = N / bs;
  PetscCall(VecGetArrayRead(coorlocal, &_coor));
  for (i = 0; i < N; i++) {
    for (b = 0; b < bs; b++) {
      gmin[b] = PetscMin(gmin[b], PetscRealPart(_coor[bs * i + b]));
      gmax[b] = PetscMax(gmax[b], PetscRealPart(_coor[bs * i + b]));
    }
  }
  PetscCall(VecRestoreArrayRead(coorlocal, &_coor));

  /* broadcast points from rank 0 if requested */
  if (redundant) {
    PetscMPIInt imy;

    my_npoints = npoints;
    PetscCallMPI(MPI_Bcast(&my_npoints, 1, MPIU_INT, 0, comm));

    if (rank > 0) { /* allocate space */
      PetscCall(PetscMalloc1(bs * my_npoints, &my_coor));
    } else {
      my_coor = coor;
    }
    PetscCall(PetscMPIIntCast(bs * my_npoints, &imy));
    PetscCallMPI(MPI_Bcast(my_coor, imy, MPIU_REAL, 0, comm));
  } else {
    my_npoints = npoints;
    my_coor    = coor;
  }

  /* determine the number of points living in the bounding box */
  n_estimate = 0;
  for (i = 0; i < my_npoints; i++) {
    PetscBool point_inside = PETSC_TRUE;

    for (b = 0; b < bs; b++) {
      if (my_coor[bs * i + b] < gmin[b]) point_inside = PETSC_FALSE;
      if (my_coor[bs * i + b] > gmax[b]) point_inside = PETSC_FALSE;
    }
    if (point_inside) n_estimate++;
  }

  /* create candidate list */
  PetscCall(VecCreate(PETSC_COMM_SELF, &pos));
  PetscCall(VecSetSizes(pos, bs * n_estimate, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(pos, bs));
  PetscCall(VecSetFromOptions(pos));
  PetscCall(VecGetArray(pos, &_pos));

  n_estimate = 0;
  for (i = 0; i < my_npoints; i++) {
    PetscBool point_inside = PETSC_TRUE;

    for (b = 0; b < bs; b++) {
      if (my_coor[bs * i + b] < gmin[b]) point_inside = PETSC_FALSE;
      if (my_coor[bs * i + b] > gmax[b]) point_inside = PETSC_FALSE;
    }
    if (point_inside) {
      for (b = 0; b < bs; b++) _pos[bs * n_estimate + b] = my_coor[bs * i + b];
      n_estimate++;
    }
  }
  PetscCall(VecRestoreArray(pos, &_pos));

  /* locate points */
  PetscCall(DMLocatePoints(dm, pos, DM_POINTLOCATION_NONE, &sfcell));

  PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) n_found++;
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    PetscCall(DMSwarmGetLocalSize(sw, &n_curr));
    n_new_est = n_curr + n_found;
    PetscCall(DMSwarmSetLocalSizes(sw, n_new_est, -1));
  }
  if (mode == INSERT_VALUES) {
    n_curr    = 0;
    n_new_est = n_found;
    PetscCall(DMSwarmSetLocalSizes(sw, n_new_est, -1));
  }

  /* initialize new coords, cell owners, pid */
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(VecGetArrayRead(pos, &_coor));
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (b = 0; b < bs; b++) swarm_coor[bs * (n_curr + n_found) + b] = PetscRealPart(_coor[bs * p + b]);
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(VecRestoreArrayRead(pos, &_coor));

  if (redundant) {
    if (rank > 0) PetscCall(PetscFree(my_coor));
  }
  PetscCall(PetscSFDestroy(&sfcell));
  PetscCall(VecDestroy(&pos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA(DM, DM, DMSwarmPICLayoutType, PetscInt);
extern PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX(DM, DM, DMSwarmPICLayoutType, PetscInt);

/*@
  DMSwarmInsertPointsUsingCellDM - Insert point coordinates within each cell

  Not Collective

  Input Parameters:
+ dm          - the `DMSWARM`
. layout_type - method used to fill each cell with the cell `DM`
- fill_param  - parameter controlling how many points per cell are added (the meaning of this parameter is dependent on the layout type)

  Level: beginner

  Notes:
  The insert method will reset any previous defined points within the `DMSWARM`.

  When using a `DMDA` both 2D and 3D are supported for all layout types provided you are using `DMDA_ELEMENT_Q1`.

  When using a `DMPLEX` the following case are supported\:
.vb
   (i) DMSWARMPIC_LAYOUT_REGULAR: 2D (triangle),
   (ii) DMSWARMPIC_LAYOUT_GAUSS: 2D and 3D provided the cell is a tri/tet or a quad/hex,
   (iii) DMSWARMPIC_LAYOUT_SUBDIVISION: 2D and 3D for quad/hex and 2D tri.
.ve

.seealso: `DMSWARM`, `DMSwarmPICLayoutType`, `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmInsertPointsUsingCellDM(DM dm, DMSwarmPICLayoutType layout_type, PetscInt fill_param)
{
  DM        celldm;
  PetscBool isDA, isPLEX;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMDA, &isDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMPLEX, &isPLEX));
  if (isDA) {
    PetscCall(private_DMSwarmInsertPointsUsingCellDM_DA(dm, celldm, layout_type, fill_param));
  } else if (isPLEX) {
    PetscCall(private_DMSwarmInsertPointsUsingCellDM_PLEX(dm, celldm, layout_type, fill_param));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMDA and DMPLEX");
  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM, DM, PetscInt, PetscReal *);

/*@C
  DMSwarmSetPointCoordinatesCellwise - Insert point coordinates (defined over the reference cell) within each cell

  Not Collective

  Input Parameters:
+ dm      - the `DMSWARM`
. npoints - the number of points to insert in each cell
- xi      - the coordinates (defined in the local coordinate system for each cell) to insert

  Level: beginner

  Notes:
  The method will reset any previous defined points within the `DMSWARM`.
  Only supported for `DMPLEX`. If you are using a `DMDA` it is recommended to either use
  `DMSwarmInsertPointsUsingCellDM()`, or extract and set the coordinates yourself the following code
.vb
    PetscReal  *coor;
    const char *coordname;
    DMSwarmGetCoordinateField(dm, &coordname);
    DMSwarmGetField(dm,coordname,NULL,NULL,(void**)&coor);
    // user code to define the coordinates here
    DMSwarmRestoreField(dm,coordname,NULL,NULL,(void**)&coor);
.ve

.seealso: `DMSWARM`, `DMSwarmSetCellDM()`, `DMSwarmInsertPointsUsingCellDM()`
@*/
PetscErrorCode DMSwarmSetPointCoordinatesCellwise(DM dm, PetscInt npoints, PetscReal xi[])
{
  DM        celldm;
  PetscBool isDA, isPLEX;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMDA, &isDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMPLEX, &isPLEX));
  PetscCheck(!isDA, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMPLEX. Recommended you use DMSwarmInsertPointsUsingCellDM()");
  PetscCheck(isPLEX, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMDA and DMPLEX");
  PetscCall(private_DMSwarmSetPointCoordinatesCellwise_PLEX(dm, celldm, npoints, xi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmCreatePointPerCellCount - Count the number of points within all cells in the cell DM

  Not Collective

  Input Parameter:
. sw - the `DMSWARM`

  Output Parameters:
+ ncells - the number of cells in the cell `DM` (optional argument, pass `NULL` to ignore)
- count  - array of length ncells containing the number of points per cell

  Level: beginner

  Notes:
  The array count is allocated internally and must be free'd by the user.

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmCreatePointPerCellCount(DM sw, PetscInt *ncells, PetscInt **count)
{
  DMSwarmCellDM celldm;
  PetscBool     isvalid;
  PetscInt      nel;
  PetscInt     *sum;
  const char   *cellid;

  PetscFunctionBegin;
  PetscCall(DMSwarmSortGetIsValid(sw, &isvalid));
  nel = 0;
  if (isvalid) {
    PetscInt e;

    PetscCall(DMSwarmSortGetSizes(sw, &nel, NULL));

    PetscCall(PetscMalloc1(nel, &sum));
    for (e = 0; e < nel; e++) PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, e, &sum[e]));
  } else {
    DM        dm;
    PetscBool isda, isplex, isshell;
    PetscInt  p, npoints;
    PetscInt *swarm_cellid;

    /* get the number of cells */
    PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
    PetscCall(DMSwarmCellDMGetDM(celldm, &dm));
    PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMDA, &isda));
    PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
    PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMSHELL, &isshell));
    if (isda) {
      PetscInt        _nel, _npe;
      const PetscInt *_element;

      PetscCall(DMDAGetElements(dm, &_nel, &_npe, &_element));
      nel = _nel;
      PetscCall(DMDARestoreElements(dm, &_nel, &_npe, &_element));
    } else if (isplex) {
      PetscInt ps, pe;

      PetscCall(DMPlexGetHeightStratum(dm, 0, &ps, &pe));
      nel = pe - ps;
    } else if (isshell) {
      PetscErrorCode (*method_DMShellGetNumberOfCells)(DM, PetscInt *);

      PetscCall(PetscObjectQueryFunction((PetscObject)dm, "DMGetNumberOfCells_C", &method_DMShellGetNumberOfCells));
      if (method_DMShellGetNumberOfCells) {
        PetscCall(method_DMShellGetNumberOfCells(dm, &nel));
      } else
        SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Cannot determine the number of cells for the DMSHELL object. User must provide a method via PetscObjectComposeFunction( (PetscObject)shelldm, \"DMGetNumberOfCells_C\", your_function_to_compute_number_of_cells);");
    } else SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Cannot determine the number of cells for a DM not of type DA, PLEX or SHELL");

    PetscCall(PetscMalloc1(nel, &sum));
    PetscCall(PetscArrayzero(sum, nel));
    PetscCall(DMSwarmGetLocalSize(sw, &npoints));
    PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
    PetscCall(DMSwarmGetField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
    for (p = 0; p < npoints; p++) {
      if (swarm_cellid[p] != DMLOCATEPOINT_POINT_NOT_FOUND) sum[swarm_cellid[p]]++;
    }
    PetscCall(DMSwarmRestoreField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  }
  if (ncells) *ncells = nel;
  *count = sum;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmGetNumSpecies - Get the number of particle species

  Not Collective

  Input Parameter:
. sw - the `DMSWARM`

  Output Parameters:
. Ns - the number of species

  Level: intermediate

.seealso: `DMSWARM`, `DMSwarmSetNumSpecies()`, `DMSwarmSetType()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmGetNumSpecies(DM sw, PetscInt *Ns)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  *Ns = swarm->Ns;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetNumSpecies - Set the number of particle species

  Not Collective

  Input Parameters:
+ sw - the `DMSWARM`
- Ns - the number of species

  Level: intermediate

.seealso: `DMSWARM`, `DMSwarmGetNumSpecies()`, `DMSwarmSetType()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmSetNumSpecies(DM sw, PetscInt Ns)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  swarm->Ns = Ns;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmGetCoordinateFunction - Get the function setting initial particle positions, if it exists

  Not Collective

  Input Parameter:
. sw - the `DMSWARM`

  Output Parameter:
. coordFunc - the function setting initial particle positions, or `NULL`, see `PetscSimplePointFn` for the calling sequence

  Level: intermediate

.seealso: `DMSWARM`, `DMSwarmSetCoordinateFunction()`, `DMSwarmGetVelocityFunction()`, `DMSwarmInitializeCoordinates()`, `PetscSimplePointFn`
@*/
PetscErrorCode DMSwarmGetCoordinateFunction(DM sw, PetscSimplePointFn **coordFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscAssertPointer(coordFunc, 2);
  *coordFunc = swarm->coordFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmSetCoordinateFunction - Set the function setting initial particle positions

  Not Collective

  Input Parameters:
+ sw        - the `DMSWARM`
- coordFunc - the function setting initial particle positions, see `PetscSimplePointFn` for the calling sequence

  Level: intermediate

.seealso: `DMSWARM`, `DMSwarmGetCoordinateFunction()`, `DMSwarmSetVelocityFunction()`, `DMSwarmInitializeCoordinates()`, `PetscSimplePointFn`
@*/
PetscErrorCode DMSwarmSetCoordinateFunction(DM sw, PetscSimplePointFn *coordFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscValidFunction(coordFunc, 2);
  swarm->coordFunc = coordFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmGetVelocityFunction - Get the function setting initial particle velocities, if it exists

  Not Collective

  Input Parameter:
. sw - the `DMSWARM`

  Output Parameter:
. velFunc - the function setting initial particle velocities, or `NULL`, see `PetscSimplePointFn` for the calling sequence

  Level: intermediate

.seealso: `DMSWARM`, `DMSwarmSetVelocityFunction()`, `DMSwarmGetCoordinateFunction()`, `DMSwarmInitializeVelocities()`, `PetscSimplePointFn`
@*/
PetscErrorCode DMSwarmGetVelocityFunction(DM sw, PetscSimplePointFn **velFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscAssertPointer(velFunc, 2);
  *velFunc = swarm->velFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmSetVelocityFunction - Set the function setting initial particle velocities

  Not Collective

  Input Parameters:
+ sw      - the `DMSWARM`
- velFunc - the function setting initial particle velocities, see `PetscSimplePointFn` for the calling sequence

  Level: intermediate

.seealso: `DMSWARM`, `DMSwarmGetVelocityFunction()`, `DMSwarmSetCoordinateFunction()`, `DMSwarmInitializeVelocities()`, `PetscSimplePointFn`
@*/
PetscErrorCode DMSwarmSetVelocityFunction(DM sw, PetscSimplePointFn *velFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscValidFunction(velFunc, 2);
  swarm->velFunc = velFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmComputeLocalSize - Compute the local number and distribution of particles based upon a density function

  Not Collective

  Input Parameters:
+ sw      - The `DMSWARM`
. N       - The target number of particles
- density - The density field for the particle layout, normalized to unity

  Level: advanced

  Note:
  One particle will be created for each species.

.seealso: `DMSWARM`, `DMSwarmComputeLocalSizeFromOptions()`
@*/
PetscErrorCode DMSwarmComputeLocalSize(DM sw, PetscInt N, PetscProbFn *density)
{
  DM               dm;
  DMSwarmCellDM    celldm;
  PetscQuadrature  quad;
  const PetscReal *xq, *wq;
  PetscReal       *n_int;
  PetscInt        *npc_s, *swarm_cellid, Ni;
  PetscReal        gmin[3], gmax[3], xi0[3];
  PetscInt         Ns, cStart, cEnd, c, dim, d, Nq, q, Np = 0, p, s;
  PetscBool        simplex;
  const char      *cellid;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  if (simplex) PetscCall(PetscDTStroudConicalQuadrature(dim, 1, 5, -1.0, 1.0, &quad));
  else PetscCall(PetscDTGaussTensorQuadrature(dim, 1, 5, -1.0, 1.0, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &xq, &wq));
  PetscCall(PetscCalloc2(Ns, &n_int, (cEnd - cStart) * Ns, &npc_s));
  /* Integrate the density function to get the number of particles in each cell */
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  for (c = 0; c < cEnd - cStart; ++c) {
    const PetscInt cell = c + cStart;
    PetscReal      v0[3], J[9], invJ[9], detJ, detJp = 2. / (gmax[0] - gmin[0]), xr[3], den;

    /* Have to transform quadrature points/weights to cell domain */
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    PetscCall(PetscArrayzero(n_int, Ns));
    for (q = 0; q < Nq; ++q) {
      CoordinatesRefToReal(dim, dim, xi0, v0, J, &xq[q * dim], xr);
      /* Have to transform mesh to domain of definition of PDF, [-1, 1], and weight PDF by |J|/2 */
      xr[0] = detJp * (xr[0] - gmin[0]) - 1.;

      for (s = 0; s < Ns; ++s) {
        PetscCall(density(xr, NULL, &den));
        n_int[s] += (detJp * den) * (detJ * wq[q]) / (PetscReal)Ns;
      }
    }
    for (s = 0; s < Ns; ++s) {
      Ni = N;
      npc_s[c * Ns + s] += (PetscInt)(Ni * n_int[s] + 0.5); // TODO Wish we wrapped round()
      Np += npc_s[c * Ns + s];
    }
  }
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  for (c = 0, p = 0; c < cEnd - cStart; ++c) {
    for (s = 0; s < Ns; ++s) {
      for (q = 0; q < npc_s[c * Ns + s]; ++q, ++p) swarm_cellid[p] = c;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(PetscFree2(n_int, npc_s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmComputeLocalSizeFromOptions - Compute the local number and distribution of particles based upon a density function determined by options

  Not Collective

  Input Parameter:
. sw - The `DMSWARM`

  Level: advanced

.seealso: `DMSWARM`, `DMSwarmComputeLocalSize()`
@*/
PetscErrorCode DMSwarmComputeLocalSizeFromOptions(DM sw)
{
  PetscProbFn *pdf;
  const char  *prefix;
  char         funcname[PETSC_MAX_PATH_LEN];
  PetscInt    *N, Ns, dim, n;
  PetscBool    flg;
  PetscMPIInt  size, rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sw), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sw), &rank));
  PetscCall(PetscCalloc1(size, &N));
  PetscOptionsBegin(PetscObjectComm((PetscObject)sw), "", "DMSwarm Options", "DMSWARM");
  n = size;
  PetscCall(PetscOptionsIntArray("-dm_swarm_num_particles", "The target number of particles", "", N, &n, NULL));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(PetscOptionsInt("-dm_swarm_num_species", "The number of species", "DMSwarmSetNumSpecies", Ns, &Ns, &flg));
  if (flg) PetscCall(DMSwarmSetNumSpecies(sw, Ns));
  PetscCall(PetscOptionsString("-dm_swarm_coordinate_function", "Function to determine particle coordinates", "DMSwarmSetCoordinateFunction", funcname, funcname, sizeof(funcname), &flg));
  PetscOptionsEnd();
  if (flg) {
    PetscSimplePointFn *coordFunc;

    PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
    PetscCall(PetscDLSym(NULL, funcname, (void **)&coordFunc));
    PetscCheck(coordFunc, PetscObjectComm((PetscObject)sw), PETSC_ERR_ARG_WRONG, "Could not locate function %s", funcname);
    PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
    PetscCall(DMSwarmSetLocalSizes(sw, N[rank] * Ns, 0));
    PetscCall(DMSwarmSetCoordinateFunction(sw, coordFunc));
  } else {
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
    PetscCall(PetscProbCreateFromOptions(dim, prefix, "-dm_swarm_coordinate_density", &pdf, NULL, NULL));
    PetscCall(DMSwarmComputeLocalSize(sw, N[rank], pdf));
  }
  PetscCall(PetscFree(N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmInitializeCoordinates - Determine the initial coordinates of particles for a PIC method

  Not Collective

  Input Parameter:
. sw - The `DMSWARM`

  Level: advanced

  Note:
  Currently, we randomly place particles in their assigned cell

.seealso: `DMSWARM`, `DMSwarmComputeLocalSize()`, `DMSwarmInitializeVelocities()`
@*/
PetscErrorCode DMSwarmInitializeCoordinates(DM sw)
{
  DMSwarmCellDM       celldm;
  PetscSimplePointFn *coordFunc;
  PetscScalar        *weight;
  PetscReal          *x;
  PetscInt           *species;
  void               *ctx;
  PetscBool           removePoints = PETSC_TRUE;
  PetscDataType       dtype;
  PetscInt            Nfc, Np, p, Ns, dim, d, bs;
  const char        **coordFields;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMSwarmGetCoordinateFunction(sw, &coordFunc));

  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);

  PetscCall(DMSwarmGetField(sw, coordFields[0], &bs, &dtype, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  if (coordFunc) {
    PetscCall(DMGetApplicationContext(sw, &ctx));
    for (p = 0; p < Np; ++p) {
      PetscScalar X[3];

      PetscCall((*coordFunc)(dim, 0., NULL, p, X, ctx));
      for (d = 0; d < dim; ++d) x[p * dim + d] = PetscRealPart(X[d]);
      weight[p]  = 1.0;
      species[p] = p % Ns;
    }
  } else {
    DM          dm;
    PetscRandom rnd;
    PetscReal   xi0[3];
    PetscInt    cStart, cEnd, c;

    PetscCall(DMSwarmGetCellDM(sw, &dm));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMGetApplicationContext(sw, &ctx));

    /* Set particle position randomly in cell, set weights to 1 */
    PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)dm), &rnd));
    PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
    PetscCall(PetscRandomSetFromOptions(rnd));
    PetscCall(DMSwarmSortGetAccess(sw));
    for (d = 0; d < dim; ++d) xi0[d] = -1.0;
    for (c = cStart; c < cEnd; ++c) {
      PetscReal v0[3], J[9], invJ[9], detJ;
      PetscInt *pidx, Npc, q;

      PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
      for (q = 0; q < Npc; ++q) {
        const PetscInt p = pidx[q];
        PetscReal      xref[3];

        for (d = 0; d < dim; ++d) PetscCall(PetscRandomGetValueReal(rnd, &xref[d]));
        CoordinatesRefToReal(dim, dim, xi0, v0, J, xref, &x[p * dim]);

        weight[p]  = 1.0 / Np;
        species[p] = p % Ns;
      }
      PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
    }
    PetscCall(PetscRandomDestroy(&rnd));
    PetscCall(DMSwarmSortRestoreAccess(sw));
  }
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));

  PetscCall(DMSwarmMigrate(sw, removePoints));
  PetscCall(DMLocalizeCoordinates(sw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmInitializeVelocities - Set the initial velocities of particles using a distribution.

  Collective

  Input Parameters:
+ sw      - The `DMSWARM` object
. sampler - A function which uniformly samples the velocity PDF
- v0      - The velocity scale for nondimensionalization for each species

  Level: advanced

  Note:
  If `v0` is zero for the first species, all velocities are set to zero. If it is zero for any other species, the effect will be to give that species zero velocity.

.seealso: `DMSWARM`, `DMSwarmComputeLocalSize()`, `DMSwarmInitializeCoordinates()`, `DMSwarmInitializeVelocitiesFromOptions()`
@*/
PetscErrorCode DMSwarmInitializeVelocities(DM sw, PetscProbFn *sampler, const PetscReal v0[])
{
  PetscSimplePointFn *velFunc;
  PetscReal          *v;
  PetscInt           *species;
  void               *ctx;
  PetscInt            dim, Np, p;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetVelocityFunction(sw, &velFunc));

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  if (v0[0] == 0.) {
    PetscCall(PetscArrayzero(v, Np * dim));
  } else if (velFunc) {
    PetscCall(DMGetApplicationContext(sw, &ctx));
    for (p = 0; p < Np; ++p) {
      PetscInt    s = species[p], d;
      PetscScalar vel[3];

      PetscCall((*velFunc)(dim, 0., NULL, p, vel, ctx));
      for (d = 0; d < dim; ++d) v[p * dim + d] = (v0[s] / v0[0]) * PetscRealPart(vel[d]);
    }
  } else {
    PetscRandom rnd;

    PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)sw), &rnd));
    PetscCall(PetscRandomSetInterval(rnd, 0, 1.));
    PetscCall(PetscRandomSetFromOptions(rnd));

    for (p = 0; p < Np; ++p) {
      PetscInt  s = species[p], d;
      PetscReal a[3], vel[3];

      for (d = 0; d < dim; ++d) PetscCall(PetscRandomGetValueReal(rnd, &a[d]));
      PetscCall(sampler(a, NULL, vel));
      for (d = 0; d < dim; ++d) v[p * dim + d] = (v0[s] / v0[0]) * vel[d];
    }
    PetscCall(PetscRandomDestroy(&rnd));
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmInitializeVelocitiesFromOptions - Set the initial velocities of particles using a distribution determined from options.

  Collective

  Input Parameters:
+ sw - The `DMSWARM` object
- v0 - The velocity scale for nondimensionalization for each species

  Level: advanced

.seealso: `DMSWARM`, `DMSwarmComputeLocalSize()`, `DMSwarmInitializeCoordinates()`, `DMSwarmInitializeVelocities()`
@*/
PetscErrorCode DMSwarmInitializeVelocitiesFromOptions(DM sw, const PetscReal v0[])
{
  PetscProbFn *sampler;
  PetscInt     dim;
  const char  *prefix;
  char         funcname[PETSC_MAX_PATH_LEN];
  PetscBool    flg;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)sw), "", "DMSwarm Options", "DMSWARM");
  PetscCall(PetscOptionsString("-dm_swarm_velocity_function", "Function to determine particle velocities", "DMSwarmSetVelocityFunction", funcname, funcname, sizeof(funcname), &flg));
  PetscOptionsEnd();
  if (flg) {
    PetscSimplePointFn *velFunc;

    PetscCall(PetscDLSym(NULL, funcname, (void **)&velFunc));
    PetscCheck(velFunc, PetscObjectComm((PetscObject)sw), PETSC_ERR_ARG_WRONG, "Could not locate function %s", funcname);
    PetscCall(DMSwarmSetVelocityFunction(sw, velFunc));
  }
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
  PetscCall(PetscProbCreateFromOptions(dim, prefix, "-dm_swarm_velocity_density", NULL, NULL, &sampler));
  PetscCall(DMSwarmInitializeVelocities(sw, sampler, v0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The input vector U is assumed to be from a PetscFE. The Swarm fields are input as auxiliary values.
PetscErrorCode DMProjectFieldLocal_Swarm(DM dm, PetscReal time, Vec U, PetscPointFn **funcs, InsertMode mode, Vec X)
{
  MPI_Comm         comm;
  DM               dmIn;
  PetscDS          ds;
  PetscTabulation *T;
  DMSwarmCellDM    celldm;
  PetscScalar     *a, *val, *u, *u_x;
  PetscFEGeom      fegeom;
  PetscReal       *xi, *v0, *J, *invJ, detJ = 1.0, v0ref[3] = {-1.0, -1.0, -1.0};
  PetscInt         dim, dE, Np, n, Nf, Nfc, Nfu, cStart, cEnd, maxC = 0, totbs = 0;
  const char     **coordFields, **fields;
  PetscReal      **coordVals, **vals;
  PetscInt        *cbs, *bs, *uOff, *uOff_x;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(VecGetDM(U, &dmIn));
  PetscCall(DMGetDimension(dmIn, &dim));
  PetscCall(DMGetCoordinateDim(dmIn, &dE));
  PetscCall(DMGetDS(dmIn, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nfu));
  PetscCall(PetscDSGetComponentOffsets(ds, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(ds, &uOff_x));
  PetscCall(PetscDSGetTabulation(ds, &T));
  PetscCall(PetscDSGetEvaluationArrays(ds, &u, NULL, &u_x));
  PetscCall(PetscMalloc3(dim, &v0, dim * dim, &J, dim * dim, &invJ));
  PetscCall(DMPlexGetHeightStratum(dmIn, 0, &cStart, &cEnd));

  fegeom.dim      = dim;
  fegeom.dimEmbed = dE;
  fegeom.v        = v0;
  fegeom.xi       = v0ref;
  fegeom.J        = J;
  fegeom.invJ     = invJ;
  fegeom.detJ     = &detJ;

  PetscCall(DMSwarmGetLocalSize(dm, &Np));
  PetscCall(VecGetLocalSize(X, &n));
  PetscCheck(n == Np, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Output vector local size %" PetscInt_FMT " != %" PetscInt_FMT " number of local particles", n, Np);
  PetscCall(DMSwarmGetCellDMActive(dm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCall(DMSwarmCellDMGetFields(celldm, &Nf, &fields));

  PetscCall(PetscMalloc2(Nfc, &coordVals, Nfc, &cbs));
  for (PetscInt i = 0; i < Nfc; ++i) PetscCall(DMSwarmGetField(dm, coordFields[i], &cbs[i], NULL, (void **)&coordVals[i]));
  PetscCall(PetscMalloc2(Nf, &vals, Nfc, &bs));
  for (PetscInt i = 0; i < Nf; ++i) {
    PetscCall(DMSwarmGetField(dm, fields[i], &bs[i], NULL, (void **)&vals[i]));
    totbs += bs[i];
  }

  PetscCall(DMSwarmSortGetAccess(dm));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscInt *pindices, Npc;

    PetscCall(DMSwarmSortGetPointsPerCell(dm, cell, &Npc, &pindices));
    maxC = PetscMax(maxC, Npc);
    PetscCall(DMSwarmSortRestorePointsPerCell(dm, cell, &Npc, &pindices));
  }
  PetscCall(PetscMalloc3(maxC * dim, &xi, maxC * totbs, &val, Nfu, &T));
  PetscCall(VecGetArray(X, &a));
  {
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      PetscInt *pindices, Npc;

      // TODO: Use DMField instead of assuming affine
      PetscCall(DMPlexComputeCellGeometryFEM(dmIn, cell, NULL, v0, J, invJ, &detJ));
      PetscCall(DMSwarmSortGetPointsPerCell(dm, cell, &Npc, &pindices));

      PetscScalar *closure = NULL;
      PetscInt     Ncl;

      // Get fields from input vector and auxiliary fields from swarm
      for (PetscInt p = 0; p < Npc; ++p) {
        PetscReal xr[8];
        PetscInt  off;

        off = 0;
        for (PetscInt i = 0; i < Nfc; ++i) {
          for (PetscInt b = 0; b < cbs[i]; ++b, ++off) xr[off] = coordVals[i][pindices[p] * cbs[i] + b];
        }
        PetscCheck(off == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The total block size of coordinates is %" PetscInt_FMT " != %" PetscInt_FMT " the DM coordinate dimension", off, dim);
        CoordinatesRealToRef(dE, dim, fegeom.xi, fegeom.v, fegeom.invJ, xr, &xi[p * dim]);
        off = 0;
        for (PetscInt i = 0; i < Nf; ++i) {
          for (PetscInt b = 0; b < bs[i]; ++b, ++off) val[p * totbs + off] = vals[i][pindices[p] * bs[i] + b];
        }
        PetscCheck(off == totbs, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The total block size of swarm fields is %" PetscInt_FMT " != %" PetscInt_FMT " the computed total block size", off, totbs);
      }
      PetscCall(DMPlexVecGetClosure(dmIn, NULL, U, cell, &Ncl, &closure));
      for (PetscInt field = 0; field < Nfu; ++field) {
        PetscFE fe;

        PetscCall(PetscDSGetDiscretization(ds, field, (PetscObject *)&fe));
        PetscCall(PetscFECreateTabulation(fe, 1, Npc, xi, 1, &T[field]));
      }
      for (PetscInt p = 0; p < Npc; ++p) {
        // Get fields from input vector
        PetscCall(PetscFEEvaluateFieldJets_Internal(ds, Nfu, 0, p, T, &fegeom, closure, NULL, u, u_x, NULL));
        (*funcs[0])(dim, 1, 1, uOff, uOff_x, u, NULL, u_x, bs, NULL, &val[p * totbs], NULL, NULL, time, &xi[p * dim], 0, NULL, &a[pindices[p]]);
      }
      PetscCall(DMSwarmSortRestorePointsPerCell(dm, cell, &Npc, &pindices));
      PetscCall(DMPlexVecRestoreClosure(dmIn, NULL, U, cell, &Ncl, &closure));
      for (PetscInt field = 0; field < Nfu; ++field) PetscCall(PetscTabulationDestroy(&T[field]));
    }
  }
  for (PetscInt i = 0; i < Nfc; ++i) PetscCall(DMSwarmRestoreField(dm, coordFields[i], &cbs[i], NULL, (void **)&coordVals[i]));
  for (PetscInt i = 0; i < Nf; ++i) PetscCall(DMSwarmRestoreField(dm, fields[i], &bs[i], NULL, (void **)&vals[i]));
  PetscCall(VecRestoreArray(X, &a));
  PetscCall(DMSwarmSortRestoreAccess(dm));
  PetscCall(PetscFree3(xi, val, T));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscCall(PetscFree2(coordVals, cbs));
  PetscCall(PetscFree2(vals, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

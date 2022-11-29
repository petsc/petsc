#define PETSCDM_DLL
#include <petsc/private/dmswarmimpl.h> /*I   "petscdmswarm.h"   I*/
#include <petscsf.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdt.h>
#include "../src/dm/impls/swarm/data_bucket.h"

#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */

/*
 Error checking to ensure the swarm type is correct and that a cell DM has been set
*/
#define DMSWARMPICVALID(dm) \
  do { \
    DM_Swarm *_swarm = (DM_Swarm *)(dm)->data; \
    PetscCheck(_swarm->swarm_type == DMSWARM_PIC, PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Valid only for DMSwarm-PIC. You must call DMSwarmSetType(dm,DMSWARM_PIC)"); \
    PetscCheck(_swarm->dmcell, PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Valid only for DMSwarmPIC if the cell DM is set. You must call DMSwarmSetCellDM(dm,celldm)"); \
  } while (0)

/* Coordinate insertition/addition API */
/*@C
   DMSwarmSetPointsUniformCoordinates - Set point coordinates in a DMSwarm on a regular (ijk) grid

   Collective on dm

   Input parameters:
+  dm - the DMSwarm
.  min - minimum coordinate values in the x, y, z directions (array of length dim)
.  max - maximum coordinate values in the x, y, z directions (array of length dim)
.  npoints - number of points in each spatial direction (array of length dim)
-  mode - indicates whether to append points to the swarm (ADD_VALUES), or over-ride existing points (INSERT_VALUES)

   Level: beginner

   Notes:
   When using mode = INSERT_VALUES, this method will reset the number of particles in the DMSwarm
   to be npoints[0]*npoints[1] (2D) or npoints[0]*npoints[1]*npoints[2] (3D). When using mode = ADD_VALUES,
   new points will be appended to any already existing in the DMSwarm

.seealso: `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSetPointsUniformCoordinates(DM dm, PetscReal min[], PetscReal max[], PetscInt npoints[], InsertMode mode)
{
  PetscReal          gmin[] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal          gmax[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt           i, j, k, N, bs, b, n_estimate, n_curr, n_new_est, p, n_found;
  Vec                coorlocal;
  const PetscScalar *_coor;
  DM                 celldm;
  PetscReal          dx[3];
  PetscInt           _npoints[] = {0, 0, 1};
  Vec                pos;
  PetscScalar       *_pos;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscSF            sfcell = NULL;
  const PetscSFNode *LA_sfcell;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(DMGetCoordinatesLocal(celldm, &coorlocal));
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
          if (xp[b] < gmin[b]) point_inside = PETSC_FALSE;
          if (xp[b] > gmax[b]) point_inside = PETSC_FALSE;
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
          if (xp[b] < gmin[b]) point_inside = PETSC_FALSE;
          if (xp[b] > gmax[b]) point_inside = PETSC_FALSE;
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
  PetscCall(DMLocatePoints(celldm, pos, DM_POINTLOCATION_NONE, &sfcell));
  PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) n_found++;
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    PetscCall(DMSwarmGetLocalSize(dm, &n_curr));
    n_new_est = n_curr + n_found;
    PetscCall(DMSwarmSetLocalSizes(dm, n_new_est, -1));
  }
  if (mode == INSERT_VALUES) {
    n_curr    = 0;
    n_new_est = n_found;
    PetscCall(DMSwarmSetLocalSizes(dm, n_new_est, -1));
  }

  /* initialize new coords, cell owners, pid */
  PetscCall(VecGetArrayRead(pos, &_coor));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (b = 0; b < bs; b++) swarm_coor[bs * (n_curr + n_found) + b] = PetscRealPart(_coor[bs * p + b]);
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&swarm_coor));
  PetscCall(VecRestoreArrayRead(pos, &_coor));

  PetscCall(PetscSFDestroy(&sfcell));
  PetscCall(VecDestroy(&pos));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSetPointCoordinates - Set point coordinates in a DMSwarm from a user defined list

   Collective on dm

   Input parameters:
+  dm - the DMSwarm
.  npoints - the number of points to insert
.  coor - the coordinate values
.  redundant - if set to PETSC_TRUE, it is assumed that npoints and coor[] are only valid on rank 0 and should be broadcast to other ranks
-  mode - indicates whether to append points to the swarm (ADD_VALUES), or over-ride existing points (INSERT_VALUES)

   Level: beginner

   Notes:
   If the user has specified redundant = PETSC_FALSE, the cell DM will attempt to locate the coordinates provided by coor[] within
   its sub-domain. If they any values within coor[] are not located in the sub-domain, they will be ignored and will not get
   added to the DMSwarm.

.seealso: `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`, `DMSwarmSetPointsUniformCoordinates()`
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinates(DM dm, PetscInt npoints, PetscReal coor[], PetscBool redundant, InsertMode mode)
{
  PetscReal          gmin[] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal          gmax[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt           i, N, bs, b, n_estimate, n_curr, n_new_est, p, n_found;
  Vec                coorlocal;
  const PetscScalar *_coor;
  DM                 celldm;
  Vec                pos;
  PetscScalar       *_pos;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscSF            sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  PetscReal         *my_coor;
  PetscInt           my_npoints;
  PetscMPIInt        rank;
  MPI_Comm           comm;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(DMGetCoordinatesLocal(celldm, &coorlocal));
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
    my_npoints = npoints;
    PetscCallMPI(MPI_Bcast(&my_npoints, 1, MPIU_INT, 0, comm));

    if (rank > 0) { /* allocate space */
      PetscCall(PetscMalloc1(bs * my_npoints, &my_coor));
    } else {
      my_coor = coor;
    }
    PetscCallMPI(MPI_Bcast(my_coor, bs * my_npoints, MPIU_REAL, 0, comm));
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
  PetscCall(DMLocatePoints(celldm, pos, DM_POINTLOCATION_NONE, &sfcell));

  PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) n_found++;
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    PetscCall(DMSwarmGetLocalSize(dm, &n_curr));
    n_new_est = n_curr + n_found;
    PetscCall(DMSwarmSetLocalSizes(dm, n_new_est, -1));
  }
  if (mode == INSERT_VALUES) {
    n_curr    = 0;
    n_new_est = n_found;
    PetscCall(DMSwarmSetLocalSizes(dm, n_new_est, -1));
  }

  /* initialize new coords, cell owners, pid */
  PetscCall(VecGetArrayRead(pos, &_coor));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
  n_found = 0;
  for (p = 0; p < n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (b = 0; b < bs; b++) swarm_coor[bs * (n_curr + n_found) + b] = PetscRealPart(_coor[bs * p + b]);
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&swarm_coor));
  PetscCall(VecRestoreArrayRead(pos, &_coor));

  if (redundant) {
    if (rank > 0) PetscCall(PetscFree(my_coor));
  }
  PetscCall(PetscSFDestroy(&sfcell));
  PetscCall(VecDestroy(&pos));
  PetscFunctionReturn(0);
}

extern PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA(DM, DM, DMSwarmPICLayoutType, PetscInt);
extern PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX(DM, DM, DMSwarmPICLayoutType, PetscInt);

/*@C
   DMSwarmInsertPointsUsingCellDM - Insert point coordinates within each cell

   Not collective

   Input parameters:
+  dm - the DMSwarm
.  layout_type - method used to fill each cell with the cell DM
-  fill_param - parameter controlling how many points per cell are added (the meaning of this parameter is dependent on the layout type)

   Level: beginner

   Notes:

   The insert method will reset any previous defined points within the DMSwarm.

   When using a DMDA both 2D and 3D are supported for all layout types provided you are using DMDA_ELEMENT_Q1.

   When using a DMPLEX the following case are supported:
   (i) DMSWARMPIC_LAYOUT_REGULAR: 2D (triangle),
   (ii) DMSWARMPIC_LAYOUT_GAUSS: 2D and 3D provided the cell is a tri/tet or a quad/hex,
   (iii) DMSWARMPIC_LAYOUT_SUBDIVISION: 2D and 3D for quad/hex and 2D tri.

.seealso: `DMSwarmPICLayoutType`, `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PETSC_EXTERN PetscErrorCode DMSwarmInsertPointsUsingCellDM(DM dm, DMSwarmPICLayoutType layout_type, PetscInt fill_param)
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
  PetscFunctionReturn(0);
}

extern PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM, DM, PetscInt, PetscReal *);

/*@C
   DMSwarmSetPointCoordinatesCellwise - Insert point coordinates (defined over the reference cell) within each cell

   Not collective

   Input parameters:
+  dm - the DMSwarm
.  celldm - the cell DM
.  npoints - the number of points to insert in each cell
-  xi - the coordinates (defined in the local coordinate system for each cell) to insert

 Level: beginner

 Notes:
 The method will reset any previous defined points within the DMSwarm.
 Only supported for DMPLEX. If you are using a DMDA it is recommended to either use
 DMSwarmInsertPointsUsingCellDM(), or extract and set the coordinates yourself the following code

$    PetscReal *coor;
$    DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&coor);
$    // user code to define the coordinates here
$    DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&coor);

.seealso: `DMSwarmSetCellDM()`, `DMSwarmInsertPointsUsingCellDM()`
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinatesCellwise(DM dm, PetscInt npoints, PetscReal xi[])
{
  DM        celldm;
  PetscBool isDA, isPLEX;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMDA, &isDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMPLEX, &isPLEX));
  PetscCheck(!isDA, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMPLEX. Recommended you use DMSwarmInsertPointsUsingCellDM()");
  if (isPLEX) {
    PetscCall(private_DMSwarmSetPointCoordinatesCellwise_PLEX(dm, celldm, npoints, xi));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMDA and DMPLEX");
  PetscFunctionReturn(0);
}

/* Field projection API */
extern PetscErrorCode private_DMSwarmProjectFields_DA(DM swarm, DM celldm, PetscInt project_type, PetscInt nfields, DMSwarmDataField dfield[], Vec vecs[]);
extern PetscErrorCode private_DMSwarmProjectFields_PLEX(DM swarm, DM celldm, PetscInt project_type, PetscInt nfields, DMSwarmDataField dfield[], Vec vecs[]);

/*@C
   DMSwarmProjectFields - Project a set of swarm fields onto the cell DM

   Collective on dm

   Input parameters:
+  dm - the DMSwarm
.  nfields - the number of swarm fields to project
.  fieldnames - the textual names of the swarm fields to project
.  fields - an array of Vec's of length nfields
-  reuse - flag indicating whether the array and contents of fields should be re-used or internally allocated

   Currently, the only available projection method consists of
     phi_i = \sum_{p=0}^{np} N_i(x_p) phi_p dJ / \sum_{p=0}^{np} N_i(x_p) dJ
   where phi_p is the swarm field at point p,
     N_i() is the cell DM basis function at vertex i,
     dJ is the determinant of the cell Jacobian and
     phi_i is the projected vertex value of the field phi.

   Level: beginner

   Notes:

   If reuse = PETSC_FALSE, this function will allocate the array of Vec's, and each individual Vec.
     The user is responsible for destroying both the array and the individual Vec objects.

   Only swarm fields registered with data type = PETSC_REAL can be projected onto the cell DM.

   Only swarm fields of block size = 1 can currently be projected.

   The only projection methods currently only support the DA (2D) and PLEX (triangles 2D).

.seealso: `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PETSC_EXTERN PetscErrorCode DMSwarmProjectFields(DM dm, PetscInt nfields, const char *fieldnames[], Vec **fields, PetscBool reuse)
{
  DM_Swarm         *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataField *gfield;
  DM                celldm;
  PetscBool         isDA, isPLEX;
  Vec              *vecs;
  PetscInt          f, nvecs;
  PetscInt          project_type = 0;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(PetscMalloc1(nfields, &gfield));
  nvecs = 0;
  for (f = 0; f < nfields; f++) {
    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, fieldnames[f], &gfield[f]));
    PetscCheck(gfield[f]->petsc_type == PETSC_REAL, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Projection only valid for fields using a data type = PETSC_REAL");
    PetscCheck(gfield[f]->bs == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Projection only valid for fields with block size = 1");
    nvecs += gfield[f]->bs;
  }
  if (!reuse) {
    PetscCall(PetscMalloc1(nvecs, &vecs));
    for (f = 0; f < nvecs; f++) {
      PetscCall(DMCreateGlobalVector(celldm, &vecs[f]));
      PetscCall(PetscObjectSetName((PetscObject)vecs[f], gfield[f]->name));
    }
  } else {
    vecs = *fields;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMDA, &isDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMPLEX, &isPLEX));
  if (isDA) {
    PetscCall(private_DMSwarmProjectFields_DA(dm, celldm, project_type, nfields, gfield, vecs));
  } else if (isPLEX) {
    PetscCall(private_DMSwarmProjectFields_PLEX(dm, celldm, project_type, nfields, gfield, vecs));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMDA and DMPLEX");

  PetscCall(PetscFree(gfield));
  if (!reuse) *fields = vecs;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmCreatePointPerCellCount - Count the number of points within all cells in the cell DM

   Not collective

   Input parameter:
.  dm - the DMSwarm

   Output parameters:
+  ncells - the number of cells in the cell DM (optional argument, pass NULL to ignore)
-  count - array of length ncells containing the number of points per cell

   Level: beginner

   Notes:
   The array count is allocated internally and must be free'd by the user.

.seealso: `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PETSC_EXTERN PetscErrorCode DMSwarmCreatePointPerCellCount(DM dm, PetscInt *ncells, PetscInt **count)
{
  PetscBool isvalid;
  PetscInt  nel;
  PetscInt *sum;

  PetscFunctionBegin;
  PetscCall(DMSwarmSortGetIsValid(dm, &isvalid));
  nel = 0;
  if (isvalid) {
    PetscInt e;

    PetscCall(DMSwarmSortGetSizes(dm, &nel, NULL));

    PetscCall(PetscMalloc1(nel, &sum));
    for (e = 0; e < nel; e++) PetscCall(DMSwarmSortGetNumberOfPointsPerCell(dm, e, &sum[e]));
  } else {
    DM        celldm;
    PetscBool isda, isplex, isshell;
    PetscInt  p, npoints;
    PetscInt *swarm_cellid;

    /* get the number of cells */
    PetscCall(DMSwarmGetCellDM(dm, &celldm));
    PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMDA, &isda));
    PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMPLEX, &isplex));
    PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMSHELL, &isshell));
    if (isda) {
      PetscInt        _nel, _npe;
      const PetscInt *_element;

      PetscCall(DMDAGetElements(celldm, &_nel, &_npe, &_element));
      nel = _nel;
      PetscCall(DMDARestoreElements(celldm, &_nel, &_npe, &_element));
    } else if (isplex) {
      PetscInt ps, pe;

      PetscCall(DMPlexGetHeightStratum(celldm, 0, &ps, &pe));
      nel = pe - ps;
    } else if (isshell) {
      PetscErrorCode (*method_DMShellGetNumberOfCells)(DM, PetscInt *);

      PetscCall(PetscObjectQueryFunction((PetscObject)celldm, "DMGetNumberOfCells_C", &method_DMShellGetNumberOfCells));
      if (method_DMShellGetNumberOfCells) {
        PetscCall(method_DMShellGetNumberOfCells(celldm, &nel));
      } else
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Cannot determine the number of cells for the DMSHELL object. User must provide a method via PetscObjectComposeFunction( (PetscObject)shelldm, \"DMGetNumberOfCells_C\", your_function_to_compute_number_of_cells);");
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Cannot determine the number of cells for a DM not of type DA, PLEX or SHELL");

    PetscCall(PetscMalloc1(nel, &sum));
    PetscCall(PetscArrayzero(sum, nel));
    PetscCall(DMSwarmGetLocalSize(dm, &npoints));
    PetscCall(DMSwarmGetField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
    for (p = 0; p < npoints; p++) {
      if (swarm_cellid[p] != DMLOCATEPOINT_POINT_NOT_FOUND) sum[swarm_cellid[p]]++;
    }
    PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&swarm_cellid));
  }
  if (ncells) *ncells = nel;
  *count = sum;
  PetscFunctionReturn(0);
}

/*@
  DMSwarmGetNumSpecies - Get the number of particle species

  Not collective

  Input parameter:
. dm - the DMSwarm

  Output parameters:
. Ns - the number of species

  Level: intermediate

.seealso: `DMSwarmSetNumSpecies()`, `DMSwarmSetType()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmGetNumSpecies(DM sw, PetscInt *Ns)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  *Ns = swarm->Ns;
  PetscFunctionReturn(0);
}

/*@
  DMSwarmSetNumSpecies - Set the number of particle species

  Not collective

  Input parameter:
+ dm - the DMSwarm
- Ns - the number of species

  Level: intermediate

.seealso: `DMSwarmGetNumSpecies()`, `DMSwarmSetType()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmSetNumSpecies(DM sw, PetscInt Ns)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  swarm->Ns = Ns;
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmGetCoordinateFunction - Get the function setting initial particle positions, if it exists

  Not collective

  Input parameter:
. dm - the DMSwarm

  Output Parameter:
. coordFunc - the function setting initial particle positions, or NULL

  Level: intermediate

.seealso: `DMSwarmSetCoordinateFunction()`, `DMSwarmGetVelocityFunction()`, `DMSwarmInitializeCoordinates()`
@*/
PetscErrorCode DMSwarmGetCoordinateFunction(DM sw, PetscSimplePointFunc *coordFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscValidPointer(coordFunc, 2);
  *coordFunc = swarm->coordFunc;
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmSetCoordinateFunction - Set the function setting initial particle positions

  Not collective

  Input parameters:
+ dm - the DMSwarm
- coordFunc - the function setting initial particle positions

  Level: intermediate

.seealso: `DMSwarmGetCoordinateFunction()`, `DMSwarmSetVelocityFunction()`, `DMSwarmInitializeCoordinates()`
@*/
PetscErrorCode DMSwarmSetCoordinateFunction(DM sw, PetscSimplePointFunc coordFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscValidFunction(coordFunc, 2);
  swarm->coordFunc = coordFunc;
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmGetCoordinateFunction - Get the function setting initial particle velocities, if it exists

  Not collective

  Input parameter:
. dm - the DMSwarm

  Output Parameter:
. velFunc - the function setting initial particle velocities, or NULL

  Level: intermediate

.seealso: `DMSwarmSetVelocityFunction()`, `DMSwarmGetCoordinateFunction()`, `DMSwarmInitializeVelocities()`
@*/
PetscErrorCode DMSwarmGetVelocityFunction(DM sw, PetscSimplePointFunc *velFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscValidPointer(velFunc, 2);
  *velFunc = swarm->velFunc;
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmSetVelocityFunction - Set the function setting initial particle velocities

  Not collective

  Input parameters:
+ dm - the DMSwarm
- coordFunc - the function setting initial particle velocities

  Level: intermediate

.seealso: `DMSwarmGetVelocityFunction()`, `DMSwarmSetCoordinateFunction()`, `DMSwarmInitializeVelocities()`
@*/
PetscErrorCode DMSwarmSetVelocityFunction(DM sw, PetscSimplePointFunc velFunc)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sw, DM_CLASSID, 1);
  PetscValidFunction(velFunc, 2);
  swarm->velFunc = velFunc;
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmComputeLocalSize - Compute the local number and distribution of particles based upon a density function

  Not collective

  Input Parameters:
+ sw      - The DMSwarm
. N       - The target number of particles
- density - The density field for the particle layout, normalized to unity

  Note: One particle will be created for each species.

  Level: advanced

.seealso: `DMSwarmComputeLocalSizeFromOptions()`
@*/
PetscErrorCode DMSwarmComputeLocalSize(DM sw, PetscInt N, PetscProbFunc density)
{
  DM               dm;
  PetscQuadrature  quad;
  const PetscReal *xq, *wq;
  PetscInt        *npc, *cellid;
  PetscReal        xi0[3];
  PetscInt         Ns, cStart, cEnd, c, dim, d, Nq, q, Np = 0, p;
  PetscBool        simplex;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  if (simplex) PetscCall(PetscDTStroudConicalQuadrature(dim, 1, 5, -1.0, 1.0, &quad));
  else PetscCall(PetscDTGaussTensorQuadrature(dim, 1, 5, -1.0, 1.0, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &xq, &wq));
  PetscCall(PetscMalloc1(cEnd - cStart, &npc));
  /* Integrate the density function to get the number of particles in each cell */
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  for (c = 0; c < cEnd - cStart; ++c) {
    const PetscInt cell = c + cStart;
    PetscReal      v0[3], J[9], invJ[9], detJ;
    PetscReal      n_int = 0.;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    for (q = 0; q < Nq; ++q) {
      PetscReal xr[3], den[3];

      CoordinatesRefToReal(dim, dim, xi0, v0, J, &xq[q * dim], xr);
      PetscCall(density(xr, NULL, den));
      n_int += den[0] * wq[q];
    }
    npc[c] = (PetscInt)(N * n_int);
    npc[c] *= Ns;
    Np += npc[c];
  }
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  for (c = 0, p = 0; c < cEnd - cStart; ++c) {
    for (q = 0; q < npc[c]; ++q, ++p) cellid[p] = c;
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(PetscFree(npc));
  PetscFunctionReturn(0);
}

/*@
  DMSwarmComputeLocalSizeFromOptions - Compute the local number and distribution of particles based upon a density function determined by options

  Not collective

  Input Parameters:
, sw - The DMSwarm

  Level: advanced

.seealso: `DMSwarmComputeLocalSize()`
@*/
PetscErrorCode DMSwarmComputeLocalSizeFromOptions(DM sw)
{
  PetscProbFunc pdf;
  const char   *prefix;
  char          funcname[PETSC_MAX_PATH_LEN];
  PetscInt     *N, Ns, dim, n;
  PetscBool     flg;
  PetscMPIInt   size, rank;

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
    PetscSimplePointFunc coordFunc;

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
  PetscFunctionReturn(0);
}

/*@
  DMSwarmInitializeCoordinates - Determine the initial coordinates of particles for a PIC method

  Not collective

  Input Parameters:
, sw - The DMSwarm

  Note: Currently, we randomly place particles in their assigned cell

  Level: advanced

.seealso: `DMSwarmComputeLocalSize()`, `DMSwarmInitializeVelocities()`
@*/
PetscErrorCode DMSwarmInitializeCoordinates(DM sw)
{
  PetscSimplePointFunc coordFunc;
  PetscScalar         *weight;
  PetscReal           *x;
  PetscInt            *species;
  void                *ctx;
  PetscBool            removePoints = PETSC_TRUE;
  PetscDataType        dtype;
  PetscInt             Np, p, Ns, dim, d, bs;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMSwarmGetCoordinateFunction(sw, &coordFunc));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, &bs, &dtype, (void **)&x));
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

        weight[p]  = 1.0;
        species[p] = p % Ns;
      }
      PetscCall(PetscFree(pidx));
    }
    PetscCall(PetscRandomDestroy(&rnd));
    PetscCall(DMSwarmSortRestoreAccess(sw));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));

  PetscCall(DMSwarmMigrate(sw, removePoints));
  PetscCall(DMLocalizeCoordinates(sw));
  PetscFunctionReturn(0);
}

/*@C
  DMSwarmInitializeVelocities - Set the initial velocities of particles using a distribution.

  Collective on dm

  Input Parameters:
+ sw      - The DMSwarm object
. sampler - A function which uniformly samples the velocity PDF
- v0      - The velocity scale for nondimensionalization for each species

  Note: If v0 is zero for the first species, all velocities are set to zero. If it is zero for any other species, the effect will be to give that species zero velocity.

  Level: advanced

.seealso: `DMSwarmComputeLocalSize()`, `DMSwarmInitializeCoordinates()`, `DMSwarmInitializeVelocitiesFromOptions()`
@*/
PetscErrorCode DMSwarmInitializeVelocities(DM sw, PetscProbFunc sampler, const PetscReal v0[])
{
  PetscSimplePointFunc velFunc;
  PetscReal           *v;
  PetscInt            *species;
  void                *ctx;
  PetscInt             dim, Np, p;

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
  PetscFunctionReturn(0);
}

/*@
  DMSwarmInitializeVelocitiesFromOptions - Set the initial velocities of particles using a distribution determined from options.

  Collective on dm

  Input Parameters:
+ sw      - The DMSwarm object
- v0      - The velocity scale for nondimensionalization for each species

  Level: advanced

.seealso: `DMSwarmComputeLocalSize()`, `DMSwarmInitializeCoordinates()`, `DMSwarmInitializeVelocities()`
@*/
PetscErrorCode DMSwarmInitializeVelocitiesFromOptions(DM sw, const PetscReal v0[])
{
  PetscProbFunc sampler;
  PetscInt      dim;
  const char   *prefix;
  char          funcname[PETSC_MAX_PATH_LEN];
  PetscBool     flg;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)sw), "", "DMSwarm Options", "DMSWARM");
  PetscCall(PetscOptionsString("-dm_swarm_velocity_function", "Function to determine particle velocities", "DMSwarmSetVelocityFunction", funcname, funcname, sizeof(funcname), &flg));
  PetscOptionsEnd();
  if (flg) {
    PetscSimplePointFunc velFunc;

    PetscCall(PetscDLSym(NULL, funcname, (void **)&velFunc));
    PetscCheck(velFunc, PetscObjectComm((PetscObject)sw), PETSC_ERR_ARG_WRONG, "Could not locate function %s", funcname);
    PetscCall(DMSwarmSetVelocityFunction(sw, velFunc));
  }
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
  PetscCall(PetscProbCreateFromOptions(dim, prefix, "-dm_swarm_velocity_density", NULL, NULL, &sampler));
  PetscCall(DMSwarmInitializeVelocities(sw, sampler, v0));
  PetscFunctionReturn(0);
}

#define PETSCDM_DLL
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include <petscsf.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include "../src/dm/impls/swarm/data_bucket.h"

/*
 Error chceking macto to ensure the swarm type is correct and that a cell DM has been set
*/
#define DMSWARMPICVALID(dm) \
{ \
  DM_Swarm *_swarm = (DM_Swarm*)(dm)->data; \
  if (_swarm->swarm_type != DMSWARM_PIC) SETERRQ(PetscObjectComm((PetscObject)(dm)),PETSC_ERR_SUP,"Only valid for DMSwarm-PIC. You must call DMSwarmSetType(dm,DMSWARM_PIC)"); \
  else \
    if (!_swarm->dmcell) SETERRQ(PetscObjectComm((PetscObject)(dm)),PETSC_ERR_SUP,"Only valid for DMSwarmPIC if the cell DM is set. You must call DMSwarmSetCellDM(dm,celldm)"); \
}

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

.seealso: DMSwarmSetType(), DMSwarmSetCellDM(), DMSwarmType
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSetPointsUniformCoordinates(DM dm,PetscReal min[],PetscReal max[],PetscInt npoints[],InsertMode mode)
{
  PetscErrorCode    ierr;
  PetscReal         gmin[] = {PETSC_MAX_REAL ,PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal         gmax[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt          i,j,k,N,bs,b,n_estimate,n_curr,n_new_est,p,n_found;
  Vec               coorlocal;
  const PetscScalar *_coor;
  DM                celldm;
  PetscReal         dx[3];
  PetscInt          _npoints[] = { 0, 0, 1 };
  Vec               pos;
  PetscScalar       *_pos;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscSF           sfcell = NULL;
  const PetscSFNode *LA_sfcell;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  ierr = DMSwarmGetCellDM(dm,&celldm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(celldm,&coorlocal);CHKERRQ(ierr);
  ierr = VecGetSize(coorlocal,&N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coorlocal,&bs);CHKERRQ(ierr);
  N = N / bs;
  ierr = VecGetArrayRead(coorlocal,&_coor);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    for (b=0; b<bs; b++) {
      gmin[b] = PetscMin(gmin[b],PetscRealPart(_coor[bs*i+b]));
      gmax[b] = PetscMax(gmax[b],PetscRealPart(_coor[bs*i+b]));
    }
  }
  ierr = VecRestoreArrayRead(coorlocal,&_coor);CHKERRQ(ierr);

  for (b=0; b<bs; b++) {
    if (npoints[b] > 1) {
      dx[b] = (max[b] - min[b])/((PetscReal)(npoints[b]-1));
    } else {
      dx[b] = 0.0;
    }

    _npoints[b] = npoints[b];
  }

  /* determine number of points living in the bounding box */
  n_estimate = 0;
  for (k=0; k<_npoints[2]; k++) {
    for (j=0; j<_npoints[1]; j++) {
      for (i=0; i<_npoints[0]; i++) {
        PetscReal xp[] = {0.0,0.0,0.0};
        PetscInt ijk[3];
        PetscBool point_inside = PETSC_TRUE;

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        for (b=0; b<bs; b++) {
          xp[b] = min[b] + ijk[b] * dx[b];
        }
        for (b=0; b<bs; b++) {
          if (xp[b] < gmin[b]) { point_inside = PETSC_FALSE; }
          if (xp[b] > gmax[b]) { point_inside = PETSC_FALSE; }
        }
        if (point_inside) { n_estimate++; }
      }
    }
  }

  /* create candidate list */
  ierr = VecCreate(PETSC_COMM_SELF,&pos);CHKERRQ(ierr);
  ierr = VecSetSizes(pos,bs*n_estimate,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(pos,bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pos);CHKERRQ(ierr);
  ierr = VecGetArray(pos,&_pos);CHKERRQ(ierr);

  n_estimate = 0;
  for (k=0; k<_npoints[2]; k++) {
    for (j=0; j<_npoints[1]; j++) {
      for (i=0; i<_npoints[0]; i++) {
        PetscReal xp[] = {0.0,0.0,0.0};
        PetscInt  ijk[3];
        PetscBool point_inside = PETSC_TRUE;

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        for (b=0; b<bs; b++) {
          xp[b] = min[b] + ijk[b] * dx[b];
        }
        for (b=0; b<bs; b++) {
          if (xp[b] < gmin[b]) { point_inside = PETSC_FALSE; }
          if (xp[b] > gmax[b]) { point_inside = PETSC_FALSE; }
        }
        if (point_inside) {
          for (b=0; b<bs; b++) {
            _pos[bs*n_estimate+b] = xp[b];
          }
          n_estimate++;
        }
      }
    }
  }
  ierr = VecRestoreArray(pos,&_pos);CHKERRQ(ierr);

  /* locate points */
  ierr = DMLocatePoints(celldm,pos,DM_POINTLOCATION_NONE,&sfcell);CHKERRQ(ierr);

  ierr = PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell);CHKERRQ(ierr);
  n_found = 0;
  for (p=0; p<n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      n_found++;
    }
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    ierr = DMSwarmGetLocalSize(dm,&n_curr);CHKERRQ(ierr);
    n_new_est = n_curr + n_found;
    ierr = DMSwarmSetLocalSizes(dm,n_new_est,-1);CHKERRQ(ierr);
  }
  if (mode == INSERT_VALUES) {
    n_curr = 0;
    n_new_est = n_found;
    ierr = DMSwarmSetLocalSizes(dm,n_new_est,-1);CHKERRQ(ierr);
  }

  /* initialize new coords, cell owners, pid */
  ierr = VecGetArrayRead(pos,&_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  n_found = 0;
  for (p=0; p<n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (b=0; b<bs; b++) {
        swarm_coor[bs*(n_curr + n_found) + b] = PetscRealPart(_coor[bs*p+b]);
      }
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(pos,&_coor);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&sfcell);CHKERRQ(ierr);
  ierr = VecDestroy(&pos);CHKERRQ(ierr);

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

.seealso: DMSwarmSetType(), DMSwarmSetCellDM(), DMSwarmType, DMSwarmSetPointsUniformCoordinates()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinates(DM dm,PetscInt npoints,PetscReal coor[],PetscBool redundant,InsertMode mode)
{
  PetscErrorCode    ierr;
  PetscReal         gmin[] = {PETSC_MAX_REAL ,PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal         gmax[] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt          i,N,bs,b,n_estimate,n_curr,n_new_est,p,n_found;
  Vec               coorlocal;
  const PetscScalar *_coor;
  DM                celldm;
  Vec               pos;
  PetscScalar       *_pos;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;
  PetscSF           sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  PetscReal         *my_coor;
  PetscInt          my_npoints;
  PetscMPIInt       rank;
  MPI_Comm          comm;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = DMSwarmGetCellDM(dm,&celldm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(celldm,&coorlocal);CHKERRQ(ierr);
  ierr = VecGetSize(coorlocal,&N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coorlocal,&bs);CHKERRQ(ierr);
  N = N / bs;
  ierr = VecGetArrayRead(coorlocal,&_coor);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    for (b=0; b<bs; b++) {
      gmin[b] = PetscMin(gmin[b],PetscRealPart(_coor[bs*i+b]));
      gmax[b] = PetscMax(gmax[b],PetscRealPart(_coor[bs*i+b]));
    }
  }
  ierr = VecRestoreArrayRead(coorlocal,&_coor);CHKERRQ(ierr);

  /* broadcast points from rank 0 if requested */
  if (redundant) {
    my_npoints = npoints;
    ierr = MPI_Bcast(&my_npoints,1,MPIU_INT,0,comm);CHKERRMPI(ierr);

    if (rank > 0) { /* allocate space */
      ierr = PetscMalloc1(bs*my_npoints,&my_coor);CHKERRQ(ierr);
    } else {
      my_coor = coor;
    }
    ierr = MPI_Bcast(my_coor,bs*my_npoints,MPIU_REAL,0,comm);CHKERRMPI(ierr);
  } else {
    my_npoints = npoints;
    my_coor = coor;
  }

  /* determine the number of points living in the bounding box */
  n_estimate = 0;
  for (i=0; i<my_npoints; i++) {
    PetscBool point_inside = PETSC_TRUE;

    for (b=0; b<bs; b++) {
      if (my_coor[bs*i+b] < gmin[b]) { point_inside = PETSC_FALSE; }
      if (my_coor[bs*i+b] > gmax[b]) { point_inside = PETSC_FALSE; }
    }
    if (point_inside) { n_estimate++; }
  }

  /* create candidate list */
  ierr = VecCreate(PETSC_COMM_SELF,&pos);CHKERRQ(ierr);
  ierr = VecSetSizes(pos,bs*n_estimate,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(pos,bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pos);CHKERRQ(ierr);
  ierr = VecGetArray(pos,&_pos);CHKERRQ(ierr);

  n_estimate = 0;
  for (i=0; i<my_npoints; i++) {
    PetscBool point_inside = PETSC_TRUE;

    for (b=0; b<bs; b++) {
      if (my_coor[bs*i+b] < gmin[b]) { point_inside = PETSC_FALSE; }
      if (my_coor[bs*i+b] > gmax[b]) { point_inside = PETSC_FALSE; }
    }
    if (point_inside) {
      for (b=0; b<bs; b++) {
        _pos[bs*n_estimate+b] = my_coor[bs*i+b];
      }
      n_estimate++;
    }
  }
  ierr = VecRestoreArray(pos,&_pos);CHKERRQ(ierr);

  /* locate points */
  ierr = DMLocatePoints(celldm,pos,DM_POINTLOCATION_NONE,&sfcell);CHKERRQ(ierr);

  ierr = PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell);CHKERRQ(ierr);
  n_found = 0;
  for (p=0; p<n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      n_found++;
    }
  }

  /* adjust size */
  if (mode == ADD_VALUES) {
    ierr = DMSwarmGetLocalSize(dm,&n_curr);CHKERRQ(ierr);
    n_new_est = n_curr + n_found;
    ierr = DMSwarmSetLocalSizes(dm,n_new_est,-1);CHKERRQ(ierr);
  }
  if (mode == INSERT_VALUES) {
    n_curr = 0;
    n_new_est = n_found;
    ierr = DMSwarmSetLocalSizes(dm,n_new_est,-1);CHKERRQ(ierr);
  }

  /* initialize new coords, cell owners, pid */
  ierr = VecGetArrayRead(pos,&_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  n_found = 0;
  for (p=0; p<n_estimate; p++) {
    if (LA_sfcell[p].index != DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (b=0; b<bs; b++) {
        swarm_coor[bs*(n_curr + n_found) + b] = PetscRealPart(_coor[bs*p+b]);
      }
      swarm_cellid[n_curr + n_found] = LA_sfcell[p].index;
      n_found++;
    }
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(pos,&_coor);CHKERRQ(ierr);

  if (redundant) {
    if (rank > 0) {
      ierr = PetscFree(my_coor);CHKERRQ(ierr);
    }
  }
  ierr = PetscSFDestroy(&sfcell);CHKERRQ(ierr);
  ierr = VecDestroy(&pos);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

extern PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_DA(DM,DM,DMSwarmPICLayoutType,PetscInt);
extern PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX(DM,DM,DMSwarmPICLayoutType,PetscInt);

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

.seealso: DMSwarmPICLayoutType, DMSwarmSetType(), DMSwarmSetCellDM(), DMSwarmType
@*/
PETSC_EXTERN PetscErrorCode DMSwarmInsertPointsUsingCellDM(DM dm,DMSwarmPICLayoutType layout_type,PetscInt fill_param)
{
  PetscErrorCode ierr;
  DM             celldm;
  PetscBool      isDA,isPLEX;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  ierr = DMSwarmGetCellDM(dm,&celldm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)celldm,DMDA,&isDA);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)celldm,DMPLEX,&isPLEX);CHKERRQ(ierr);
  if (isDA) {
    ierr = private_DMSwarmInsertPointsUsingCellDM_DA(dm,celldm,layout_type,fill_param);CHKERRQ(ierr);
  } else if (isPLEX) {
    ierr = private_DMSwarmInsertPointsUsingCellDM_PLEX(dm,celldm,layout_type,fill_param);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only supported for cell DMs of type DMDA and DMPLEX");

  PetscFunctionReturn(0);
}


extern PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM,DM,PetscInt,PetscReal*);

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

.seealso: DMSwarmSetCellDM(), DMSwarmInsertPointsUsingCellDM()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinatesCellwise(DM dm,PetscInt npoints,PetscReal xi[])
{
  PetscErrorCode ierr;
  DM             celldm;
  PetscBool      isDA,isPLEX;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  ierr = DMSwarmGetCellDM(dm,&celldm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)celldm,DMDA,&isDA);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)celldm,DMPLEX,&isPLEX);CHKERRQ(ierr);
  if (isDA) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only supported for cell DMs of type DMPLEX. Recommended you use DMSwarmInsertPointsUsingCellDM()");
  else if (isPLEX) {
    ierr = private_DMSwarmSetPointCoordinatesCellwise_PLEX(dm,celldm,npoints,xi);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only supported for cell DMs of type DMDA and DMPLEX");

  PetscFunctionReturn(0);
}


/* Field projection API */
extern PetscErrorCode private_DMSwarmProjectFields_DA(DM swarm,DM celldm,PetscInt project_type,PetscInt nfields,DMSwarmDataField dfield[],Vec vecs[]);
extern PetscErrorCode private_DMSwarmProjectFields_PLEX(DM swarm,DM celldm,PetscInt project_type,PetscInt nfields,DMSwarmDataField dfield[],Vec vecs[]);

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

.seealso: DMSwarmSetType(), DMSwarmSetCellDM(), DMSwarmType
@*/
PETSC_EXTERN PetscErrorCode DMSwarmProjectFields(DM dm,PetscInt nfields,const char *fieldnames[],Vec **fields,PetscBool reuse)
{
  DM_Swarm         *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataField *gfield;
  DM               celldm;
  PetscBool        isDA,isPLEX;
  Vec              *vecs;
  PetscInt         f,nvecs;
  PetscInt         project_type = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  ierr = DMSwarmGetCellDM(dm,&celldm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nfields,&gfield);CHKERRQ(ierr);
  nvecs = 0;
  for (f=0; f<nfields; f++) {
    ierr = DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,fieldnames[f],&gfield[f]);CHKERRQ(ierr);
    if (gfield[f]->petsc_type != PETSC_REAL) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Projection only valid for fields using a data type = PETSC_REAL");
    if (gfield[f]->bs != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Projection only valid for fields with block size = 1");
    nvecs += gfield[f]->bs;
  }
  if (!reuse) {
    ierr = PetscMalloc1(nvecs,&vecs);CHKERRQ(ierr);
    for (f=0; f<nvecs; f++) {
      ierr = DMCreateGlobalVector(celldm,&vecs[f]);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)vecs[f],gfield[f]->name);CHKERRQ(ierr);
    }
  } else {
    vecs = *fields;
  }

  ierr = PetscObjectTypeCompare((PetscObject)celldm,DMDA,&isDA);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)celldm,DMPLEX,&isPLEX);CHKERRQ(ierr);
  if (isDA) {
    ierr = private_DMSwarmProjectFields_DA(dm,celldm,project_type,nfields,gfield,vecs);CHKERRQ(ierr);
  } else if (isPLEX) {
    ierr = private_DMSwarmProjectFields_PLEX(dm,celldm,project_type,nfields,gfield,vecs);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only supported for cell DMs of type DMDA and DMPLEX");

  ierr = PetscFree(gfield);CHKERRQ(ierr);
  if (!reuse) {
    *fields = vecs;
  }

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

.seealso: DMSwarmSetType(), DMSwarmSetCellDM(), DMSwarmType
@*/
PETSC_EXTERN PetscErrorCode DMSwarmCreatePointPerCellCount(DM dm,PetscInt *ncells,PetscInt **count)
{
  PetscErrorCode ierr;
  PetscBool      isvalid;
  PetscInt       nel;
  PetscInt       *sum;

  PetscFunctionBegin;
  ierr = DMSwarmSortGetIsValid(dm,&isvalid);CHKERRQ(ierr);
  nel = 0;
  if (isvalid) {
    PetscInt e;

    ierr = DMSwarmSortGetSizes(dm,&nel,NULL);CHKERRQ(ierr);

    ierr = PetscMalloc1(nel,&sum);CHKERRQ(ierr);
    for (e=0; e<nel; e++) {
      ierr = DMSwarmSortGetNumberOfPointsPerCell(dm,e,&sum[e]);CHKERRQ(ierr);
    }
  } else {
    DM        celldm;
    PetscBool isda,isplex,isshell;
    PetscInt  p,npoints;
    PetscInt *swarm_cellid;

    /* get the number of cells */
    ierr = DMSwarmGetCellDM(dm,&celldm);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)celldm,DMDA,&isda);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)celldm,DMPLEX,&isplex);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)celldm,DMSHELL,&isshell);CHKERRQ(ierr);
    if (isda) {
      PetscInt       _nel,_npe;
      const PetscInt *_element;

      ierr = DMDAGetElements(celldm,&_nel,&_npe,&_element);CHKERRQ(ierr);
      nel = _nel;
      ierr = DMDARestoreElements(celldm,&_nel,&_npe,&_element);CHKERRQ(ierr);
    } else if (isplex) {
      PetscInt ps,pe;

      ierr = DMPlexGetHeightStratum(celldm,0,&ps,&pe);CHKERRQ(ierr);
      nel = pe - ps;
    } else if (isshell) {
      PetscErrorCode (*method_DMShellGetNumberOfCells)(DM,PetscInt*);

      ierr = PetscObjectQueryFunction((PetscObject)celldm,"DMGetNumberOfCells_C",&method_DMShellGetNumberOfCells);CHKERRQ(ierr);
      if (method_DMShellGetNumberOfCells) {
        ierr = method_DMShellGetNumberOfCells(celldm,&nel);CHKERRQ(ierr);
      } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot determine the number of cells for the DMSHELL object. User must provide a method via PetscObjectComposeFunction( (PetscObject)shelldm, \"DMGetNumberOfCells_C\", your_function_to_compute_number_of_cells);");
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot determine the number of cells for a DM not of type DA, PLEX or SHELL");

    ierr = PetscMalloc1(nel,&sum);CHKERRQ(ierr);
    ierr = PetscArrayzero(sum,nel);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if (swarm_cellid[p] != DMLOCATEPOINT_POINT_NOT_FOUND) {
        sum[ swarm_cellid[p] ]++;
      }
    }
    ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid);CHKERRQ(ierr);
  }
  if (ncells) { *ncells = nel; }
  *count  = sum;
  PetscFunctionReturn(0);
}

#include <petscsf.h>
#include <petscdmswarm.h>
#include <petscdmda.h>
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include "../src/dm/impls/swarm/data_bucket.h"
#include "../src/dm/impls/swarm/data_ex.h"

/*
 User loads desired location (MPI rank) into field DMSwarm_rank
*/
PetscErrorCode DMSwarmMigrate_Push_Basic(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataEx  de;
  PetscInt       p,npoints,*rankval,n_points_recv;
  PetscMPIInt    rank,nrank;
  void           *point_buffer,*recv_points;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));

  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0, &de));
  CHKERRQ(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < npoints; ++p) {
    nrank = rankval[p];
    if (nrank != rank) {
      CHKERRQ(DMSwarmDataExTopologyAddNeighbour(de,nrank));
    }
  }
  CHKERRQ(DMSwarmDataExTopologyFinalize(de));
  CHKERRQ(DMSwarmDataExInitializeSendCount(de));
  for (p=0; p<npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      CHKERRQ(DMSwarmDataExAddToSendCount(de,nrank,1));
    }
  }
  CHKERRQ(DMSwarmDataExFinalizeSendCount(de));
  CHKERRQ(DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer));
  CHKERRQ(DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point));
  for (p=0; p<npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      /* copy point into buffer */
      CHKERRQ(DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer));
      /* insert point buffer into DMSwarmDataExchanger */
      CHKERRQ(DMSwarmDataExPackData(de,nrank,1,point_buffer));
    }
  }
  CHKERRQ(DMSwarmDataExPackFinalize(de));
  CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));

  if (remove_sent_points) {
    DMSwarmDataField gfield;

    CHKERRQ(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&gfield));
    CHKERRQ(DMSwarmDataFieldGetAccess(gfield));
    CHKERRQ(DMSwarmDataFieldGetEntries(gfield,(void**)&rankval));

    /* remove points which left processor */
    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
    for (p=0; p<npoints; p++) {
      nrank = rankval[p];
      if (nrank != rank) {
        /* kill point */
        CHKERRQ(DMSwarmDataFieldRestoreAccess(gfield));

        CHKERRQ(DMSwarmDataBucketRemovePointAtIndex(swarm->db,p));

        CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL)); /* you need to update npoints as the list size decreases! */
        CHKERRQ(DMSwarmDataFieldGetAccess(gfield));
        CHKERRQ(DMSwarmDataFieldGetEntries(gfield,(void**)&rankval));
        p--; /* check replacement point */
      }
    }
    CHKERRQ(DMSwarmDataFieldRestoreEntries(gfield,(void**)&rankval));
    CHKERRQ(DMSwarmDataFieldRestoreAccess(gfield));
  }
  CHKERRQ(DMSwarmDataExBegin(de));
  CHKERRQ(DMSwarmDataExEnd(de));
  CHKERRQ(DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    CHKERRQ(DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p));
  }
  CHKERRQ(DMSwarmDataExView(de));
  CHKERRQ(DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer));
  CHKERRQ(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_DMNeighborScatter(DM dm,DM dmcell,PetscBool remove_sent_points,PetscInt *npoints_prior_migration)
{
  DM_Swarm          *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataEx     de;
  PetscInt          r,p,npoints,*rankval,n_points_recv;
  PetscMPIInt       rank,_rank;
  const PetscMPIInt *neighbourranks;
  void              *point_buffer,*recv_points;
  size_t            sizeof_dmswarm_point;
  PetscInt          nneighbors;
  PetscMPIInt       mynneigh,*myneigh;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de));
  CHKERRQ(DMGetNeighbors(dmcell,&nneighbors,&neighbourranks));
  CHKERRQ(DMSwarmDataExTopologyInitialize(de));
  for (r=0; r<nneighbors; r++) {
    _rank = neighbourranks[r];
    if ((_rank != rank) && (_rank > 0)) {
      CHKERRQ(DMSwarmDataExTopologyAddNeighbour(de,_rank));
    }
  }
  CHKERRQ(DMSwarmDataExTopologyFinalize(de));
  CHKERRQ(DMSwarmDataExTopologyGetNeighbours(de,&mynneigh,&myneigh));
  CHKERRQ(DMSwarmDataExInitializeSendCount(de));
  for (p=0; p<npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r=0; r<mynneigh; r++) {
        _rank = myneigh[r];
        CHKERRQ(DMSwarmDataExAddToSendCount(de,_rank,1));
      }
    }
  }
  CHKERRQ(DMSwarmDataExFinalizeSendCount(de));
  CHKERRQ(DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer));
  CHKERRQ(DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point));
  for (p=0; p<npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r=0; r<mynneigh; r++) {
        _rank = myneigh[r];
        /* copy point into buffer */
        CHKERRQ(DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer));
        /* insert point buffer into DMSwarmDataExchanger */
        CHKERRQ(DMSwarmDataExPackData(de,_rank,1,point_buffer));
      }
    }
  }
  CHKERRQ(DMSwarmDataExPackFinalize(de));
  CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  if (remove_sent_points) {
    DMSwarmDataField PField;

    CHKERRQ(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField));
    CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval));
    /* remove points which left processor */
    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
    for (p=0; p<npoints; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        CHKERRQ(DMSwarmDataBucketRemovePointAtIndex(swarm->db,p));
        CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL)); /* you need to update npoints as the list size decreases! */
        CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval)); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,npoints_prior_migration,NULL,NULL));
  CHKERRQ(DMSwarmDataExBegin(de));
  CHKERRQ(DMSwarmDataExEnd(de));
  CHKERRQ(DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    CHKERRQ(DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p));
  }
  CHKERRQ(DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer));
  CHKERRQ(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_CellDMScatter(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm          *swarm = (DM_Swarm*)dm->data;
  PetscInt          p,npoints,npointsg=0,npoints2,npoints2g,*rankval,npoints_prior_migration;
  PetscSF           sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  DM                dmcell;
  Vec               pos;
  PetscBool         error_check = swarm->migrate_error_on_missing_point;
  PetscMPIInt       size,rank;

  PetscFunctionBegin;
  CHKERRQ(DMSwarmGetCellDM(dm,&dmcell));
  PetscCheck(dmcell,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));

#if 1
  {
    PetscInt *p_cellid;
    PetscInt npoints_curr,range = 0;
    PetscSFNode *sf_cells;

    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL));
    CHKERRQ(PetscMalloc1(npoints_curr, &sf_cells));

    CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
    CHKERRQ(DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid));
    for (p=0; p<npoints_curr; p++) {

      sf_cells[p].rank  = 0;
      sf_cells[p].index = p_cellid[p];
      if (p_cellid[p] > range) {
        range = p_cellid[p];
      }
    }
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid));
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));

    /*CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)dm),&sfcell));*/
    CHKERRQ(PetscSFCreate(PETSC_COMM_SELF,&sfcell));
    CHKERRQ(PetscSFSetGraph(sfcell, range, npoints_curr, NULL, PETSC_OWN_POINTER, sf_cells, PETSC_OWN_POINTER));
  }
#endif

  CHKERRQ(DMSwarmCreateLocalVectorFromField(dm, DMSwarmPICField_coor, &pos));
  CHKERRQ(DMLocatePoints(dmcell, pos, DM_POINTLOCATION_NONE, &sfcell));
  CHKERRQ(DMSwarmDestroyLocalVectorFromField(dm, DMSwarmPICField_coor, &pos));

  if (error_check) {
    CHKERRQ(DMSwarmGetSize(dm,&npointsg));
  }
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
  for (p=0; p<npoints; p++) {
    rankval[p] = LA_sfcell[p].index;
  }
  CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(PetscSFDestroy(&sfcell));

  if (size > 1) {
    CHKERRQ(DMSwarmMigrate_DMNeighborScatter(dm,dmcell,remove_sent_points,&npoints_prior_migration));
  } else {
    DMSwarmDataField PField;
    PetscInt npoints_curr;

    /* remove points which the domain */
    CHKERRQ(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField));
    CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval));

    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL));
    for (p=0; p<npoints_curr; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        CHKERRQ(DMSwarmDataBucketRemovePointAtIndex(swarm->db,p));
        CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL)); /* you need to update npoints as the list size decreases! */
        CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval)); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
    CHKERRQ(DMSwarmGetSize(dm,&npoints_prior_migration));

  }

  /* locate points newly received */
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL));

#if 0
  { /* safe alternative - however this performs two point locations on: (i) the initial points set and; (ii) the (initial + received) point set */
    PetscScalar *LA_coor;
    PetscInt bs;
    DMSwarmDataField PField;

    CHKERRQ(DMSwarmGetField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,bs*npoints2,(const PetscScalar*)LA_coor,&pos));
    CHKERRQ(DMLocatePoints(dmcell,pos,DM_POINTLOCATION_NONE,&sfcell));

    CHKERRQ(VecDestroy(&pos));
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor));

    CHKERRQ(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
    CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
    for (p=0; p<npoints2; p++) {
      rankval[p] = LA_sfcell[p].index;
    }
    CHKERRQ(PetscSFDestroy(&sfcell));
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));

    /* remove points which left processor */
    CHKERRQ(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField));
    CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval));

    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL));
    for (p=0; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        CHKERRQ(DMSwarmDataBucketRemovePointAtIndex(swarm->db,p));
        CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL)); /* you need to update npoints as the list size decreases! */
        CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval)); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
#endif

  { /* perform two point locations: (i) on the initial points set prior to communication; and (ii) on the new (received) points */
    PetscScalar      *LA_coor;
    PetscInt         npoints_from_neighbours,bs;
    DMSwarmDataField PField;

    npoints_from_neighbours = npoints2 - npoints_prior_migration;

    CHKERRQ(DMSwarmGetField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,bs*npoints_from_neighbours,(const PetscScalar*)&LA_coor[bs*npoints_prior_migration],&pos));

    CHKERRQ(DMLocatePoints(dmcell,pos,DM_POINTLOCATION_NONE,&sfcell));

    CHKERRQ(VecDestroy(&pos));
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor));

    CHKERRQ(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
    CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
    for (p=0; p<npoints_from_neighbours; p++) {
      rankval[npoints_prior_migration + p] = LA_sfcell[p].index;
    }
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
    CHKERRQ(PetscSFDestroy(&sfcell));

    /* remove points which left processor */
    CHKERRQ(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField));
    CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval));

    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL));
    for (p=npoints_prior_migration; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        CHKERRQ(DMSwarmDataBucketRemovePointAtIndex(swarm->db,p));
        CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL)); /* you need to update npoints as the list size decreases! */
        CHKERRQ(DMSwarmDataFieldGetEntries(PField,(void**)&rankval)); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }

  {
    PetscInt *p_cellid;

    CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL));
    CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
    CHKERRQ(DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid));
    for (p=0; p<npoints2; p++) {
      p_cellid[p] = rankval[p];
    }
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid));
    CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  }

  /* check for error on removed points */
  if (error_check) {
    CHKERRQ(DMSwarmGetSize(dm,&npoints2g));
    PetscCheckFalse(npointsg != npoints2g,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Points from the DMSwarm must remain constant during migration (initial %D - final %D)",npointsg,npoints2g);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_CellDMExact(DM dm,PetscBool remove_sent_points)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
 Redundant as this assumes points can only be sent to a single rank
*/
PetscErrorCode DMSwarmMigrate_GlobalToLocal_Basic(DM dm,PetscInt *globalsize)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataEx  de;
  PetscInt       p,npoints,*rankval,n_points_recv;
  PetscMPIInt    rank,nrank,negrank;
  void           *point_buffer,*recv_points;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  *globalsize = npoints;
  CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de));
  CHKERRQ(DMSwarmDataExTopologyInitialize(de));
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      CHKERRQ(DMSwarmDataExTopologyAddNeighbour(de,nrank));
    }
  }
  CHKERRQ(DMSwarmDataExTopologyFinalize(de));
  CHKERRQ(DMSwarmDataExInitializeSendCount(de));
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      CHKERRQ(DMSwarmDataExAddToSendCount(de,nrank,1));
    }
  }
  CHKERRQ(DMSwarmDataExFinalizeSendCount(de));
  CHKERRQ(DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer));
  CHKERRQ(DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point));
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      rankval[p] = nrank;
      /* copy point into buffer */
      CHKERRQ(DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer));
      /* insert point buffer into DMSwarmDataExchanger */
      CHKERRQ(DMSwarmDataExPackData(de,nrank,1,point_buffer));
      rankval[p] = negrank;
    }
  }
  CHKERRQ(DMSwarmDataExPackFinalize(de));
  CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(DMSwarmDataExBegin(de));
  CHKERRQ(DMSwarmDataExEnd(de));
  CHKERRQ(DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    CHKERRQ(DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p));
  }
  CHKERRQ(DMSwarmDataExView(de));
  CHKERRQ(DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer));
  CHKERRQ(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscMPIInt owner_rank;
  PetscReal   min[3],max[3];
} CollectBBox;

PETSC_EXTERN PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM dm,PetscInt *globalsize)
{
  DM_Swarm *        swarm = (DM_Swarm*)dm->data;
  DMSwarmDataEx     de;
  PetscInt          p,pk,npoints,*rankval,n_points_recv,n_bbox_recv,dim,neighbour_cells;
  PetscMPIInt       rank,nrank;
  void              *point_buffer,*recv_points;
  size_t            sizeof_dmswarm_point,sizeof_bbox_ctx;
  PetscBool         isdmda;
  CollectBBox       *bbox,*recv_bbox;
  const PetscMPIInt *dmneighborranks;
  DM                dmcell;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));

  CHKERRQ(DMSwarmGetCellDM(dm,&dmcell));
  PetscCheck(dmcell,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");
  isdmda = PETSC_FALSE;
  PetscObjectTypeCompare((PetscObject)dmcell,DMDA,&isdmda);
  PetscCheck(isdmda,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only DMDA support for CollectBoundingBox");

  CHKERRQ(DMGetDimension(dm,&dim));
  sizeof_bbox_ctx = sizeof(CollectBBox);
  PetscMalloc1(1,&bbox);
  bbox->owner_rank = rank;

  /* compute the bounding box based on the overlapping / stenctil size */
  {
    Vec lcoor;

    CHKERRQ(DMGetCoordinatesLocal(dmcell,&lcoor));
    if (dim >= 1) {
      CHKERRQ(VecStrideMin(lcoor,0,NULL,&bbox->min[0]));
      CHKERRQ(VecStrideMax(lcoor,0,NULL,&bbox->max[0]));
    }
    if (dim >= 2) {
      CHKERRQ(VecStrideMin(lcoor,1,NULL,&bbox->min[1]));
      CHKERRQ(VecStrideMax(lcoor,1,NULL,&bbox->max[1]));
    }
    if (dim == 3) {
      CHKERRQ(VecStrideMin(lcoor,2,NULL,&bbox->min[2]));
      CHKERRQ(VecStrideMax(lcoor,2,NULL,&bbox->max[2]));
    }
  }
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  *globalsize = npoints;
  CHKERRQ(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de));
  /* use DMDA neighbours */
  CHKERRQ(DMDAGetNeighbors(dmcell,&dmneighborranks));
  if (dim == 1) {
    neighbour_cells = 3;
  } else if (dim == 2) {
    neighbour_cells = 9;
  } else {
    neighbour_cells = 27;
  }
  CHKERRQ(DMSwarmDataExTopologyInitialize(de));
  for (p=0; p<neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) {
      CHKERRQ(DMSwarmDataExTopologyAddNeighbour(de,dmneighborranks[p]));
    }
  }
  CHKERRQ(DMSwarmDataExTopologyFinalize(de));
  CHKERRQ(DMSwarmDataExInitializeSendCount(de));
  for (p=0; p<neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) {
      CHKERRQ(DMSwarmDataExAddToSendCount(de,dmneighborranks[p],1));
    }
  }
  CHKERRQ(DMSwarmDataExFinalizeSendCount(de));
  /* send bounding boxes */
  CHKERRQ(DMSwarmDataExPackInitialize(de,sizeof_bbox_ctx));
  for (p=0; p<neighbour_cells; p++) {
    nrank = dmneighborranks[p];
    if ((nrank >= 0) && (nrank != rank)) {
      /* insert bbox buffer into DMSwarmDataExchanger */
      CHKERRQ(DMSwarmDataExPackData(de,nrank,1,bbox));
    }
  }
  CHKERRQ(DMSwarmDataExPackFinalize(de));
  /* recv bounding boxes */
  CHKERRQ(DMSwarmDataExBegin(de));
  CHKERRQ(DMSwarmDataExEnd(de));
  CHKERRQ(DMSwarmDataExGetRecvData(de,&n_bbox_recv,(void**)&recv_bbox));
  /*  Wrong, should not be using PETSC_COMM_WORLD */
  for (p=0; p<n_bbox_recv; p++) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[rank %d]: box from %d : range[%+1.4e,%+1.4e]x[%+1.4e,%+1.4e]\n",rank,recv_bbox[p].owner_rank,
                                    (double)recv_bbox[p].min[0],(double)recv_bbox[p].max[0],(double)recv_bbox[p].min[1],(double)recv_bbox[p].max[1]));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));
  /* of course this is stupid as this "generic" function should have a better way to know what the coordinates are called */
  CHKERRQ(DMSwarmDataExInitializeSendCount(de));
  for (pk=0; pk<n_bbox_recv; pk++) {
    PetscReal *array_x,*array_y;

    CHKERRQ(DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y));
    for (p=0; p<npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) {
          CHKERRQ(DMSwarmDataExAddToSendCount(de,recv_bbox[pk].owner_rank,1));
        }
      }
    }
    CHKERRQ(DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y));
    CHKERRQ(DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x));
  }
  CHKERRQ(DMSwarmDataExFinalizeSendCount(de));
  CHKERRQ(DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer));
  CHKERRQ(DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point));
  for (pk=0; pk<n_bbox_recv; pk++) {
    PetscReal *array_x,*array_y;

    CHKERRQ(DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x));
    CHKERRQ(DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y));
    for (p=0; p<npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) {
          /* copy point into buffer */
          CHKERRQ(DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer));
          /* insert point buffer into DMSwarmDataExchanger */
          CHKERRQ(DMSwarmDataExPackData(de,recv_bbox[pk].owner_rank,1,point_buffer));
        }
      }
    }
    CHKERRQ(DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y));
    CHKERRQ(DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x));
  }
  CHKERRQ(DMSwarmDataExPackFinalize(de));
  CHKERRQ(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
  CHKERRQ(DMSwarmDataExBegin(de));
  CHKERRQ(DMSwarmDataExEnd(de));
  CHKERRQ(DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    CHKERRQ(DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p));
  }
  CHKERRQ(DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer));
  PetscFree(bbox);
  CHKERRQ(DMSwarmDataExView(de));
  CHKERRQ(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(0);
}

/* General collection when no order, or neighbour information is provided */
/*
 User provides context and collect() method
 Broadcast user context

 for each context / rank {
   collect(swarm,context,n,list)
 }
*/
PETSC_EXTERN PetscErrorCode DMSwarmCollect_General(DM dm,PetscErrorCode (*collect)(DM,void*,PetscInt*,PetscInt**),size_t ctx_size,void *ctx,PetscInt *globalsize)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  DMSwarmDataEx  de;
  PetscInt       p,r,npoints,n_points_recv;
  PetscMPIInt    size,rank;
  void           *point_buffer,*recv_points;
  void           *ctxlist;
  PetscInt       *n2collect,**collectlist;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  *globalsize = npoints;
  /* Broadcast user context */
  PetscMalloc(ctx_size*size,&ctxlist);
  CHKERRMPI(MPI_Allgather(ctx,ctx_size,MPI_CHAR,ctxlist,ctx_size,MPI_CHAR,PetscObjectComm((PetscObject)dm)));
  CHKERRQ(PetscMalloc1(size,&n2collect));
  CHKERRQ(PetscMalloc1(size,&collectlist));
  for (r=0; r<size; r++) {
    PetscInt _n2collect;
    PetscInt *_collectlist;
    void     *_ctx_r;

    _n2collect   = 0;
    _collectlist = NULL;
    if (r != rank) { /* don't collect data from yourself */
      _ctx_r = (void*)( (char*)ctxlist + r * ctx_size);
      CHKERRQ(collect(dm,_ctx_r,&_n2collect,&_collectlist));
    }
    n2collect[r]   = _n2collect;
    collectlist[r] = _collectlist;
  }
  CHKERRQ(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de));
  /* Define topology */
  CHKERRQ(DMSwarmDataExTopologyInitialize(de));
  for (r=0; r<size; r++) {
    if (n2collect[r] > 0) {
      CHKERRQ(DMSwarmDataExTopologyAddNeighbour(de,(PetscMPIInt)r));
    }
  }
  CHKERRQ(DMSwarmDataExTopologyFinalize(de));
  /* Define send counts */
  CHKERRQ(DMSwarmDataExInitializeSendCount(de));
  for (r=0; r<size; r++) {
    if (n2collect[r] > 0) {
      CHKERRQ(DMSwarmDataExAddToSendCount(de,r,n2collect[r]));
    }
  }
  CHKERRQ(DMSwarmDataExFinalizeSendCount(de));
  /* Pack data */
  CHKERRQ(DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer));
  CHKERRQ(DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point));
  for (r=0; r<size; r++) {
    for (p=0; p<n2collect[r]; p++) {
      CHKERRQ(DMSwarmDataBucketFillPackedArray(swarm->db,collectlist[r][p],point_buffer));
      /* insert point buffer into the data exchanger */
      CHKERRQ(DMSwarmDataExPackData(de,r,1,point_buffer));
    }
  }
  CHKERRQ(DMSwarmDataExPackFinalize(de));
  /* Scatter */
  CHKERRQ(DMSwarmDataExBegin(de));
  CHKERRQ(DMSwarmDataExEnd(de));
  /* Collect data in DMSwarm container */
  CHKERRQ(DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points));
  CHKERRQ(DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL));
  CHKERRQ(DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    CHKERRQ(DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p));
  }
  /* Release memory */
  for (r=0; r<size; r++) {
    if (collectlist[r]) PetscFree(collectlist[r]);
  }
  CHKERRQ(PetscFree(collectlist));
  CHKERRQ(PetscFree(n2collect));
  CHKERRQ(PetscFree(ctxlist));
  CHKERRQ(DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer));
  CHKERRQ(DMSwarmDataExView(de));
  CHKERRQ(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(0);
}

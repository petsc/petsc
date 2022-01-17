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
  PetscErrorCode ierr;
  DMSwarmDataEx  de;
  PetscInt       p,npoints,*rankval,n_points_recv;
  PetscMPIInt    rank,nrank;
  void           *point_buffer,*recv_points;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);

  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0, &de);CHKERRQ(ierr);
  ierr = DMSwarmDataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    nrank = rankval[p];
    if (nrank != rank) {
      ierr = DMSwarmDataExTopologyAddNeighbour(de,nrank);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      ierr = DMSwarmDataExAddToSendCount(de,nrank,1);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      /* copy point into buffer */
      ierr = DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
      /* insert point buffer into DMSwarmDataExchanger */
      ierr = DMSwarmDataExPackData(de,nrank,1,point_buffer);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);

  if (remove_sent_points) {
    DMSwarmDataField gfield;

    ierr = DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&gfield);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldGetAccess(gfield);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldGetEntries(gfield,(void**)&rankval);CHKERRQ(ierr);

    /* remove points which left processor */
    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      nrank = rankval[p];
      if (nrank != rank) {
        /* kill point */
        ierr = DMSwarmDataFieldRestoreAccess(gfield);CHKERRQ(ierr);

        ierr = DMSwarmDataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);

        ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DMSwarmDataFieldGetAccess(gfield);CHKERRQ(ierr);
        ierr = DMSwarmDataFieldGetEntries(gfield,(void**)&rankval);CHKERRQ(ierr);
        p--; /* check replacement point */
      }
    }
    ierr = DMSwarmDataFieldRestoreEntries(gfield,(void**)&rankval);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldRestoreAccess(gfield);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataExBegin(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExEnd(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    ierr = DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataExView(de);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_DMNeighborScatter(DM dm,DM dmcell,PetscBool remove_sent_points,PetscInt *npoints_prior_migration)
{
  DM_Swarm          *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode    ierr;
  DMSwarmDataEx     de;
  PetscInt          r,p,npoints,*rankval,n_points_recv;
  PetscMPIInt       rank,_rank;
  const PetscMPIInt *neighbourranks;
  void              *point_buffer,*recv_points;
  size_t            sizeof_dmswarm_point;
  PetscInt          nneighbors;
  PetscMPIInt       mynneigh,*myneigh;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  ierr = DMGetNeighbors(dmcell,&nneighbors,&neighbourranks);CHKERRQ(ierr);
  ierr = DMSwarmDataExTopologyInitialize(de);CHKERRQ(ierr);
  for (r=0; r<nneighbors; r++) {
    _rank = neighbourranks[r];
    if ((_rank != rank) && (_rank > 0)) {
      ierr = DMSwarmDataExTopologyAddNeighbour(de,_rank);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExTopologyGetNeighbours(de,&mynneigh,&myneigh);CHKERRQ(ierr);
  ierr = DMSwarmDataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r=0; r<mynneigh; r++) {
        _rank = myneigh[r];
        ierr = DMSwarmDataExAddToSendCount(de,_rank,1);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSwarmDataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r=0; r<mynneigh; r++) {
        _rank = myneigh[r];
        /* copy point into buffer */
        ierr = DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
        /* insert point buffer into DMSwarmDataExchanger */
        ierr = DMSwarmDataExPackData(de,_rank,1,point_buffer);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSwarmDataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  if (remove_sent_points) {
    DMSwarmDataField PField;

    ierr = DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);
    /* remove points which left processor */
    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DMSwarmDataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
  ierr = DMSwarmDataBucketGetSizes(swarm->db,npoints_prior_migration,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmDataExBegin(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExEnd(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    ierr = DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_CellDMScatter(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm          *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode    ierr;
  PetscInt          p,npoints,npointsg=0,npoints2,npoints2g,*rankval,npoints_prior_migration;
  PetscSF           sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  DM                dmcell;
  Vec               pos;
  PetscBool         error_check = swarm->migrate_error_on_missing_point;
  PetscMPIInt       size,rank;

  PetscFunctionBegin;
  ierr = DMSwarmGetCellDM(dm,&dmcell);CHKERRQ(ierr);
  if (!dmcell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);

#if 1
  {
    PetscInt *p_cellid;
    PetscInt npoints_curr,range = 0;
    PetscSFNode *sf_cells;

    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(npoints_curr, &sf_cells);CHKERRQ(ierr);

    ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid);CHKERRQ(ierr);
    for (p=0; p<npoints_curr; p++) {

      sf_cells[p].rank  = 0;
      sf_cells[p].index = p_cellid[p];
      if (p_cellid[p] > range) {
        range = p_cellid[p];
      }
    }
    ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);

    /*ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm),&sfcell);CHKERRQ(ierr);*/
    ierr = PetscSFCreate(PETSC_COMM_SELF,&sfcell);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfcell, range, npoints_curr, NULL, PETSC_OWN_POINTER, sf_cells, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
#endif

  ierr = DMSwarmCreateLocalVectorFromField(dm, DMSwarmPICField_coor, &pos);CHKERRQ(ierr);
  ierr = DMLocatePoints(dmcell, pos, DM_POINTLOCATION_NONE, &sfcell);CHKERRQ(ierr);
  ierr = DMSwarmDestroyLocalVectorFromField(dm, DMSwarmPICField_coor, &pos);CHKERRQ(ierr);

  if (error_check) {
    ierr = DMSwarmGetSize(dm,&npointsg);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    rankval[p] = LA_sfcell[p].index;
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfcell);CHKERRQ(ierr);

  if (size > 1) {
    ierr = DMSwarmMigrate_DMNeighborScatter(dm,dmcell,remove_sent_points,&npoints_prior_migration);CHKERRQ(ierr);
  } else {
    DMSwarmDataField PField;
    PetscInt npoints_curr;

    /* remove points which the domain */
    ierr = DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);

    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints_curr; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DMSwarmDataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
    ierr = DMSwarmGetSize(dm,&npoints_prior_migration);CHKERRQ(ierr);

  }

  /* locate points newly received */
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);

#if 0
  { /* safe alternative - however this performs two point locations on: (i) the initial points set and; (ii) the (initial + received) point set */
    PetscScalar *LA_coor;
    PetscInt bs;
    DMSwarmDataField PField;

    ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,bs*npoints2,(const PetscScalar*)LA_coor,&pos);CHKERRQ(ierr);
    ierr = DMLocatePoints(dmcell,pos,DM_POINTLOCATION_NONE,&sfcell);CHKERRQ(ierr);

    ierr = VecDestroy(&pos);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor);CHKERRQ(ierr);

    ierr = PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    for (p=0; p<npoints2; p++) {
      rankval[p] = LA_sfcell[p].index;
    }
    ierr = PetscSFDestroy(&sfcell);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);

    /* remove points which left processor */
    ierr = DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);

    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DMSwarmDataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
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

    ierr = DMSwarmGetField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,bs*npoints_from_neighbours,(const PetscScalar*)&LA_coor[bs*npoints_prior_migration],&pos);CHKERRQ(ierr);

    ierr = DMLocatePoints(dmcell,pos,DM_POINTLOCATION_NONE,&sfcell);CHKERRQ(ierr);

    ierr = VecDestroy(&pos);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor);CHKERRQ(ierr);

    ierr = PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    for (p=0; p<npoints_from_neighbours; p++) {
      rankval[npoints_prior_migration + p] = LA_sfcell[p].index;
    }
    ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfcell);CHKERRQ(ierr);

    /* remove points which left processor */
    ierr = DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);

    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
    for (p=npoints_prior_migration; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DMSwarmDataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DMSwarmDataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }

  {
    PetscInt *p_cellid;

    ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid);CHKERRQ(ierr);
    for (p=0; p<npoints2; p++) {
      p_cellid[p] = rankval[p];
    }
    ierr = DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&p_cellid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  }

  /* check for error on removed points */
  if (error_check) {
    ierr = DMSwarmGetSize(dm,&npoints2g);CHKERRQ(ierr);
    if (npointsg != npoints2g) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Points from the DMSwarm must remain constant during migration (initial %D - final %D)",npointsg,npoints2g);
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
  PetscErrorCode ierr;
  DMSwarmDataEx  de;
  PetscInt       p,npoints,*rankval,n_points_recv;
  PetscMPIInt    rank,nrank,negrank;
  void           *point_buffer,*recv_points;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  ierr = DMSwarmDataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      ierr = DMSwarmDataExTopologyAddNeighbour(de,nrank);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      ierr = DMSwarmDataExAddToSendCount(de,nrank,1);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      rankval[p] = nrank;
      /* copy point into buffer */
      ierr = DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
      /* insert point buffer into DMSwarmDataExchanger */
      ierr = DMSwarmDataExPackData(de,nrank,1,point_buffer);CHKERRQ(ierr);
      rankval[p] = negrank;
    }
  }
  ierr = DMSwarmDataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DMSwarmDataExBegin(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExEnd(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    ierr = DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataExView(de);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscMPIInt owner_rank;
  PetscReal   min[3],max[3];
} CollectBBox;

PETSC_EXTERN PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM dm,PetscInt *globalsize)
{
  DM_Swarm *        swarm = (DM_Swarm*)dm->data;
  PetscErrorCode    ierr;
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
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);

  ierr = DMSwarmGetCellDM(dm,&dmcell);CHKERRQ(ierr);
  if (!dmcell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");
  isdmda = PETSC_FALSE;
  PetscObjectTypeCompare((PetscObject)dmcell,DMDA,&isdmda);
  if (!isdmda) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only DMDA support for CollectBoundingBox");

  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  sizeof_bbox_ctx = sizeof(CollectBBox);
  PetscMalloc1(1,&bbox);
  bbox->owner_rank = rank;

  /* compute the bounding box based on the overlapping / stenctil size */
  {
    Vec lcoor;

    ierr = DMGetCoordinatesLocal(dmcell,&lcoor);CHKERRQ(ierr);
    if (dim >= 1) {
      ierr = VecStrideMin(lcoor,0,NULL,&bbox->min[0]);CHKERRQ(ierr);
      ierr = VecStrideMax(lcoor,0,NULL,&bbox->max[0]);CHKERRQ(ierr);
    }
    if (dim >= 2) {
      ierr = VecStrideMin(lcoor,1,NULL,&bbox->min[1]);CHKERRQ(ierr);
      ierr = VecStrideMax(lcoor,1,NULL,&bbox->max[1]);CHKERRQ(ierr);
    }
    if (dim == 3) {
      ierr = VecStrideMin(lcoor,2,NULL,&bbox->min[2]);CHKERRQ(ierr);
      ierr = VecStrideMax(lcoor,2,NULL,&bbox->max[2]);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  /* use DMDA neighbours */
  ierr = DMDAGetNeighbors(dmcell,&dmneighborranks);CHKERRQ(ierr);
  if (dim == 1) {
    neighbour_cells = 3;
  } else if (dim == 2) {
    neighbour_cells = 9;
  } else {
    neighbour_cells = 27;
  }
  ierr = DMSwarmDataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p=0; p<neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) {
      ierr = DMSwarmDataExTopologyAddNeighbour(de,dmneighborranks[p]);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) {
      ierr = DMSwarmDataExAddToSendCount(de,dmneighborranks[p],1);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExFinalizeSendCount(de);CHKERRQ(ierr);
  /* send bounding boxes */
  ierr = DMSwarmDataExPackInitialize(de,sizeof_bbox_ctx);CHKERRQ(ierr);
  for (p=0; p<neighbour_cells; p++) {
    nrank = dmneighborranks[p];
    if ((nrank >= 0) && (nrank != rank)) {
      /* insert bbox buffer into DMSwarmDataExchanger */
      ierr = DMSwarmDataExPackData(de,nrank,1,bbox);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExPackFinalize(de);CHKERRQ(ierr);
  /* recv bounding boxes */
  ierr = DMSwarmDataExBegin(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExEnd(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExGetRecvData(de,&n_bbox_recv,(void**)&recv_bbox);CHKERRQ(ierr);
  /*  Wrong, should not be using PETSC_COMM_WORLD */
  for (p=0; p<n_bbox_recv; p++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[rank %d]: box from %d : range[%+1.4e,%+1.4e]x[%+1.4e,%+1.4e]\n",rank,recv_bbox[p].owner_rank,
           (double)recv_bbox[p].min[0],(double)recv_bbox[p].max[0],(double)recv_bbox[p].min[1],(double)recv_bbox[p].max[1]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout);CHKERRQ(ierr);
  /* of course this is stupid as this "generic" function should have a better way to know what the coordinates are called */
  ierr = DMSwarmDataExInitializeSendCount(de);CHKERRQ(ierr);
  for (pk=0; pk<n_bbox_recv; pk++) {
    PetscReal *array_x,*array_y;

    ierr = DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) {
          ierr = DMSwarmDataExAddToSendCount(de,recv_bbox[pk].owner_rank,1);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (pk=0; pk<n_bbox_recv; pk++) {
    PetscReal *array_x,*array_y;

    ierr = DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) {
          /* copy point into buffer */
          ierr = DMSwarmDataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
          /* insert point buffer into DMSwarmDataExchanger */
          ierr = DMSwarmDataExPackData(de,recv_bbox[pk].owner_rank,1,point_buffer);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DMSwarmDataExBegin(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExEnd(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    ierr = DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  PetscFree(bbox);
  ierr = DMSwarmDataExView(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExDestroy(de);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DMSwarmDataEx  de;
  PetscInt       p,r,npoints,n_points_recv;
  PetscMPIInt    size,rank;
  void           *point_buffer,*recv_points;
  void           *ctxlist;
  PetscInt       *n2collect,**collectlist;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  /* Broadcast user context */
  PetscMalloc(ctx_size*size,&ctxlist);
  ierr = MPI_Allgather(ctx,ctx_size,MPI_CHAR,ctxlist,ctx_size,MPI_CHAR,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
  ierr = PetscMalloc1(size,&n2collect);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&collectlist);CHKERRQ(ierr);
  for (r=0; r<size; r++) {
    PetscInt _n2collect;
    PetscInt *_collectlist;
    void     *_ctx_r;

    _n2collect   = 0;
    _collectlist = NULL;
    if (r != rank) { /* don't collect data from yourself */
      _ctx_r = (void*)( (char*)ctxlist + r * ctx_size);
      ierr = collect(dm,_ctx_r,&_n2collect,&_collectlist);CHKERRQ(ierr);
    }
    n2collect[r]   = _n2collect;
    collectlist[r] = _collectlist;
  }
  ierr = DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  /* Define topology */
  ierr = DMSwarmDataExTopologyInitialize(de);CHKERRQ(ierr);
  for (r=0; r<size; r++) {
    if (n2collect[r] > 0) {
      ierr = DMSwarmDataExTopologyAddNeighbour(de,(PetscMPIInt)r);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExTopologyFinalize(de);CHKERRQ(ierr);
  /* Define send counts */
  ierr = DMSwarmDataExInitializeSendCount(de);CHKERRQ(ierr);
  for (r=0; r<size; r++) {
    if (n2collect[r] > 0) {
      ierr = DMSwarmDataExAddToSendCount(de,r,n2collect[r]);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExFinalizeSendCount(de);CHKERRQ(ierr);
  /* Pack data */
  ierr = DMSwarmDataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (r=0; r<size; r++) {
    for (p=0; p<n2collect[r]; p++) {
      ierr = DMSwarmDataBucketFillPackedArray(swarm->db,collectlist[r][p],point_buffer);CHKERRQ(ierr);
      /* insert point buffer into the data exchanger */
      ierr = DMSwarmDataExPackData(de,r,1,point_buffer);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmDataExPackFinalize(de);CHKERRQ(ierr);
  /* Scatter */
  ierr = DMSwarmDataExBegin(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExEnd(de);CHKERRQ(ierr);
  /* Collect data in DMSwarm container */
  ierr = DMSwarmDataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketSetSizes(swarm->db,npoints + n_points_recv,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point);

    ierr = DMSwarmDataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  /* Release memory */
  for (r=0; r<size; r++) {
    if (collectlist[r]) PetscFree(collectlist[r]);
  }
  ierr = PetscFree(collectlist);CHKERRQ(ierr);
  ierr = PetscFree(n2collect);CHKERRQ(ierr);
  ierr = PetscFree(ctxlist);CHKERRQ(ierr);
  ierr = DMSwarmDataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DMSwarmDataExView(de);CHKERRQ(ierr);
  ierr = DMSwarmDataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

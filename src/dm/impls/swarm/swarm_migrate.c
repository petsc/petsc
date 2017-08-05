#include <petscsf.h>
#include <petscdmswarm.h>
#include <petscdmda.h>
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include "data_bucket.h"
#include "data_ex.h"

/*
 User loads desired location (MPI rank) into field DMSwarm_rank
*/
PetscErrorCode DMSwarmMigrate_Push_Basic(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataEx de;
  PetscInt p,npoints,*rankval,n_points_recv;
  PetscMPIInt rank,nrank;
  void *point_buffer,*recv_points;
  size_t sizeof_dmswarm_point;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DataExCreate(PetscObjectComm((PetscObject)dm),0, &de);CHKERRQ(ierr);
  ierr = DataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    nrank = rankval[p];
    if (nrank != rank) {
      ierr = DataExTopologyAddNeighbour(de,nrank);CHKERRQ(ierr);
    }
  }
  ierr = DataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      ierr = DataExAddToSendCount(de,nrank,1);CHKERRQ(ierr);
    }
  }
  ierr = DataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      /* copy point into buffer */
      ierr = DataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
      /* insert point buffer into DataExchanger */
      ierr = DataExPackData(de,nrank,1,point_buffer);CHKERRQ(ierr);
    }
  }
  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);

  if (remove_sent_points) {
    DataField gfield;

    ierr = DataBucketGetDataFieldByName(swarm->db,DMSwarmField_rank,&gfield);CHKERRQ(ierr);
    ierr = DataFieldGetAccess(gfield);CHKERRQ(ierr);
    ierr = DataFieldGetEntries(gfield,(void**)&rankval);CHKERRQ(ierr);

    /* remove points which left processor */
    ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      nrank = rankval[p];
      if (nrank != rank) {
        /* kill point */
        ierr = DataFieldRestoreAccess(gfield);CHKERRQ(ierr);
        
        ierr = DataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);

        ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DataFieldGetAccess(gfield);CHKERRQ(ierr);
        ierr = DataFieldGetEntries(gfield,(void**)&rankval);CHKERRQ(ierr);
        p--; /* check replacement point */
      }
    }
    ierr = DataFieldRestoreEntries(gfield,(void**)&rankval);CHKERRQ(ierr);
    ierr = DataFieldRestoreAccess(gfield);CHKERRQ(ierr);
  }
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );

    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DataExView(de);CHKERRQ(ierr);
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_DMNeighborScatter(DM dm,DM dmcell,PetscBool remove_sent_points,PetscInt *npoints_prior_migration)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataEx de;
  PetscInt r,p,npoints,*rankval,n_points_recv;
  PetscMPIInt rank,_rank;
  const PetscMPIInt *neighbourranks;
  void *point_buffer,*recv_points;
  size_t sizeof_dmswarm_point;
  PetscInt nneighbors;
  PetscMPIInt mynneigh,*myneigh;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  ierr = DMGetNeighbors(dmcell,&nneighbors,&neighbourranks);CHKERRQ(ierr);
  ierr = DataExTopologyInitialize(de);CHKERRQ(ierr);
  for (r=0; r<nneighbors; r++) {
    _rank = neighbourranks[r];
    if ((_rank != rank) && (_rank > 0)) {
      ierr = DataExTopologyAddNeighbour(de,_rank);CHKERRQ(ierr);
    }
  }
  ierr = DataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DataExTopologyGetNeighbours(de,&mynneigh,&myneigh);CHKERRQ(ierr);
  ierr = DataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r=0; r<mynneigh; r++) {
        _rank = myneigh[r];
        ierr = DataExAddToSendCount(de,_rank,1);CHKERRQ(ierr);
      }
    }
  }
  ierr = DataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r=0; r<mynneigh; r++) {
        _rank = myneigh[r];
        /* copy point into buffer */
        ierr = DataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
        /* insert point buffer into DataExchanger */
        ierr = DataExPackData(de,_rank,1,point_buffer);CHKERRQ(ierr);
      }
    }
  }
  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  if (remove_sent_points) {
    DataField PField;

    ierr = DataBucketGetDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);
    /* remove points which left processor */
    ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
  ierr = DataBucketGetSizes(swarm->db,npoints_prior_migration,NULL,NULL);CHKERRQ(ierr);
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );

    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmMigrate_CellDMScatter(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscInt p,npoints,npointsg=0,npoints2,npoints2g,*rankval,npoints_prior_migration;
  PetscSF sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  DM dmcell;
  Vec pos;
  PetscBool error_check = swarm->migrate_error_on_missing_point;
  PetscMPIInt commsize,rank;

  PetscFunctionBegin;
  ierr = DMSwarmGetCellDM(dm,&dmcell);CHKERRQ(ierr);
  if (!dmcell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

#if 1
  {
    PetscInt *p_cellid;
    PetscInt npoints_curr,range = 0;
    PetscSFNode *sf_cells;

    
    ierr = DataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL);CHKERRQ(ierr);
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
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    rankval[p] = LA_sfcell[p].index;
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfcell);CHKERRQ(ierr);

  if (commsize > 1) {
    ierr = DMSwarmMigrate_DMNeighborScatter(dm,dmcell,remove_sent_points,&npoints_prior_migration);CHKERRQ(ierr);
  } else {
    DataField PField;
    PetscInt npoints_curr;
    
    /* remove points which the domain */
    ierr = DataBucketGetDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);
    
    ierr = DataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints_curr; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DataBucketGetSizes(swarm->db,&npoints_curr,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
    ierr = DMSwarmGetSize(dm,&npoints_prior_migration);CHKERRQ(ierr);
    
  }

  /* locate points newly recevied */
  ierr = DataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
  
#if 0
  { /* safe alternative - however this performs two point locations on: (i) the intial points set and; (ii) the (intial + recieved) point set */
    PetscScalar *LA_coor;
    PetscInt bs;
    DataField PField;

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
    ierr = DataBucketGetDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);

    ierr = DataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
#endif

  { /* this performs two point locations: (i) on the intial points set prior to communication; and (ii) on the new (recieved) points */
    PetscScalar *LA_coor;
    PetscInt npoints_from_neighbours,bs;
    DataField PField;
    
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
    ierr = DataBucketGetDataFieldByName(swarm->db,DMSwarmField_rank,&PField);CHKERRQ(ierr);
    ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr);
    
    ierr = DataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
    for (p=npoints_prior_migration; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        ierr = DataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        ierr = DataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        ierr = DataFieldGetEntries(PField,(void**)&rankval);CHKERRQ(ierr); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
  
  {
    PetscInt *p_cellid;
    
    ierr = DataBucketGetSizes(swarm->db,&npoints2,NULL,NULL);CHKERRQ(ierr);
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
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataEx de;
  PetscInt p,npoints,*rankval,n_points_recv;
  PetscMPIInt rank,nrank,negrank;
  void *point_buffer,*recv_points;
  size_t sizeof_dmswarm_point;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  ierr = DataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      ierr = DataExTopologyAddNeighbour(de,nrank);CHKERRQ(ierr);
    }
  }
  ierr = DataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      ierr = DataExAddToSendCount(de,nrank,1);CHKERRQ(ierr);
    }
  }
  ierr = DataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      rankval[p] = nrank;
      /* copy point into buffer */
      ierr = DataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
      /* insert point buffer into DataExchanger */
      ierr = DataExPackData(de,nrank,1,point_buffer);CHKERRQ(ierr);
      rankval[p] = negrank;
    }
  }
  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );

    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DataExView(de);CHKERRQ(ierr);
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscMPIInt owner_rank;
  PetscReal min[3],max[3];
} CollectBBox;

PETSC_EXTERN PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM dm,PetscInt *globalsize)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataEx de;
  PetscInt p,pk,npoints,*rankval,n_points_recv,n_bbox_recv,dim,neighbour_cells;
  PetscMPIInt rank,nrank;
  void *point_buffer,*recv_points;
  size_t sizeof_dmswarm_point,sizeof_bbox_ctx;
  PetscBool isdmda;
  CollectBBox *bbox,*recv_bbox;
  const PetscMPIInt *dmneighborranks;
  DM dmcell;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

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
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  ierr = DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  /* use DMDA neighbours */
  ierr = DMDAGetNeighbors(dmcell,&dmneighborranks);CHKERRQ(ierr);
  if (dim == 1) {
    neighbour_cells = 3;
  } else if (dim == 2) {
    neighbour_cells = 9;
  } else {
    neighbour_cells = 27;
  }
  ierr = DataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p=0; p<neighbour_cells; p++) {
    if ( (dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank) ) {
      ierr = DataExTopologyAddNeighbour(de,dmneighborranks[p]);CHKERRQ(ierr);
    }
  }
  ierr = DataExTopologyFinalize(de);CHKERRQ(ierr);
  ierr = DataExInitializeSendCount(de);CHKERRQ(ierr);
  for (p=0; p<neighbour_cells; p++) {
    if ( (dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank) ) {
      ierr = DataExAddToSendCount(de,dmneighborranks[p],1);CHKERRQ(ierr);
    }
  }
  ierr = DataExFinalizeSendCount(de);CHKERRQ(ierr);
  /* send bounding boxes */
  ierr = DataExPackInitialize(de,sizeof_bbox_ctx);CHKERRQ(ierr);
  for (p=0; p<neighbour_cells; p++) {
    nrank = dmneighborranks[p];
    if ( (nrank >= 0) && (nrank != rank) ) {
      /* insert bbox buffer into DataExchanger */
      ierr = DataExPackData(de,nrank,1,bbox);CHKERRQ(ierr);
    }
  }
  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  /* recv bounding boxes */
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  ierr = DataExGetRecvData(de,&n_bbox_recv,(void**)&recv_bbox);CHKERRQ(ierr);
  for (p=0; p<n_bbox_recv; p++) {
    PetscPrintf(PETSC_COMM_SELF,"[rank %d]: box from %d : range[%+1.4e,%+1.4e]x[%+1.4e,%+1.4e]\n",rank,recv_bbox[p].owner_rank,
           (double)recv_bbox[p].min[0],(double)recv_bbox[p].max[0],(double)recv_bbox[p].min[1],(double)recv_bbox[p].max[1]);
  }
  /* of course this is stupid as this "generic" function should have a better way to know what the coordinates are called */
  ierr = DataExInitializeSendCount(de);CHKERRQ(ierr);
  for (pk=0; pk<n_bbox_recv; pk++) {
    PetscReal *array_x,*array_y;

    ierr = DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0]) ) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1]) ) {
          ierr = DataExAddToSendCount(de,recv_bbox[pk].owner_rank,1);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  }
  ierr = DataExFinalizeSendCount(de);CHKERRQ(ierr);
  ierr = DataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (pk=0; pk<n_bbox_recv; pk++) {
    PetscReal *array_x,*array_y;

    ierr = DMSwarmGetField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0]) ) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1]) ) {
          /* copy point into buffer */
          ierr = DataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
          /* insert point buffer into DataExchanger */
          ierr = DataExPackData(de,recv_bbox[pk].owner_rank,1,point_buffer);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  }
  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );

    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  PetscFree(bbox);
  ierr = DataExView(de);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
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
  DataEx         de;
  PetscInt       p,r,npoints,n_points_recv;
  PetscMPIInt    commsize,rank;
  void           *point_buffer,*recv_points;
  void           *ctxlist;
  PetscInt       *n2collect,**collectlist;
  size_t         sizeof_dmswarm_point;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  /* Broadcast user context */
  PetscMalloc(ctx_size*commsize,&ctxlist);
  ierr = MPI_Allgather(ctx,ctx_size,MPI_CHAR,ctxlist,ctx_size,MPI_CHAR,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = PetscMalloc1(commsize,&n2collect);CHKERRQ(ierr);
  ierr = PetscMalloc1(commsize,&collectlist);CHKERRQ(ierr);
  for (r=0; r<commsize; r++) {
    PetscInt _n2collect;
    PetscInt *_collectlist;
    void     *_ctx_r;

    _n2collect   = 0;
    _collectlist = NULL;
    if (r != rank) { /* don't collect data from yourself */
      _ctx_r = (void*)( (char*)ctxlist + r * ctx_size );
      ierr = collect(dm,_ctx_r,&_n2collect,&_collectlist);CHKERRQ(ierr);
    }
    n2collect[r]   = _n2collect;
    collectlist[r] = _collectlist;
  }
  ierr = DataExCreate(PetscObjectComm((PetscObject)dm),0,&de);CHKERRQ(ierr);
  /* Define topology */
  ierr = DataExTopologyInitialize(de);CHKERRQ(ierr);
  for (r=0; r<commsize; r++) {
    if (n2collect[r] > 0) {
      ierr = DataExTopologyAddNeighbour(de,(PetscMPIInt)r);CHKERRQ(ierr);
    }
  }
  ierr = DataExTopologyFinalize(de);CHKERRQ(ierr);
  /* Define send counts */
  ierr = DataExInitializeSendCount(de);CHKERRQ(ierr);
  for (r=0; r<commsize; r++) {
    if (n2collect[r] > 0) {
      ierr = DataExAddToSendCount(de,r,n2collect[r]);CHKERRQ(ierr);
    }
  }
  ierr = DataExFinalizeSendCount(de);CHKERRQ(ierr);
  /* Pack data */
  ierr = DataBucketCreatePackedArray(swarm->db,&sizeof_dmswarm_point,&point_buffer);CHKERRQ(ierr);
  ierr = DataExPackInitialize(de,sizeof_dmswarm_point);CHKERRQ(ierr);
  for (r=0; r<commsize; r++) {
    for (p=0; p<n2collect[r]; p++) {
      ierr = DataBucketFillPackedArray(swarm->db,collectlist[r][p],point_buffer);CHKERRQ(ierr);
      /* insert point buffer into the data exchanger */
      ierr = DataExPackData(de,r,1,point_buffer);CHKERRQ(ierr);
    }
  }
  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  /* Scatter */
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  /* Collect data in DMSwarm container */
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );

    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  /* Release memory */
  for (r=0; r<commsize; r++) {
    if (collectlist[r]) PetscFree(collectlist[r]);
  }
  ierr = PetscFree(collectlist);CHKERRQ(ierr);
  ierr = PetscFree(n2collect);CHKERRQ(ierr);
  ierr = PetscFree(ctxlist);CHKERRQ(ierr);
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DataExView(de);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


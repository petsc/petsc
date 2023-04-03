#include <petscsf.h>
#include <petscdmswarm.h>
#include <petscdmda.h>
#include <petsc/private/dmswarmimpl.h> /*I   "petscdmswarm.h"   I*/
#include "../src/dm/impls/swarm/data_bucket.h"
#include "../src/dm/impls/swarm/data_ex.h"

/*
 User loads desired location (MPI rank) into field DMSwarm_rank
*/
PetscErrorCode DMSwarmMigrate_Push_Basic(DM dm, PetscBool remove_sent_points)
{
  DM_Swarm     *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx de;
  PetscInt      p, npoints, *rankval, n_points_recv;
  PetscMPIInt   rank, nrank;
  void         *point_buffer, *recv_points;
  size_t        sizeof_dmswarm_point;
  PetscBool     debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm), 0, &de));
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < npoints; ++p) {
    nrank = rankval[p];
    if (nrank != rank) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, nrank));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) PetscCall(DMSwarmDataExAddToSendCount(de, nrank, 1));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (p = 0; p < npoints; p++) {
    nrank = rankval[p];
    if (nrank != rank) {
      /* copy point into buffer */
      PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
      /* insert point buffer into DMSwarmDataExchanger */
      PetscCall(DMSwarmDataExPackData(de, nrank, 1, point_buffer));
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));

  if (remove_sent_points) {
    DMSwarmDataField gfield;

    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &gfield));
    PetscCall(DMSwarmDataFieldGetAccess(gfield));
    PetscCall(DMSwarmDataFieldGetEntries(gfield, (void **)&rankval));

    /* remove points which left processor */
    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
    for (p = 0; p < npoints; p++) {
      nrank = rankval[p];
      if (nrank != rank) {
        /* kill point */
        PetscCall(DMSwarmDataFieldRestoreAccess(gfield));

        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));

        PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetAccess(gfield));
        PetscCall(DMSwarmDataFieldGetEntries(gfield, (void **)&rankval));
        p--; /* check replacement point */
      }
    }
    PetscCall(DMSwarmDataFieldRestoreEntries(gfield, (void **)&rankval));
    PetscCall(DMSwarmDataFieldRestoreAccess(gfield));
  }
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, (void **)&recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  if (debug) PetscCall(DMSwarmDataExView(de));
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmMigrate_DMNeighborScatter(DM dm, DM dmcell, PetscBool remove_sent_points, PetscInt *npoints_prior_migration)
{
  DM_Swarm          *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx      de;
  PetscInt           r, p, npoints, *rankval, n_points_recv;
  PetscMPIInt        rank, _rank;
  const PetscMPIInt *neighbourranks;
  void              *point_buffer, *recv_points;
  size_t             sizeof_dmswarm_point;
  PetscInt           nneighbors;
  PetscMPIInt        mynneigh, *myneigh;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm), 0, &de));
  PetscCall(DMGetNeighbors(dmcell, &nneighbors, &neighbourranks));
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (r = 0; r < nneighbors; r++) {
    _rank = neighbourranks[r];
    if ((_rank != rank) && (_rank > 0)) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, _rank));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExTopologyGetNeighbours(de, &mynneigh, &myneigh));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r = 0; r < mynneigh; r++) {
        _rank = myneigh[r];
        PetscCall(DMSwarmDataExAddToSendCount(de, _rank, 1));
      }
    }
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (p = 0; p < npoints; p++) {
    if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r = 0; r < mynneigh; r++) {
        _rank = myneigh[r];
        /* copy point into buffer */
        PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
        /* insert point buffer into DMSwarmDataExchanger */
        PetscCall(DMSwarmDataExPackData(de, _rank, 1, point_buffer));
      }
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  if (remove_sent_points) {
    DMSwarmDataField PField;

    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &PField));
    PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&rankval));
    /* remove points which left processor */
    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
    for (p = 0; p < npoints; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));
        PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&rankval));      /* update date point increase realloc performed */
        p--;                                                                   /* check replacement point */
      }
    }
  }
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, npoints_prior_migration, NULL, NULL));
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, (void **)&recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmMigrate_CellDMScatter(DM dm, PetscBool remove_sent_points)
{
  DM_Swarm          *swarm = (DM_Swarm *)dm->data;
  PetscInt           p, npoints, npointsg = 0, npoints2, npoints2g, *rankval, npoints_prior_migration;
  PetscSF            sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  DM                 dmcell;
  Vec                pos;
  PetscBool          error_check = swarm->migrate_error_on_missing_point;
  PetscMPIInt        size, rank;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(dm, &dmcell));
  PetscCheck(dmcell, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only valid if cell DM provided");

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

#if 1
  {
    PetscInt    *p_cellid;
    PetscInt     npoints_curr, range = 0;
    PetscSFNode *sf_cells;

    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL));
    PetscCall(PetscMalloc1(npoints_curr, &sf_cells));

    PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    PetscCall(DMSwarmGetField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&p_cellid));
    for (p = 0; p < npoints_curr; p++) {
      sf_cells[p].rank  = 0;
      sf_cells[p].index = p_cellid[p];
      if (p_cellid[p] > range) range = p_cellid[p];
    }
    PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&p_cellid));
    PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));

    /* PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm),&sfcell)); */
    PetscCall(PetscSFCreate(PETSC_COMM_SELF, &sfcell));
    PetscCall(PetscSFSetGraph(sfcell, range, npoints_curr, NULL, PETSC_OWN_POINTER, sf_cells, PETSC_OWN_POINTER));
  }
#endif

  PetscCall(DMSwarmCreateLocalVectorFromField(dm, DMSwarmPICField_coor, &pos));
  PetscCall(DMLocatePoints(dmcell, pos, DM_POINTLOCATION_NONE, &sfcell));
  PetscCall(DMSwarmDestroyLocalVectorFromField(dm, DMSwarmPICField_coor, &pos));

  if (error_check) PetscCall(DMSwarmGetSize(dm, &npointsg));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
  for (p = 0; p < npoints; p++) rankval[p] = LA_sfcell[p].index;
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(PetscSFDestroy(&sfcell));

  if (size > 1) {
    PetscCall(DMSwarmMigrate_DMNeighborScatter(dm, dmcell, remove_sent_points, &npoints_prior_migration));
  } else {
    DMSwarmDataField PField;
    PetscInt         npoints_curr;

    /* remove points which the domain */
    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &PField));
    PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&rankval));

    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL));
    for (p = 0; p < npoints_curr; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));
        PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&rankval));           /* update date point increase realloc performed */
        p--;                                                                        /* check replacement point */
      }
    }
    PetscCall(DMSwarmGetSize(dm, &npoints_prior_migration));
  }

  /* locate points newly received */
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL));

#if 0
  { /* safe alternative - however this performs two point locations on: (i) the initial points set and; (ii) the (initial + received) point set */
    PetscScalar *LA_coor;
    PetscInt bs;
    DMSwarmDataField PField;

    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,bs,bs*npoints2,(const PetscScalar*)LA_coor,&pos));
    PetscCall(DMLocatePoints(dmcell,pos,DM_POINTLOCATION_NONE,&sfcell));

    PetscCall(VecDestroy(&pos));
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_coor,&bs,NULL,(void**)&LA_coor));

    PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
    PetscCall(DMSwarmGetField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));
    for (p=0; p<npoints2; p++) {
      rankval[p] = LA_sfcell[p].index;
    }
    PetscCall(PetscSFDestroy(&sfcell));
    PetscCall(DMSwarmRestoreField(dm,DMSwarmField_rank,NULL,NULL,(void**)&rankval));

    /* remove points which left processor */
    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db,DMSwarmField_rank,&PField));
    PetscCall(DMSwarmDataFieldGetEntries(PField,(void**)&rankval));

    PetscCall(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL));
    for (p=0; p<npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db,p));
        PetscCall(DMSwarmDataBucketGetSizes(swarm->db,&npoints2,NULL,NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetEntries(PField,(void**)&rankval)); /* update date point increase realloc performed */
        p--; /* check replacement point */
      }
    }
  }
#endif

  { /* perform two point locations: (i) on the initial points set prior to communication; and (ii) on the new (received) points */
    PetscScalar     *LA_coor;
    PetscInt         npoints_from_neighbours, bs;
    DMSwarmDataField PField;

    npoints_from_neighbours = npoints2 - npoints_prior_migration;

    PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, &bs, NULL, (void **)&LA_coor));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, bs, bs * npoints_from_neighbours, (const PetscScalar *)&LA_coor[bs * npoints_prior_migration], &pos));

    PetscCall(DMLocatePoints(dmcell, pos, DM_POINTLOCATION_NONE, &sfcell));

    PetscCall(VecDestroy(&pos));
    PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, &bs, NULL, (void **)&LA_coor));

    PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
    PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    for (p = 0; p < npoints_from_neighbours; p++) rankval[npoints_prior_migration + p] = LA_sfcell[p].index;
    PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    PetscCall(PetscSFDestroy(&sfcell));

    /* remove points which left processor */
    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &PField));
    PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&rankval));

    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL));
    for (p = npoints_prior_migration; p < npoints2; p++) {
      if (rankval[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));
        PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&rankval));       /* update date point increase realloc performed */
        p--;                                                                    /* check replacement point */
      }
    }
  }

  {
    PetscInt *p_cellid;

    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL));
    PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    PetscCall(DMSwarmGetField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&p_cellid));
    for (p = 0; p < npoints2; p++) p_cellid[p] = rankval[p];
    PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_cellid, NULL, NULL, (void **)&p_cellid));
    PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  }

  /* check for error on removed points */
  if (error_check) {
    PetscCall(DMSwarmGetSize(dm, &npoints2g));
    PetscCheck(npointsg == npoints2g, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Points from the DMSwarm must remain constant during migration (initial %" PetscInt_FMT " - final %" PetscInt_FMT ")", npointsg, npoints2g);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmMigrate_CellDMExact(DM dm, PetscBool remove_sent_points)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Redundant as this assumes points can only be sent to a single rank
*/
PetscErrorCode DMSwarmMigrate_GlobalToLocal_Basic(DM dm, PetscInt *globalsize)
{
  DM_Swarm     *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx de;
  PetscInt      p, npoints, *rankval, n_points_recv;
  PetscMPIInt   rank, nrank, negrank;
  void         *point_buffer, *recv_points;
  size_t        sizeof_dmswarm_point;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  *globalsize = npoints;
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm), 0, &de));
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      PetscCall(DMSwarmDataExTopologyAddNeighbour(de, nrank));
    }
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank = -negrank - 1;
      PetscCall(DMSwarmDataExAddToSendCount(de, nrank, 1));
    }
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (p = 0; p < npoints; p++) {
    negrank = rankval[p];
    if (negrank < 0) {
      nrank      = -negrank - 1;
      rankval[p] = nrank;
      /* copy point into buffer */
      PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
      /* insert point buffer into DMSwarmDataExchanger */
      PetscCall(DMSwarmDataExPackData(de, nrank, 1, point_buffer));
      rankval[p] = negrank;
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, (void **)&recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  PetscCall(DMSwarmDataExView(de));
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscMPIInt owner_rank;
  PetscReal   min[3], max[3];
} CollectBBox;

PETSC_EXTERN PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM dm, PetscInt *globalsize)
{
  DM_Swarm          *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx      de;
  PetscInt           p, pk, npoints, *rankval, n_points_recv, n_bbox_recv, dim, neighbour_cells;
  PetscMPIInt        rank, nrank;
  void              *point_buffer, *recv_points;
  size_t             sizeof_dmswarm_point, sizeof_bbox_ctx;
  PetscBool          isdmda;
  CollectBBox       *bbox, *recv_bbox;
  const PetscMPIInt *dmneighborranks;
  DM                 dmcell;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  PetscCall(DMSwarmGetCellDM(dm, &dmcell));
  PetscCheck(dmcell, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only valid if cell DM provided");
  isdmda = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)dmcell, DMDA, &isdmda));
  PetscCheck(isdmda, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only DMDA support for CollectBoundingBox");

  PetscCall(DMGetDimension(dm, &dim));
  sizeof_bbox_ctx = sizeof(CollectBBox);
  PetscCall(PetscMalloc1(1, &bbox));
  bbox->owner_rank = rank;

  /* compute the bounding box based on the overlapping / stenctil size */
  {
    Vec lcoor;

    PetscCall(DMGetCoordinatesLocal(dmcell, &lcoor));
    if (dim >= 1) {
      PetscCall(VecStrideMin(lcoor, 0, NULL, &bbox->min[0]));
      PetscCall(VecStrideMax(lcoor, 0, NULL, &bbox->max[0]));
    }
    if (dim >= 2) {
      PetscCall(VecStrideMin(lcoor, 1, NULL, &bbox->min[1]));
      PetscCall(VecStrideMax(lcoor, 1, NULL, &bbox->max[1]));
    }
    if (dim == 3) {
      PetscCall(VecStrideMin(lcoor, 2, NULL, &bbox->min[2]));
      PetscCall(VecStrideMax(lcoor, 2, NULL, &bbox->max[2]));
    }
  }
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  *globalsize = npoints;
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm), 0, &de));
  /* use DMDA neighbours */
  PetscCall(DMDAGetNeighbors(dmcell, &dmneighborranks));
  if (dim == 1) {
    neighbour_cells = 3;
  } else if (dim == 2) {
    neighbour_cells = 9;
  } else {
    neighbour_cells = 27;
  }
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, dmneighborranks[p]));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) PetscCall(DMSwarmDataExAddToSendCount(de, dmneighborranks[p], 1));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  /* send bounding boxes */
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_bbox_ctx));
  for (p = 0; p < neighbour_cells; p++) {
    nrank = dmneighborranks[p];
    if ((nrank >= 0) && (nrank != rank)) {
      /* insert bbox buffer into DMSwarmDataExchanger */
      PetscCall(DMSwarmDataExPackData(de, nrank, 1, bbox));
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  /* recv bounding boxes */
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_bbox_recv, (void **)&recv_bbox));
  /*  Wrong, should not be using PETSC_COMM_WORLD */
  for (p = 0; p < n_bbox_recv; p++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[rank %d]: box from %d : range[%+1.4e,%+1.4e]x[%+1.4e,%+1.4e]\n", rank, recv_bbox[p].owner_rank, (double)recv_bbox[p].min[0], (double)recv_bbox[p].max[0], (double)recv_bbox[p].min[1],
                                      (double)recv_bbox[p].max[1]));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));
  /* of course this is stupid as this "generic" function should have a better way to know what the coordinates are called */
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (pk = 0; pk < n_bbox_recv; pk++) {
    PetscReal *array_x, *array_y;

    PetscCall(DMSwarmGetField(dm, "coorx", NULL, NULL, (void **)&array_x));
    PetscCall(DMSwarmGetField(dm, "coory", NULL, NULL, (void **)&array_y));
    for (p = 0; p < npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) PetscCall(DMSwarmDataExAddToSendCount(de, recv_bbox[pk].owner_rank, 1));
      }
    }
    PetscCall(DMSwarmRestoreField(dm, "coory", NULL, NULL, (void **)&array_y));
    PetscCall(DMSwarmRestoreField(dm, "coorx", NULL, NULL, (void **)&array_x));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (pk = 0; pk < n_bbox_recv; pk++) {
    PetscReal *array_x, *array_y;

    PetscCall(DMSwarmGetField(dm, "coorx", NULL, NULL, (void **)&array_x));
    PetscCall(DMSwarmGetField(dm, "coory", NULL, NULL, (void **)&array_y));
    for (p = 0; p < npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) {
          /* copy point into buffer */
          PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
          /* insert point buffer into DMSwarmDataExchanger */
          PetscCall(DMSwarmDataExPackData(de, recv_bbox[pk].owner_rank, 1, point_buffer));
        }
      }
    }
    PetscCall(DMSwarmRestoreField(dm, "coory", NULL, NULL, (void **)&array_y));
    PetscCall(DMSwarmRestoreField(dm, "coorx", NULL, NULL, (void **)&array_x));
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, (void **)&recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(PetscFree(bbox));
  PetscCall(DMSwarmDataExView(de));
  PetscCall(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* General collection when no order, or neighbour information is provided */
/*
 User provides context and collect() method
 Broadcast user context

 for each context / rank {
   collect(swarm,context,n,list)
 }
*/
PETSC_EXTERN PetscErrorCode DMSwarmCollect_General(DM dm, PetscErrorCode (*collect)(DM, void *, PetscInt *, PetscInt **), size_t ctx_size, void *ctx, PetscInt *globalsize)
{
  DM_Swarm     *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx de;
  PetscInt      p, r, npoints, n_points_recv;
  PetscMPIInt   size, rank;
  void         *point_buffer, *recv_points;
  void         *ctxlist;
  PetscInt     *n2collect, **collectlist;
  size_t        sizeof_dmswarm_point;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  *globalsize = npoints;
  /* Broadcast user context */
  PetscCall(PetscMalloc(ctx_size * size, &ctxlist));
  PetscCallMPI(MPI_Allgather(ctx, ctx_size, MPI_CHAR, ctxlist, ctx_size, MPI_CHAR, PetscObjectComm((PetscObject)dm)));
  PetscCall(PetscMalloc1(size, &n2collect));
  PetscCall(PetscMalloc1(size, &collectlist));
  for (r = 0; r < size; r++) {
    PetscInt  _n2collect;
    PetscInt *_collectlist;
    void     *_ctx_r;

    _n2collect   = 0;
    _collectlist = NULL;
    if (r != rank) { /* don't collect data from yourself */
      _ctx_r = (void *)((char *)ctxlist + r * ctx_size);
      PetscCall(collect(dm, _ctx_r, &_n2collect, &_collectlist));
    }
    n2collect[r]   = _n2collect;
    collectlist[r] = _collectlist;
  }
  PetscCall(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm), 0, &de));
  /* Define topology */
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (r = 0; r < size; r++) {
    if (n2collect[r] > 0) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, (PetscMPIInt)r));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  /* Define send counts */
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (r = 0; r < size; r++) {
    if (n2collect[r] > 0) PetscCall(DMSwarmDataExAddToSendCount(de, r, n2collect[r]));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  /* Pack data */
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (r = 0; r < size; r++) {
    for (p = 0; p < n2collect[r]; p++) {
      PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, collectlist[r][p], point_buffer));
      /* insert point buffer into the data exchanger */
      PetscCall(DMSwarmDataExPackData(de, r, 1, point_buffer));
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  /* Scatter */
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  /* Collect data in DMSwarm container */
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, (void **)&recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  /* Release memory */
  for (r = 0; r < size; r++) {
    if (collectlist[r]) PetscCall(PetscFree(collectlist[r]));
  }
  PetscCall(PetscFree(collectlist));
  PetscCall(PetscFree(n2collect));
  PetscCall(PetscFree(ctxlist));
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(DMSwarmDataExView(de));
  PetscCall(DMSwarmDataExDestroy(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmGetMigrateType - Get the style of point migration

  Logically Collective

  Input parameter:
. dm    - the `DMSWARM`

  Output parameter:
. mtype - The migration type, see `DMSwarmMigrateType`

  Level: intermediate

.seealso: `DM`, `DMSWARM`, `DMSwarmMigrateType`, `DMSwarmGetMigrateType()`, `DMSwarmMigrateType`, `DMSwarmMigrate()`
@*/
PetscErrorCode DMSwarmGetMigrateType(DM dm, DMSwarmMigrateType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(mtype, 2);
  *mtype = ((DM_Swarm *)dm->data)->migrate_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetMigrateType - Set the style of point migration

  Logically Collective

  Input parameters:
+ dm    - the `DMSWARM`
- mtype - The migration type, see `DMSwarmMigrateType`

  Level: intermediate

.seealso: `DM`, `DMSWARM`, `DMSwarmMigrateType`, `DMSwarmGetMigrateType()`, `DMSwarmMigrateType`, `DMSwarmMigrate()`
@*/
PetscErrorCode DMSwarmSetMigrateType(DM dm, DMSwarmMigrateType mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, mtype, 2);
  ((DM_Swarm *)dm->data)->migrate_type = mtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

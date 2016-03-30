
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include "data_bucket.h"
#include "data_ex.h"


/*
 User loads desired location (MPI rank) into field DMSwarm_rank
*/
#undef __FUNCT__
#define __FUNCT__ "DMSwarmMigrate_Push_Basic"
PetscErrorCode DMSwarmMigrate_Push_Basic(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataEx de;
  PetscInt p,npoints,*rankval,n_points_recv;
  PetscMPIInt rank,nrank;
  void *point_buffer,*recv_points;
  size_t sizeof_dmswarm_point;
  
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  de = DataExCreate(PetscObjectComm((PetscObject)dm),0);CHKERRQ(ierr);
  
  ierr = DataExTopologyInitialize(de);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
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

  
  if (remove_sent_points) {
    /* remove points which left processor */
    ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      nrank = rankval[p];
      if (nrank != rank) {
        /* kill point */
        ierr = DataBucketRemovePointAtIndex(swarm->db,p);CHKERRQ(ierr);
        DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr); /* you need to update npoints as the list size decreases! */
        p--; /* check replacement point */
      }
    }		
  }
  ierr = DMSwarmRestoreField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);

	ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
	ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,-1);CHKERRQ(ierr);
	for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );
    
    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }
  ierr = DataExView(de);CHKERRQ(ierr);
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

//DMLocatePoints

#undef __FUNCT__
#define __FUNCT__ "DMSwarmMigrate_CellDM"
PetscErrorCode DMSwarmMigrate_CellDM(DM dm,PetscBool remove_sent_points)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscInt p,npoints,*rankval;
  const PetscInt *LA_iscell;
  DM dmcell;
  IS iscell;
  Vec pos;
  
  ierr = DMSwarmGetCellDM(dm,&dmcell);CHKERRQ(ierr);
  if (!dmcell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");
  
  ierr = DMSwarmCreateGlobalVectorFromField(dm,"coor",&pos);CHKERRQ(ierr);
  ierr = DMLocatePoints(dmcell,pos,&iscell);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dm,"coor",&pos);CHKERRQ(ierr);

  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = ISGetIndices(iscell,&LA_iscell);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    if (LA_iscell[p] == -1) {
      rankval[p] = -1;
    }
  }
  ierr = ISRestoreIndices(iscell,&LA_iscell);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  ierr = ISDestroy(&iscell);CHKERRQ(ierr);
  
  ierr = DMSwarmMigrate_Push_Basic(dm,remove_sent_points);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Redundant as this assumes points can only be sent to a single rank
*/
#undef __FUNCT__
#define __FUNCT__ "DMSwarmMigrate_GlobalToLocal_Basic"
PetscErrorCode DMSwarmMigrate_GlobalToLocal_Basic(DM dm,PetscInt *globalsize)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataEx de;
  PetscInt p,npoints,*rankval,n_points_recv;
  PetscMPIInt rank,nrank,negrank;
  void *point_buffer,*recv_points;
  size_t sizeof_dmswarm_point;
  
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;
  ierr = DMSwarmGetField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  de = DataExCreate(PetscObjectComm((PetscObject)dm),0);CHKERRQ(ierr);
  
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
  
  ierr = DMSwarmRestoreField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  
	ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
	ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,-1);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMSwarmCollect_DMDABoundingBox"
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
  
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

  ierr = DMSwarmGetCellDM(dm,&dmcell);CHKERRQ(ierr);
  if (!dmcell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid if cell DM provided");
  
  isdmda = PETSC_FALSE;
  PetscObjectTypeCompare((PetscObject)dmcell,DMDA,&isdmda);
  if (!isdmda) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only DMDA support for CollectBoundingBox");

  ierr = DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (dim == 1) {
    neighbour_cells = 3;
  } else if (dim == 2) {
    neighbour_cells = 9;
  } else {
    neighbour_cells = 27;
  }
  
  sizeof_bbox_ctx = sizeof(CollectBBox);
  PetscMalloc1(1,&bbox);
  bbox->owner_rank = rank;

  /* compute the bounding box based on the overlapping / stenctil size */
  //ierr = DMDAGetLocalBoundingBox(dmcell,bbox->min,bbox->max);CHKERRQ(ierr);
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
  ierr = DMSwarmGetField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  de = DataExCreate(PetscObjectComm((PetscObject)dm),0);CHKERRQ(ierr);

  /* use DMDA neighbours */
	ierr = DMDAGetNeighbors(dmcell,&dmneighborranks);CHKERRQ(ierr);

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
    printf("[rank %d]: box from %d : range[%+1.4e,%+1.4e]x[%+1.4e,%+1.4e]\n",rank,recv_bbox[p].owner_rank,
           recv_bbox[p].min[0],recv_bbox[p].max[0],recv_bbox[p].min[1],recv_bbox[p].max[1]);
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
          // copy point into buffer //
          ierr = DataBucketFillPackedArray(swarm->db,p,point_buffer);CHKERRQ(ierr);
          // insert point buffer into DataExchanger //
          ierr = DataExPackData(de,recv_bbox[pk].owner_rank,1,point_buffer);CHKERRQ(ierr);
        }
      }
    }
  
    ierr = DMSwarmRestoreField(dm,"coory",NULL,NULL,(void**)&array_y);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,"coorx",NULL,NULL,(void**)&array_x);CHKERRQ(ierr);
  }

  ierr = DataExPackFinalize(de);CHKERRQ(ierr);
  
  ierr = DMSwarmRestoreField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  ierr = DataExBegin(de);CHKERRQ(ierr);
  ierr = DataExEnd(de);CHKERRQ(ierr);
  
  ierr = DataExGetRecvData(de,&n_points_recv,(void**)&recv_points);CHKERRQ(ierr);
  
	ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
	ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,-1);CHKERRQ(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "DMSwarmCollect_General"
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
  
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  *globalsize = npoints;

  /* Broadcast user context */
  PetscMalloc(ctx_size*commsize,&ctxlist);
  ierr = MPI_Allgather(ctx,ctx_size,MPI_CHAR,ctxlist,ctx_size,MPI_CHAR,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  
  PetscMalloc1(commsize,&n2collect);
  PetscMalloc1(commsize,&collectlist);
  
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
  
  de = DataExCreate(PetscObjectComm((PetscObject)dm),0);CHKERRQ(ierr);
  
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
	ierr = DataBucketSetSizes(swarm->db,npoints + n_points_recv,-1);CHKERRQ(ierr);
	for (p=0; p<n_points_recv; p++) {
    void *data_p = (void*)( (char*)recv_points + p*sizeof_dmswarm_point );
    
    ierr = DataBucketInsertPackedArray(swarm->db,npoints+p,data_p);CHKERRQ(ierr);
  }

  /* Release memory */
  for (r=0; r<commsize; r++) {
    if (collectlist[r]) PetscFree(collectlist[r]);
  }
  PetscFree(collectlist);
  PetscFree(n2collect);
  PetscFree(ctxlist);
  ierr = DataBucketDestroyPackedArray(swarm->db,&point_buffer);CHKERRQ(ierr);
  ierr = DataExView(de);CHKERRQ(ierr);
  ierr = DataExDestroy(de);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


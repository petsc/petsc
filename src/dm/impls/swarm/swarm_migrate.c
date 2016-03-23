
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include "data_bucket.h"
#include "data_ex.h"


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

#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVES)

typedef struct {
  SFBASICHEADER;
  MPI_Comm      comms[2];       /* Communicators with distributed topology in both directions */
  PetscBool     initialized[2]; /* Are the two communicators initialized? */
  PetscMPIInt   *rootdispls,*rootcounts,*leafdispls,*leafcounts; /* displs/counts for non-distinguished ranks */
  PetscInt      rootdegree,leafdegree;
} PetscSF_Neighbor;

/*===================================================================================*/
/*              Internal utility routines                                            */
/*===================================================================================*/

/* Get the communicator with distributed graph topology, which is not cheap to build so we do it on demand (instead of at PetscSFSetUp time) */
static PetscErrorCode PetscSFGetDistComm_Neighbor(PetscSF sf,PetscSFDirection direction,MPI_Comm *distcomm)
{
  PetscErrorCode    ierr;
  PetscSF_Neighbor  *dat = (PetscSF_Neighbor*)sf->data;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,&rootranks,NULL,NULL);CHKERRQ(ierr);      /* Which ranks will access my roots (I am a destination) */
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,&leafranks,NULL,NULL,NULL);CHKERRQ(ierr); /* My leaves will access whose roots (I am a source) */

  if (!dat->initialized[direction]) {
    const PetscMPIInt indegree  = nrootranks-ndrootranks,*sources      = rootranks+ndrootranks;
    const PetscMPIInt outdegree = nleafranks-ndleafranks,*destinations = leafranks+ndleafranks;
    MPI_Comm          *mycomm   = &dat->comms[direction];
    ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
    if (direction == PETSCSF_LEAF2ROOT) {
      ierr = MPI_Dist_graph_create_adjacent(comm,indegree,sources,dat->rootcounts/*src weights*/,outdegree,destinations,dat->leafcounts/*dest weights*/,MPI_INFO_NULL,1/*reorder*/,mycomm);CHKERRMPI(ierr);
    } else { /* PETSCSF_ROOT2LEAF, reverse src & dest */
      ierr = MPI_Dist_graph_create_adjacent(comm,outdegree,destinations,dat->leafcounts/*src weights*/,indegree,sources,dat->rootcounts/*dest weights*/,MPI_INFO_NULL,1/*reorder*/,mycomm);CHKERRMPI(ierr);
    }
    dat->initialized[direction] = PETSC_TRUE;
  }
  *distcomm = dat->comms[direction];
  PetscFunctionReturn(0);
}

/*===================================================================================*/
/*              Implementations of SF public APIs                                    */
/*===================================================================================*/
static PetscErrorCode PetscSFSetUp_Neighbor(PetscSF sf)
{
  PetscErrorCode   ierr;
  PetscSF_Neighbor *dat = (PetscSF_Neighbor*)sf->data;
  PetscInt         i,j,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt   *rootoffset,*leafoffset;
  PetscMPIInt      m,n;

  PetscFunctionBegin;
  /* SFNeighbor inherits from Basic */
  ierr = PetscSFSetUp_Basic(sf);CHKERRQ(ierr);
  /* SFNeighbor specific */
  sf->persistent  = PETSC_FALSE;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
  dat->rootdegree = nrootranks-ndrootranks;
  dat->leafdegree = nleafranks-ndleafranks;
  sf->nleafreqs   = 0;
  dat->nrootreqs  = 1;

  /* Only setup MPI displs/counts for non-distinguished ranks. Distinguished ranks use shared memory */
  ierr = PetscMalloc4(dat->rootdegree,&dat->rootdispls,dat->rootdegree,&dat->rootcounts,dat->leafdegree,&dat->leafdispls,dat->leafdegree,&dat->leafcounts);CHKERRQ(ierr);
  for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
    ierr = PetscMPIIntCast(rootoffset[i]-rootoffset[ndrootranks],&m);CHKERRQ(ierr); dat->rootdispls[j] = m;
    ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],        &n);CHKERRQ(ierr); dat->rootcounts[j] = n;
  }

  for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
    ierr = PetscMPIIntCast(leafoffset[i]-leafoffset[ndleafranks],&m);CHKERRQ(ierr); dat->leafdispls[j] = m;
    ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],        &n);CHKERRQ(ierr); dat->leafcounts[j] = n;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReset_Neighbor(PetscSF sf)
{
  PetscErrorCode       ierr;
  PetscInt             i;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;

  PetscFunctionBegin;
  if (dat->inuse) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  ierr = PetscFree4(dat->rootdispls,dat->rootcounts,dat->leafdispls,dat->leafcounts);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    if (dat->initialized[i]) {
      ierr = MPI_Comm_free(&dat->comms[i]);CHKERRMPI(ierr);
      dat->initialized[i] = PETSC_FALSE;
    }
  }
  ierr = PetscSFReset_Basic(sf);CHKERRQ(ierr); /* Common part */
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFDestroy_Neighbor(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReset_Neighbor(sf);CHKERRQ(ierr);
  ierr = PetscFree(sf->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastBegin_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFLink          link;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;
  MPI_Comm             distcomm;
  void                 *rootbuf = NULL,*leafbuf = NULL;
  MPI_Request          *req;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata);CHKERRQ(ierr);
  /* Do neighborhood alltoallv for remote ranks */
  ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */);CHKERRQ(ierr);
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_ROOT2LEAF,&distcomm);CHKERRQ(ierr);
  ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL);CHKERRQ(ierr);
  ierr = PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  ierr = MPI_Start_ineighbor_alltoallv(dat->rootdegree,dat->leafdegree,rootbuf,dat->rootcounts,dat->rootdispls,unit,leafbuf,dat->leafcounts,dat->leafdispls,unit,distcomm,req);CHKERRMPI(ierr);
  ierr = PetscSFLinkScatterLocal(sf,link,PETSCSF_ROOT2LEAF,(void*)rootdata,leafdata,op);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLeafToRootBegin_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *out)
{
  PetscErrorCode       ierr;
  PetscSFLink          link;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;
  MPI_Comm             distcomm = MPI_COMM_NULL;
  void                 *rootbuf = NULL,*leafbuf = NULL;
  MPI_Request          *req = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,sfop,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata);CHKERRQ(ierr);
  /* Do neighborhood alltoallv for remote ranks */
  ierr = PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */);CHKERRQ(ierr);
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_LEAF2ROOT,&distcomm);CHKERRQ(ierr);
  ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL);CHKERRQ(ierr);
  ierr = PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_LEAF2ROOT);CHKERRQ(ierr);
  ierr = MPI_Start_ineighbor_alltoallv(dat->leafdegree,dat->rootdegree,leafbuf,dat->leafcounts,dat->leafdispls,unit,rootbuf,dat->rootcounts,dat->rootdispls,unit,distcomm,req);CHKERRMPI(ierr);
  *out = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFLink          link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLeafToRootBegin_Neighbor(sf,unit,leafmtype,leafdata,rootmtype,rootdata,op,PETSCSF_REDUCE,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkScatterLocal(sf,link,PETSCSF_LEAF2ROOT,rootdata,(void*)leafdata,op);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpBegin_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFLink          link = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLeafToRootBegin_Neighbor(sf,unit,leafmtype,leafdata,rootmtype,rootdata,op,PETSCSF_FETCH,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkFetchAndOpLocal(sf,link,rootdata,leafdata,leafupdate,op);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Neighbor(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = NULL;
  MPI_Comm          comm = MPI_COMM_NULL;
  PetscSF_Neighbor  *dat = (PetscSF_Neighbor*)sf->data;
  void              *rootbuf = NULL,*leafbuf = NULL;

  PetscFunctionBegin;
  ierr = PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkFinishCommunication(sf,link,PETSCSF_LEAF2ROOT);CHKERRQ(ierr);
  /* Process remote fetch-and-op */
  ierr = PetscSFLinkFetchAndOpRemote(sf,link,rootdata,op);CHKERRQ(ierr);
  /* Bcast the updated rootbuf back to leaves */
  ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */);CHKERRQ(ierr);
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_ROOT2LEAF,&comm);CHKERRQ(ierr);
  ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  ierr = MPI_Start_neighbor_alltoallv(dat->rootdegree,dat->leafdegree,rootbuf,dat->rootcounts,dat->rootdispls,unit,leafbuf,dat->leafcounts,dat->leafdispls,unit,comm);CHKERRMPI(ierr);
  ierr = PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_FALSE/* host2device after recving */);CHKERRQ(ierr);
  ierr = PetscSFLinkUnpackLeafData(sf,link,PETSCSF_REMOTE,leafupdate,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFLinkReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Neighbor(PetscSF sf)
{
  PetscErrorCode   ierr;
  PetscSF_Neighbor *dat;

  PetscFunctionBegin;
  sf->ops->CreateEmbeddedRootSF = PetscSFCreateEmbeddedRootSF_Basic;
  sf->ops->BcastEnd             = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd            = PetscSFReduceEnd_Basic;
  sf->ops->GetLeafRanks         = PetscSFGetLeafRanks_Basic;
  sf->ops->View                 = PetscSFView_Basic;

  sf->ops->SetUp                = PetscSFSetUp_Neighbor;
  sf->ops->Reset                = PetscSFReset_Neighbor;
  sf->ops->Destroy              = PetscSFDestroy_Neighbor;
  sf->ops->BcastBegin           = PetscSFBcastBegin_Neighbor;
  sf->ops->ReduceBegin          = PetscSFReduceBegin_Neighbor;
  sf->ops->FetchAndOpBegin      = PetscSFFetchAndOpBegin_Neighbor;
  sf->ops->FetchAndOpEnd        = PetscSFFetchAndOpEnd_Neighbor;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
#endif

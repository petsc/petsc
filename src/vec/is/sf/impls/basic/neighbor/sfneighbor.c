#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVES)

/* SF Neighbor completely relies on the two sided info built by SF Basic. Therefore we build Neighbor as a subclass of Basic */

typedef struct {
  SPPACKBASICHEADER;
} *PetscSFPack_Neighbor;

typedef struct {
  SFBASICHEADER;
  MPI_Comm      comms[2];       /* Communicators with distributed topology in both directions */
  PetscBool     initialized[2]; /* Are the two communicators initialized? */
  PetscMPIInt   *rootdispls,*rootcounts,*leafdispls,*leafcounts; /* displs/counts for non-distinguished ranks */
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
    if (direction == PETSCSF_LEAF2ROOT_REDUCE) {
      ierr = MPI_Dist_graph_create_adjacent(comm,indegree,sources,dat->rootcounts/*src weights*/,outdegree,destinations,dat->leafcounts/*dest weights*/,MPI_INFO_NULL,1/*reorder*/,mycomm);CHKERRQ(ierr);
    } else { /* PETSCSF_ROOT2LEAF_BCAST, reverse src & dest */
      ierr = MPI_Dist_graph_create_adjacent(comm,outdegree,destinations,dat->leafcounts/*src weights*/,indegree,sources,dat->rootcounts/*dest weights*/,MPI_INFO_NULL,1/*reorder*/,mycomm);CHKERRQ(ierr);
    }
    dat->initialized[direction] = PETSC_TRUE;
  }
  *distcomm = dat->comms[direction];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFPackGet_Neighbor(PetscSF sf,MPI_Datatype unit,const void *rootdata,const void *leafdata,PetscSFPack_Neighbor *mylink)
{
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* Unlike SFBasic, SFNeighbor only needs one request. But we allocate two, to anticipate MPI will support persistent Neighbor requests in future. */
  ierr = PetscSFPackGet_Basic_Common(sf,unit,rootdata,leafdata,1/*half*/,(PetscSFPack_Basic*)mylink);CHKERRQ(ierr);
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
  ierr = PetscSFSetUp_Basic(sf);CHKERRQ(ierr);
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);

  /* Only setup MPI displs/counts for non-distinguished ranks. Distinguished ranks use shared memory */
  ierr = PetscMalloc4(nrootranks-ndrootranks,&dat->rootdispls,nrootranks-ndrootranks,&dat->rootcounts,nleafranks-ndleafranks,&dat->leafdispls,nleafranks-ndleafranks,&dat->leafcounts);CHKERRQ(ierr);
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
      ierr = MPI_Comm_free(&dat->comms[i]);CHKERRQ(ierr);
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

static PetscErrorCode PetscSFBcastAndOpBegin_Neighbor(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack_Neighbor link;
  PetscInt             nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt       *rootoffset,*rootloc;
  PetscMPIInt          ind,outd;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;
  MPI_Comm             distcomm;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFPackGet_Neighbor(sf,unit,rootdata,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscSFPackRootData(sf,(PetscSFPack)link,nrootranks,ndrootranks,rootoffset,rootloc,rootdata);CHKERRQ(ierr);

  /* Do neighborhood alltoallv for non-distinguished ranks */
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_ROOT2LEAF_BCAST,&distcomm);CHKERRQ(ierr);
  outd = nrootranks - ndrootranks;
  ind  = nleafranks - ndleafranks;
  ierr = MPI_Start_ineighbor_alltoallv(outd,ind,link->rootbuf,dat->rootcounts,dat->rootdispls,unit,link->leafbuf,dat->leafcounts,dat->leafdispls,unit,distcomm,&link->requests[PETSCSF_ROOT2LEAF_BCAST]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Neighbor(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscInt             nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt       *leafoffset,*leafloc;
  PetscMPIInt          ind,outd;
  PetscSFPack_Neighbor link;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;
  MPI_Comm             distcomm;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,&leafloc,NULL);CHKERRQ(ierr);
  ierr = PetscSFPackGet_Neighbor(sf,unit,rootdata,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscSFPackLeafData(sf,(PetscSFPack)link,nleafranks,ndleafranks,leafoffset,leafloc,leafdata);CHKERRQ(ierr);

  /* Do neighborhood alltoallv for non-distinguished ranks */
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_LEAF2ROOT_REDUCE,&distcomm);CHKERRQ(ierr);
  ind  = nrootranks - ndrootranks;
  outd = nleafranks - ndleafranks;
  ierr = MPI_Start_ineighbor_alltoallv(outd,ind,link->leafbuf,dat->leafcounts,dat->leafdispls,unit,link->rootbuf,dat->rootcounts,dat->rootdispls,unit,distcomm,&link->requests[PETSCSF_LEAF2ROOT_REDUCE]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Neighbor(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack_Basic link;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Comm          comm;
  PetscMPIInt       ind,outd;
  PetscSF_Neighbor  *dat = (PetscSF_Neighbor*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);

  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,&leafloc,NULL);CHKERRQ(ierr);

  /* Process local fetch-and-op */
  ierr = PetscSFFetchAndOpRootData(sf,(PetscSFPack)link,nrootranks,ndrootranks,rootoffset,rootloc,rootdata,op);CHKERRQ(ierr);

  /* Bcast the updated root buffer back to leaves */
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_ROOT2LEAF_BCAST,&comm);CHKERRQ(ierr);
  outd = nrootranks - ndrootranks;
  ind  = nleafranks - ndleafranks;
  ierr = MPI_Start_neighbor_alltoallv(outd,ind,link->rootbuf,dat->rootcounts,dat->rootdispls,unit,link->leafbuf,dat->leafcounts,dat->leafdispls,unit,comm);CHKERRQ(ierr);
  ierr = PetscSFUnpackAndOpLeafData(sf,(PetscSFPack)link,nleafranks,ndleafranks,leafoffset,leafloc,leafupdate,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Neighbor(PetscSF sf)
{
  PetscErrorCode   ierr;
  PetscSF_Neighbor *dat;

  PetscFunctionBegin;
  sf->ops->CreateEmbeddedSF     = PetscSFCreateEmbeddedSF_Basic;
  sf->ops->CreateEmbeddedLeafSF = PetscSFCreateEmbeddedLeafSF_Basic;
  sf->ops->BcastAndOpEnd        = PetscSFBcastAndOpEnd_Basic;
  sf->ops->ReduceEnd            = PetscSFReduceEnd_Basic;
  sf->ops->FetchAndOpBegin      = PetscSFFetchAndOpBegin_Basic;
  sf->ops->GetLeafRanks         = PetscSFGetLeafRanks_Basic;
  sf->ops->View                 = PetscSFView_Basic;

  sf->ops->SetUp                = PetscSFSetUp_Neighbor;
  sf->ops->Reset                = PetscSFReset_Neighbor;
  sf->ops->Destroy              = PetscSFDestroy_Neighbor;
  sf->ops->BcastAndOpBegin      = PetscSFBcastAndOpBegin_Neighbor;
  sf->ops->ReduceBegin          = PetscSFReduceBegin_Neighbor;
  sf->ops->FetchAndOpEnd        = PetscSFFetchAndOpEnd_Neighbor;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
#endif

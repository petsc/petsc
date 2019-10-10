#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVES)

typedef struct {
  SFBASICHEADER;
  MPI_Comm      comms[2];       /* Communicators with distributed topology in both directions */
  PetscBool     initialized[2]; /* Are the two communicators initialized? */
  PetscMPIInt   *rootdispls,*rootcounts,*leafdispls,*leafcounts; /* displs/counts for non-distinguished ranks */
  PetscInt      rootdegree,leafdegree; /* Number of non-distinguished root/leaf ranks, equal to outdegree or indegree in neigborhood collectives, depending on PetscSFDirection */
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

static PetscErrorCode PetscSFPackGet_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,PetscSFPack *mylink)
{
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Basic_Common(sf,unit,rootmtype,rootdata,leafmtype,leafdata,1/*nrootreqs*/,1/*nleafreqs*/,mylink);CHKERRQ(ierr);
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

  dat->rootdegree = nrootranks-ndrootranks;
  dat->leafdegree = nleafranks-ndleafranks;

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

static PetscErrorCode PetscSFBcastAndOpBegin_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack          link;
  const PetscInt       *rootloc = NULL;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;
  MPI_Comm             distcomm;
  PetscMemType         rootmtype_mpi,leafmtype_mpi; /* memtypes seen by MPI */

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Neighbor(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscSFGetRootIndicesWithMemType_Basic(sf,rootmtype,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFPackRootData(sf,link,rootloc,rootdata,PETSC_TRUE);CHKERRQ(ierr);

  /* Do neighborhood alltoallv for non-distinguished ranks */
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_ROOT2LEAF_BCAST,&distcomm);CHKERRQ(ierr);
  if (use_gpu_aware_mpi) {
    rootmtype_mpi = rootmtype;
    leafmtype_mpi = leafmtype;
  } else {
    rootmtype_mpi = PETSC_MEMTYPE_HOST;
    leafmtype_mpi = PETSC_MEMTYPE_HOST;
  }
  ierr = MPI_Start_ineighbor_alltoallv(dat->rootdegree,dat->leafdegree,link->rootbuf[rootmtype_mpi],dat->rootcounts,dat->rootdispls,unit,link->leafbuf[leafmtype_mpi],dat->leafcounts,dat->leafdispls,unit,distcomm,link->rootreqs[PETSCSF_ROOT2LEAF_BCAST][rootmtype_mpi]);CHKERRQ(ierr);
  if (rootmtype != leafmtype) {ierr = PetscMemcpyWithMemType(leafmtype,rootmtype,link->selfbuf[leafmtype],link->selfbuf[rootmtype],link->selfbuflen*link->unitbytes);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  const PetscInt       *leafloc = NULL;
  PetscSFPack          link;
  PetscSF_Neighbor     *dat = (PetscSF_Neighbor*)sf->data;
  MPI_Comm             distcomm;
  PetscMemType         rootmtype_mpi,leafmtype_mpi; /* memtypes seen by MPI */

  PetscFunctionBegin;
  ierr = PetscSFGetLeafIndicesWithMemType_Basic(sf,leafmtype,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFPackGet_Neighbor(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscSFPackLeafData(sf,link,leafloc,leafdata,PETSC_TRUE);CHKERRQ(ierr);

  /* Do neighborhood alltoallv for non-distinguished ranks */
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_LEAF2ROOT_REDUCE,&distcomm);CHKERRQ(ierr);
  if (use_gpu_aware_mpi) {
    rootmtype_mpi = rootmtype;
    leafmtype_mpi = leafmtype;
  } else {
    rootmtype_mpi = PETSC_MEMTYPE_HOST;
    leafmtype_mpi = PETSC_MEMTYPE_HOST;
  }
  ierr = MPI_Start_ineighbor_alltoallv(dat->leafdegree,dat->rootdegree,link->leafbuf[leafmtype_mpi],dat->leafcounts,dat->leafdispls,unit,link->rootbuf[rootmtype_mpi],dat->rootcounts,dat->rootdispls,unit,distcomm,link->rootreqs[PETSCSF_LEAF2ROOT_REDUCE][rootmtype_mpi]);CHKERRQ(ierr);
  if (rootmtype != leafmtype) {ierr = PetscMemcpyWithMemType(rootmtype,leafmtype,link->selfbuf[rootmtype],link->selfbuf[leafmtype],link->selfbuflen*link->unitbytes);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Neighbor(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack       link;
  const PetscInt    *rootloc = NULL,*leafloc = NULL;
  MPI_Comm          comm;
  PetscSF_Neighbor  *dat = (PetscSF_Neighbor*)sf->data;
  PetscMemType      rootmtype_mpi,leafmtype_mpi; /* memtypes seen by MPI */

  PetscFunctionBegin;
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);
  ierr = PetscSFGetRootIndicesWithMemType_Basic(sf,rootmtype,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFGetLeafIndicesWithMemType_Basic(sf,leafmtype,&leafloc);CHKERRQ(ierr);
  /* Process local fetch-and-op */
  ierr = PetscSFFetchAndOpRootData(sf,link,rootloc,rootdata,op,PETSC_TRUE);CHKERRQ(ierr);

  /* Bcast the updated rootbuf back to leaves */
  ierr = PetscSFGetDistComm_Neighbor(sf,PETSCSF_ROOT2LEAF_BCAST,&comm);CHKERRQ(ierr);
  if (use_gpu_aware_mpi) {
    rootmtype_mpi = rootmtype;
    leafmtype_mpi = leafmtype;
  } else {
    rootmtype_mpi = PETSC_MEMTYPE_HOST;
    leafmtype_mpi = PETSC_MEMTYPE_HOST;
  }
  ierr = MPI_Start_neighbor_alltoallv(dat->rootdegree,dat->leafdegree,link->rootbuf[rootmtype_mpi],dat->rootcounts,dat->rootdispls,unit,link->leafbuf[leafmtype_mpi],dat->leafcounts,dat->leafdispls,unit,comm);CHKERRQ(ierr);
  if (rootmtype != leafmtype) {ierr = PetscMemcpyWithMemType(leafmtype,rootmtype,link->selfbuf[leafmtype],link->selfbuf[rootmtype],link->selfbuflen*link->unitbytes);CHKERRQ(ierr);}
  ierr = PetscSFUnpackAndOpLeafData(sf,link,leafloc,leafupdate,MPIU_REPLACE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);
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

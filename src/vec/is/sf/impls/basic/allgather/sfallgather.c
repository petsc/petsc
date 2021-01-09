#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Allgather on rank != 0, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Allgather;

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpBegin_Gather(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);

PetscErrorCode PetscSFSetUp_Allgather(PetscSF sf)
{
  PetscInt              i;
  PetscSF_Allgather     *dat = (PetscSF_Allgather*)sf->data;

  PetscFunctionBegin;
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    sf->leafbuflen[i]  = 0;
    sf->leafstart[i]   = 0;
    sf->leafcontig[i]  = PETSC_TRUE;
    sf->leafdups[i]    = PETSC_FALSE;
    dat->rootbuflen[i] = 0;
    dat->rootstart[i]  = 0;
    dat->rootcontig[i] = PETSC_TRUE;
    dat->rootdups[i]   = PETSC_FALSE;
  }

  sf->leafbuflen[PETSCSF_REMOTE]  = sf->nleaves;
  dat->rootbuflen[PETSCSF_REMOTE] = sf->nroots;
  sf->persistent = PETSC_FALSE;
  sf->nleafreqs  = 0; /* MPI collectives only need one request. We treat it as a root request. */
  dat->nrootreqs = 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastAndOpBegin_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode        ierr;
  PetscSFLink           link;
  PetscMPIInt           sendcount;
  MPI_Comm              comm;
  void                  *rootbuf = NULL,*leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request           *req;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
  ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL);CHKERRQ(ierr);
  ierr = MPIU_Iallgather(rootbuf,sendcount,unit,leafbuf,sendcount,unit,comm,req);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode        ierr;
  PetscSFLink           link;
  PetscInt              rstart;
  MPI_Comm              comm;
  PetscMPIInt           rank,count,recvcount;
  void                  *rootbuf = NULL,*leafbuf = NULL; /* buffer seen by MPI */
  PetscSF_Allgather     *dat = (PetscSF_Allgather*)sf->data;
  MPI_Request           *req;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_REDUCE,&link);CHKERRQ(ierr);
  if (op == MPIU_REPLACE) {
    /* REPLACE is only meaningful when all processes have the same leafdata to reduce. Therefore copy from local leafdata is fine */
    ierr = PetscLayoutGetRange(sf->map,&rstart,NULL);CHKERRQ(ierr);
    ierr = (*link->Memcpy)(link,rootmtype,rootdata,leafmtype,(const char*)leafdata+(size_t)rstart*link->unitbytes,(size_t)sf->nroots*link->unitbytes);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE)
    if (PetscMemTypeHost(rootmtype)  && PetscMemTypeDevice(leafmtype)) {ierr = (*link->d_SyncStream)(link);CHKERRQ(ierr);}
#endif
  } else {
    ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata);CHKERRQ(ierr);
    ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(dat->rootbuflen[PETSCSF_REMOTE],&recvcount);CHKERRQ(ierr);
    if (!rank && !link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]) {
      ierr = PetscSFMalloc(sf,link->leafmtype_mpi,sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]);CHKERRQ(ierr);
    }
    if (!rank && link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi] == leafbuf) leafbuf = MPI_IN_PLACE;
    ierr = PetscMPIIntCast(sf->nleaves*link->bs,&count);CHKERRQ(ierr);
    ierr = MPI_Reduce(leafbuf,link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi],count,link->basicunit,op,0,comm);CHKERRMPI(ierr);
    ierr = MPIU_Iscatter(link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi],recvcount,unit,rootbuf,recvcount,unit,0/*rank 0*/,comm,req);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastToZero_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata)
{
  PetscErrorCode        ierr;
  PetscSFLink           link;
  PetscMPIInt           rank;

  PetscFunctionBegin;
  ierr = PetscSFBcastAndOpBegin_Gather(sf,unit,rootmtype,rootdata,leafmtype,leafdata,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkMPIWaitall(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRMPI(ierr);
  if (!rank && leafmtype == PETSC_MEMTYPE_DEVICE && !sf->use_gpu_aware_mpi) {
    ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,leafdata,PETSC_MEMTYPE_HOST,link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST],sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes);CHKERRQ(ierr);
  }
  ierr = PetscSFLinkReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Allgather(PetscSF sf)
{
  PetscErrorCode    ierr;
  PetscSF_Allgather *dat = (PetscSF_Allgather*)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastAndOpEnd   = PetscSFBcastAndOpEnd_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Basic;

  /* Inherit from Allgatherv */
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;

  /* Allgather stuff */
  sf->ops->SetUp           = PetscSFSetUp_Allgather;
  sf->ops->BcastAndOpBegin = PetscSFBcastAndOpBegin_Allgather;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Allgather;
  sf->ops->BcastToZero     = PetscSFBcastToZero_Allgather;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}

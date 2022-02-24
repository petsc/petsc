#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Allgather on rank != 0, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Allgather;

PETSC_INTERN PetscErrorCode PetscSFBcastBegin_Gather(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);

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

static PetscErrorCode PetscSFBcastBegin_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSFLink           link;
  PetscMPIInt           sendcount;
  MPI_Comm              comm;
  void                  *rootbuf = NULL,*leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request           *req;

  PetscFunctionBegin;
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link));
  CHKERRQ(PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata));
  CHKERRQ(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  CHKERRQ(PetscMPIIntCast(sf->nroots,&sendcount));
  CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL));
  CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF));
  CHKERRMPI(MPIU_Iallgather(rootbuf,sendcount,unit,leafbuf,sendcount,unit,comm,req));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSFLink           link;
  PetscInt              rstart;
  MPI_Comm              comm;
  PetscMPIInt           rank,count,recvcount;
  void                  *rootbuf = NULL,*leafbuf = NULL; /* buffer seen by MPI */
  PetscSF_Allgather     *dat = (PetscSF_Allgather*)sf->data;
  MPI_Request           *req;

  PetscFunctionBegin;
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_REDUCE,&link));
  if (op == MPI_REPLACE) {
    /* REPLACE is only meaningful when all processes have the same leafdata to reduce. Therefore copy from local leafdata is fine */
    CHKERRQ(PetscLayoutGetRange(sf->map,&rstart,NULL));
    CHKERRQ((*link->Memcpy)(link,rootmtype,rootdata,leafmtype,(const char*)leafdata+(size_t)rstart*link->unitbytes,(size_t)sf->nroots*link->unitbytes));
    if (PetscMemTypeDevice(leafmtype) && PetscMemTypeHost(rootmtype)) CHKERRQ((*link->SyncStream)(link)); /* Sync the device to host memcpy */
  } else {
    CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
    CHKERRQ(PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata));
    CHKERRQ(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
    CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL));
    CHKERRQ(PetscMPIIntCast(dat->rootbuflen[PETSCSF_REMOTE],&recvcount));
    if (rank == 0 && !link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]) {
      CHKERRQ(PetscSFMalloc(sf,link->leafmtype_mpi,sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]));
    }
    if (rank == 0 && link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi] == leafbuf) leafbuf = MPI_IN_PLACE;
    CHKERRQ(PetscMPIIntCast(sf->nleaves*link->bs,&count));
    CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_LEAF2ROOT));
    CHKERRMPI(MPI_Reduce(leafbuf,link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi],count,link->basicunit,op,0,comm)); /* Must do reduce with MPI builltin datatype basicunit */
    CHKERRMPI(MPIU_Iscatter(link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi],recvcount,unit,rootbuf,recvcount,unit,0/*rank 0*/,comm,req));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastToZero_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata)
{
  PetscSFLink           link;
  PetscMPIInt           rank;

  PetscFunctionBegin;
  CHKERRQ(PetscSFBcastBegin_Gather(sf,unit,rootmtype,rootdata,leafmtype,leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link));
  CHKERRQ(PetscSFLinkFinishCommunication(sf,link,PETSCSF_ROOT2LEAF));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank));
  if (rank == 0 && PetscMemTypeDevice(leafmtype) && !sf->use_gpu_aware_mpi) {
    CHKERRQ((*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,leafdata,PETSC_MEMTYPE_HOST,link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST],sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes));
  }
  CHKERRQ(PetscSFLinkReclaim(sf,&link));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Allgather(PetscSF sf)
{
  PetscSF_Allgather *dat = (PetscSF_Allgather*)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Allgatherv;

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
  sf->ops->BcastBegin      = PetscSFBcastBegin_Allgather;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Allgather;
  sf->ops->BcastToZero     = PetscSFBcastToZero_Allgather;

  CHKERRQ(PetscNewLog(sf,&dat));
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}

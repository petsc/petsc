
#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>

/* Reuse the type. The difference is some fields (displs, recvcounts) are only significant
   on rank 0 in Gatherv. On other ranks they are harmless NULL.
 */
typedef PetscSF_Allgatherv PetscSF_Gatherv;

PETSC_INTERN PetscErrorCode PetscSFBcastBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSFLink          link;
  PetscMPIInt          sendcount;
  MPI_Comm             comm;
  PetscSF_Gatherv      *dat = (PetscSF_Gatherv*)sf->data;
  void                 *rootbuf = NULL,*leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request          *req;

  PetscFunctionBegin;
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link));
  CHKERRQ(PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata));
  CHKERRQ(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  CHKERRQ(PetscMPIIntCast(sf->nroots,&sendcount));
  CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL));
  CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF));
  CHKERRMPI(MPIU_Igatherv(rootbuf,sendcount,unit,leafbuf,dat->recvcounts,dat->displs,unit,0/*rank 0*/,comm,req));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSFLink          link;
  PetscMPIInt          recvcount;
  MPI_Comm             comm;
  PetscSF_Gatherv      *dat = (PetscSF_Gatherv*)sf->data;
  void                 *rootbuf = NULL,*leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request          *req;

  PetscFunctionBegin;
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_REDUCE,&link));
  CHKERRQ(PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata));
  CHKERRQ(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  CHKERRQ(PetscMPIIntCast(sf->nroots,&recvcount));
  CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL));
  CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_LEAF2ROOT));
  CHKERRMPI(MPIU_Iscatterv(leafbuf,dat->recvcounts,dat->displs,unit,rootbuf,recvcount,unit,0,comm,req));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscFunctionBegin;
  /* In Gatherv, each root only has one leaf. So we just need to bcast rootdata to leafupdate and then reduce leafdata to rootdata */
  CHKERRQ(PetscSFBcastBegin(sf,unit,rootdata,leafupdate,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf,unit,rootdata,leafupdate,MPI_REPLACE));
  CHKERRQ(PetscSFReduceBegin(sf,unit,leafdata,rootdata,op));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Gatherv(PetscSF sf)
{
  PetscSF_Gatherv *dat = (PetscSF_Gatherv*)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Basic;

  /* Inherit from Allgatherv */
  sf->ops->SetUp           = PetscSFSetUp_Allgatherv;
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;

  /* Gatherv stuff */
  sf->ops->BcastBegin      = PetscSFBcastBegin_Gatherv;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Gatherv;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  CHKERRQ(PetscNewLog(sf,&dat));
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
